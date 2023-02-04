# Wrapper for bi-level optimization solver
import abc
import collections
import dataclasses
import logging
import math
import statistics
from collections.abc import Callable, Iterable, Iterator

import functorch
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from hypergrad.utils import Params


class Recorder(object):
    # A helper class to accumulate statistics
    def __init__(self):
        self._tmp: dict[str, list[float]] = collections.defaultdict(list)
        self._archive: dict[str, list[list[float]]] = collections.defaultdict(list)

    def add(self,
            name: str,
            value: float | Tensor
            ) -> None:
        # add value
        if isinstance(value, Tensor):
            value = value.cpu().item()
        self._tmp[name].append(value)
        self._archive[name].append(value)

    def flush(self) -> dict[str, float]:
        # average recorded values so far and clear the temporary storage
        out = {}
        for k, v in self._tmp.items():
            out[k] = statistics.mean(v)
        self._tmp = collections.defaultdict(list)
        return out

    @property
    def archive(self):
        return self._archive.copy()


@dataclasses.dataclass
class ForwardOutput:
    loss: Tensor
    output: Tensor

    def __iter__(self):
        # intended to use as `dataclasses.astuple`, but
        # it is 50 times faster than using dataclasses.astuple
        return iter(self.__dict__.values())


class BaseSolver(abc.ABC):

    def __init__(self,
                 inner: nn.Module,
                 outer: nn.Module,
                 inner_loader: Iterable,
                 outer_loader: Iterable,
                 num_iters: int,
                 unroll_steps: int,
                 inner_optimizer: Optimizer | Callable,
                 outer_optimizer: Optimizer,
                 outer_patience_iters: int = 0,
                 logger: logging.Logger = None,
                 log_freq: int = 100,
                 inner_requires_grad: bool = False,
                 outer_requires_grad: bool = False,
                 skip_sync_inner: bool = False,
                 device: torch.device | None = None,
                 ):

        if num_iters <= 0:
            raise ValueError()
        if unroll_steps <= 0:
            raise ValueError()
        if num_iters < unroll_steps:
            raise ValueError
        if outer_patience_iters < 0:
            raise ValueError()

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.inner = inner.to(self.device)
        self.outer = outer.to(self.device)
        self.outer.requires_grad_(outer_requires_grad)

        self._inner_loader = inner_loader
        self._outer_loader = outer_loader
        self._num_iters = num_iters
        self._outer_patience_iters = outer_patience_iters
        self._unroll_steps = unroll_steps
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self.logger = logger or logging.getLogger(self.__class__.__name__.lower())
        self._log_freq = log_freq

        self._inner_requires_grad = inner_requires_grad
        self._outer_requires_grad = outer_requires_grad
        self._skip_sync_inner = skip_sync_inner

        # functional inner model
        self.inner_func = None
        self.inner_params = None
        self.inner_optim_state = None

        self.reset_inner_model()

        self.recorder = Recorder()
        self._inner_step = 0
        self._outer_step = 0

    @property
    def inner_step(self) -> int:
        return self._inner_step

    @property
    def outer_step(self) -> int:
        return self._outer_step

    def stdout_results(self,
                       results: dict[str, float]
                       ) -> None:
        out = f"[inner step {self.inner_step - 1:>{int(math.log10(self._num_iters))}}] "
        out += f"(outer step {self.outer_step:>{int(math.log10(self._num_iters // self._unroll_steps))}}) "
        for k, v in results.items():
            out += f"{k}={v:.4f} / "
        self.logger.info(out[:-3])

    @property
    def inner_loader(self) -> Iterator:
        # infinite data loader
        while True:
            for data in self._inner_loader:
                yield [d.to(self.device, non_blocking=True) if isinstance(d, Tensor) else d for d in data]

    @property
    def outer_loader(self) -> Iterator:
        # infinite data loader
        while True:
            for data in self._outer_loader:
                yield [d.to(self.device, non_blocking=True) if isinstance(d, Tensor) else d for d in data]

    def run(self) -> None:
        for i in range(self._num_iters):
            self.inner_update()
            self._inner_step += 1

            if i > 0 and i >= self._outer_patience_iters and i % self._unroll_steps == 0:
                self.outer_update()
                self.post_outer_update()
                self.sync_inner()
                self._outer_step += 1

            if i > 0 and i % self._log_freq == 0:
                self.stdout_results(self.recorder.flush())

    def reset_inner_model(self):
        self.inner_func, self.inner_params = functorch.make_functional(self.inner, not self._inner_requires_grad)

    @abc.abstractmethod
    def inner_forward(self,
                      in_params: Params,
                      out_params: Params,
                      input: Tensor,
                      target: Tensor
                      ) -> ForwardOutput:
        # returns loss, output
        ...

    @abc.abstractmethod
    def inner_update(self) -> None:
        ...

    @abc.abstractmethod
    def outer_forward(self,
                      in_params: Params,
                      out_params: Params,
                      input: Tensor,
                      target: Tensor
                      ) -> ForwardOutput:
        # returns loss, output
        ...

    @abc.abstractmethod
    def outer_update(self) -> None:
        ...

    def post_outer_update(self) -> None:
        """ Clean up
        """
        ...

    def sync_inner(self) -> None:
        """ Sync the inner model in nn.Module and its functional version
        """
        if not self._skip_sync_inner:
            for p, f_p in zip(self.inner.parameters(), self.inner_params):
                p.data.copy_(f_p.data)


class BaseImplicitSolver(BaseSolver):
    def __init__(self,
                 inner: nn.Module,
                 outer: nn.Module,
                 inner_loader: Iterable,
                 outer_loader: Iterable,
                 approx_ihvp: Callable,
                 num_iters: int,
                 unroll_steps: int,
                 inner_optimizer: Optimizer | Callable,
                 outer_optimizer: Optimizer,
                 outer_patience_iters: int = 0,
                 logger: logging.Logger = None,
                 log_freq: int = 100,
                 inner_requires_grad: bool = False,
                 skip_sync_inner: bool = False,
                 device: torch.device | None = None,
                 ):
        super().__init__(inner, outer, inner_loader, outer_loader, num_iters, unroll_steps, inner_optimizer,
                         outer_optimizer, outer_patience_iters, logger, log_freq, inner_requires_grad,
                         outer_requires_grad=False, skip_sync_inner=skip_sync_inner, device=device)
        self.approx_ihvp = approx_ihvp

    @property
    def outer_grad(self):
        for p in self.outer.parameters():
            yield p.grad

    @outer_grad.setter
    def outer_grad(self,
                   grads: Params
                   ) -> None:
        for p, g in zip(self.outer.parameters(), grads):
            if p.grad is None:
                p.grad = g
            else:
                p.grad = p.grad + g
