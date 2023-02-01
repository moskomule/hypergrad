import copy
import functools
from collections.abc import Callable

import functorch
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hypergrad.approximate_ihvp import conjugate_gradient, neumann, nystrom
from hypergrad.optimizers import diff_sgd
from hypergrad.solver import BaseImplicitSolver
from hypergrad.utils import Objective, Params


# generate simple data
def generate_data(num_data: int,
                  dim_data: int
                  ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    w = torch.randn(dim_data)
    x = torch.randn(num_data, dim_data)
    y = (x @ w + 0.1 * torch.randn(num_data) > 0).float()
    train_size = int(0.5 * num_data)
    return (x[:train_size], y[:train_size][:, None]), (x[train_size:], y[train_size:][:, None])


class Solver(BaseImplicitSolver):
    def inner_update(self) -> None:
        input, target = next(self.inner_loader)
        grads, (loss, output) = functorch.grad_and_value(self.inner_obj,
                                                         has_aux=True)(self.f_params,
                                                                       tuple(self.outer.parameters()), input, target)
        self.f_params, self.f_optim_state = self.inner_optimizer(list(self.f_params), list(grads), self.f_optim_state)
        self.recorder.add('inner_loss', loss.detach())

    def inner_obj(self,
                  in_params: Params,
                  out_params: Params,
                  input: Tensor,
                  target: Tensor
                  ) -> tuple[Tensor, Tensor]:
        output = self.f_model(in_params, input)
        loss = F.binary_cross_entropy_with_logits(output, target)
        wd = sum([(_in.pow(2) * _out).sum() for _in, _out in zip(in_params, out_params)])
        return loss + wd / 2, output

    def outer_update(self) -> None:
        in_input, in_target = next(self.inner_loader)
        out_input, out_target = next(self.outer_loader)
        in_params = tuple(self.f_params)
        _, out_params = functorch.make_functional(self.outer, disable_autograd_tracking=True)

        self.recorder.add('outer_loss', self.outer_obj(in_params, out_params, out_input, out_target)[0].detach())
        ihvp, out_out_g = self.approx_ihvp(lambda i, o: self.inner_obj(i, o, in_input, in_target)[0],
                                           lambda i, o: self.outer_obj(i, o, out_input, out_target)[0],
                                           in_params, out_params)
        _, implicit_grads = functorch.jvp(
            lambda i_p: functorch.grad(lambda i, o: self.inner_obj(i, o, in_input, in_target)[0],
                                       argnums=(0, 1))(i_p, out_params)[1],
            (in_params,), (ihvp,))
        self.set_out_grad(out_out_g)
        self.set_out_grad(tuple(-g for g in implicit_grads))
        self.outer_optimizer.step()
        self.outer.zero_grad(set_to_none=True)

    def outer_obj(self,
                  in_params: Params,
                  out_params: Params,
                  input: Tensor,
                  target: Tensor
                  ) -> tuple[Tensor, Tensor]:
        output = self.f_model(in_params, input)
        return F.binary_cross_entropy_with_logits(output, target), output

    def post_outer_update(self) -> None:
        self.f_params[0].zero_()
        self.f_params[0].detach_()
        for p in self.outer.parameters():
            p.data.clamp_(min=1e-8)


class CGSolver(Solver):

    def approx_ihvp(self,
                    inner_obj: Objective,
                    outer_obj: Objective,
                    in_params: Params,
                    out_params: Params
                    ) -> tuple[Params, Params]:
        return conjugate_gradient(inner_obj, outer_obj, in_params, out_params, self.hyper_config['num_iters'],
                                  self.hyper_config['lr'])


class NeumannSolver(Solver):

    def approx_ihvp(self,
                    inner_obj: Objective,
                    outer_obj: Objective,
                    in_params: Params,
                    out_params: Params
                    ) -> tuple[Params, Params]:
        return neumann(inner_obj, outer_obj, in_params, out_params, self.hyper_config['num_iters'],
                       self.hyper_config['lr'])


class NystromSolver(Solver):
    def approx_ihvp(self,
                    inner_obj: Callable,
                    outer_obj: Callable,
                    in_params: Params,
                    out_params: Params
                    ) -> tuple[Params, Params]:
        return nystrom(inner_obj, outer_obj, in_params, out_params, self.hyper_config['rank'], self.hyper_config['rho'])


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num_data', type=int, default=1_000)
    p.add_argument('--dim_data', type=int, default=100)
    p.add_argument('--num_iters', type=int, default=10_00)
    p.add_argument('--unroll_steps', type=int, default=100)
    p.add_argument("--alpha", "--rho", type=float, default=0.1)
    p.add_argument("--cost", default=5, type=int)
    p.add_argument("--method", choices=('nystrom', 'neumann', 'cg',))
    args = p.parse_args()

    torch.random.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    train_data, val_data = generate_data(args.num_data, args.dim_data)

    model = nn.Linear(args.dim_data, 1, bias=False)
    nn.init.zeros_(model.weight)
    hyper = copy.deepcopy(model)
    nn.init.ones_(hyper.weight)

    inner_optimizer = functools.partial(diff_sgd, momentum=0, weight_decay=0, lr=0.1)
    outer_optimizer = torch.optim.SGD(hyper.parameters(), weight_decay=0, momentum=0.9, lr=1)

    match args.method:
        case 'neumann':
            approx_ihvp = functools.partial(neumann, num_iters=args.cost, lr=args.alpha)
        case 'cg':
            approx_ihvp = functools.partial(neumann, num_iters=args.cost, lr=args.alpha)
        case 'nystrom':
            approx_ihvp = functools.partial(nystrom, rank=args.cost, rho=args.alpha)
        case _:
            raise NotImplementedError

    solver = Solver(model, hyper, [train_data], [val_data], approx_ihvp,
                    inner_optimizer=inner_optimizer, outer_optimizer=outer_optimizer,
                    num_iters=args.num_iters, unroll_steps=args.unroll_steps)
    solver.run()
