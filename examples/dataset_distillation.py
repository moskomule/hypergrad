import functools

import functorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from hypergrad.approx_hypergrad import conjugate_gradient, neumann, nystrom
from hypergrad.optimizers import diff_sgd
from hypergrad.solver import BaseImplicitSolver
from hypergrad.utils import Params


@torch.no_grad()
def accuracy(input: torch.Tensor,
             target: torch.Tensor,
             return_sum: bool = False
             ) -> torch.Tensor:
    if target.ndim == 2:
        target = target.argmax(dim=-1)
    pred_idx = input.argmax(dim=-1, keepdim=True)
    target = target.view(-1, 1).expand_as(pred_idx)

    out = (pred_idx == target).float().sum(dim=1)
    if return_sum:
        return out.sum()
    else:
        return out.mean()


@torch.no_grad()
def test(f_model,
         f_params,
         data_loader
         ) -> float:
    device = f_params[0].device
    correct = 0
    num_example = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        output = f_model(f_params, input)
        correct += accuracy(output, target, return_sum=True)
        num_example += output.size(0)
    return correct.item() / num_example


class Solver(BaseImplicitSolver):
    def __init__(self, *args, **kwargs):
        self.val_loader = kwargs.pop('val_loader')
        super().__init__(*args, **kwargs)

    def target(self):
        return torch.arange(10, device=self.device).repeat(self.outer.num_per_class)

    def inner_obj(self,
                  in_params: Params,
                  out_params: Params,
                  input: torch.Tensor,
                  target: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        input, = out_params
        output = self.f_model(in_params, input)
        loss = F.cross_entropy(output, target)
        return loss, output

    def outer_update(self) -> None:
        in_input, = tuple(self.outer.parameters())
        in_target = self.target()
        out_input, out_target = next(self.outer_loader)
        in_params = tuple(self.f_params)
        _, out_params = functorch.make_functional(self.outer, disable_autograd_tracking=True)

        implicit_grads = self.approx_ihvp(lambda i, o: self.inner_obj(i, o, in_input, in_target)[0],
                                          lambda i, o: self.outer_obj(i, o, out_input, out_target)[0],
                                          in_params, out_params)
        self.set_out_grad(implicit_grads)
        self.outer_optimizer.step()
        self.outer.zero_grad(set_to_none=True)

        loss, output = self.outer_obj(in_params, out_params, out_input, out_target)
        self.recorder.add('outer_loss', loss.detach())
        self.recorder.add('outer_acc', accuracy(output, out_target))

    def outer_obj(self,
                  in_params: Params,
                  out_params: Params,
                  input: torch.Tensor,
                  target: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.f_model(in_params, input)
        return F.cross_entropy(output, target), output

    def inner_update(self) -> None:
        target = self.target()
        grads, (loss, output) = functorch.grad_and_value(self.inner_obj,
                                                         has_aux=True)(self.f_params,
                                                                       tuple(self.outer.parameters()), None,
                                                                       target)
        self.f_params, self.f_optim_state = self.inner_optimizer(list(self.f_params), list(grads), self.f_optim_state)
        self.recorder.add('inner_loss', loss.detach())
        self.recorder.add('inner_acc', accuracy(output, target))

    def post_outer_update(self) -> None:
        self.recorder.add('val_acc', test(self.f_model, self.f_params, self.val_loader))
        self.reset_inner_model()

    def reset_inner_model(self) -> None:
        self.f_model, self.f_params = functorch.make_functional(self.inner, self._functorch_requires_grad)

    def sync_inner(self) -> None:
        pass


class DistilledData(nn.Module):
    def __init__(self, num_per_class):
        super().__init__()
        self.num_per_class = num_per_class
        self.data = nn.Parameter(torch.randn(self.num_per_class * 10, 1, 28, 28))


if __name__ == '__main__':
    import argparse
    from rich.traceback import install

    install(show_locals=True, width=150)

    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num_iters', type=int, default=500_000)
    p.add_argument('--unroll_steps', type=int, default=100)
    p.add_argument('--num_per_class', type=int, default=5)
    p.add_argument("--alpha", "--rho", type=float, default=0.1)
    p.add_argument("--cost", default=5, type=int)
    p.add_argument("--method", choices=('neumann', 'cg', 'iterative', 'nystrom'))
    args = p.parse_args()

    torch.random.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    model = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2),
                          nn.LeakyReLU(),
                          nn.MaxPool2d(2),
                          nn.Conv2d(6, 16, 5),
                          nn.LeakyReLU(),
                          nn.MaxPool2d(2),
                          nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120),
                          nn.LeakyReLU(),
                          nn.Linear(120, 84),
                          nn.LeakyReLU(),
                          nn.Linear(84, 10))
    hyper = DistilledData(args.num_per_class)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    dataset = MNIST('~/.torch/data', train=True, transform=transform, download=True)
    train_set, val_set = random_split(dataset, (int(len(dataset) * 0.9), int(len(dataset) * 0.1)),
                                      generator=torch.Generator().manual_seed(0))
    test_set = MNIST('~/.torch/data', train=False, transform=transform, download=True)
    data_loader = DataLoader(train_set, batch_size=1024, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, pin_memory=False, drop_last=True)

    match args.method:
        case 'neumann':
            approx_ihvp = functools.partial(neumann, num_iters=args.cost, lr=args.alpha)

        case 'cg':
            approx_ihvp = functools.partial(conjugate_gradient, num_iters=args.cost, lr=args.alpha)
        case 'nystrom':
            approx_ihvp = functools.partial(nystrom, rank=args.cost, rho=args.alpha)
        case _:
            raise NotImplementedError

    solver = Solver(model, hyper, [], data_loader, approx_ihvp, args.num_iters, args.unroll_steps,
                    inner_optimizer=functools.partial(diff_sgd, momentum=0, weight_decay=0, lr=0.01),
                    outer_optimizer=torch.optim.Adam(hyper.parameters(), lr=1e-3, betas=(0.9, 0.999)),
                    val_loader=val_loader)
    solver.run()
    test(solver.f_model, solver.f_params, test_loader)
