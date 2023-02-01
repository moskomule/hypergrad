# functions to approximate inverse Hessian vector product and returns hypergradient
import functools
from typing import Callable, ParamSpec

import functorch
import torch
from torch import Tensor

from hypergrad.utils import Objective, Params, foreach, hvp, vector_to_tree

_P = ParamSpec('_P')


def implicit_grad(f: Callable[_P, tuple[Params, Params]]) -> Callable[_P, Params]:
    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Params:
        inner_obj = kwargs.get('inner_obj') or args[0]
        in_params = kwargs.get('in_params') or args[2]
        out_params = kwargs.get('out_params') or args[3]
        ihvp, out_out_g = f(*args, **kwargs)
        _, implicit_grads = functorch.jvp(
            lambda i_p: functorch.grad(lambda i, o: inner_obj(i, o),
                                       argnums=(0, 1))(i_p, out_params)[1],
            (in_params,), (ihvp,))
        return foreach(out_out_g, implicit_grads, torch.sub)

    return wrapper


@implicit_grad
def conjugate_gradient(inner_obj: Objective,
                       outer_obj: Objective,
                       in_params: Params,
                       out_params: Params,
                       num_iters: int,
                       lr: float
                       ) -> tuple[Params, Params]:
    """ Conjugate gradient method to approximate inverse Hessian vector product

    Args:
        inner_obj: Inner objective
        outer_obj: Outer objective
        in_params: Inner parameters
        out_params: Outer parameters
        num_iters: Number of iterations
        lr: step size

    Returns: approximated implicit gradients
    """

    in_out_g, out_out_g = functorch.grad(outer_obj, argnums=(0, 1))(in_params, out_params)
    xs = tuple(torch.zeros_like(g) for g in in_out_g)
    rs = tuple(torch.clone(g) for g in in_out_g)
    ps = tuple(torch.clone(g) for g in in_out_g)
    for _ in range(num_iters):
        hvps = hvp(lambda params: inner_obj(params, out_params), in_params, ps)
        num = sum([p.sum() for p in foreach(rs, rs, torch.mul)])
        denom = sum([p.sum() for p in foreach(hvps, ps, torch.mul)])
        alpha = num / (denom * lr)
        _xs = foreach(xs, ps, torch.add, alpha=alpha)
        _rs = foreach(rs, hvps, torch.sub, alpha=alpha)
        beta = sum([p.sum() for p in foreach(_rs, _rs, torch.mul)]) / num
        _ps = foreach(rs, ps, torch.add, alpha=beta)
        xs, ps, rs = _xs, _ps, _rs

    xs = tuple(lr * x for x in xs)
    return xs, out_out_g


@implicit_grad
def neumann(inner_obj: Objective,
            outer_obj: Objective,
            in_params: Params,
            out_params: Params,
            num_iters: int,
            lr: float
            ) -> tuple[Params, Params]:
    """ Neumann-series approximation method to approximate inverse Hessian vector product

    Args:
        inner_obj: Inner objective
        outer_obj: Outer objective
        in_params: Inner parameters
        out_params: Outer parameters
        num_iters: Number of iterations
        lr: step size

    Returns: approximated implicit gradients
    """

    in_out_g, out_out_g = functorch.grad(outer_obj, argnums=(0, 1))(in_params, out_params)

    vs = in_out_g
    ps = in_out_g
    for _ in range(num_iters):
        hvps = hvp(lambda params: inner_obj(params, out_params), in_params, vs)
        vs = foreach(vs, hvps, torch.sub, alpha=lr)
        ps = foreach(vs, ps, torch.add)

    ps = tuple(lr * p for p in ps)
    return ps, out_out_g


@implicit_grad
def nystrom(inner_obj: Objective,
            outer_obj: Objective,
            in_params: Params,
            out_params: Params,
            rank: int,
            rho: float
            ) -> tuple[Params, Params]:
    """ Nystrom method to approximate inverse Hessian vector product

    Args:
        inner_obj: Inner objective
        outer_obj: Outer objective
        in_params: Inner parameters
        out_params: Outer parameters
        rank: Rank of low-rank approximation
        rho: additive constant to improve numerical stability

    Returns: approximated implicit gradients
    """

    in_out_g, out_out_g = functorch.grad(outer_obj, argnums=(0, 1))(in_params, out_params)
    indices = torch.randperm(sum([p.numel() for p in in_params]), device=in_params[0].device)[:rank]

    def select_grad_row(in_params: Params, indices: Tensor) -> Tensor:
        grad = functorch.grad(lambda params: inner_obj(params, out_params))(in_params)
        grad = torch.cat([g.reshape(-1) for g in grad])
        # gather can be vmap'ed
        return functorch.vmap(lambda i: grad.gather(0, i))(indices)

    hessian_rows = functorch.jacrev(select_grad_row)(in_params, indices)
    c = torch.cat([v.reshape(rank, -1) for v in hessian_rows], dim=1)

    # Woodbury matrix identity
    m = c.take_along_dim(indices[None], dim=1)
    v = torch.cat([v.view(-1) for v in in_out_g])
    x = 1 / rho * v - 1 / (rho ** 2) * c.t() @ torch.linalg.solve(m + 1 / rho * c @ c.t(), c @ v)

    ihvp = vector_to_tree(x, in_out_g)
    return ihvp, out_out_g
