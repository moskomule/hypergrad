from collections.abc import Callable
from typing import TypeAlias

import functorch
from torch import Tensor

Params: TypeAlias = tuple[Tensor, ...]  # functorch's parameters
Objective: TypeAlias = Callable[[Params, Params], Tensor]  # objective function


def hvp(f: Objective,
        primal: Params,
        tangent: Params,
        *,
        rev_only: bool = False
        ) -> Params:
    # computing hessian vector product
    # rev_only=True if forward-mode AD is not available
    if rev_only:
        _, vjp_fn = functorch.vjp(functorch.grad(f), primal)
        return vjp_fn(tangent)[0]
    else:
        return functorch.jvp(functorch.grad(f), (primal,), (tangent,))[1]


def foreach(p1: Params,
            p2: Params,
            op: Callable[[Tensor, Tensor], Tensor],
            alpha: float | Tensor = 1
            ) -> Params:
    # this may be replaced with nested_tensor in the future
    if len(p1) != len(p2):
        # need more sophisticated check
        raise ValueError('p1 and p2 must have the same structure')
    return tuple(op(t1, alpha * t2) for t1, t2 in zip(p1, p2))


def vector_to_tree(vec: Tensor,
                   tree: Params
                   ) -> Params:
    pointer = 0
    out = []
    for t in tree:
        size = t.numel()
        out.append(vec[pointer: pointer + size].view_as(t))
        pointer += size
    return tuple(out)
