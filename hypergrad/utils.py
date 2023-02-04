from collections.abc import Callable
from typing import TypeAlias

import functorch
from torch import Tensor

# types
Params: TypeAlias = tuple[Tensor, ...]  # functorch's parameters
Objective: TypeAlias = Callable[[Params, Params], Tensor]  # objective function


def hvp(f: Objective,
        primal: Params,
        tangent: Params,
        *,
        rev_only: bool = False
        ) -> Params:
    """ Hessian vector product using both forward- and reverse-mode ADs (if `rev_only=False`)
    or reverse-mode AD twice (if `rev_only=True`).

    Args:
        f: Objective function
        primal: Parameters to differentiate `f`
        tangent: "Vector" of Hessian-vector product
        rev_only: True if reverse-mode AD only

    Returns: Results of Hessian vector product
    """

    if rev_only:
        _, vjp_fn = functorch.vjp(functorch.grad(f), primal)
        return vjp_fn(tangent)[0]
    else:
        return functorch.jvp(functorch.grad(f), (primal,), (tangent,))[1]


# utilities for parameters
def foreach(p1: Params,
            p2: Params,
            op: Callable[[Tensor, Tensor], Tensor],
            alpha: float | Tensor = 1
            ) -> Params:
    # this may be replaced with nested_tensor in the future
    if len(p1) != len(p2):
        # need more sophisticated check
        raise ValueError('p1 and p2 must have the same structure')
    return tuple(op(p1, alpha * p2) for p1, p2 in zip(p1, p2))


def vector_to_params(vec: Tensor,
                     params: Params
                     ) -> Params:
    pointer = 0
    out = []
    for p in params:
        size = p.numel()
        out.append(vec[pointer: pointer + size].view_as(p))
        pointer += size
    return tuple(out)
