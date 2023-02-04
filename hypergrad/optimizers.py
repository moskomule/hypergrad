from hypergrad.utils import Params


def diff_sgd(params: Params,
             grads: Params,
             state: Params | None,
             weight_decay: float,
             momentum: float,
             lr: float,
             dampening: float = 0,
             nesterov: bool = False,
             ) -> tuple[Params, Params]:
    # differentiable SGD
    params = list(params)
    grads = list(grads)
    if state is None:
        state = [None for _ in params]
    else:
        state = list(state)
    for i, param in enumerate(params):
        grad = grads[i]
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        if momentum != 0:
            buf = state[i]

            if buf is None:
                state[i] = grad.clone().detach()
            else:
                state[i] = momentum * buf + grad * (1 - dampening)

            if nesterov:
                grad = grad.add(state[i], alpha=momentum)
            else:
                grad = state[i]

        params[i] = param.add(grad, alpha=-lr)
    return tuple(params), tuple(state)
