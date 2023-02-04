import copy

import pytest
import torch
from torch.optim.sgd import sgd

from hypergrad.optimizers import diff_sgd


@pytest.mark.parametrize("wd", [0, 1e-3])
@pytest.mark.parametrize("momentum", [0, 0.9])
def test_sgd(momentum, wd):
    params = [torch.randn(3, 3) for _ in range(3)]
    grads = [torch.randn_like(p) for p in params]
    buffer = [torch.randn_like(p) if momentum > 0 else None for p in params]
    _params, _grads, _buffer = copy.deepcopy([params, grads, buffer])
    params = [p.requires_grad_(True) for p in params]
    out, _ = diff_sgd(params, grads, buffer, wd, momentum, 0.1)
    out[0].sum().backward()

    sgd(_params, _grads, _buffer, weight_decay=wd, momentum=momentum, lr=0.1,
        dampening=0, nesterov=False, maximize=False, has_sparse_grad=False)

    assert torch.allclose(out[0].data, _params[0].data)
