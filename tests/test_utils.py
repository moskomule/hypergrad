import torch

from hypergrad.utils import foreach, vector_to_params


def test_foreach():
    p1 = tuple(torch.randn(3, 3) for _ in range(4))
    p2 = tuple(torch.randn(3, 3) for _ in range(4))
    out = foreach(p1, p2, torch.add, alpha=0.1)

    expected = []
    for _p1, _p2 in zip(p1, p2):
        expected.append(_p1 + 0.1 * _p2)

    for _out, _expected in zip(out, expected):
        assert torch.equal(_out, _expected)


def test_vector_to_params():
    vector = torch.randn(4 * 3 * 3)
    expected = tuple(vector[i * 9:(i + 1) * 9].view(3, 3) for i in range(4))
    out = vector_to_params(vector, expected)

    for _out, _expected in zip(out, expected):
        assert torch.equal(_out, _expected)
