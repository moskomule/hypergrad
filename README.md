# hypergrad 

![pytest](https://github.com/moskomule/hypergrad/workflows/pytest/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/hypergrad.svg)](https://pypi.org/project/hypergrad)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hypergrad.svg)](https://pypi.org/project/hypergrad)
[![document](https://img.shields.io/static/v1?label=doc&message=hypergrad&color=blue)](https://moskomule.github.io/hypergrad)

Simple and extensible hypergradient for PyTorch

## Installation

First, install `torch` and its accompanying `torchvision` appropriately. Then,

```console
pip install hypergrad
```

## Methods

### Implicit hypergradient approximation (via approximated inverse Hessian-vector product)

* conjugate gradient
* [Neumann-series approximation](https://proceedings.mlr.press/v108/lorraine20a.html)
* [Nyström method](to_be_updated)

Implementation of these methods can be found in `hypergrad/approximate_ihvp.py`

## Citation

To cite this repository,

```bibtex
@software{hypergrad,
    author = {Ryuichiro Hataya},
    title = {{hypergrad}},
    url = {https://github.com/moskomule/hypergrad},
    year = {2023}
}
```

`hypergrad` is developed as a part of the following research projects:

```bibtex
@inproceedings{hataya2023nystrom,
    author = {Ryuichiro Hataya and Makoto Yamada},
    title = {{Nystr\"om Method for Accurate and Scalable Implicit Differentiation}},
    booktitle = {AISTATS},
    year = {2023}
}
```