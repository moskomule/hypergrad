# hypergrad ![pytest](https://github.com/moskomule/hypergrad/workflows/pytest/badge.svg)

Simple and extensible hypergradient for PyTorch

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/hypergrad.svg)](https://pypi.org/project/hypergrad)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hypergrad.svg)](https://pypi.org/project/hypergrad) -->

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

First, install PyTorch appropriately. Then,

```console
pip install hypergrad
```

## Methods

### Implicit hypergradient approximation (via approximated inverse Hessian-vector product)

* conjugate gradient
* [Neumann-series approsimation](https://proceedings.mlr.press/v108/lorraine20a.html)
* [Nyström method](to_be_updated)

Implementation of these methods can be found in `hypergrad/approximate_ihvp.py`

## Citation

```bibtex
@inproceedings{hataya2023nystrom,
    author = {Ryuichiro Hataya and Makoto Yamada},
    title = {{Nystr\"om Method for Accurate and Scalable Implicit Differentiation}},
    booktitle = {AISTATS},
    year = {2023}
}
```