# hypergrad

Simple and extensible hypergradient for PyTorch

## What is `hypergrad`?

`hypergrad` is a PyTorch tool to approximate hypergradients easily and efficiently.



Usually, a machine learning problem we consider is something like

$$
\newcommand{\argmin}{\mathop{\rm argmin}\limits}
\argmin_\theta f(\theta; \mathcal{T}),
$$

where $f$ is an objective function, $\theta$ is parameters, and $\mathcal{T}$ is training data.
This optimization is often solved by gradient-based optimizers, such as SGD.

Practically, we also need to take hyperparameter optimization into account. That is,

\[\argmin_\phi g(\theta^\ast(\phi), \phi; \mathcal{V})\quad\text{s.t.}~\theta^\ast(\phi)\in\argmin_\theta f(\theta,
\phi; \mathcal{T}),\]

where $g$ is a validation objective, $\phi$ is hyperparameters, $\mathcal{V}$ is validation data.
In some cases, $\nabla_\phi g$ can be obtained, so that this hyperparameter optimization problem can also be solved
in a gradient-based manner.
This idea is also applicable to gradient-based meta-learning, such as MAML.
So, for generality, we call $f$ an *inner objective* and $g$ an *outer objective*, and so on, in this repository.

The essential challenge of such nested problems is that $\nabla_\phi g$ needs backpropagation through the inner optimization $\argmin_\theta
f(\theta, \phi; \mathcal{T})$.
One way to address this is to use *approximated implicit differentiation* methods.
`hypergrad` currently supports [conjugate gradient-based approximation](), [Neumann-series approximation](),
and [Nyström method approximation]() like:

```python
from hypergrad.approx_hypergrad import nystrom

nystrom(f, g, theta, phi, rank=5, rho=0.1)
```

The functions in `hypergrad.approx_hypergrad` return approximation of $\nabla_\phi g$.
`hypergrad` also offers useful wrappers to make gradient-based hyperparameter optimization and meta-learning easier.
See [examples](https://github.com/moskomule/hypergrad/examples) for working examples.

## Installation

First, install `torch` and its accompanying `torchvision` [appropriately](https://pytorch.org). Then,

```console
pip install hypergrad
```

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