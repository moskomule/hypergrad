# Solver for bilevel optimization

`hypergrad.approx_hypergrad`'s functions alone are enough to implement gradient-based bi-level optimization problems.
`hypergrad` additionally offers `Solver` to make the implementation easy.

As a simple example, we take a logistic regression with weight decay for each parameter 
(see [the example](https://github.com/moskomule/hypergrad/examples/logistic_regression.py) as well).
Users need to define `inner_forward`, `inner_update`, `outer_forward`, and `outer_update` methods.

```python
from functorch import grad_and_value, make_functional
from torch import Tensor, nn
from torch.nn import functional as F

from hypergrad.approx_hypergrad import nystrom
from hypergrad.optimizers import diff_sgd
from hypergrad.solver import BaseImplicitSolver
from hypergrad.utils import Params


class Solver(BaseImplicitSolver):
    def inner_forward(self,
                      in_params: Params,
                      out_params: Params,
                      input: Tensor,
                      target: Tensor
                      ) -> Tensor:
        # forward computation of the inner problem
        output = self.inner_func(in_params, input)
        loss = F.binary_cross_entropy_with_logits(output, target)
        weight_decay = sum([(_in.pow(2) * _out).sum() 
                            for _in, _out in zip(in_params, out_params)])
        return loss + weight_decay / 2  # (1)

    def inner_update(self) -> None:
        # update rule of the inner problem
        input, target = next(self.inner_loader)  # (2)
        grads, loss = grad_and_value(self.inner_forward)(self.inner_params, 
                                                         tuple(self.outer.parameters()), 
                                                         input,
                                                         target)
        self.inner_params, self.inner_optim_state = self.inner_optimizer(self.inner_params,
                                                                         grads,
                                                                         self.inner_optim_state)

    def outer_forward(self,
                      in_params: Params,
                      out_params: Params,
                      input: Tensor,
                      target: Tensor
                      ) -> Tensor:
        # forward computation of the outer problem
        output = self.inner_func(in_params, input)  # (3)
        return F.binary_cross_entropy_with_logits(output, target)

    def outer_update(self) -> None:
        # update rule of the outer problem
        in_input, in_target = next(self.inner_loader)
        out_input, out_target = next(self.outer_loader)
        _, out_params = make_functional(self.outer, disable_autograd_tracking=True)

        # compute implicit gradients
        implicit_grads = self.approx_ihvp(lambda i, o: self.inner_forward(i, o, in_input, in_target),
                                          lambda i, o: self.outer_forward(i, o, out_input, out_target),
                                          self.inner_params, out_params)

        self.outer_grad = implicit_grads # (4)
        self.outer_optimizer.step()
        self.outer.zero_grad(set_to_none=True)
```

1. To compute gradients, the outputs of `inner_forward` and `outer_forward` are scalar loss values. 
These methods are only used in `inner_update` and `outer_update`, so if users can remember this, the outputs can be something else.
See [dataset distillation example](https://github.com/moskomule/hypergrad/examples/dataset_distillation.py) for example.
2. `inner_loader` and `outer_loader` are data loaders and can be iterated infinitely.
3. `inner_func` and `inner_params` are the functional model and its parameters of the given inner `nn.Module` model
(in this example, `inner`).
These are extracted by [`functorch.make_functional`](https://pytorch.org/functorch/stable/generated/functorch.make_functional.html#functorch.make_functional).
4. Set computed gradients to outer parameters, so that `outer_optimizer.step` can update the outer model.

Then, define models and optimizers.

```python
inner = nn.Linear(dim_data, 1, bias=False)
nn.init.zeros_(inner.weight)
outer = copy.deepcopy(inner)
nn.init.ones_(outer.weight)

inner_optimizer = functools.partial(diff_sgd, momentum=0, weight_decay=0, lr=0.1)
outer_optimizer = SGD(outer.parameters(), weight_decay=0, momentum=0.9, lr=1)
```

Now, the hyperparameter optimization problem can be solved by `solver.run`.

```python
approx_ihvp = functools.partial(nystrom, rank=5, rho=0.1)
solver = Solver(inner, outer, inner_loader, outer_loader, approx_ihvp,
                inner_optimizer=inner_optimizer, outer_optimizer=outer_optimizer,
                num_iters=num_iters, unroll_steps=unroll_steps, outer_patience_iters=0)
solver.run()
```

`solver.run` is something like:

```python
for i in range(num_iters):
    solver.inner_update()

    if i > 0 and i >= outer_patience_iters and i % unroll_steps == 0:
        solver.outer_update()
        solver.post_outer_update()
```

`solver.post_outer_update` can be used to clean up after each outer update.