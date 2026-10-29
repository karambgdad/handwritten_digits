# Derive the network before memorizing the code

Read `cbem/network.py` alongside this document. B = batch size, D = input
features, H = hidden neurons, C = classes. Our examples occupy rows.

## Forward pass

| Quantity | Formula | Shape | Meaning |
| --- | --- | --- | --- |
| X | input batch | B x D | Scaled pixels |
| W1, b1 | learned parameters | D x H, H | Hidden-layer weights and biases |
| Z1 | X @ W1 + b1 | B x H | Hidden weighted sums |
| A1 | tanh(Z1) | B x H | Hidden activations |
| W2, b2 | learned parameters | H x C, C | Output weights and biases |
| Z2 | A1 @ W2 + b2 | B x C | Logits |
| P | softmax(Z2), by row | B x C | Class probabilities |

`@` sums products along matching dimensions. `.T` transposes a matrix: rows
become columns. Adding a one-dimensional bias uses broadcasting to apply the
same bias to every example. In MATLAB, `*` is matrix multiplication; in NumPy,
`*` is elementwise and `@` is matrix multiplication. Python indexes from zero.

For one example and class k:

```text
p_k = exp(z_k) / sum_j exp(z_j)
loss = -log(p_y) = -z_y + log(sum_j exp(z_j))
```

Subtract the largest score before exponentiating. Since numerator and
denominator receive the same multiplicative scaling, softmax is unchanged.
Calculate log probabilities directly to avoid taking log of a probability
that underflows to zero. This is a numerical issue, separate from learning.

## Start the backward pass at the loss

Differentiate the expression above with respect to each score z_k:

```text
d(loss)/dz_k = -1[k == y] + exp(z_k)/sum_j exp(z_j)
             = p_k - 1[k == y]
```

Here `1[k == y]` is 1 for the true class and 0 otherwise. For the mean over B
examples, divide by B. This gives the matrix formula:

```text
dZ2 = (P - one_hot(y)) / B
```

`dZ2` is shorthand for dL/dZ2, not a small modification we directly apply to
Z2. We update parameters, not temporary activations.

## Differentiate the output affine layer

The score for example i and class k is:

```text
Z2[i,k] = sum_h A1[i,h] * W2[h,k] + b2[k]
```

Changing W2[h,k] affects all examples, so add their contributions:

```text
dW2[h,k] = sum_i A1[i,h] * dZ2[i,k]
dW2 = A1.T @ dZ2             # (H,B) @ (B,C) -> (H,C)
db2 = sum(dZ2, axis=0)       # sum examples, preserve classes
dA1 = dZ2 @ W2.T             # (B,C) @ (C,H) -> (B,H)
```

We already averaged in dZ2. Dividing again here would incorrectly shrink the
gradient a second time. The bias gradient sums over examples because every
example shares the same bias.

## Differentiate the hidden activation and first layer

```text
d(tanh(z))/dz = 1 - tanh(z)^2
dZ1 = dA1 * (1 - A1^2)       # elementwise chain rule
dW1 = X.T @ dZ1              # (D,B) @ (B,H) -> (D,H)
db1 = sum(dZ1, axis=0)       # (H,)
```

Every gradient must have the same shape as the parameter it will update.
Shapes are a powerful debugging tool, but matching shapes alone do not prove
the mathematics is correct.

Finally, for each parameter theta:

```text
theta = theta - learning_rate * dL/dtheta
```

Compute all derivatives using the same parameter values before updating any
of them. Otherwise, the backward pass no longer differentiates the forward
pass you actually performed.

## Check calculus with arithmetic

A central finite difference estimates one parameter's derivative:

```text
dL/dtheta ~= (L(theta + epsilon) - L(theta - epsilon)) / (2 * epsilon)
```

The gradient test perturbs every parameter in a tiny network. It compares
these numerical slopes with the backpropagation result. This is slow for large
networks, so use a tiny batch and a tiny model. Too large an epsilon introduces
approximation error; too small introduces floating-point cancellation.

Run `python -m unittest discover -s tests -v`. Then intentionally remove `/ B`
from dZ2 and rerun the test. Observe the failure, explain it, and restore the
correct formula. A failing test can teach more than a successful training run.

## Exercises

1. Calculate all shapes for B=16, D=784, H=32, C=10.
2. Count parameters: D*H + H + H*C + C = 25,450 for the default architecture.
3. Why can equal initial hidden weights make neurons learn the same thing?
4. tanh approaches -1 or 1 for large-magnitude inputs. What happens to its
   derivative and to the gradient reaching W1?
5. Why does omitting the hidden activation turn two affine layers into one?

For (3), equal hidden weights and equal outgoing weights give identical
activations and symmetric updates. Random initialization breaks this symmetry.
For (4), the derivative approaches zero, weakening the gradient: saturation.
For (5), substitute `(X @ W1 + b1) @ W2 + b2` and regroup the products.
