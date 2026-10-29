# What to build next, and why

The repository implements the foundations and a NumPy digit classifier. The
items below are planned learning extensions, not features already implemented.
Move on when you can explain the current stage and reproduce its key idea.

## 1. Build a linear classifier before adding more hidden layers

Implement softmax regression: `logits = X @ W + b`, with the same cross-entropy
loss and split. Compare it with the hidden-layer network. This isolates what
the nonlinear hidden layer contributes. Derive and gradient-check its two
parameter derivatives before comparing accuracy.

Also learn linear regression and binary logistic regression on small synthetic
datasets. Draw their predictions. These models connect linear algebra,
probability, optimization, and evaluation with fewer moving parts.

## 2. Learn the mathematics just when it becomes useful

- Linear algebra: dot products, matrix products, transposes, and broadcasting.
- Calculus: derivatives, partial derivatives, the chain rule, and gradients.
- Probability: categorical distributions, likelihood, and cross-entropy.
- Statistics: sampling variation, uncertainty, leakage, and generalization.

For each idea, use a numerical example first, a formula second, and a small
implementation third. Do not postpone all experimentation until you have
finished a large mathematics course.

## 3. Expand data and evaluation

Obtain the full MNIST training split from an established dataset distribution,
record its source and terms, and hold out validation examples from training.
Never supplement training with test examples to get a better score.
Compare learning curves as training set size grows. Use repeated seeds and
report variability, not just the best observed run.

## 4. Add regularization as a controlled experiment

Implement L2 regularization, adding both its loss term and its derivative.
Gradient-check the modified objective. Compare several strengths while keeping
the split fixed. Then compare checkpoint selection with explicit early stopping.
Ask whether the effect is reduced overfitting or simply worse optimization.

## 5. Reproduce the same network in PyTorch

Match architecture, dtype, initial weights, inputs, targets, and mean loss
reduction. Compare outputs and gradients before training. PyTorch's autograd
computes derivatives for you; the earlier stages explain what it is computing.

Use the [official autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
Learn why gradients accumulate and why a training loop clears them. After
equivalence works, explore optimizers such as momentum and Adam.

## 6. Add a convolutional network

Reshape vectors back to images and investigate filters, local receptive fields,
and shared weights. Compare parameter counts and generalization with the dense
model using the same evaluation discipline. Visualize filters or intermediate
activations, but avoid treating a pleasing visualization as proof of reasoning.

## 7. Finish with a small inference application

A drawing canvas can be a useful final exercise. First match preprocessing:
image size, grayscale range, foreground/background polarity, and positioning.
User-drawn inputs may differ from MNIST even when both contain digits. Record
errors and investigate this distribution shift before changing the architecture.

The central habit at every stage is: **predict, run, inspect, explain**. Rich
comments help you read a solution; exercises and controlled experiments help
you learn to produce and challenge one yourself.
