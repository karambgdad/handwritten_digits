# Foundations: what is actually learning?

## 1. A dataset is a collection of examples

For digit recognition, an example has an input (a picture) and a target (the
correct digit). Each pixel brightness is a **feature**. The target is also
called a **label**. This is supervised learning because training examples
include the answer we want the model to learn to predict.

A 28 by 28 image has 784 pixels. Flattening it changes the arrangement into a
vector of length 784 without changing the values. A dense network does not
explicitly use the image's spatial arrangement; later, a convolutional network
will provide a useful comparison.

In this code, `X.shape == (B, D)` means B examples, each with D features.
`y.shape == (B,)` holds one integer label per example. A shape is a description
of an array's dimensions, not a mathematical operation.

## 2. Start with one neuron

The simplest model in this project is:

```text
prediction = weight * input + bias
```

The weight controls how much changing the input changes the prediction. The
bias moves the prediction up or down even when the input is zero. Both are
parameters: numbers adjusted during training.

Suppose x = 2, the target is 5, and weight = 1, bias = 0. The prediction is 2,
the error is -3, and squared error is 9. A loss function turns prediction
quality into a number we can try to minimize.

For several examples, mean squared error is:

```text
L = (1/B) * sum((w*x_i + b - y_i)^2)
```

Run `lessons/01_one_neuron.py`. It learns the line y = 2*x + 1. No special
"intelligence" command exists: a repeated arithmetic update changes two numbers.

## 3. What does a derivative tell us?

A derivative describes how fast one quantity changes when another changes
slightly. If dL/dw is positive, increasing w slightly increases loss locally;
decreasing w slightly should lower it. For one squared-error example:

```text
e = w*x + b - y
L = e^2
dL/de = 2*e
de/dw = x
dL/dw = (dL/de) * (de/dw) = 2*e*x
dL/db = 2*e
```

This multiplication of local derivatives is the **chain rule**. It connects a
parameter to the loss through the intermediate calculations. For a mean loss,
average the derivatives across examples too.

## 4. Gradient descent

```text
new_weight = old_weight - learning_rate * dL/dw
new_bias   = old_bias   - learning_rate * dL/db
```

The gradient is the collection of derivatives with respect to all parameters.
The learning rate controls step size. Small steps can be slow; large steps can
overshoot or diverge. The negative sign moves opposite the direction of local
increase. The guarantee is local, so a large step need not lower the loss.

Training repeats four actions: predict, measure loss, differentiate, update.
Inference only predicts, using fixed parameters.

## 5. From numbers to classes

Digit labels 0 through 9 are categories. Their numerical distance is not our
objective: predicting 8 instead of 9 is not automatically a smaller mistake
than predicting 1 instead of 9.

The network therefore outputs ten scores, called logits. Softmax converts
these into positive probabilities that sum to one. Cross-entropy for a single
example is `-log(probability_of_the_true_class)`.

If the true digit is 7, assigning it probability 0.9 gives loss about 0.105;
probability 0.1 gives loss about 2.303 (natural logarithms). Loss considers the
probabilities; accuracy only asks whether the largest one identifies the right
class. Two models can have identical accuracy and different losses.

## 6. What makes it a neural network?

Each hidden neuron computes a weighted sum plus a bias and applies a nonlinear
function. Here that function is tanh, whose output lies between -1 and 1.
The output layer combines the hidden activations into digit scores.

Without a nonlinear activation, stacking affine layers produces another affine
map. It cannot gain the same expressive power by simply adding more such layers.
The XOR test is a tiny concrete example of a classification problem that needs
a nonlinear decision boundary.

## 7. Learning is not the same as memorizing

Training data changes the parameters. Validation data helps choose settings
and a checkpoint. Test data evaluates the final choice. If you repeatedly use
test performance to choose settings, that test is no longer an untouched check.

A model can learn training examples almost perfectly and still fail on new
ones. This is why the program plots training and validation curves separately.
Pixel scaling uses a fixed 255 divisor. If you later subtract a learned mean
or divide by a learned standard deviation, calculate them from training data
only and reuse those same values for validation and test data.

## Check your understanding

1. With x = 2, y = 5, w = 1, b = 0, calculate both derivatives. Which way do w
   and b move when the learning rate is positive?
2. Why does the bias derivative not contain x?
3. What does dividing a pixel value by 255 change? What does it preserve?
4. Can training loss fall while validation loss rises? Explain a mechanism.
5. What is the difference between a parameter and a hyperparameter?

Answers: (1) dL/dw = -12 and dL/db = -6, so both increase; (2) the derivative
of w*x+b with respect to b is 1; (3) the scale changes, pixel order and relative
brightness remain; (4) the model may fit training-specific details that do not
generalize; (5) parameters are learned by updates, hyperparameters configure
the model or training procedure.
