"""Learn y = 2*x + 1 with one neuron, using ordinary Python and no ML library.

Run: python lessons/01_one_neuron.py
Predict the direction of the first weight update before running it.
"""

examples = [(-1.0, -1.0), (0.0, 1.0), (1.0, 3.0)]
weight, bias = 0.0, 0.0
learning_rate = 0.1

for step in range(101):
    loss, d_weight, d_bias = 0.0, 0.0, 0.0
    for x, target in examples:
        prediction = weight * x + bias
        error = prediction - target
        # Squaring prevents positive and negative errors from cancelling out.
        loss += error**2 / len(examples)
        # Chain rule: d(error^2)/dw = 2*error * d(prediction)/dw = 2*error*x.
        # A derivative tells us how sensitive the loss is to a small change.
        d_weight += 2 * error * x / len(examples)
        d_bias += 2 * error / len(examples)
    if step % 20 == 0:
        print(f"step={step:3d} loss={loss:.6f} weight={weight:.4f} bias={bias:.4f}")
    if step < 100:
        # Move AGAINST the gradient to reduce loss locally. A huge step can
        # overshoot; gradient descent does not guarantee each large step helps.
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

print(f"For a new x=2, prediction={weight * 2 + bias:.4f}; expected 5.")
# Experiments: set learning_rate to 0.001, then 2.0. Explain what changes.
# This tiny noiseless line is a teaching example, not evidence that a model
# trained on three real-world observations would generalize reliably.
