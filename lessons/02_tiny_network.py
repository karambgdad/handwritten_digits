"""Trace one complete learning step with a two-input, two-hidden, two-class model.

Run from the repository root: python lessons/02_tiny_network.py
Read docs/02_tiny_network_walkthrough.md alongside the output.
Only Python's standard library is needed. The same calculations are expressed
as NumPy matrix operations in cbem/network.py; here every connection is visible.
"""

from copy import deepcopy
from math import exp, log, tanh


def forward(x, parameters, target):
    """Predict with fixed parameters and record the intermediate calculations."""
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    # W1[input][hidden]: each COLUMN contains one hidden neuron's weights.
    # The first index is a row, the second a column. Python starts counting at 0.
    z1 = [
        x[0] * W1[0][0] + x[1] * W1[1][0] + b1[0],
        x[0] * W1[0][1] + x[1] * W1[1][1] + b1[1],
    ]
    hidden = [tanh(z1[0]), tanh(z1[1])]

    # W2[hidden][class]: each output combines BOTH hidden activations.
    logits = [
        hidden[0] * W2[0][0] + hidden[1] * W2[1][0] + b2[0],
        hidden[0] * W2[0][1] + hidden[1] * W2[1][1] + b2[1],
    ]
    # Subtracting a common value preserves softmax probabilities and avoids
    # exp(very_large_number). Natural log and exp are inverse functions.
    shifted = [score - max(logits) for score in logits]
    exponentials = [exp(score) for score in shifted]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    loss = log(total) - shifted[target]  # stable form of -log(p_true)
    return dict(z1=z1, hidden=hidden, logits=logits, shifted=shifted,
                exponentials=exponentials, total=total,
                probabilities=probabilities, loss=loss)


def worked_example():
    """Return the full trace, without printing or modifying any shared state.

    Fixed starting values make the written arithmetic reproducible. Real
    training normally initializes weights randomly, as the main model does.
    """
    x = [1.0, 0.5]                 # one example with two invented features
    target = 1                     # correct class ID, not an input feature
    learning_rate = 0.1
    parameters = {
        "W1": [[0.1, -0.2], [0.4, 0.2]],
        "b1": [0.0, 0.2],
        "W2": [[0.3, -0.1], [-0.2, 0.4]],
        "b2": [0.1, -0.1],
    }
    before = forward(x, parameters, target)
    h = before["hidden"]
    p = before["probabilities"]

    # d_logits[k] means d(loss)/d(logit_k), a sensitivity, not a weight update.
    # Softmax + cross-entropy gives probability minus the true-class indicator.
    # There is one example, so averaging over the batch would divide by 1.
    d_logits = [p[0] - float(target == 0), p[1] - float(target == 1)]

    # Output weights: upstream input * downstream loss sensitivity.
    dW2 = [[h[0] * d_logits[0], h[0] * d_logits[1]],
           [h[1] * d_logits[0], h[1] * d_logits[1]]]
    db2 = d_logits.copy()  # derivative of (weighted_sum + bias) w.r.t. bias is 1

    # Each hidden neuron influences BOTH class scores. Add both chain-rule
    # paths, using the OLD output weights from the forward pass.
    W2 = parameters["W2"]
    d_hidden = [
        d_logits[0] * W2[0][0] + d_logits[1] * W2[0][1],
        d_logits[0] * W2[1][0] + d_logits[1] * W2[1][1],
    ]
    tanh_slopes = [1 - h[0] ** 2, 1 - h[1] ** 2]
    d_z1 = [d_hidden[0] * tanh_slopes[0], d_hidden[1] * tanh_slopes[1]]
    dW1 = [[x[0] * d_z1[0], x[0] * d_z1[1]],
           [x[1] * d_z1[0], x[1] * d_z1[1]]]
    db1 = d_z1.copy()
    gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ALL gradients are ready before ANY parameter changes. deepcopy preserves
    # the nested lists so we can compare original and updated parameters.
    updated = deepcopy(parameters)
    for name in ("W1", "W2"):
        for row in range(2):
            for column in range(2):
                updated[name][row][column] -= learning_rate * gradients[name][row][column]
    for name in ("b1", "b2"):
        for index in range(2):
            updated[name][index] -= learning_rate * gradients[name][index]

    after = forward(x, updated, target)
    return dict(x=x, target=target, learning_rate=learning_rate,
                parameters=parameters, before=before, d_logits=d_logits,
                d_hidden=d_hidden, tanh_slopes=tanh_slopes, d_z1=d_z1,
                gradients=gradients, updated=updated, after=after)


def display(value):
    """Round only for display. All calculations above retain full precision."""
    if isinstance(value, list):
        return "[" + ", ".join(display(item) for item in value) + "]"
    return f"{value:.6f}"


def main():
    result = worked_example()
    before, after = result["before"], result["after"]
    print("ONE EXAMPLE, ONE UPDATE (all displayed numbers rounded to 6 decimals)")
    print(f"Input = {display(result['x'])}; target class = {result['target']}")
    print(f"Learning rate = {display(result['learning_rate'])}")
    print("\n1. Starting parameters (rows = source, columns = destination)")
    for name, value in result["parameters"].items():
        print(f"{name} = {display(value)}")
    print("\n2. Forward pass")
    for name in ("z1", "hidden", "logits", "shifted", "exponentials", "total",
                 "probabilities", "loss"):
        print(f"{name} = {display(before[name])}")
    print("\n3. Backward pass (d_ means derivative of loss with respect to)")
    for name in ("d_logits", "d_hidden", "tanh_slopes", "d_z1"):
        print(f"{name} = {display(result[name])}")
    for name, value in result["gradients"].items():
        print(f"d{name} = {display(value)}")
    print("\n4. Update: new parameter = old parameter - learning rate * gradient")
    for name, value in result["updated"].items():
        print(f"{name} = {display(value)}")
    print("\n5. Predict again with the updated parameters")
    for name in ("z1", "hidden", "logits", "probabilities", "loss"):
        print(f"{name} = {display(after[name])}")
    print(f"Loss change = {after['loss'] - before['loss']:.6f}")
    old_class = max(range(2), key=lambda k: before["probabilities"][k])
    new_class = max(range(2), key=lambda k: after["probabilities"][k])
    print(f"Predicted class: {old_class} -> {new_class}; true class: {result['target']}")
    print("One example can demonstrate an update; it cannot establish generalization.")


if __name__ == "__main__":
    main()
