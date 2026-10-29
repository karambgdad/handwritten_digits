"""One hidden layer, with every derivative written explicitly in NumPy.

Read docs/01_foundations.md first and docs/02_backpropagation.md alongside this
file. We use ROWS for examples, so a batch X has shape (batch_size, inputs).
This convention differs from tutorials that put examples in columns.
"""

import numpy as np


class NeuralNetwork:
    """Map input features to class probabilities: affine -> tanh -> affine -> softmax.

    A parameter is a number learned from data (weights and biases). Hidden width,
    learning rate, and seed are hyperparameters: choices made outside training.
    """

    def __init__(self, inputs=784, hidden=32, outputs=10, seed=42):
        if any(not isinstance(n, int) or n < 1 for n in (inputs, hidden, outputs)):
            raise ValueError("Layer sizes must be positive integers.")
        if outputs < 2:
            raise ValueError("Classification requires at least two output classes.")
        rng = np.random.default_rng(seed)
        # Different random weights break symmetry: hidden neurons must not all
        # start identically. Xavier scaling keeps initial signals moderate.
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / (inputs + hidden)), (inputs, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, np.sqrt(2 / (hidden + outputs)), (hidden, outputs)),
            "b2": np.zeros(outputs),
        }

    def _check_inputs(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.params["W1"].shape[0]:
            raise ValueError("X must be a nonempty (examples, input_features) matrix.")
        if not np.isfinite(X).all():
            raise ValueError("Input features must be finite.")
        return X

    def _forward(self, X):
        p = self.params
        # @ multiplies matrices; + b1 broadcasts one bias vector to every row.
        # Z1: (B, H). Each entry is a weighted sum of pixels plus a bias.
        Z1 = X @ p["W1"] + p["b1"]
        A1 = np.tanh(Z1)
        # A nonlinear activation lets stacked layers express more than a single
        # linear map. tanh is smooth, bounded, and has a simple derivative.
        logits = A1 @ p["W2"] + p["b2"]  # (B, C), unnormalized class scores

        # Softmax is unchanged by subtracting the same number from each row.
        # Subtracting the row maximum prevents overflow in exp(large_score).
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return A1, log_probs, np.exp(log_probs)

    def predict_proba(self, X):
        """Return (examples, classes) probabilities; each row sums to one.

        These are model estimates, not guarantees that confidence is calibrated.
        """
        X = self._check_inputs(X)
        return self._forward(X)[2]

    def predict(self, X):
        """Choose the class with the largest estimated probability in each row."""
        return self.predict_proba(X).argmax(axis=1)

    def loss_and_gradients(self, X, y):
        """Return mean cross-entropy and derivatives for all four parameters.

        y contains integer class IDs, shape (B,). Computing derivatives does
        NOT change parameters; train_batch performs the separate update step.
        """
        X = self._check_inputs(X)
        y = np.asarray(y)
        classes = self.params["b2"].size
        if (y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer)
                or np.any(y < 0) or np.any(y >= classes)):
            raise ValueError("y must contain one integer class ID per input row.")
        A1, log_probs, probs = self._forward(X)
        B = len(X)
        # Penalize low probability assigned to the correct class. Use log_probs
        # directly: log(exp(log_probs)) can lose information through underflow.
        loss = -log_probs[np.arange(B), y].mean()

        # For softmax + mean cross-entropy, dL/dlogits = (P - one_hot(y)) / B.
        # Subtract 1 only at the true class position, avoiding a one-hot array.
        dZ2 = probs.copy()
        dZ2[np.arange(B), y] -= 1
        dZ2 /= B

        # Chain rule, moving backward. Each weight accumulates its input
        # activation times the derivative of loss with respect to its output.
        dW2 = A1.T @ dZ2                       # (H, B) @ (B, C) -> (H, C)
        db2 = dZ2.sum(axis=0)                  # one shared bias per class
        dA1 = dZ2 @ self.params["W2"].T        # (B, C) @ (C, H) -> (B, H)
        dZ1 = dA1 * (1 - A1**2)                # tanh'(z) = 1 - tanh(z)^2
        dW1 = X.T @ dZ1                        # (D, B) @ (B, H) -> (D, H)
        db1 = dZ1.sum(axis=0)
        return float(loss), {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def train_batch(self, X, y, learning_rate):
        """Take one gradient-descent step and return the loss BEFORE the step."""
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        loss, gradients = self.loss_and_gradients(X, y)
        # Compute ALL gradients before updating ANY weights. Otherwise earlier
        # layers would backpropagate through already-modified later layers.
        for name in self.params:
            self.params[name] -= learning_rate * gradients[name]
        return loss
