"""Checks of mathematical behavior, including an independent gradient estimate."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cbem.data import load_mnist_csv, stratified_split
from cbem.network import NeuralNetwork


class LearningTests(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        model = NeuralNetwork(2, 3, 2, seed=7)
        X = np.array([[0.2, -0.7], [0.8, 0.1], [-0.4, 0.5]])
        y = np.array([0, 1, 0])
        _, gradients = model.loss_and_gradients(X, y)
        epsilon = 1e-5
        for name, parameter in model.params.items():
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus, _ = model.loss_and_gradients(X, y)
                parameter[index] = original - epsilon
                minus, _ = model.loss_and_gradients(X, y)
                parameter[index] = original
                # The slope between nearby points independently approximates
                # the derivative derived with the chain rule in network.py.
                numerical[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(gradients[name], numerical, rtol=1e-5, atol=1e-7)

    def test_nonlinear_xor_can_be_learned(self):
        X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = np.array([0, 1, 1, 0])
        model = NeuralNetwork(2, 8, 2, seed=3)
        initial, _ = model.loss_and_gradients(X, y)
        for _ in range(2000):
            model.train_batch(X, y, 0.1)
        final, _ = model.loss_and_gradients(X, y)
        self.assertLess(final, initial / 10)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_probabilities_and_extreme_logits(self):
        model = NeuralNetwork(2, 3, 2)
        model.params["b2"][:] = [1000, -1000]
        probabilities = model.predict_proba([[1, 2], [3, 4]])
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        loss, gradients = model.loss_and_gradients([[1, 2]], np.array([1]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(g).all() for g in gradients.values()))

    def test_seed_reproduces_initialization(self):
        a, b = NeuralNetwork(seed=4), NeuralNetwork(seed=4)
        for name in a.params:
            np.testing.assert_array_equal(a.params[name], b.params[name])

    def test_split_disjoint_reproducible_and_covers_data(self):
        X = np.arange(40).reshape(20, 2)
        y = np.repeat(np.arange(10), 2)
        a = stratified_split(X, y)
        b = stratified_split(X, y)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(set(a[0][:, 0]) & set(a[2][:, 0]))
        self.assertEqual(set(a[0][:, 0]) | set(a[2][:, 0]), set(X[:, 0]))
        self.assertEqual(set(a[1]), set(y))
        self.assertEqual(set(a[3]), set(y))

    def test_csv_normalization_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digits.csv"
            row = np.zeros((1, 785))
            row[0, 0], row[0, 1] = 7, 255
            np.savetxt(path, row, delimiter=",")
            X, y = load_mnist_csv(path)
            self.assertEqual(X.shape, (1, 784))
            self.assertEqual(X[0, 0], 1)
            self.assertEqual(y[0], 7)
            for label, pixel in [(1.5, 0), (10, 0), (1, 256), (1, float("nan"))]:
                row[0, 0], row[0, 1] = label, pixel
                np.savetxt(path, row, delimiter=",")
                with self.assertRaises(ValueError):
                    load_mnist_csv(path)


if __name__ == "__main__":
    unittest.main()
