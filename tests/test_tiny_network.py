"""Verify the scalar lesson independently and keep its published output accurate."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import runpy
import unittest

import numpy as np

from cbem.network import NeuralNetwork

ROOT = Path(__file__).resolve().parents[1]
LESSON = runpy.run_path(str(ROOT / "lessons" / "02_tiny_network.py"))


class TinyNetworkTests(unittest.TestCase):
    def test_scalar_step_matches_main_network(self):
        trace = LESSON["worked_example"]()
        model = NeuralNetwork(2, 2, 2)
        model.params = {name: np.array(values) for name, values in trace["parameters"].items()}
        X, y = np.array([trace["x"]]), np.array([trace["target"]])
        loss, gradients = model.loss_and_gradients(X, y)
        self.assertAlmostEqual(loss, trace["before"]["loss"], places=12)
        np.testing.assert_allclose(model.predict_proba(X)[0], trace["before"]["probabilities"], atol=1e-12)
        for name in gradients:
            np.testing.assert_allclose(gradients[name], trace["gradients"][name], rtol=1e-12, atol=1e-12)
        model.train_batch(X, y, trace["learning_rate"])
        for name in model.params:
            np.testing.assert_allclose(model.params[name], trace["updated"][name], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.predict_proba(X)[0], trace["after"]["probabilities"], atol=1e-12)
        self.assertLess(trace["after"]["loss"], trace["before"]["loss"])

    def test_all_scalar_gradients_match_numerical_slopes(self):
        trace = LESSON["worked_example"]()
        epsilon = 1e-5
        for name, values in trace["parameters"].items():
            for index in np.ndindex(np.array(values).shape):
                plus, minus = deepcopy(trace["parameters"]), deepcopy(trace["parameters"])
                if len(index) == 2:
                    row, column = index
                    plus[name][row][column] += epsilon
                    minus[name][row][column] -= epsilon
                    analytic = trace["gradients"][name][row][column]
                else:
                    plus[name][index[0]] += epsilon
                    minus[name][index[0]] -= epsilon
                    analytic = trace["gradients"][name][index[0]]
                upper = LESSON["forward"](trace["x"], plus, trace["target"])["loss"]
                lower = LESSON["forward"](trace["x"], minus, trace["target"])["loss"]
                self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=8)

    def test_walkthrough_transcript_matches_actual_output(self):
        output = StringIO()
        with redirect_stdout(output):
            LESSON["main"]()
        doc = (ROOT / "docs" / "02_tiny_network_walkthrough.md").read_text(encoding="utf-8")
        expected = doc.split("<!-- tiny-network-output:start -->\n```text\n", 1)[1]
        expected = expected.split("```\n<!-- tiny-network-output:end -->", 1)[0]
        self.assertEqual(output.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
