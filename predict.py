"""Load saved parameters and inspect one labelled CSV example."""

import argparse

import numpy as np

from cbem.data import load_mnist_csv
from cbem.network import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    X, y = load_mnist_csv(args.data)
    if not 0 <= args.index < len(y):
        parser.error("index must identify a row in the CSV (starting at 0).")
    with np.load(args.model, allow_pickle=False) as saved:
        parameters = {name: saved[name].copy() for name in ("W1", "b1", "W2", "b2")}
    model = NeuralNetwork(parameters["W1"].shape[0], parameters["W1"].shape[1],
                          parameters["W2"].shape[1])
    for name, value in parameters.items():
        if value.shape != model.params[name].shape or not np.isfinite(value).all():
            parser.error(f"Invalid parameter array: {name}")
    model.params = parameters
    probabilities = model.predict_proba(X[args.index:args.index + 1])[0]
    print(f"True digit: {y[args.index]}; predicted: {probabilities.argmax()}")
    for digit, probability in enumerate(probabilities):
        print(f"  {digit}: {probability:.2%}")


if __name__ == "__main__":
    main()
