"""Train a readable MNIST classifier. Run `python train.py --help` for options."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a desktop display.
import matplotlib.pyplot as plt
import numpy as np

from cbem.data import load_mnist_csv, stratified_split
from cbem.network import NeuralNetwork

ROOT = Path(__file__).resolve().parent


def evaluate(model, X, y):
    """Evaluate fixed weights; no parameter updates happen here."""
    loss, _ = model.loss_and_gradients(X, y)
    accuracy = float(np.mean(model.predict(X) == y))
    return loss, accuracy


def save_plots(history, X, y, predictions, out):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].plot(epochs, [r[f"{split}_loss"] for r in history], label=split)
        axes[1].plot(epochs, [r[f"{split}_accuracy"] for r in history], label=split)
    for ax, ylabel in zip(axes, ("Mean cross-entropy", "Accuracy")):
        ax.set(xlabel="Epoch (0 = before training)", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.2)
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)

    # Looking at concrete errors is often more informative than a single score.
    mistakes = np.flatnonzero(y != predictions)[:12]
    if len(mistakes):
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, mistakes):
            ax.imshow(X[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"True {y[index]}, predicted {predictions[index]}")
        fig.suptitle("Validation mistakes (first 12)")
        fig.tight_layout()
        fig.savefig(out / "validation_mistakes.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "mnist_train_100.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "mnist_test.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate test data once, after choosing settings on validation data.")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "first_run")
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden < 1 or args.batch_size < 1:
        parser.error("epochs, hidden, and batch-size must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive.")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new --out to preserve experiments.")
    if args.evaluate_test and args.train.resolve() == args.test.resolve():
        parser.error("Training and test paths must differ.")
    try:
        X, y = load_mnist_csv(args.train)
        X_train, y_train, X_val, y_val = stratified_split(
            X, y, args.validation_fraction, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f"Training: {len(y_train)}; validation: {len(y_val)}; features: {X.shape[1]}")
    if len(y_train) < 1000:
        print("Small teaching dataset: use accuracy to explore, not to claim robust performance.")
    model = NeuralNetwork(X.shape[1], args.hidden, 10, args.seed)
    rng = np.random.default_rng(args.seed)
    history = []
    best_loss = float("inf")
    best_parameters, best_epoch = None, 0

    # Epoch 0 measures the random starting model. An epoch thereafter means
    # one complete pass over the training split, in shuffled mini-batches.
    for epoch in range(args.epochs + 1):
        if epoch:
            order = rng.permutation(len(y_train))
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                model.train_batch(X_train[batch], y_train[batch], args.learning_rate)
        train_loss, train_accuracy = evaluate(model, X_train, y_train)
        val_loss, val_accuracy = evaluate(model, X_val, y_val)
        if not np.isfinite(train_loss + val_loss):
            parser.error("Training diverged. Try a smaller learning rate.")
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                            validation_loss=val_loss, validation_accuracy=val_accuracy))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            # Copies matter: references would change with future training steps.
            best_parameters = {name: value.copy() for name, value in model.params.items()}
        if epoch == 0 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:3d} train loss={train_loss:.4f} acc={train_accuracy:.1%} "
                  f"validation loss={val_loss:.4f} acc={val_accuracy:.1%}")

    # Select the checkpoint using VALIDATION loss, never test accuracy.
    model.params = best_parameters
    val_predictions = model.predict(X_val)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y_val, val_predictions), 1)
    majority_class = int(np.bincount(y_train, minlength=10).argmax())
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "train_examples": len(y_train), "validation_examples": len(y_val),
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_loss,
        "selected_validation_accuracy": float(np.mean(val_predictions == y_val)),
        "majority_class": majority_class,
        "majority_baseline_validation_accuracy": float(np.mean(y_val == majority_class)),
        "numpy_version": np.__version__,
        "confusion_matrix_convention": "rows = true digits; columns = predicted digits",
    }
    if args.evaluate_test:
        try:
            X_test, y_test = load_mnist_csv(args.test)
        except ValueError as error:
            parser.error(str(error))
        test_loss, test_accuracy = evaluate(model, X_test, y_test)
        summary.update(test_examples=len(y_test), test_loss=test_loss, test_accuracy=test_accuracy)
        print(f"Final test: loss={test_loss:.4f}, accuracy={test_accuracy:.1%} (n={len(y_test)})")

    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savetxt(args.out / "validation_confusion.csv", confusion, fmt="%d", delimiter=",")
    np.savez(args.out / "model.npz", **model.params)
    save_plots(history, X_val, y_val, val_predictions, args.out)
    print(f"Selected epoch {best_epoch} by validation loss. Results: {args.out}")


if __name__ == "__main__":
    main()
