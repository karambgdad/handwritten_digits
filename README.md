# Machine Learning from Zero

A small Python learning project that grows from one adjustable number to a
handwritten-digit classifier. The aim is to understand what the calculations
mean, derive the gradients, and test explanations through experiments.

The main model uses **NumPy**, with forward propagation and backpropagation
written explicitly. Matplotlib draws the results. No deep-learning framework
is needed for this first stage.

## Start here

Use Python 3.11 or newer. Run commands from this repository's root directory.

Windows PowerShell (activation is optional):

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe lessons/00_python_and_arrays.py
.venv\Scripts\python.exe lessons/01_one_neuron.py
.venv\Scripts\python.exe lessons/02_tiny_network.py
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lessons/00_python_and_arrays.py
.venv/bin/python lessons/01_one_neuron.py
.venv/bin/python lessons/02_tiny_network.py
.venv/bin/python train.py
.venv/bin/python -m unittest discover -s tests -v
```

The rest of this guide uses `python` for your virtual environment's interpreter.
On Windows, replace it with `.venv\Scripts\python.exe` if you have not activated
the environment. No GPU or dataset download is required to run the examples.

## Learn in this order

| Step | Read / run | What you should be able to explain |
| --- | --- | --- |
| 0 | [Python and arrays](lessons/00_python_and_arrays.py) | Variables, functions, indexing, shapes, `@`, and broadcasting |
| 1 | [Foundations](docs/01_foundations.md) and [one neuron](lessons/01_one_neuron.py) | Features, targets, weights, biases, loss, derivatives, and learning rate |
| 2 | [Tiny network walkthrough](docs/02_tiny_network_walkthrough.md) and [matching code](lessons/02_tiny_network.py) | One complete prediction, all 12 gradients, and one update with actual numbers |
| 3 | [Backpropagation](docs/02_backpropagation.md) and [network code](cbem/network.py) | Every matrix shape and how the chain rule produces each gradient |
| 4 | [Experiments](docs/03_experiments.md) and [training code](train.py) | Generalization, data splits, baselines, overfitting, and error analysis |
| 5 | [Next steps](docs/04_next_steps.md) | A progression toward linear classifiers, PyTorch, and convolutional networks |

Before changing a setting, write a prediction. Run the experiment, inspect its
curves and mistakes, then explain whether the evidence supports your prediction.

## What the model does

```text
28 x 28 grayscale image
        -> 784 pixel values in [0, 1]
        -> weighted sums + biases
        -> 32 tanh hidden activations
        -> 10 class scores
        -> softmax probabilities for digits 0 through 9
```

Training compares those probabilities with known digit labels using
cross-entropy, then changes the weights and biases using gradient descent.
The code comments explain the reasons and the shapes, not just the syntax.

## Run an experiment

```bash
python train.py --epochs 100 --hidden 32 --learning-rate 0.1 --seed 42 --out runs/baseline
python train.py --epochs 100 --hidden 32 --learning-rate 0.01 --seed 42 --out runs/slower
```

Each run creates:

- `history.csv`: training and validation loss/accuracy, including epoch zero.
- `learning_curves.png`: plots to compare learning and generalization.
- `validation_mistakes.png`: up to 12 errors (only when mistakes exist).
- `validation_confusion.csv`: rows are true digits, columns are predicted digits.
- `summary.json`: settings, split sizes, selected epoch, and majority baseline.
- `model.npz`: parameters from the epoch with the lowest validation loss.

An existing output directory is never silently overwritten. Choose a fresh
`--out` for each experiment. Curves show the whole trajectory; the saved model
is the selected checkpoint, which may be earlier than the final epoch.

Once you have chosen settings using validation data, evaluate the test set:

```bash
python train.py --epochs 100 --hidden 32 --learning-rate 0.1 --seed 42 --evaluate-test --out runs/final
python predict.py --model runs/final/model.npz --data mnist_test.csv --index 0
```

The second command inspects a prediction; repeatedly choosing settings from
test predictions would turn the test set into another validation set.

## Data and limitations

The included files come from the original
[CBEM project](https://github.com/karambgdad/cbem_project), a fork of
[maniafshani/CBEM_Project](https://github.com/maniafshani/CBEM_Project).
The original `neuralNetwork.m` exercise skeleton is preserved for reference;
the runnable implementation is Python. The original repository had no README
or completed training/prediction implementation when this version was started.

Each CSV row contains a digit label followed by 784 grayscale pixel values.
`mnist_train_100.csv` contains only 100 examples. A stratified split reserves
roughly 20% for validation; the remainder is used for training. `mnist_test.csv`
is reserved for the final evaluation. `mnist_test_10.csv` is a tiny inspection
sample, not an additional source of training data or an independent benchmark.

This is deliberately a small teaching dataset. Tiny validation sets give noisy
estimates, and repeated tuning can overfit the validation set itself. Do not
interpret a good training score as reliable performance on unseen handwriting.
The repository does not establish independent dataset licensing; consult the
upstream dataset's terms before redistribution outside this learning context.

## Verification

The tests compare every backpropagation gradient in a tiny network with a
numerical finite-difference estimate, learn the nonlinear XOR problem, check
stable probabilities at extreme scores, and check data loading and split
separation. These checks validate mechanics, not real-world model quality.

See [the verified example run](docs/05_verified_run.md) for actual metrics and
learning curves, including the gap between training and test accuracy.

Useful references: [NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html),
[NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html),
and, after the manual derivatives make sense,
[PyTorch automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
