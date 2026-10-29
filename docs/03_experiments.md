# Turn code into experiments

Use the tiny dataset to investigate mechanisms. Keep a notebook or a simple
Markdown journal containing: question, prediction, settings, observations,
and interpretation. Change one factor at a time at first.

## Baseline and comparison

Run `python train.py --out runs/baseline`. The summary includes a majority-class
baseline: always predict the most common training digit. This asks whether the
network adds value beyond a very simple rule. It does not prove robustness.

The default split reserves about 20 examples for validation. One additional
correct prediction changes accuracy by approximately five percentage points.
Treat small differences cautiously. The split is stratified by label but very
small per-class counts still make conclusions uncertain.

## Experiments to try

| Question | Change | Inspect / explain |
| --- | --- | --- |
| What does step size do? | `--learning-rate 0.01`, `0.1`, then `1.0` | Convergence speed, oscillation, or divergence; do not assume a fixed outcome |
| Does capacity help? | `--hidden 4`, `32`, then `128` | Training fit versus validation loss |
| Does more training always help? | `--epochs 300` | Whether validation loss bottoms out before training ends |
| Why use mini-batches? | `--batch-size 1`, `16`, then `100` | Update count per epoch and noise; equal epochs do not mean equal updates |
| How much depends on chance? | `--seed 1`, `2`, then `3` | Initialization, split, and shuffle all change with this seed |
| What does the model confuse? | Open the confusion CSV and mistake plot | Repeated pairs, unusual shapes, ambiguous images |

Give each run a new `--out` directory. Keep the seed fixed when comparing other
settings. The current single seed changes several sources of randomness at
once; to isolate initialization alone, extend the script with independent
split/model/shuffle seeds as a later exercise.

## Read the learning curves

- Both losses stay high: possible underfitting, poor optimization, or a bug.
- Training loss falls while validation loss rises: evidence consistent with
  overfitting, especially if sustained; tiny samples can also fluctuate.
- Both losses fall: promising, but validation performance still has uncertainty.
- Loss changes while accuracy stays fixed: probabilities can improve without
  changing which class has the largest probability.

The model checkpoint is selected by minimum validation loss, not maximum
training accuracy. Repeated model selection itself can overfit validation data.
With more training examples, cross-validation and repeated splits can help
characterize variability. Keep the final test set separate throughout.

## Inspect mistakes

The confusion matrix's row 7, column 1 counts true sevens classified as ones.
Raw counts depend on the number of examples in each class. If comparing
per-class error rates, divide each row by its row total, handling empty rows.

Inspect actual images before deciding why a mistake happened. A digit that
looks obvious to you may have unusual positioning, thickness, or style.
Softmax confidence does not necessarily equal empirical correctness.

## Add a deliberate failure experiment

Copy the training script into a scratch file, then shuffle *only the training
labels*, keeping validation labels correct. Can a sufficiently flexible model
fit random labels? Does that help validation? This distinguishes memorization
from learning a relationship that generalizes. Keep the original experiment
and outputs for comparison.

## Use the test set once settings are chosen

Run with `--evaluate-test` and a new output directory. The test score describes
this selected model on this particular test set. Do not tune settings using
that score and keep calling it a final independent test.

The ten-row test file is useful for checking file loading or inspecting digit
images. It should not be treated as independent from the larger test file.

## An experiment journal template

```text
Question:
Prediction and why:
Training data / split:
Settings changed (all others fixed):
Training loss / accuracy:
Validation loss / accuracy:
What the plots and mistakes show:
Alternative explanations:
What I would test next:
```
