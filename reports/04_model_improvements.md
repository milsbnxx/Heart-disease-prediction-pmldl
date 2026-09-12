# 4. Model Improvements

This section takes the proposed MLP from Section 3 and improves it. It covers the tuning
procedure, the experiments that were run, the selection of the final version of the model, and a
corrected comparison against the Section 2 baselines.

All measurements in this section are made on the **validation split**. The test split is not
loaded anywhere in `src/tuning.py` or `src/threshold_analysis.py`: it stays untouched for the
final evaluation in Section 5.

## What Was Tuned

Two different kinds of improvement were explored.

**1. The decision threshold.** Section 3 converts probabilities into labels at a fixed threshold
of 0.5. That threshold is not neutral here: the loss is weighted with
`pos_weight = n_negative / n_positive ≈ 9.69` to compensate for the ~9.36% positive rate, which
deliberately pushes the predicted probabilities upward. Keeping 0.5 therefore produces a model
that predicts the positive class far too often - recall 0.77 against precision 0.24. The
threshold is treated here as a hyperparameter and selected on the validation split.

**2. The training hyperparameters.** Learning rate, dropout, hidden-layer sizes, batch size,
weight decay, batch normalisation, and a learning-rate scheduler.

A full grid over these seven axes would be several hundred runs. Instead a **coordinate search**
was used: one axis is varied at a time while the other axes stay at their Section 3 values, and
the winning value of every axis is then combined into one final configuration (`exp99_combined`).
This gives 24 single-axis experiments plus one combined configuration.

Two further groups of runs were added afterwards to interpret the sweep rather than to extend it:
five repeats of the winning configuration under different random seeds, to measure how much of
the observed spread is noise, and four values of `pos_weight`, to check whether it and the
decision threshold do the same job.

## Experimental Setup

The tuning code is `src/tuning.py`. `src/mlp.py` was refactored so that the training loop accepts
a configuration dictionary (`build_config` / `train_mlp`) instead of reading module-level
constants; `DEFAULT_CONFIG` holds exactly the Section 3 values, and `run_mlp()` is unchanged in
behaviour, so Section 3 remains reproducible.

- The preprocessing transformer is fitted **once** on the training split and the resulting tensors
  are reused by every experiment, so all configurations are compared on identical inputs.
- Unless an experiment varies it, every run uses seed 42 for `torch`, `numpy`, and the
  `DataLoader` shuffle.
- Tuning budget: up to 30 epochs with early stopping patience 5, instead of the 50/7 used in
  Section 3. This is safe: the Section 3 run reached its best epoch at 10 and stopped at 17, and
  the baseline configuration re-run under the shorter budget (`exp01_baseline`) reproduces the
  Section 3 result exactly - best epoch 10, F1 0.3655 at threshold 0.5.
- Checkpoint selection inside a run is by validation F1 at threshold 0.5, as in Section 3.
  Configuration selection across runs is by validation F1 at the **tuned** threshold.
- Hardware: Apple M-series GPU via the PyTorch MPS backend. The 25 runs of the main sweep took
  750 seconds in total, the 9 additional runs 135 seconds.

For every experiment three operating points are recorded:

- **@0.5** - the fixed Section 3 threshold;
- **@best** - the threshold maximising validation F1, scanned over 0.01 to 0.99 in steps of 0.01;
- **@recall 0.70** - the highest-precision threshold that still keeps recall at or above 0.70,
  reported because this is a screening task where a missed positive case is costlier than a
  false alarm.

## Results of the Main Sweep

Full results are in `results/tuning_experiments.csv`; per-epoch curves for every run are in
`results/tuning_histories.csv`.

| Experiment | Axis | Value | F1 @0.5 | ROC-AUC | Best thr. | Precision @best | Recall @best | **F1 @best** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| exp01_baseline | baseline | Section 3 | 0.3655 | 0.8376 | 0.74 | 0.3479 | 0.4903 | 0.4070 |
| exp_lr_0.01 | learning_rate | 0.01 | 0.3691 | 0.8353 | 0.67 | 0.3431 | 0.4925 | 0.4044 |
| exp_lr_0.003 | learning_rate | 0.003 | 0.3674 | 0.8369 | 0.69 | 0.3309 | 0.5303 | 0.4075 |
| exp_lr_0.0003 | learning_rate | 0.0003 | 0.3606 | 0.8379 | 0.70 | 0.3344 | 0.5204 | 0.4072 |
| exp_lr_0.0001 | learning_rate | 0.0001 | 0.3575 | 0.8363 | 0.70 | 0.3232 | 0.5454 | 0.4059 |
| exp_dropout_0 | dropout | 0.0 | 0.3704 | 0.8365 | 0.64 | 0.3102 | 0.5813 | 0.4045 |
| exp_dropout_0.1 | dropout | 0.1 | 0.3673 | 0.8369 | 0.67 | 0.3187 | 0.5583 | 0.4057 |
| exp_dropout_0.2 | dropout | 0.2 | 0.3639 | 0.8374 | 0.74 | 0.3532 | 0.4791 | 0.4067 |
| exp_dropout_0.5 | dropout | 0.5 | 0.3630 | 0.8381 | 0.71 | 0.3317 | 0.5306 | 0.4082 |
| **exp_hidden_256** | **hidden_layers** | **[256]** | 0.3618 | 0.8376 | **0.70** | **0.3291** | **0.5401** | **0.4090** |
| exp_hidden_64x32 | hidden_layers | [64, 32] | 0.3654 | 0.8380 | 0.69 | 0.3234 | 0.5434 | 0.4055 |
| exp_hidden_256x128 | hidden_layers | [256, 128] | 0.3655 | 0.8375 | 0.73 | 0.3416 | 0.5087 | 0.4087 |
| exp_hidden_128x64x32 | hidden_layers | [128, 64, 32] | 0.3670 | 0.8373 | 0.70 | 0.3338 | 0.5209 | 0.4068 |
| exp_hidden_256x128x64 | hidden_layers | [256, 128, 64] | 0.3699 | 0.8369 | 0.69 | 0.3331 | 0.5226 | 0.4068 |
| exp_batch_128 | batch_size | 128 | 0.3680 | 0.8380 | 0.65 | 0.3232 | 0.5557 | 0.4087 |
| exp_batch_256 | batch_size | 256 | 0.3625 | 0.8370 | 0.69 | 0.3270 | 0.5322 | 0.4051 |
| exp_batch_1024 | batch_size | 1024 | 0.3586 | 0.8377 | 0.70 | 0.3354 | 0.5185 | 0.4073 |
| exp_batch_2048 | batch_size | 2048 | 0.3589 | 0.8355 | 0.69 | 0.3188 | 0.5590 | 0.4060 |
| exp_wd_1e-05 | weight_decay | 1e-05 | 0.3642 | 0.8377 | 0.69 | 0.3147 | 0.5730 | 0.4062 |
| exp_wd_0.0001 | weight_decay | 0.0001 | 0.3636 | 0.8378 | 0.73 | 0.3405 | 0.5080 | 0.4077 |
| exp_wd_0.001 | weight_decay | 0.001 | 0.3677 | 0.8366 | 0.66 | 0.3169 | 0.5716 | 0.4077 |
| exp_batchnorm | batch_norm | True | 0.3562 | 0.8376 | 0.71 | 0.3244 | 0.5470 | 0.4073 |
| exp_sched_cosine | scheduler | cosine | 0.3644 | 0.8377 | 0.73 | 0.3419 | 0.5031 | 0.4071 |
| exp_sched_plateau | scheduler | plateau | 0.3628 | 0.8378 | 0.72 | 0.3344 | 0.5160 | 0.4058 |
| exp99_combined | combined | all axis winners | 0.3610 | 0.8353 | 0.66 | 0.3230 | 0.5363 | 0.4031 |

Two observations. First, moving the baseline configuration from threshold 0.5 to its optimal
0.74 raises F1 from 0.3655 to 0.4070, **+0.0415 (+11.4%)**, without retraining anything. Second,
all 25 configurations land between 0.4031 and 0.4090 - a total spread of 0.0059 - and ROC-AUC
spans only 0.8353 to 0.8381. The combined configuration, built from the winning value of every
axis, is the *worst* run in the table.

## How Much of That Spread Is Noise?

To find out, the winning configuration was re-run with five different random seeds, changing
nothing else. Results are in `results/tuning_extra.csv`.

| Experiment | Seed | Best epoch | F1 @0.5 | ROC-AUC | Best thr. | F1 @best |
|---|---:|---:|---:|---:|---:|---:|
| exp_seed_0 | 0 | 3 | 0.3613 | 0.8375 | 0.69 | 0.4063 |
| exp_seed_1 | 1 | 1 | 0.3629 | 0.8367 | 0.71 | 0.4038 |
| exp_seed_7 | 7 | 1 | 0.3603 | 0.8361 | 0.70 | 0.4051 |
| exp_seed_42 | 42 | 10 | 0.3618 | 0.8376 | 0.70 | 0.4090 |
| exp_seed_2024 | 2024 | 2 | 0.3607 | 0.8374 | 0.68 | 0.4074 |

Mean 0.4063, standard deviation 0.0020, **range 0.0052**. The entire main sweep - 25 different
architectures, learning rates, batch sizes and regularisation settings - spans 0.0059. In other
words **89% of the spread across the whole sweep is reproduced by changing nothing but the random
seed.** No single-axis result in the table above can be called an improvement: the differences
between them are smaller than, or comparable to, the noise floor.

This also puts the selected model in perspective. `exp_hidden_256` and `exp_seed_42` are the same
run, and its 0.4090 is the *maximum* of the five seed draws, 1.3 standard deviations above the
mean of its own configuration. Selecting the argmax of 25 noisy runs is guaranteed to pick an
upward fluctuation.

A secondary finding: the best epoch varies wildly between seeds (1, 1, 2, 3, 10). Early stopping
watches validation F1 at threshold 0.5, which fluctuates by ±0.005 from epoch to epoch while
ROC-AUC moves by ±0.001. With patience 5 this means training frequently stops on a noise dip.
Selecting the checkpoint on ROC-AUC, which is threshold-free and far smoother, would be a more
robust choice; that is left as a recommendation rather than a change, because it would make the
Section 3 comparison inconsistent.

## Is `pos_weight` Just the Threshold in Disguise?

Every run above used `pos_weight ≈ 9.69`. Both `pos_weight` and the decision threshold trade
recall against precision, so four more values were tried, holding everything else at the winning
configuration.

| Experiment | pos_weight | F1 @0.5 | ROC-AUC | Best thr. | F1 @best |
|---|---:|---:|---:|---:|---:|
| exp_posw_1 | 1.0 (unweighted) | 0.1639 | 0.8379 | 0.20 | 0.4083 |
| exp_posw_3 | 3.0 | 0.4015 | 0.8375 | 0.44 | 0.4070 |
| exp_posw_5 | 5.0 | 0.4041 | 0.8371 | 0.55 | 0.4071 |
| exp_posw_15 | 15.0 | 0.3261 | 0.8374 | 0.79 | 0.4069 |
| (main sweep) | 9.69 | 0.3618 | 0.8376 | 0.70 | 0.4090 |

At the fixed 0.5 threshold, `pos_weight` looks enormously important: F1 ranges from 0.164 to
0.404, a factor of 2.5. Once each model is read at its own optimal threshold, the same runs span
0.4069 to 0.4090 - a spread of 0.0021, well inside the seed noise of 0.0052. ROC-AUC is flat at
0.837 throughout.

So the two mechanisms are interchangeable: weighting the loss and moving the threshold shift the
same decision boundary, and doing both is redundant. Note in particular that `pos_weight = 5`
reaches F1 0.404 at the default threshold of 0.5 - nearly the tuned optimum - which is another
way of saying that the Section 3 model was mis-calibrated for its own threshold rather than
under-trained. The unweighted model (`pos_weight = 1`) is just as good once its threshold is set
to 0.20, which confirms that the class imbalance never needed to be handled in the loss at all.

## Fair Comparison Against the Baselines

Section 3 compares the MLP at threshold 0.5 with baselines also at threshold 0.5, and this
section tunes the MLP's threshold. Comparing a threshold-tuned model with fixed-threshold
baselines would overstate the improvement, so `src/threshold_analysis.py` re-scores **every**
model that has saved validation probabilities at its own optimal threshold. Nothing is
retrained; only the saved probability columns are re-read. Output:
`results/threshold_comparison.csv`.

| Model | F1 @0.5 | Best thr. | Precision @best | Recall @best | **F1 @best** | Gain | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MLP tuned (Section 4)** | 0.3618 | 0.70 | 0.3291 | 0.5401 | **0.4090** | +0.0472 | 0.8376 | 0.3561 |
| MLP (Section 3) | 0.3655 | 0.72 | 0.3349 | 0.5157 | 0.4061 | +0.0406 | 0.8376 | 0.3568 |
| Logistic Regression | 0.3620 | 0.72 | 0.3418 | 0.4875 | 0.4018 | +0.0398 | 0.8346 | 0.3542 |
| Random Forest | 0.1494 | 0.20 | 0.2868 | 0.5147 | 0.3684 | **+0.2190** | 0.8012 | 0.2828 |
| Decision Tree | 0.2353 | 0.96 | 0.2329 | 0.2393 | 0.2361 | +0.0007 | 0.5789 | 0.1271 |

This changes the conclusions of Section 2 and Section 3 in two ways.

**Random Forest was not a bad model; it was read at the wrong threshold.** Its F1 rises from
0.149 to 0.368, a factor of 2.5, purely from moving the threshold to 0.20. Its apparent failure
in Section 2 - recall 0.085 - was an artefact of `class_weight="balanced"` combined with a
threshold of 0.5, not evidence about the model. Its ROC-AUC of 0.80 was already saying this.

**The MLP's advantage over Logistic Regression is real but very small.** On a like-for-like
comparison it is 0.4090 against 0.4018, **+0.0072 (+1.8%)** - and against the seed-mean of the
MLP configuration, 0.4063, it is +0.0045 (+1.1%), which is inside the seed noise. PR-AUC, which
is threshold-free and, unlike ROC-AUC, sensitive to the 9.36% positive rate, tells the same
story: 0.3561 for the tuned MLP against 0.3542 for Logistic Regression, a difference of 0.002.

The headline number one could quote from Section 3 - "the MLP beats the best baseline by 13%" -
is therefore an artefact of the comparison protocol, not a property of the model.

## Selected Model

The best configuration by validation F1 at the tuned threshold is **`exp_hidden_256`**: a single
hidden layer of 256 units, with the Section 3 values on every other axis, evaluated at
**threshold 0.70**.

| | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression @0.72 | 0.8642 | 0.3418 | 0.4875 | 0.4018 | 0.8346 | 0.3542 |
| MLP, Section 3 @0.72 | 0.8588 | 0.3349 | 0.5157 | 0.4061 | 0.8376 | 0.3568 |
| **MLP tuned @0.70** | **0.8540** | 0.3291 | **0.5401** | **0.4090** | 0.8376 | 0.3561 |

An alternative operating point is also available for every model. At the screening-oriented
threshold of 0.57, the tuned MLP gives precision 0.261 at recall 0.709 (F1 0.382). Which point to
deploy is a clinical decision, not a metric decision, and the max-F1 point roughly halves recall
to buy precision.

It should be stated plainly what this section did and did not achieve:

- **Achieved:** +0.0472 F1 over the Section 3 model as reported (+13.1%), essentially all of it
  from correcting the decision threshold, plus a corrected comparison protocol that repairs the
  Section 2 ranking of Random Forest.
- **Not achieved:** any improvement from architecture or optimisation. The single-layer network
  is kept because it is the argmax of the search and because it is the *simpler* model - 14 337
  parameters against 15 361 for the two-layer Section 3 network - so nothing is paid for the
  choice. Its 0.0020 F1 advantage over the Section 3 architecture is one standard deviation of
  seed noise and should not be read as a real difference. ROC-AUC is identical to Section 3 at
  0.8376.

The underlying reason is visible throughout: 34 runs covering seven hyperparameter axes, two
class-balancing mechanisms and five seeds all produce ROC-AUC between 0.835 and 0.838. The task
is limited by the information in the 17 self-reported BRFSS features, not by model capacity, and
no amount of tuning of this model family will move it.

## Limitations

The checkpoint, the configuration, and the threshold are all selected on the same validation
split, so the numbers in this section are optimistically biased, and the bias is largest exactly
where the search was widest. The unbiased estimate is the one measured on the untouched test
split in Section 5, and some regression there is expected - particularly for the threshold, which
was fitted to the validation split at a resolution of 0.01, and for the selected configuration,
which is the maximum of 25 noisy draws.

Section 5 should therefore report the test-split metrics of every model at the threshold chosen
on validation (never re-tuned on test), and should use the corrected comparison of this section
rather than the threshold-0.5 table from Section 2.

## Reproducibility

Run from the project root:

```bash
python3 src/tuning.py                  # main sweep, 25 runs, ~13 minutes on Apple MPS
python3 src/tuning.py --list           # show the experiment plan without training
python3 src/tuning.py --stage extra    # seed variance and pos_weight, 9 runs, ~2 minutes
python3 src/threshold_analysis.py      # fair comparison, no training
python3 src/tuning.py --export-only    # rebuild the deliverables from existing runs
```

Each stage appends to its results CSV after every experiment and skips experiments already
recorded there, so an interrupted sweep can be restarted without losing finished runs.

Artefacts produced:

- `results/tuning_experiments.csv` - one row per main-sweep experiment, all three operating points;
- `results/tuning_histories.csv` - per-epoch loss and validation metrics for every experiment;
- `results/tuning_extra.csv`, `results/tuning_extra_histories.csv` - seed and `pos_weight` runs;
- `results/threshold_comparison.csv` - every model at its own tuned threshold, with PR-AUC;
- `results/tuning_best_history.csv` - per-epoch curves of the selected model, for Section 5;
- `results/mlp_tuned_metrics.csv` - metrics of the selected model;
- `results/models/mlp_tuned_model.pt` - the selected checkpoint;
- `results/models/mlp_tuned_params.json` - its configuration and decision threshold;
- `results/predictions/mlp_tuned_validation.csv` - its validation predictions and probabilities.

Intermediate per-experiment checkpoints are written to
`results/models/tuning_checkpoints/` and are excluded from version control.
