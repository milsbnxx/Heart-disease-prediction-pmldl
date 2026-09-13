# 4. Model Improvements

This section takes the proposed MLP from Section 3, tunes it, selects a final version, and
corrects the protocol used to compare it against the Section 2 baselines. Everything is measured
on the **validation split**; the test split is not loaded anywhere in `src/tuning.py` or
`src/threshold_analysis.py`.

## What Was Tuned

**The decision threshold.** Section 3 converts probabilities into labels at 0.5, which is not a
neutral choice: the loss is weighted with `pos_weight = n_negative / n_positive ≈ 9.69` to
compensate for the ~9.36% positive rate, and that deliberately pushes the predicted probabilities
upward. Keeping 0.5 gives recall 0.77 against precision 0.24. The threshold is therefore treated
as a hyperparameter and selected on validation.

**The training hyperparameters.** Learning rate, dropout, hidden-layer sizes, batch size, weight
decay, batch normalisation and a learning-rate scheduler. A full grid over these seven axes would
be several hundred runs, so a **coordinate search** was used: one axis is varied at a time while
the others stay at their Section 3 values, and the winning value of every axis is then combined
into one configuration (`exp99_combined`). That gives 24 single-axis runs plus the combination.

Two further groups of runs were added to interpret the sweep rather than extend it: five repeats
of the winning configuration under different seeds, and four values of `pos_weight`.

## Experimental Setup

The tuning code is `src/tuning.py`. `src/mlp.py` was refactored so that the training loop takes a
configuration dictionary (`build_config` / `train_mlp`) instead of reading module-level constants;
`DEFAULT_CONFIG` holds exactly the Section 3 values and `run_mlp()` is unchanged in behaviour, so
Section 3 stays reproducible.

- The preprocessing transformer is fitted **once** on the training split and the tensors are
  reused by every experiment, so all configurations see identical inputs.
- Unless an experiment varies it, every run uses seed 42 for `torch`, `numpy` and the
  `DataLoader` shuffle.
- Budget: 30 epochs, early stopping patience 5, against 50/7 in Section 3. This is safe - the
  baseline configuration re-run under the shorter budget (`exp01_baseline`) reproduces the
  Section 3 result exactly: best epoch 10, F1 0.3655 at threshold 0.5.
- Checkpoint selection inside a run is by validation F1 at 0.5, as in Section 3. Configuration
  selection across runs is by validation F1 at the **tuned** threshold.
- Hardware: Apple M-series GPU via the PyTorch MPS backend. 750 seconds for the 25 runs of the
  main sweep, 135 seconds for the 9 additional ones.

Three operating points are recorded per run: **@0.5**, the fixed Section 3 threshold; **@best**,
the threshold maximising validation F1, scanned from 0.01 to 0.99; and **@recall 0.70**, the
highest-precision threshold keeping recall at or above 0.70, which matters because this is a
screening task where a missed positive case is costlier than a false alarm.

## Results of the Main Sweep

Full results in `results/tuning_experiments.csv`, per-epoch curves in `results/tuning_histories.csv`.

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

Two observations. Moving the baseline configuration from threshold 0.5 to its optimal 0.74 raises
F1 from 0.3655 to 0.4070, **+0.0415 (+11.4%)**, without retraining anything. And all 25
configurations land between 0.4031 and 0.4090 - a spread of 0.0059, with ROC-AUC spanning only
0.8353 to 0.8381 - while the combined configuration, built from the winner of every axis, is the
*worst* run in the table.

## How Much of That Spread Is Noise?

The winning configuration was re-run with five seeds, changing nothing else
(`results/tuning_extra.csv`).

| Experiment | Seed | Best epoch | F1 @0.5 | ROC-AUC | Best thr. | F1 @best |
|---|---:|---:|---:|---:|---:|---:|
| exp_seed_0 | 0 | 3 | 0.3613 | 0.8375 | 0.69 | 0.4063 |
| exp_seed_1 | 1 | 1 | 0.3629 | 0.8367 | 0.71 | 0.4038 |
| exp_seed_7 | 7 | 1 | 0.3603 | 0.8361 | 0.70 | 0.4051 |
| exp_seed_42 | 42 | 10 | 0.3618 | 0.8376 | 0.70 | 0.4090 |
| exp_seed_2024 | 2024 | 2 | 0.3607 | 0.8374 | 0.68 | 0.4074 |

Mean 0.4063, standard deviation 0.0020, **range 0.0052** - against 0.0059 for the entire sweep of
25 different architectures, learning rates and regularisation settings. **89% of the spread across
the whole sweep is reproduced by changing nothing but the random seed**, so no single-axis result
above can be called an improvement. This also places the selected model: `exp_hidden_256` and
`exp_seed_42` are the same run, and its 0.4090 is the *maximum* of the five seed draws, 1.3
standard deviations above the mean of its own configuration. Taking the argmax of 25 noisy runs
is guaranteed to pick an upward fluctuation.

A secondary finding: the best epoch swings between seeds (1, 1, 2, 3, 10). Early stopping watches
validation F1 at 0.5, which fluctuates by ±0.005 between epochs while ROC-AUC moves by ±0.001, so
with patience 5 training often stops on a noise dip. Selecting the checkpoint on ROC-AUC would be
more robust; that is left as a recommendation, since changing it would break comparability with
Section 3.

## Is `pos_weight` Just the Threshold in Disguise?

Every run above used `pos_weight ≈ 9.69`. Both it and the decision threshold trade recall against
precision, so four more values were tried on the winning configuration.

| Experiment | pos_weight | F1 @0.5 | ROC-AUC | Best thr. | F1 @best |
|---|---:|---:|---:|---:|---:|
| exp_posw_1 | 1.0 (unweighted) | 0.1639 | 0.8379 | 0.20 | 0.4083 |
| exp_posw_3 | 3.0 | 0.4015 | 0.8375 | 0.44 | 0.4070 |
| exp_posw_5 | 5.0 | 0.4041 | 0.8371 | 0.55 | 0.4071 |
| exp_posw_15 | 15.0 | 0.3261 | 0.8374 | 0.79 | 0.4069 |
| (main sweep) | 9.69 | 0.3618 | 0.8376 | 0.70 | 0.4090 |

At a fixed 0.5 threshold `pos_weight` looks decisive: F1 ranges from 0.164 to 0.404. Read at each
model's own optimal threshold the same runs span 0.4069 to 0.4090, a spread of 0.0021 - well
inside the seed noise - with ROC-AUC flat at 0.837 throughout. The two mechanisms are therefore
interchangeable, and applying both is redundant. Note that `pos_weight = 5` reaches F1 0.404 at
the default threshold, nearly the tuned optimum: the Section 3 model was mis-calibrated for its
own threshold rather than under-trained. The unweighted model is equally good once its threshold
is set to 0.20, so the class imbalance never needed handling in the loss at all.

## Fair Comparison Against the Baselines

Section 2 and Section 3 report every model at a fixed 0.5 threshold, and this section tunes the
MLP's threshold. Comparing the two directly would overstate the improvement, so
`src/threshold_analysis.py` re-scores **every** model with saved validation probabilities at its
own optimal threshold. Nothing is retrained; only the saved probability columns are re-read.
Output: `results/threshold_comparison.csv`.

| Model | F1 @0.5 | Best thr. | Precision @best | Recall @best | **F1 @best** | Gain | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MLP tuned (Section 4)** | 0.3618 | 0.70 | 0.3291 | 0.5401 | **0.4090** | +0.0472 | 0.8376 | 0.3561 |
| MLP (Section 3) | 0.3655 | 0.72 | 0.3349 | 0.5157 | 0.4061 | +0.0406 | 0.8376 | 0.3568 |
| Logistic Regression | 0.3620 | 0.72 | 0.3418 | 0.4875 | 0.4018 | +0.0398 | 0.8346 | 0.3542 |
| Random Forest | 0.1494 | 0.20 | 0.2868 | 0.5147 | 0.3684 | **+0.2190** | 0.8012 | 0.2828 |
| Decision Tree | 0.2353 | 0.96 | 0.2329 | 0.2393 | 0.2361 | +0.0007 | 0.5789 | 0.1271 |

This changes two conclusions from earlier sections.

**Random Forest was not a bad model; it was read at the wrong threshold.** Its F1 rises from 0.149
to 0.368 purely by moving the threshold to 0.20. The recall of 0.085 reported in Section 2 was an
artefact of `class_weight="balanced"` combined with a 0.5 threshold, not evidence about the model
- as its ROC-AUC of 0.80 was already indicating.

**The MLP's advantage over Logistic Regression is real but very small.** Like for like it is
0.4090 against 0.4018, **+0.0072 (+1.8%)**; measured against the seed-mean of 0.4063 it is +0.0045
(+1.1%), which is inside the seed noise. PR-AUC, which is threshold-free and, unlike ROC-AUC,
sensitive to the 9.36% positive rate, agrees: 0.3561 against 0.3542. The figure one could quote
from Section 3 - the MLP beating the best baseline by 13% - is an artefact of the comparison
protocol, not a property of the model.

## Selected Model

The best configuration by validation F1 at the tuned threshold is **`exp_hidden_256`**: a single
hidden layer of 256 units, Section 3 values on every other axis, evaluated at **threshold 0.70**.
Its metrics and those of the two models it should be compared with are the first three rows of
the table above. An alternative operating point is available for every model: at threshold 0.57
the tuned MLP gives precision 0.261 at recall 0.709 (F1 0.382), which is the screening-oriented
choice, since the max-F1 point roughly halves recall to buy precision.

What this section did and did not achieve:

- **Achieved:** +0.0472 F1 over the Section 3 model as reported (+13.1%), essentially all of it
  from correcting the decision threshold, plus a comparison protocol that repairs the Section 2
  ranking of Random Forest.
- **Not achieved:** any improvement from architecture or optimisation. The single-layer network is
  kept because it is the argmax of the search and because it is the *simpler* model - 14 337
  parameters against 15 361 for the two-layer Section 3 network - so nothing is paid for the
  choice. Its 0.0020 F1 advantage over the Section 3 architecture is one standard deviation of
  seed noise and should not be read as a real difference; ROC-AUC is identical at 0.8376.

The reason is visible throughout: 34 runs covering seven hyperparameter axes, two class-balancing
mechanisms and five seeds all produce ROC-AUC between 0.835 and 0.838. The task is limited by the
information in the 17 self-reported BRFSS features, not by model capacity.

## Limitations

The checkpoint, the configuration and the threshold are all selected on the same validation split,
so these numbers are optimistically biased, most of all where the search was widest. The unbiased
estimate is the test-split measurement in Section 5, and some regression is expected - especially
for the threshold, fitted to validation at a resolution of 0.01, and for the selected
configuration, the maximum of 25 noisy draws. Section 5 should report test-split metrics at the
threshold chosen on validation, never re-tuned on test, and should use the comparison table above
rather than the fixed-threshold table from Section 2.

## Reproducibility

```bash
python3 src/tuning.py                  # main sweep, 25 runs, ~13 min on Apple MPS
python3 src/tuning.py --list           # experiment plan, no training
python3 src/tuning.py --stage extra    # seed variance and pos_weight, 9 runs, ~2 min
python3 src/threshold_analysis.py      # fair comparison, no training
python3 src/tuning.py --export-only    # rebuild deliverables from existing runs
```

Each stage appends to its results CSV after every experiment and skips runs already recorded
there, so an interrupted sweep can be restarted without losing them.

Artefacts: per-experiment metrics in `results/tuning_experiments.csv`, `tuning_extra.csv` and
`threshold_comparison.csv`; per-epoch curves for every run in `tuning_histories.csv`; the selected
model itself in `results/mlp_tuned_metrics.csv`,
`results/models/mlp_tuned_model.pt`, `results/models/mlp_tuned_params.json` and
`results/predictions/mlp_tuned_validation.csv`. Intermediate per-experiment checkpoints go to
`results/models/tuning_checkpoints/` and are excluded from version control.
