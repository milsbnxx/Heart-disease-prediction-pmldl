# 3. Proposed Model

This section satisfies the Stage 2 requirement of proposing a new model trained specifically for the task, reporting its metrics, to be compared against the baselines from Section 2.

## Architecture

The proposed model is a feed-forward multilayer perceptron (MLP) implemented in PyTorch, defined in `src/mlp.py`.

- Input layer: 54 features, produced by the shared preprocessing transformer in `src/preprocessing.py` (4 scaled numerical features, 6 passthrough binary features, 44 one-hot encoded categorical features).
- Hidden layer 1: `Linear(54 -> 128)` → `ReLU` → `Dropout(p=0.3)`
- Hidden layer 2: `Linear(128 -> 64)` → `ReLU` → `Dropout(p=0.3)`
- Output layer: `Linear(64 -> 1)`, producing a single raw logit (no output activation)

The model outputs a logit rather than a probability so that the numerically stable `BCEWithLogitsLoss` can be used directly during training; `torch.sigmoid` is applied only at inference time to obtain class probabilities.

## Training / Fine-Tuning Procedure

- **Preprocessing**: the same `ColumnTransformer` used for the baselines (`get_preprocessor` via `fit_preprocessor`/`transform_features`) is fit on the training split only, then applied to the validation split. No test-split data is used anywhere in this section.
- **Loss**: `BCEWithLogitsLoss` with `pos_weight = n_negative / n_positive` computed on the training split (≈ 9.69), to counteract the ~9.36% positive-class imbalance. This mirrors the `class_weight="balanced"` approach used for the baseline models, so the imbalance is handled consistently across all models being compared.
- **Optimizer**: Adam, learning rate `1e-3`.
- **Batch size**: 512, with a seeded shuffle (`random_state = 42`) each epoch.
- **Epochs**: up to 50, with early stopping (patience = 7 epochs without improvement).
- **Model selection**: after every epoch, metrics are computed on the validation split; the checkpoint with the highest validation **F1-score** is kept as the best model. F1 was chosen for the same reason it was used to rank the baselines: accuracy is a poor signal on a dataset where the positive class is under 10% of the rows.
- **Decision threshold**: 0.5 on the sigmoid output.
- **Reproducibility**: a fixed random seed (`RANDOM_STATE = 42`) is applied to `torch`, `numpy`, and the training `DataLoader`'s shuffling generator.

Training ran for 17 epochs before early stopping triggered; the best checkpoint was found at **epoch 10** (~74 seconds total on CPU).

## Metrics

Validation-split metrics for the proposed MLP, alongside the Section 2 baselines for direct comparison:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7445 | 0.2362 | 0.7745 | 0.3620 | 0.8346 |
| Decision Tree | 0.8535 | 0.2300 | 0.2409 | 0.2353 | 0.5789 |
| Random Forest | 0.9029 | 0.4091 | 0.0850 | 0.1408 | 0.8012 |
| **MLP (proposed)** | **0.7493** | **0.2393** | **0.7704** | **0.3651** | **0.8374** |

The MLP achieves the best F1-score and ROC-AUC among all models tested so far, marginally ahead of Logistic Regression on both metrics, while keeping a comparably high recall (0.77): important for a screening-style medical classification task, where missing an actual positive case is costlier than a false alarm. Random Forest and the Decision Tree reach higher accuracy but at the cost of much lower recall, which is misleading given the class imbalance.

The proposed model is only marginally ahead of the linear baseline, which suggests the current architecture and hyperparameters (hidden sizes, dropout, learning rate) leave place for improvement. This is addressed in Section 4 (Model Improvements).

## Reproducibility

Run the training from the project root:

```bash
python3 src/mlp.py
```

Requires `torch` in addition to the dependencies used for the baselines. The script saves:

- `results/mlp_metrics.csv`
- `results/predictions/mlp_validation.csv`
- `results/models/mlp_model.pt` (best checkpoint's `state_dict`)
- `results/models/mlp_params.json` (architecture, hyperparameters, best epoch)