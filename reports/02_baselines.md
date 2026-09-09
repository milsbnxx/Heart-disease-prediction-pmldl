# 2. Baselines

This section evaluates several standard machine learning baselines for the heart disease prediction task. The goal is to satisfy the Stage 2 requirement that existing solutions are tested and evaluated with metrics before comparing them with the proposed model.

## Experimental Setup

All baseline models use the same prepared dataset and the same preprocessing pipeline:

- train split: used for fitting preprocessing and model parameters;
- validation split: used for baseline comparison;
- test split: not used for baseline selection, hyperparameter tuning, threshold selection, or preprocessing fit.

The shared preprocessing is implemented in `src/preprocessing.py`:

- numerical features are scaled with `StandardScaler`;
- binary features are passed through unchanged;
- categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")`.

All baseline experiments use `random_state = 42` where supported.

## Models

The baseline models are implemented in `src/baselines.py`:

- Logistic Regression;
- Decision Tree;
- Random Forest.

The models use fixed parameters rather than validation-driven hyperparameter tuning. Class weights are set to `balanced` for these baselines because the positive class is substantially underrepresented.

## Metrics

The following metrics are reported on the validation split:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.

Because the dataset is imbalanced, F1-score, Recall, and ROC-AUC should be considered alongside Accuracy.

## Reproducibility

Run the baseline experiments from the project root:

```bash
python3 src/baselines.py
```

The script saves:

- `results/baselines_metrics.csv`;
- `results/predictions/logistic_regression_validation.csv`;
- `results/predictions/decision_tree_validation.csv`;
- `results/predictions/random_forest_validation.csv`;
- model parameter JSON files in `results/models/`.

## Validation Results

The validation results are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7445 | 0.2362 | 0.7745 | 0.3620 | 0.8346 |
| Decision Tree | 0.8535 | 0.2300 | 0.2409 | 0.2353 | 0.5789 |
| Random Forest | 0.9029 | 0.4091 | 0.0850 | 0.1408 | 0.8012 |

Logistic Regression achieves the best validation F1-score and ROC-AUC among the tested baselines. Random Forest has the highest accuracy, but its recall is much lower, which is important for this imbalanced medical classification task.
