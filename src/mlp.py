from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import RANDOM_STATE, load_train_validation
from src.preprocessing import fit_preprocessor, transform_features


RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PARAMS_DIR = RESULTS_DIR / "models"

MODEL_NAME = "mlp"

# Model / training hyperparameters. Kept as module-level constants (rather than
# hardcoded inline) so that Stage 2.4 (Model Improvements) can import and vary
# them without rewriting the training loop.
HIDDEN_LAYERS = [128, 64]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
DECISION_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Feed-forward network for binary heart-disease classification.

    Outputs a single raw logit (no sigmoid); use BCEWithLogitsLoss for
    training and torch.sigmoid(...) to obtain probabilities at inference.
    """

    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.to_numpy(dtype=np.float32))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "row_index": y_true.index,
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )
    predictions.to_csv(
        PREDICTIONS_DIR / f"{model_name}_validation.csv",
        index=False,
    )


def save_params(
    model_name: str,
    hyperparameters: dict[str, Any],
    input_dim: int,
    best_epoch: int,
    trained_epochs: int,
) -> None:
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "input_dim": input_dim,
        "hidden_layers": hyperparameters["hidden_layers"],
        "dropout": hyperparameters["dropout"],
        "learning_rate": hyperparameters["learning_rate"],
        "batch_size": hyperparameters["batch_size"],
        "max_epochs": hyperparameters["max_epochs"],
        "early_stopping_patience": hyperparameters["early_stopping_patience"],
        "decision_threshold": DECISION_THRESHOLD,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss(pos_weight=n_negative/n_positive)",
        "best_epoch": best_epoch,
        "trained_epochs": trained_epochs,
        "model_selection_metric": "f1 (validation)",
        "random_state": RANDOM_STATE,
    }

    with open(PARAMS_DIR / f"{model_name}_params.json", "w", encoding="utf-8") as file:
        json.dump(params, file, indent=2, sort_keys=True)


def run_mlp() -> pd.DataFrame:
    set_seed(RANDOM_STATE)

    X_train, y_train, X_validation, y_validation = load_train_validation()

    # Fit the shared preprocessing transformer on the training split only,
    # exactly as done for the baselines in src/baselines.py.
    preprocessor = fit_preprocessor(X_train)
    X_train_transformed = transform_features(X_train, preprocessor)
    X_validation_transformed = transform_features(X_validation, preprocessor)

    input_dim = X_train_transformed.shape[1]

    train_features = to_tensor(X_train_transformed)
    train_targets = torch.tensor(y_train.to_numpy(dtype=np.float32))
    validation_features = to_tensor(X_validation_transformed).to(DEVICE)
    validation_targets = y_validation.to_numpy()

    train_dataset = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    # Class imbalance (~9.36% positive) is handled the same way the baselines
    # handle it with class_weight="balanced": weight the positive class in
    # the loss instead of resampling the data.
    n_positive = float(y_train.sum())
    n_negative = float(len(y_train) - n_positive)
    pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32).to(DEVICE)

    model = MLP(input_dim=input_dim, hidden_layers=HIDDEN_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    trained_epochs = 0

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        trained_epochs = epoch
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features)
            validation_proba = torch.sigmoid(validation_logits).cpu().numpy()
        validation_pred = (validation_proba >= DECISION_THRESHOLD).astype(int)

        epoch_metrics = calculate_metrics(validation_targets, validation_pred, validation_proba)

        # Best model is selected by validation F1, the same metric used to
        # rank the baselines in results/baselines_metrics.csv, because
        # accuracy is misleading on this imbalanced target.
        if epoch_metrics["f1"] > best_f1:
            best_f1 = epoch_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = epoch_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:02d} | "
            f"f1={epoch_metrics['f1']:.4f} "
            f"roc_auc={epoch_metrics['roc_auc']:.4f} "
            f"recall={epoch_metrics['recall']:.4f} "
            f"precision={epoch_metrics['precision']:.4f}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
            break

    elapsed_seconds = time.perf_counter() - start_time

    assert best_state is not None, "training did not complete a single epoch"
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_logits = model(validation_features)
        validation_proba = torch.sigmoid(validation_logits).cpu().numpy()
    validation_pred = (validation_proba >= DECISION_THRESHOLD).astype(int)

    save_predictions(MODEL_NAME, y_validation, validation_pred, validation_proba)

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, PARAMS_DIR / f"{MODEL_NAME}_model.pt")

    hyperparameters = {
        "hidden_layers": HIDDEN_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }
    save_params(MODEL_NAME, hyperparameters, input_dim, best_epoch, trained_epochs)

    results = [
        {
            "model": MODEL_NAME,
            **best_metrics,
            "fit_and_validation_seconds": elapsed_seconds,
            "random_state": RANDOM_STATE,
            "evaluated_split": "validation",
        }
    ]
    metrics_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(RESULTS_DIR / "mlp_metrics.csv", index=False)

    return metrics_df


if __name__ == "__main__":
    metrics = run_mlp()
    print(metrics.to_string(index=False))