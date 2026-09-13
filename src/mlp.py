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
import joblib
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

DEFAULT_CONFIG: dict[str, Any] = {
    "hidden_layers": HIDDEN_LAYERS,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "weight_decay": 0.0,
    "batch_norm": False,
    "scheduler": None,  # None | "cosine" | "plateau"
    "selection_metric": "f1",  # f1 | roc_auc | f1_tuned
    "decision_threshold": DECISION_THRESHOLD,
    "seed": RANDOM_STATE,
    # None derives n_negative / n_positive from the training split.
    "pos_weight": None,
}


def build_config(**overrides: Any) -> dict[str, Any]:
    unknown = sorted(set(overrides) - set(DEFAULT_CONFIG))
    if unknown:
        raise KeyError(f"Unknown configuration keys: {unknown}")

    config = {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in DEFAULT_CONFIG.items()
    }
    config.update(overrides)
    return config


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()

# Wide enough for both an unweighted loss, whose optimal threshold sits near
# the positive rate (~0.09), and the pos_weight-weighted one (~0.70).
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


class MLP(nn.Module):
    """Feed-forward network for binary heart-disease classification.

    Outputs a single raw logit (no sigmoid); use BCEWithLogitsLoss for
    training and torch.sigmoid(...) to obtain probabilities at inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
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


def metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return calculate_metrics(y_true, y_pred, y_proba)


# 0.5 is not a neutral threshold here: pos_weight deliberately pushes the
# predicted probabilities upward, so the threshold is tuned on validation too.
def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if grid is None:
        grid = THRESHOLD_GRID

    best_threshold = float(grid[0])
    best_f1 = -1.0
    for threshold in grid:
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, metrics_at_threshold(y_true, y_proba, best_threshold)


# Screening operating point: a missed positive case is costlier than a false
# alarm, so recall is constrained instead of maximising F1.
def find_threshold_at_min_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.70,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if grid is None:
        grid = THRESHOLD_GRID

    best_threshold: float | None = None
    best_precision = -1.0
    for threshold in grid:
        y_pred = (y_proba >= threshold).astype(int)
        if recall_score(y_true, y_pred, zero_division=0) < min_recall:
            continue
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision > best_precision:
            best_precision = float(precision)
            best_threshold = float(threshold)

    if best_threshold is None:
        best_threshold = float(grid[0])

    return best_threshold, metrics_at_threshold(y_true, y_proba, best_threshold)


# Fitted once and reused by every experiment, so all configurations are
# compared on identical inputs.
def prepare_data() -> dict[str, Any]:
    X_train, y_train, X_validation, y_validation = load_train_validation()

    # Fit the shared preprocessing transformer on the training split only,
    # exactly as done for the baselines in src/baselines.py.
    preprocessor = fit_preprocessor(X_train)
    X_train_transformed = transform_features(X_train, preprocessor)
    X_validation_transformed = transform_features(X_validation, preprocessor)

    # Class imbalance (~9.36% positive) is handled the same way the baselines
    # handle it with class_weight="balanced": weight the positive class in
    # the loss instead of resampling the data.
    n_positive = float(y_train.sum())
    n_negative = float(len(y_train) - n_positive)

    return {
        "train_features": to_tensor(X_train_transformed),
        "train_targets": torch.tensor(y_train.to_numpy(dtype=np.float32)),
        "validation_features": to_tensor(X_validation_transformed),
        "validation_targets": y_validation.to_numpy(),
        "y_validation": y_validation,
        "input_dim": X_train_transformed.shape[1],
        "preprocessor": preprocessor,
        "pos_weight": n_negative / n_positive,
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str | None,
    max_epochs: int,
) -> Any:
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
    raise ValueError(f"Unknown scheduler: {name}")


def train_mlp(
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    config = build_config() if config is None else config
    set_seed(config["seed"])

    validation_targets = data["validation_targets"]
    validation_features = data["validation_features"].to(DEVICE)

    train_dataset = TensorDataset(data["train_features"], data["train_targets"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    pos_weight_value = (
        data["pos_weight"] if config["pos_weight"] is None else config["pos_weight"]
    )
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)

    model = MLP(
        input_dim=data["input_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = _build_scheduler(optimizer, config["scheduler"], config["max_epochs"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    selection_metric = config["selection_metric"]
    fixed_threshold = config["decision_threshold"]

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_proba: np.ndarray | None = None
    epochs_without_improvement = 0
    trained_epochs = 0
    history: list[dict[str, float]] = []

    start_time = time.perf_counter()

    for epoch in range(1, config["max_epochs"] + 1):
        trained_epochs = epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features)
            validation_proba = torch.sigmoid(validation_logits).cpu().numpy()
            validation_loss = float(criterion(
                validation_logits,
                torch.as_tensor(validation_targets, dtype=torch.float32, device=DEVICE),
            ).cpu())

        epoch_metrics = metrics_at_threshold(
            validation_targets, validation_proba, fixed_threshold
        )

        # Best model is selected by validation F1, the same metric used to
        # rank the baselines in results/baselines_metrics.csv, because
        # accuracy is misleading on this imbalanced target.
        if selection_metric == "f1_tuned":
            _, tuned = find_best_threshold(validation_targets, validation_proba)
            score = tuned["f1"]
        elif selection_metric == "roc_auc":
            score = epoch_metrics["roc_auc"]
        elif selection_metric == "f1":
            score = epoch_metrics["f1"]
        else:
            raise ValueError(f"Unknown selection_metric: {selection_metric}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "val_loss": validation_loss,
                "selection_score": score,
                **{f"val_{key}": value for key, value in epoch_metrics.items()},
            }
        )

        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_proba = validation_proba
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"epoch {epoch:02d} | "
                f"loss={epoch_loss / max(n_batches, 1):.4f} "
                f"f1={epoch_metrics['f1']:.4f} "
                f"roc_auc={epoch_metrics['roc_auc']:.4f} "
                f"recall={epoch_metrics['recall']:.4f} "
                f"precision={epoch_metrics['precision']:.4f}"
            )

        if scheduler is not None:
            if config["scheduler"] == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()

        if epochs_without_improvement >= config["early_stopping_patience"]:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
            break

    elapsed_seconds = time.perf_counter() - start_time

    assert best_state is not None and best_proba is not None, (
        "training did not complete a single epoch"
    )

    default_metrics = metrics_at_threshold(
        validation_targets, best_proba, fixed_threshold
    )
    best_threshold, tuned_metrics = find_best_threshold(validation_targets, best_proba)
    recall_threshold, recall_metrics = find_threshold_at_min_recall(
        validation_targets, best_proba, min_recall=0.70
    )

    return {
        "config": config,
        "best_epoch": best_epoch,
        "trained_epochs": trained_epochs,
        "seconds": elapsed_seconds,
        "metrics_at_default": default_metrics,
        "best_threshold": best_threshold,
        "metrics_at_best_threshold": tuned_metrics,
        "recall_threshold": recall_threshold,
        "metrics_at_recall_threshold": recall_metrics,
        "history": history,
        "pos_weight_used": float(pos_weight_value),
        "best_state": best_state,
        "validation_proba": best_proba,
        "device": str(DEVICE),
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
    config: dict[str, Any],
    input_dim: int,
    best_epoch: int,
    trained_epochs: int,
    decision_threshold: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "input_dim": input_dim,
        "hidden_layers": config["hidden_layers"],
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
        "early_stopping_patience": config["early_stopping_patience"],
        "weight_decay": config["weight_decay"],
        "batch_norm": config["batch_norm"],
        "scheduler": config["scheduler"],
        "decision_threshold": (
            config["decision_threshold"]
            if decision_threshold is None
            else decision_threshold
        ),
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss(pos_weight=n_negative/n_positive)",
        "best_epoch": best_epoch,
        "trained_epochs": trained_epochs,
        "model_selection_metric": config["selection_metric"],
        "seed": config["seed"],
        "pos_weight": config["pos_weight"],
        "random_state": RANDOM_STATE,
    }
    if extra:
        params.update(extra)

    with open(PARAMS_DIR / f"{model_name}_params.json", "w", encoding="utf-8") as file:
        json.dump(params, file, indent=2, sort_keys=True)


def run_mlp() -> pd.DataFrame:
    data = prepare_data()
    config = build_config()
    result = train_mlp(data, config)

    best_metrics = result["metrics_at_default"]
    validation_pred = (
        result["validation_proba"] >= config["decision_threshold"]
    ).astype(int)

    save_predictions(
        MODEL_NAME, data["y_validation"], validation_pred, result["validation_proba"]
    )

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(result["best_state"], PARAMS_DIR / f"{MODEL_NAME}_model.pt")
    joblib.dump(data["preprocessor"], PARAMS_DIR / f"{MODEL_NAME}_preprocessor.joblib")
    pd.DataFrame(result["history"]).to_csv(RESULTS_DIR / "mlp_history.csv", index=False)

    save_params(
        MODEL_NAME,
        config,
        data["input_dim"],
        result["best_epoch"],
        result["trained_epochs"],
    )

    results = [
        {
            "model": MODEL_NAME,
            **best_metrics,
            "fit_and_validation_seconds": result["seconds"],
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
