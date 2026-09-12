"""Like-for-like comparison of every model at its own tuned decision threshold.

Section 4 tunes the decision threshold of the proposed MLP, which raises its
F1 substantially. The Section 2 baselines and the Section 3 model are, however,
reported at a fixed threshold of 0.5. Comparing a threshold-tuned model against
fixed-threshold baselines overstates the improvement, so this script re-scores
every model that has saved validation probabilities at:

- the fixed 0.5 threshold (as originally reported);
- the threshold that maximises that model's own validation F1;
- the highest-precision threshold keeping recall at or above 0.70.

It also reports average precision (PR-AUC), which is threshold-free and, unlike
ROC-AUC, is sensitive to the ~9.36% positive rate of this dataset.

No model is retrained: the script only reads the probability columns already
saved in results/predictions/. The test split is not used.

Usage (from the project root):

    python3 src/threshold_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.mlp import THRESHOLD_GRID

RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
OUTPUT_CSV = RESULTS_DIR / "threshold_comparison.csv"

MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "mlp",
    "mlp_tuned",
]


def load_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = PREDICTIONS_DIR / f"{model_name}_validation.csv"
    if not path.exists():
        print(f"skip {model_name}: {path.name} not found")
        return None

    frame = pd.read_csv(path)
    return frame["y_true"].to_numpy(), frame["y_proba"].to_numpy()


def scores(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    f1_scores = [
        f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
        for threshold in THRESHOLD_GRID
    ]
    return float(THRESHOLD_GRID[int(np.argmax(f1_scores))])


def recall_constrained_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.70,
) -> float:
    best_threshold, best_precision = float(THRESHOLD_GRID[0]), -1.0
    for threshold in THRESHOLD_GRID:
        y_pred = (y_proba >= threshold).astype(int)
        if recall_score(y_true, y_pred, zero_division=0) < min_recall:
            continue
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision > best_precision:
            best_precision, best_threshold = float(precision), float(threshold)
    return best_threshold


def analyse() -> pd.DataFrame:
    rows = []
    for model_name in MODELS:
        loaded = load_predictions(model_name)
        if loaded is None:
            continue
        y_true, y_proba = loaded

        fixed = scores(y_true, y_proba, 0.5)
        tuned = scores(y_true, y_proba, best_f1_threshold(y_true, y_proba))
        screening = scores(
            y_true, y_proba, recall_constrained_threshold(y_true, y_proba)
        )

        rows.append(
            {
                "model": model_name,
                "accuracy_at_0.5": fixed["accuracy"],
                "precision_at_0.5": fixed["precision"],
                "recall_at_0.5": fixed["recall"],
                "f1_at_0.5": fixed["f1"],
                "best_threshold": tuned["threshold"],
                "accuracy_at_best": tuned["accuracy"],
                "precision_at_best": tuned["precision"],
                "recall_at_best": tuned["recall"],
                "f1_at_best": tuned["f1"],
                "recall70_threshold": screening["threshold"],
                "precision_at_recall70": screening["precision"],
                "recall_at_recall70": screening["recall"],
                "f1_at_recall70": screening["f1"],
                "roc_auc": roc_auc_score(y_true, y_proba),
                "pr_auc": average_precision_score(y_true, y_proba),
                "evaluated_split": "validation",
            }
        )

    frame = pd.DataFrame(rows).sort_values("f1_at_best", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False)
    return frame


if __name__ == "__main__":
    results = analyse()

    print("\nAll models at their own tuned threshold (validation split):\n")
    print(
        results[
            [
                "model",
                "f1_at_0.5",
                "best_threshold",
                "precision_at_best",
                "recall_at_best",
                "f1_at_best",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    gains = results["f1_at_best"] - results["f1_at_0.5"]
    print(
        f"\nF1 gained from threshold tuning alone: "
        f"min {gains.min():.4f}, max {gains.max():.4f}"
    )
    print(f"Saved to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
