from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data import RANDOM_STATE, load_train_validation
from src.preprocessing import get_preprocessor


RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PARAMS_DIR = RESULTS_DIR / "models"


def get_baseline_models() -> dict[str, Any]:
    """Return fixed baseline models used for validation comparison."""
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
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
    y_pred: pd.Series,
    y_proba: pd.Series,
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


def save_params(model_name: str, pipeline: Pipeline) -> None:
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    model = pipeline.named_steps["model"]
    params = {
        key: str(value)
        for key, value in model.get_params().items()
    }

    with open(PARAMS_DIR / f"{model_name}_params.json", "w", encoding="utf-8") as file:
        json.dump(params, file, indent=2, sort_keys=True)


def run_baselines() -> pd.DataFrame:
    X_train, y_train, X_validation, y_validation = load_train_validation()
    baseline_models = get_baseline_models()

    results = []

    for model_name, model in baseline_models.items():
        start_time = time.perf_counter()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", get_preprocessor(scale_numerical=True)),
                ("model", model),
            ]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*encountered in matmul",
                category=RuntimeWarning,
            )
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_validation)
            y_proba = pipeline.predict_proba(X_validation)[:, 1]

        if not np.isfinite(y_proba).all():
            raise ValueError(f"{model_name} produced non-finite probabilities.")

        metrics = calculate_metrics(y_validation, y_pred, y_proba)
        elapsed_seconds = time.perf_counter() - start_time

        save_predictions(model_name, y_validation, y_pred, y_proba)
        save_params(model_name, pipeline)

        results.append(
            {
                "model": model_name,
                **metrics,
                "fit_and_validation_seconds": elapsed_seconds,
                "random_state": RANDOM_STATE,
                "evaluated_split": "validation",
            }
        )

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values("f1", ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(RESULTS_DIR / "baselines_metrics.csv", index=False)

    return metrics_df


if __name__ == "__main__":
    metrics = run_baselines()
    print(metrics.to_string(index=False))
