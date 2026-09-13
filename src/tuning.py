# Coordinate search around the proposed model: one hyperparameter axis is
# varied at a time while the others stay at their default values. A full grid
# over seven axes would be several hundred runs at ~30 seconds each.
#
# --stage extra adds two diagnostic groups: the same configuration under
# several seeds, which measures the noise floor of the whole sweep, and
# several pos_weight values, which competes with the decision threshold.
#
# Everything is measured on validation; the test split is never loaded here.
#
#   python3 src/tuning.py                  main sweep
#   python3 src/tuning.py --list           plan only, no training
#   python3 src/tuning.py --stage extra    seed variance and pos_weight
#
# Results are appended after every experiment and finished runs are skipped,
# so an interrupted sweep can be restarted without losing them.

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data import RANDOM_STATE
from src.mlp import (
    DEVICE,
    MLP,
    PARAMS_DIR,
    RESULTS_DIR,
    build_config,
    prepare_data,
    save_predictions,
    train_mlp,
)


EXPERIMENTS_CSV = RESULTS_DIR / "tuning_experiments.csv"
HISTORIES_CSV = RESULTS_DIR / "tuning_histories.csv"
EXTRA_CSV = RESULTS_DIR / "tuning_extra.csv"
EXTRA_HISTORIES_CSV = RESULTS_DIR / "tuning_extra_histories.csv"
CHECKPOINTS_DIR = PARAMS_DIR / "tuning_checkpoints"

TUNED_MODEL_NAME = "mlp_tuned"

# Shorter budget than Section 3 (50 epochs / patience 7): the Section 3 run
# reached its best epoch at 10 and stopped at 17, so 30 epochs with patience 5
# is enough to separate configurations while keeping the whole sweep affordable.
TUNING_MAX_EPOCHS = 30
TUNING_PATIENCE = 5

SEEDS = [0, 1, 7, 42, 2024]
POS_WEIGHTS = [1.0, 3.0, 5.0, 15.0]


def stage_paths(stage: str) -> tuple[Path, Path]:
    if stage == "main":
        return EXPERIMENTS_CSV, HISTORIES_CSV
    if stage == "extra":
        return EXTRA_CSV, EXTRA_HISTORIES_CSV
    raise ValueError(f"Unknown stage: {stage}")


def get_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = [
        {
            "name": "exp01_baseline",
            "axis": "baseline",
            "value": "section 3 config",
            "overrides": {},
        }
    ]

    for learning_rate in [1e-2, 3e-3, 3e-4, 1e-4]:
        experiments.append(
            {
                "name": f"exp_lr_{learning_rate:g}",
                "axis": "learning_rate",
                "value": learning_rate,
                "overrides": {"learning_rate": learning_rate},
            }
        )

    for dropout in [0.0, 0.1, 0.2, 0.5]:
        experiments.append(
            {
                "name": f"exp_dropout_{dropout:g}",
                "axis": "dropout",
                "value": dropout,
                "overrides": {"dropout": dropout},
            }
        )

    for hidden_layers in [[256], [64, 32], [256, 128], [128, 64, 32], [256, 128, 64]]:
        label = "x".join(str(size) for size in hidden_layers)
        experiments.append(
            {
                "name": f"exp_hidden_{label}",
                "axis": "hidden_layers",
                "value": label,
                "overrides": {"hidden_layers": hidden_layers},
            }
        )

    for batch_size in [128, 256, 1024, 2048]:
        experiments.append(
            {
                "name": f"exp_batch_{batch_size}",
                "axis": "batch_size",
                "value": batch_size,
                "overrides": {"batch_size": batch_size},
            }
        )

    for weight_decay in [1e-5, 1e-4, 1e-3]:
        experiments.append(
            {
                "name": f"exp_wd_{weight_decay:g}",
                "axis": "weight_decay",
                "value": weight_decay,
                "overrides": {"weight_decay": weight_decay},
            }
        )

    experiments.append(
        {
            "name": "exp_batchnorm",
            "axis": "batch_norm",
            "value": True,
            "overrides": {"batch_norm": True},
        }
    )

    for scheduler in ["cosine", "plateau"]:
        experiments.append(
            {
                "name": f"exp_sched_{scheduler}",
                "axis": "scheduler",
                "value": scheduler,
                "overrides": {"scheduler": scheduler},
            }
        )

    return experiments


def best_experiment_overrides() -> tuple[str, dict[str, Any]]:
    if not EXPERIMENTS_CSV.exists():
        raise RuntimeError(
            f"{EXPERIMENTS_CSV.name} not found - run the main stage first."
        )

    frame = pd.read_csv(EXPERIMENTS_CSV)
    name = str(frame.loc[frame["f1_at_best"].idxmax()]["experiment"])
    with open(CHECKPOINTS_DIR / f"{name}.json", encoding="utf-8") as file:
        config = json.load(file)["config"]

    defaults = build_config()
    overrides = {
        key: value
        for key, value in config.items()
        if key in defaults and value != defaults[key]
        and key not in {"max_epochs", "early_stopping_patience"}
    }
    return name, overrides


def get_extra_experiments() -> list[dict[str, Any]]:
    name, overrides = best_experiment_overrides()
    print(f"Extra stage is based on {name}: {overrides or 'section 3 defaults'}")

    experiments: list[dict[str, Any]] = []

    # Repeat the winning configuration under different random seeds. The spread
    # of these runs is the noise floor of the whole sweep: any difference
    # between configurations smaller than it cannot be called an improvement.
    for seed in SEEDS:
        experiments.append(
            {
                "name": f"exp_seed_{seed}",
                "axis": "seed",
                "value": seed,
                "overrides": {**overrides, "seed": seed},
            }
        )

    # pos_weight was held at n_negative / n_positive (~9.69) for every run of
    # the main sweep. It moves the predicted probabilities in the same
    # direction the decision threshold does, so it is varied here to check
    # whether the two are interchangeable once the threshold is tuned.
    for pos_weight in POS_WEIGHTS:
        experiments.append(
            {
                "name": f"exp_posw_{pos_weight:g}",
                "axis": "pos_weight",
                "value": pos_weight,
                "overrides": {**overrides, "pos_weight": pos_weight},
            }
        )

    return experiments


def flatten_result(
    experiment: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    config = result["config"]
    default_metrics = result["metrics_at_default"]
    tuned_metrics = result["metrics_at_best_threshold"]
    recall_metrics = result["metrics_at_recall_threshold"]

    return {
        "experiment": experiment["name"],
        "axis": experiment["axis"],
        "value": experiment["value"],
        "hidden_layers": "x".join(str(size) for size in config["hidden_layers"]),
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "weight_decay": config["weight_decay"],
        "batch_norm": config["batch_norm"],
        "scheduler": config["scheduler"] or "none",
        "seed": config["seed"],
        "pos_weight": result["pos_weight_used"],
        "best_epoch": result["best_epoch"],
        "trained_epochs": result["trained_epochs"],
        "seconds": round(result["seconds"], 2),
        "accuracy_at_0.5": default_metrics["accuracy"],
        "precision_at_0.5": default_metrics["precision"],
        "recall_at_0.5": default_metrics["recall"],
        "f1_at_0.5": default_metrics["f1"],
        "roc_auc": default_metrics["roc_auc"],
        "best_threshold": result["best_threshold"],
        "accuracy_at_best": tuned_metrics["accuracy"],
        "precision_at_best": tuned_metrics["precision"],
        "recall_at_best": tuned_metrics["recall"],
        "f1_at_best": tuned_metrics["f1"],
        "recall70_threshold": result["recall_threshold"],
        "precision_at_recall70": recall_metrics["precision"],
        "recall_at_recall70": recall_metrics["recall"],
        "f1_at_recall70": recall_metrics["f1"],
        "device": result["device"],
        "random_state": RANDOM_STATE,
        "evaluated_split": "validation",
    }


def append_row(row: dict[str, Any], path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_history(name: str, history: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(history)
    frame.insert(0, "experiment", name)
    # Older logs lack val_loss. Align by column name when extending the schema,
    # otherwise appending a wider row under the old CSV header corrupts the log.
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True, sort=False)
    frame.to_csv(path, index=False)


def save_checkpoint(name: str, result: dict[str, Any]) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(result["best_state"], CHECKPOINTS_DIR / f"{name}.pt")
    payload = {
        "config": result["config"],
        "best_epoch": result["best_epoch"],
        "trained_epochs": result["trained_epochs"],
        "best_threshold": result["best_threshold"],
        "metrics_at_default": result["metrics_at_default"],
        "metrics_at_best_threshold": result["metrics_at_best_threshold"],
    }
    with open(CHECKPOINTS_DIR / f"{name}.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def completed_experiments(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["experiment"].astype(str))


def run_experiment(
    data: dict[str, Any],
    experiment: dict[str, Any],
    max_epochs: int,
    patience: int,
    verbose: bool,
    results_csv: Path,
    histories_csv: Path,
) -> dict[str, Any]:
    config = build_config(
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        **experiment["overrides"],
    )

    print(f"\n=== {experiment['name']} ({experiment['axis']} = {experiment['value']}) ===")
    result = train_mlp(data, config, verbose=verbose)

    row = flatten_result(experiment, result)
    append_row(row, results_csv)
    append_history(experiment["name"], result["history"], histories_csv)
    save_checkpoint(experiment["name"], result)

    print(
        f"-> f1@0.5={row['f1_at_0.5']:.4f} | "
        f"f1@{row['best_threshold']:.2f}={row['f1_at_best']:.4f} | "
        f"roc_auc={row['roc_auc']:.4f} | {row['seconds']:.0f}s"
    )
    return row


# Winning value of every axis, combined. Axes that did not beat the baseline
# keep their default value.
def build_combined_overrides(frame: pd.DataFrame) -> dict[str, Any]:
    baseline = frame.loc[frame["experiment"] == "exp01_baseline"]
    if baseline.empty:
        raise RuntimeError("exp01_baseline is missing from the results file")
    baseline_f1 = float(baseline["f1_at_best"].iloc[0])

    overrides: dict[str, Any] = {}
    for axis in [
        "learning_rate",
        "dropout",
        "hidden_layers",
        "batch_size",
        "weight_decay",
        "batch_norm",
        "scheduler",
    ]:
        axis_rows = frame.loc[frame["axis"] == axis]
        if axis_rows.empty:
            continue
        best_row = axis_rows.loc[axis_rows["f1_at_best"].idxmax()]
        if float(best_row["f1_at_best"]) <= baseline_f1:
            continue

        if axis == "hidden_layers":
            overrides["hidden_layers"] = [
                int(size) for size in str(best_row["hidden_layers"]).split("x")
            ]
        elif axis == "batch_size":
            overrides["batch_size"] = int(best_row["batch_size"])
        elif axis == "batch_norm":
            overrides["batch_norm"] = bool(best_row["batch_norm"])
        elif axis == "scheduler":
            scheduler = str(best_row["scheduler"])
            overrides["scheduler"] = None if scheduler == "none" else scheduler
        else:
            overrides[axis] = float(best_row[axis])

    return overrides


def summarise_extra() -> None:
    frame = pd.read_csv(EXTRA_CSV)

    seeds = frame.loc[frame["axis"] == "seed"]
    if not seeds.empty:
        print("\n--- Seed variance (same configuration, different seeds) ---")
        print(
            seeds[["experiment", "seed", "f1_at_0.5", "roc_auc", "best_threshold", "f1_at_best"]]
            .to_string(index=False)
        )
        print(
            f"f1_at_best: mean={seeds['f1_at_best'].mean():.4f} "
            f"std={seeds['f1_at_best'].std():.4f} "
            f"range={seeds['f1_at_best'].max() - seeds['f1_at_best'].min():.4f}"
        )
        if EXPERIMENTS_CSV.exists():
            main = pd.read_csv(EXPERIMENTS_CSV)
            spread = main["f1_at_best"].max() - main["f1_at_best"].min()
            print(f"main sweep spread across 25 configurations: {spread:.4f}")

    pos_weights = frame.loc[frame["axis"] == "pos_weight"]
    if not pos_weights.empty:
        print("\n--- pos_weight vs decision threshold ---")
        print(
            pos_weights[
                ["experiment", "pos_weight", "f1_at_0.5", "roc_auc", "best_threshold", "f1_at_best"]
            ].to_string(index=False)
        )


def export_best_model(data: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(EXPERIMENTS_CSV)
    best_row = frame.loc[frame["f1_at_best"].idxmax()]
    name = str(best_row["experiment"])

    with open(CHECKPOINTS_DIR / f"{name}.json", encoding="utf-8") as file:
        payload = json.load(file)
    config = payload["config"]
    threshold = float(payload["best_threshold"])

    model = MLP(
        input_dim=data["input_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / f"{name}.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(data["validation_features"].to(DEVICE))
        proba = torch.sigmoid(logits).cpu().numpy()
    predictions = (proba >= threshold).astype(int)

    save_predictions(TUNED_MODEL_NAME, data["y_validation"], predictions, proba)
    torch.save(model.state_dict(), PARAMS_DIR / f"{TUNED_MODEL_NAME}_model.pt")
    import joblib
    joblib.dump(data["preprocessor"], PARAMS_DIR / f"{TUNED_MODEL_NAME}_preprocessor.joblib")

    params = dict(payload)
    params.update(
        {
            "source_experiment": name,
            "input_dim": data["input_dim"],
            "decision_threshold": threshold,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss(pos_weight=n_negative/n_positive)",
            "random_state": RANDOM_STATE,
        }
    )
    with open(
        PARAMS_DIR / f"{TUNED_MODEL_NAME}_params.json", "w", encoding="utf-8"
    ) as file:
        json.dump(params, file, indent=2, sort_keys=True)

    tuned = payload["metrics_at_best_threshold"]
    metrics_df = pd.DataFrame(
        [
            {
                "model": TUNED_MODEL_NAME,
                **tuned,
                "decision_threshold": threshold,
                "source_experiment": name,
                "random_state": RANDOM_STATE,
                "evaluated_split": "validation",
            }
        ]
    )
    metrics_df.to_csv(RESULTS_DIR / "mlp_tuned_metrics.csv", index=False)

    histories = pd.read_csv(HISTORIES_CSV)
    histories.loc[histories["experiment"] == name].to_csv(
        RESULTS_DIR / "tuning_best_history.csv", index=False
    )

    print(f"\nBest experiment: {name} (threshold {threshold:.2f})")
    print(metrics_df.to_string(index=False))
    return {"experiment": name, "threshold": threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 4 hyperparameter tuning")
    parser.add_argument("--stage", choices=["main", "extra"], default="main")
    parser.add_argument("--max-epochs", type=int, default=TUNING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=TUNING_PATIENCE)
    parser.add_argument("--only", type=str, default=None, help="comma-separated names")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--rerun", action="store_true", help="ignore finished runs")
    parser.add_argument("--quiet", action="store_true", help="hide per-epoch output")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="skip training, only rebuild the Section 4 deliverables",
    )
    args = parser.parse_args()

    results_csv, histories_csv = stage_paths(args.stage)

    if args.list:
        experiments = (
            get_experiments() if args.stage == "main" else get_extra_experiments()
        )
        for experiment in experiments:
            print(f"{experiment['name']:<24} {experiment['axis']} = {experiment['value']}")
        extra_note = " + 1 combined configuration" if args.stage == "main" else ""
        print(f"\n{len(experiments)} experiments{extra_note}")
        return

    print(f"Device: {DEVICE}")
    data = prepare_data()
    print(f"Input dimension: {data['input_dim']}, pos_weight: {data['pos_weight']:.2f}")

    if args.export_only:
        export_best_model(data)
        return

    if args.rerun:
        for path in (results_csv, histories_csv):
            if path.exists():
                path.unlink()
        if args.stage == "main" and CHECKPOINTS_DIR.exists():
            shutil.rmtree(CHECKPOINTS_DIR)

    experiments = get_experiments() if args.stage == "main" else get_extra_experiments()

    selected = None
    if args.only:
        selected = {name.strip() for name in args.only.split(",")}

    done = completed_experiments(results_csv)

    for experiment in experiments:
        if selected is not None and experiment["name"] not in selected:
            continue
        if experiment["name"] in done:
            print(f"skip {experiment['name']} (already in {results_csv.name})")
            continue
        run_experiment(
            data,
            experiment,
            args.max_epochs,
            args.patience,
            not args.quiet,
            results_csv,
            histories_csv,
        )

    if selected is not None:
        return

    if args.stage == "extra":
        summarise_extra()
        return

    # Stage 2 of the main sweep: combine the winning value of every axis.
    frame = pd.read_csv(EXPERIMENTS_CSV)
    if "exp99_combined" not in set(frame["experiment"].astype(str)):
        overrides = build_combined_overrides(frame)
        print(f"\nCombined configuration: {overrides or 'baseline (no axis improved)'}")
        combined = {
            "name": "exp99_combined",
            "axis": "combined",
            "value": json.dumps(overrides, sort_keys=True),
            "overrides": overrides,
        }
        run_experiment(
            data,
            combined,
            args.max_epochs,
            args.patience,
            not args.quiet,
            results_csv,
            histories_csv,
        )

    export_best_model(data)


if __name__ == "__main__":
    main()
