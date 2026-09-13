"""Shared validation and scoring rules for Stage 5 (no training or test tuning)."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/evaluation'
MODELS = ['logistic_regression', 'decision_tree', 'random_forest', 'mlp', 'mlp_tuned']
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_predictions(frame, target):
    required = {'row_index', 'y_true', 'y_proba'}
    if not required.issubset(frame):
        raise ValueError(f'Missing prediction columns: {required - set(frame)}')
    frame = frame.sort_values('row_index').reset_index(drop=True)
    if not np.array_equal(frame.row_index.to_numpy(), np.arange(len(target))):
        raise ValueError('Prediction row IDs must exactly cover the dataset row positions.')
    if not np.array_equal(frame.y_true.to_numpy(), np.asarray(target)):
        raise ValueError('Prediction targets do not match the dataset in row order.')
    if not set(frame.y_true.unique()).issubset({0, 1}):
        raise ValueError('Target must be binary.')
    probability = frame.y_proba.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Invalid probabilities.')
    return frame


def scores(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(
        threshold=float(threshold), n_samples=len(y), positive_rate=float(y.mean()),
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, p) if both else np.nan,
        average_precision=average_precision_score(y, p) if y.sum() else np.nan,
        specificity=tn / (tn + fp) if tn + fp else np.nan,
        false_positive_rate=fp / (tn + fp) if tn + fp else np.nan,
        false_negative_rate=fn / (fn + tp) if fn + tp else np.nan,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def select_threshold(y, p):
    # Stable tie rule: smallest threshold among grid points with maximal F1.
    values = [f1_score(y, p >= t, zero_division=0) for t in GRID]
    return float(GRID[int(np.argmax(values))])
