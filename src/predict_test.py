"""Reconstruct/verify models on validation, freeze thresholds, then predict test.

Legacy baselines have parameters but no fitted pipelines. Their fixed configurations
are fitted on train only; original prediction CSVs are never overwritten. The
validation replay is audited before test is loaded. Run once after model selection.
"""
from __future__ import annotations
import json
import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.pipeline import Pipeline
from src.baselines import get_baseline_models
from src.data import load_train_validation, load_final_test
from src.preprocessing import fit_preprocessor, get_preprocessor, transform_features
from src.mlp import MLP
from src.evaluation_common import OUT, MODELS, sha256, select_threshold, validate_predictions

ARTIFACTS = OUT / 'models'
PREDICTIONS = OUT / 'predictions'


def mlp_probability(model, features):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            x = torch.as_tensor(features[start:start + 4096], dtype=torch.float32)
            chunks.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(chunks)


def prediction_frame(y, p, threshold):
    return pd.DataFrame({'row_index': np.arange(len(y)), 'y_true': np.asarray(y),
                         'y_pred': (p >= threshold).astype(int), 'y_proba': p})


def main():
    global OUT, ARTIFACTS, PREDICTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='New directory for a separate frozen evaluation (relative to cwd).')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    ARTIFACTS, PREDICTIONS = OUT / 'models', OUT / 'predictions'
    for directory in (OUT, ARTIFACTS, PREDICTIONS):
        directory.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Evaluation already frozen. Use evaluation.py to regenerate analysis. '
                           'For a new experiment use a separate output directory/version.')
    torch.set_num_threads(4)
    X_train, y_train, X_val, y_val = load_train_validation()
    preprocessor = fit_preprocessor(X_train)
    val_features = transform_features(X_val, preprocessor).to_numpy(dtype=np.float32)
    joblib.dump(preprocessor, ARTIFACTS / 'mlp_preprocessor.joblib')
    predictors, validation, audit, thresholds = {}, {}, [], {}
    baseline_models = get_baseline_models()
    sources = {f'data/processed/{s}_ready.csv': sha256(ROOT / f'data/processed/{s}_ready.csv')
               for s in ('train', 'validation')}
    for name in MODELS:
        params_path = ROOT / f'results/models/{name}_params.json'
        sources[str(params_path.relative_to(ROOT))] = sha256(params_path)
        params = json.loads(params_path.read_text())
        old_path = ROOT / f'results/predictions/{name}_validation.csv'
        old = validate_predictions(pd.read_csv(old_path), y_val)
        sources[str(old_path.relative_to(ROOT))] = sha256(old_path)
        if name in baseline_models:
            estimator = baseline_models[name]
            # Refuse silent changes to the saved baseline configurations.
            for key, value in params.items():
                # sklearn 1.8 replaced legacy LR defaults with l1_ratio=0.
                # Binary liblinear with these defaults is still L2 regularized.
                equivalent_lr_defaults = {
                    'l1_ratio': ('None', '0.0'),
                    'penalty': ('l2', 'deprecated'),
                    'multi_class': ('deprecated', 'None'),
                }
                if (name == 'logistic_regression' and
                    equivalent_lr_defaults.get(key) ==
                        (value, str(estimator.get_params().get(key)))):
                    continue
                if key not in estimator.get_params() or str(estimator.get_params()[key]) != value:
                    raise ValueError(f'{name}: saved parameter differs: {key}={value}')
            pipeline = Pipeline([('preprocessor', get_preprocessor()), ('model', estimator)])
            print(f'Refitting fixed {name} on train...', flush=True)
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_val)[:, 1]
            joblib.dump(pipeline, ARTIFACTS / f'{name}_pipeline.joblib', compress=3)
            predictors[name] = pipeline
            source = 'fixed baseline refitted on train in recorded environment'
        else:
            config = params.get('config', params)
            if params['input_dim'] != val_features.shape[1]:
                raise ValueError(f'{name}: preprocessing dimension mismatch')
            model = MLP(params['input_dim'], config['hidden_layers'], config['dropout'],
                        config.get('batch_norm', False))
            weights = ROOT / f'results/models/{name}_model.pt'
            sources[str(weights.relative_to(ROOT))] = sha256(weights)
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
            model.eval()
            probability = mlp_probability(model, val_features)
            predictors[name] = model
            source = 'original saved weights, train-reconstructed preprocessing, CPU inference'
            if not np.allclose(probability, old.y_proba, atol=2e-5, rtol=2e-5):
                raise ValueError(f'{name}: reconstructed preprocessing/weights fail validation replay')
        validate_predictions(prediction_frame(y_val, probability, 0.5), y_val)
        delta = np.abs(probability - old.y_proba.to_numpy())
        audit.append({'model': name, 'source': source,
                      'validation_max_probability_difference': float(delta.max()),
                      'validation_mean_probability_difference': float(delta.mean()),
                      'matches_legacy_at_2e-5': bool(np.allclose(probability, old.y_proba, atol=2e-5, rtol=2e-5)),
                      'legacy_label_disagreements_at_0.5': int((old.y_pred != (old.y_proba >= 0.5)).sum())})
        validation[name] = probability
        # Select only on the validation replay of the exact evaluated estimator.
        thresholds[name] = select_threshold(y_val.to_numpy(), probability)
        print(f'{name}: max replay difference={delta.max():.8g}, frozen threshold={thresholds[name]:.2f}', flush=True)
    pd.DataFrame(audit).to_csv(OUT / 'validation_replay_audit.csv', index=False)
    protocol = {'selection_split': 'validation', 'selection_metric': 'f1',
                'threshold_grid': '0.01 to 0.99 inclusive, step 0.01',
                'tie_rule': 'smallest threshold', 'prediction_rule': 'probability >= threshold',
                'thresholds': thresholds, 'selected_model': 'mlp_tuned',
                'selected_model_reason': 'Stage 4 validation selection; unchanged by test results'}
    (OUT / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for name, p in validation.items():
        prediction_frame(y_val, p, thresholds[name]).to_csv(PREDICTIONS / f'{name}_validation.csv', index=False)
    print('All thresholds frozen. Loading final test for inference only.', flush=True)
    X_test, y_test = load_final_test()
    test_features = transform_features(X_test, preprocessor).to_numpy(dtype=np.float32)
    for name in MODELS:
        predictor = predictors[name]
        p = (predictor.predict_proba(X_test)[:, 1] if name in baseline_models
             else mlp_probability(predictor, test_features))
        frame = validate_predictions(prediction_frame(y_test, p, thresholds[name]), y_test)
        frame.to_csv(PREDICTIONS / f'{name}_test.csv', index=False)
    sources['data/processed/test_ready.csv'] = sha256(ROOT / 'data/processed/test_ready.csv')
    outputs = {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob('*'))
               if p.is_file() and (p.parent in (PREDICTIONS, ARTIFACTS) or p.name == 'frozen_protocol.json')}
    manifest = {'schema_version': 2, 'artifacts_relative_to': 'output_dir', 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'scikit_learn': sklearn.__version__, 'torch': torch.__version__, 'device': 'cpu',
                'sources_sha256': sources, 'artifacts_sha256': outputs,
                'train_n': len(y_train), 'validation_n': len(y_val), 'test_n': len(y_test)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved test predictions and provenance to {OUT}', flush=True)


if __name__ == '__main__':
    main()
