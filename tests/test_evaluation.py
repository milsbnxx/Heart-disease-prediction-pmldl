"""Protocol checks: alignment, thresholds, metric edge cases and paired inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.evaluation_common import MODELS, select_threshold, scores, validate_predictions


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 0, 1])
        self.frame = pd.DataFrame({'row_index': [0, 1, 2, 3], 'y_true': self.y,
                                   'y_proba': [0.1, 0.8, 0.5, 0.4]})

    def test_reorders_by_id_and_rejects_misalignment(self):
        pd.testing.assert_frame_equal(validate_predictions(self.frame.iloc[::-1], self.y), self.frame)
        corrupt = self.frame.copy()
        corrupt.loc[0, 'y_true'] = 1
        with self.assertRaises(ValueError):
            validate_predictions(corrupt, self.y)
        corrupt = self.frame.copy()
        corrupt.loc[0, 'row_index'] = 1
        with self.assertRaises(ValueError):
            validate_predictions(corrupt, self.y)

    def test_rejects_missing_and_invalid_probabilities(self):
        for invalid in [np.nan, np.inf, -0.1, 1.1]:
            corrupt = self.frame.copy()
            corrupt.loc[0, 'y_proba'] = invalid
            with self.assertRaises(ValueError):
                validate_predictions(corrupt, self.y)
        with self.assertRaises(ValueError):
            validate_predictions(self.frame.iloc[:-1], self.y)

    def test_explicit_threshold_equality_and_error_counts(self):
        result = scores(self.y, self.frame.y_proba, 0.5)
        self.assertEqual([result[k] for k in ['tn', 'fp', 'fn', 'tp']], [1, 1, 1, 1])
        self.assertEqual(result['f1'], 0.5)
        all_negative = scores(self.y, np.zeros(4), 0.5)
        self.assertEqual(all_negative['precision'], 0)
        self.assertEqual(all_negative['recall'], 0)
        self.assertEqual(all_negative['f1'], 0)
        single_class = scores([0, 0], [0.2, 0.1], 0.5)
        self.assertTrue(np.isnan(single_class['roc_auc']))

    def test_threshold_selection_stable_ties(self):
        # Every threshold from .21 through .80 perfectly separates these scores.
        self.assertEqual(select_threshold(np.array([0, 1]), np.array([0.2, 0.8])), 0.21)

    def test_paired_bootstrap_identical_models_have_zero_difference(self):
        from src import evaluation
        # Enough rows to avoid a one-class resample for this deterministic seed.
        frame = pd.concat([self.frame] * 20, ignore_index=True)
        frames = {m: frame.copy() for m in MODELS}
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluation, 'OUT', Path(directory)):
            _, differences = evaluation.bootstrap(frames, dict.fromkeys(MODELS, 0.5), 100, 42)
        np.testing.assert_array_equal(differences[['estimate', 'lower_95', 'upper_95']], 0)

    def test_training_exports_finite_validation_loss(self):
        from src import mlp
        rng = np.random.default_rng(42)
        data = {'train_features': torch.tensor(rng.normal(size=(16, 3)), dtype=torch.float32),
                'train_targets': torch.tensor([0, 1] * 8, dtype=torch.float32),
                'validation_features': torch.tensor(rng.normal(size=(8, 3)), dtype=torch.float32),
                'validation_targets': np.array([0, 1] * 4), 'input_dim': 3, 'pos_weight': 1.0}
        config = mlp.build_config(hidden_layers=[4], batch_size=8, max_epochs=2, dropout=0.0)
        with patch.object(mlp, 'DEVICE', torch.device('cpu')):
            result = mlp.train_mlp(data, config, verbose=False)
        self.assertEqual(len(result['history']), 2)
        self.assertTrue(all(np.isfinite(h['val_loss']) and h['val_loss'] > 0 for h in result['history']))

    def test_extending_legacy_history_keeps_columns_aligned(self):
        from src.tuning import append_history
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.csv'
            append_history('old', [{'epoch': 1, 'train_loss': 0.9, 'val_f1': 0.4}], path)
            append_history('new', [{'epoch': 1, 'train_loss': 0.8, 'val_loss': 0.85, 'val_f1': 0.5}], path)
            result = pd.read_csv(path)
        self.assertEqual(result.val_f1.tolist(), [0.4, 0.5])
        self.assertTrue(np.isnan(result.val_loss.iloc[0]))
        self.assertEqual(result.val_loss.iloc[1], 0.85)

    def test_report_keeps_small_positive_interval_bound(self):
        from src.evaluation import markdown_table
        table = markdown_table(pd.DataFrame([{'lower_95': 0.000029846}]), ['lower_95'], digits=6)
        self.assertIn('0.000030', table)

    def test_custom_output_is_portable_and_detects_tampering(self):
        from src import evaluation
        from src.evaluation_common import sha256
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'predictions').mkdir()
            data = output / 'predictions/model.csv'
            data.write_text('y_true,y_proba\n1,0.8\n')
            manifest = {'sources_sha256': {}, 'artifacts_relative_to': 'output_dir',
                        'artifacts_sha256': {'predictions/model.csv': sha256(data),
                                             'models/optional.joblib': 'not-distributed'}}
            with patch.object(evaluation, 'OUT', output):
                evaluation.verify_manifest(manifest)
                self.assertEqual(evaluation.report_path(), output / '05_results.md')
                data.write_text('changed')
                with self.assertRaises(ValueError):
                    evaluation.verify_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
