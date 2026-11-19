import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src.tuning as tuning


class TuningReproducibilityTests(unittest.TestCase):
    def test_best_config_is_recovered_without_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as directory:
            old_path = tuning.EXPERIMENTS_CSV
            old_checkpoints = tuning.CHECKPOINTS_DIR
            try:
                tuning.EXPERIMENTS_CSV = Path(directory) / "tuning_experiments.csv"
                tuning.CHECKPOINTS_DIR = Path(directory) / "missing_checkpoints"
                pd.DataFrame([
                    {
                        "experiment": "baseline",
                        "f1_at_best": 0.40,
                        "hidden_layers": "128x64",
                        "dropout": 0.3,
                        "learning_rate": 0.001,
                        "batch_size": 512,
                        "weight_decay": 0.0,
                        "batch_norm": False,
                        "scheduler": "none",
                    },
                    {
                        "experiment": "best",
                        "f1_at_best": 0.41,
                        "hidden_layers": "256",
                        "dropout": 0.2,
                        "learning_rate": 0.0003,
                        "batch_size": 256,
                        "weight_decay": 0.0001,
                        "batch_norm": True,
                        "scheduler": "cosine",
                    },
                ]).to_csv(tuning.EXPERIMENTS_CSV, index=False)

                name, overrides = tuning.best_experiment_overrides()

                self.assertEqual(name, "best")
                self.assertEqual(overrides["hidden_layers"], [256])
                self.assertEqual(overrides["dropout"], 0.2)
                self.assertEqual(overrides["learning_rate"], 0.0003)
                self.assertEqual(overrides["batch_size"], 256)
                self.assertTrue(overrides["batch_norm"])
                self.assertEqual(overrides["scheduler"], "cosine")
            finally:
                tuning.EXPERIMENTS_CSV = old_path
                tuning.CHECKPOINTS_DIR = old_checkpoints

    def test_append_row_keeps_legacy_csv_columns_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.csv"
            pd.DataFrame([{"experiment": "old", "f1_at_best": 0.4}]).to_csv(
                path, index=False
            )
            tuning.append_row(
                {"experiment": "new", "f1_at_best": 0.41, "max_epochs": 30},
                path,
            )
            frame = pd.read_csv(path)
            self.assertEqual(frame.loc[1, "experiment"], "new")
            self.assertAlmostEqual(frame.loc[1, "f1_at_best"], 0.41)
            self.assertEqual(frame.loc[1, "max_epochs"], 30)
            self.assertTrue(pd.isna(frame.loc[0, "max_epochs"]))


if __name__ == "__main__":
    unittest.main()
