from pathlib import Path
from typing import Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_ready.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_ready.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test_ready.csv"

TARGET_COLUMN = "target"
RANDOM_STATE = 42


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Load one project dataset and fail clearly if the file is missing."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")

    return pd.read_csv(dataset_path)


def load_train() -> pd.DataFrame:
    return load_dataset(TRAIN_PATH)


def load_validation() -> pd.DataFrame:
    return load_dataset(VALIDATION_PATH)


def load_test() -> pd.DataFrame:
    return load_dataset(TEST_PATH)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits.

    The test split is returned for final evaluation only. It must not be used
    for model selection, feature selection, preprocessing fit, or tuning.
    """
    return load_train(), load_validation(), load_test()


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(f"Target column is missing: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def load_train_validation() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train/validation data for model development."""
    train_df = load_train()
    validation_df = load_validation()

    X_train, y_train = split_features_target(train_df)
    X_validation, y_validation = split_features_target(validation_df)

    return X_train, y_train, X_validation, y_validation


def load_final_test() -> tuple[pd.DataFrame, pd.Series]:
    """Load the untouched test split for final model evaluation only."""
    test_df = load_test()
    return split_features_target(test_df)
