from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import TARGET_COLUMN


NUMERICAL_FEATURES = [
    "age",
    "bmi",
    "poor_physical_health_days",
    "poor_mental_health_days",
]

BINARY_FEATURES = [
    "sex",
    "physical_activity",
    "stroke",
    "kidney_disease",
    "copd",
    "difficulty_walking",
]

CATEGORICAL_FEATURES = [
    "smoking_status",
    "diabetes",
    "general_health",
    "education",
    "income",
    "marital_status",
    "employment_status",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(df.columns))
    if missing_columns:
        raise KeyError(f"Missing feature columns: {missing_columns}")


def get_preprocessor(scale_numerical: bool = True) -> ColumnTransformer:
    """Create the shared preprocessing transformer for all models.

    The returned transformer must be fitted on the training data only.
    Validation and test data should only be transformed with the fitted object.
    """
    numerical_steps = []
    if scale_numerical:
        numerical_steps.append(("scaler", StandardScaler()))

    numerical_transformer = (
        Pipeline(steps=numerical_steps)
        if numerical_steps
        else "passthrough"
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, NUMERICAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessor(
    train_df: pd.DataFrame,
    scale_numerical: bool = True,
) -> ColumnTransformer:
    validate_feature_columns(train_df)
    X_train = train_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    preprocessor = get_preprocessor(scale_numerical=scale_numerical)
    return preprocessor.fit(X_train)


def transform_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    validate_feature_columns(df)
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=df.index)
