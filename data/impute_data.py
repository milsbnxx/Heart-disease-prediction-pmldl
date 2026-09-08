from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"


# ============================================================
# LOAD
# ============================================================

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df = pd.read_parquet(DATA_DIR / "validation.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_columns = [
    "bmi",
    "poor_physical_health_days",
    "poor_mental_health_days",
]

# Calculate medians ONLY on train
for column in numerical_columns:
    median_value = train_df[column].median()

    train_df[column] = train_df[column].fillna(median_value)
    val_df[column] = val_df[column].fillna(median_value)
    test_df[column] = test_df[column].fillna(median_value)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "smoking_status",
    "physical_activity",
    "stroke",
    "diabetes",
    "general_health",
    "kidney_disease",
    "copd",
    "difficulty_walking",
    "education",
    "income",
    "marital_status",
    "employment_status",
]

# Calculate most frequent value ONLY on train
for column in categorical_columns:
    mode_value = train_df[column].mode()[0]

    train_df[column] = train_df[column].fillna(mode_value)
    val_df[column] = val_df[column].fillna(mode_value)
    test_df[column] = test_df[column].fillna(mode_value)


# ============================================================
# TYPES
# ============================================================

integer_columns = [
    "target",
    "age",
    "sex",
    "smoking_status",
    "physical_activity",
    "stroke",
    "diabetes",
    "general_health",
    "poor_physical_health_days",
    "poor_mental_health_days",
    "kidney_disease",
    "copd",
    "difficulty_walking",
    "education",
    "income",
    "marital_status",
    "employment_status",
]

for df in [train_df, val_df, test_df]:
    for column in integer_columns:
        df[column] = df[column].astype(int)

    df["bmi"] = df["bmi"].astype(float)


# ============================================================
# SAVE
# ============================================================

train_df.to_parquet(
    DATA_DIR / "train_ready.parquet",
    index=False,
)

val_df.to_parquet(
    DATA_DIR / "validation_ready.parquet",
    index=False,
)

test_df.to_parquet(
    DATA_DIR / "test_ready.parquet",
    index=False,
)

train_df.to_csv(
    DATA_DIR / "train_ready.csv",
    index=False,
)

val_df.to_csv(
    DATA_DIR / "validation_ready.csv",
    index=False,
)

test_df.to_csv(
    DATA_DIR / "test_ready.csv",
    index=False,
)