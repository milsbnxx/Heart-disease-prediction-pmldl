from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "heart_disease_clean.parquet"
)

SPLIT_DIR = ROOT_DIR / "data" / "processed"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(DATA_PATH)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 70% train
# 30% temporary set

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["target"],
)


# Second split:
# 30% temporary -> 15% validation + 15% test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["target"],
)


# ============================================================
# SAVE
# ============================================================

train_df.to_parquet(
    SPLIT_DIR / "train.parquet",
    index=False,
)

val_df.to_parquet(
    SPLIT_DIR / "validation.parquet",
    index=False,
)

test_df.to_parquet(
    SPLIT_DIR / "test.parquet",
    index=False,
)


train_df.to_csv(
    SPLIT_DIR / "train.csv",
    index=False,
)

val_df.to_csv(
    SPLIT_DIR / "validation.csv",
    index=False,
)

test_df.to_csv(
    SPLIT_DIR / "test.csv",
    index=False,
)