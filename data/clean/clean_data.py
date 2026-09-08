from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT_DIR / "data" / "raw" / "brfss_2024_raw.parquet"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


COLUMN_NAMES = {
    "CVDINFR4": "myocardial_infarction",
    "CVDCRHD4": "coronary_heart_disease",
    "_AGE80": "age",
    "_SEX": "sex",
    "_BMI5": "bmi",
    "_SMOKER3": "smoking_status",
    "EXERANY2": "physical_activity",
    "CVDSTRK3": "stroke",
    "DIABETE4": "diabetes",
    "GENHLTH": "general_health",
    "PHYSHLTH": "poor_physical_health_days",
    "MENTHLTH": "poor_mental_health_days",
    "CHCKDNY2": "kidney_disease",
    "CHCCOPD3": "copd",
    "DIFFWALK": "difficulty_walking",
    "EDUCA": "education",
    "INCOME3": "income",
    "MARITAL": "marital_status",
    "EMPLOY1": "employment_status",
}


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(RAW_PATH)


# ============================================================
# DUPLICATES
# ============================================================

# Remove duplicate respondents before selecting features.
# SEQNO identifies the respondent within the state.

if {"_STATE", "SEQNO"}.issubset(df.columns):
    df = df.drop_duplicates(
        subset=["_STATE", "SEQNO"]
    )
else:
    df = df.drop_duplicates()


# ============================================================
# SELECT COLUMNS
# ============================================================

df = df[list(COLUMN_NAMES.keys())].copy()

df = df.rename(
    columns=COLUMN_NAMES
)


# ============================================================
# TARGET
# ============================================================

def create_target(row):
    mi = row["myocardial_infarction"]
    chd = row["coronary_heart_disease"]

    # Heart disease exists
    if mi == 1 or chd == 1:
        return 1

    # No heart disease
    if mi == 2 and chd == 2:
        return 0

    # Unknown / refused / missing
    return pd.NA


df["target"] = df.apply(
    create_target,
    axis=1
)

# Target must be known
df = df.dropna(
    subset=["target"]
).copy()

df["target"] = df["target"].astype(int)


# Remove variables used to create target
df = df.drop(
    columns=[
        "myocardial_infarction",
        "coronary_heart_disease",
    ]
)


# ============================================================
# AGE
# ============================================================

df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

# _AGE80: valid values are 18-80
# 80 means 80 years or older
df.loc[
    ~df["age"].between(18, 80),
    "age"
] = pd.NA

df = df.dropna(
    subset=["age"]
)

df["age"] = df["age"].astype("Int64")


# ============================================================
# SEX
# ============================================================

# Original:
# 1 = Male
# 2 = Female
#
# New:
# 0 = Male
# 1 = Female

df["sex"] = df["sex"].map({
    1: 0,
    2: 1,
})

df = df.dropna(
    subset=["sex"]
)

df["sex"] = df["sex"].astype("Int64")


# ============================================================
# BMI
# ============================================================

df["bmi"] = pd.to_numeric(
    df["bmi"],
    errors="coerce"
)

# _BMI5 stores BMI multiplied by 100
df["bmi"] = df["bmi"] / 100

# Invalid/extreme values according to BRFSS calculation rules
df.loc[
    (df["bmi"] < 12)
    | (df["bmi"] >= 100),
    "bmi"
] = pd.NA


# ============================================================
# SMOKING STATUS
# ============================================================

# 1 = smokes every day
# 2 = smokes some days
# 3 = former smoker
# 4 = never smoked
# 9 = unknown/refused/missing

df["smoking_status"] = df["smoking_status"].where(
    df["smoking_status"].isin([1, 2, 3, 4]),
    pd.NA
)


# ============================================================
# BINARY FEATURES
# ============================================================

# BRFSS:
# 1 = Yes
# 2 = No
#
# New:
# 1 = Yes
# 0 = No
#
# 7 / 9 automatically become missing

binary_columns = [
    "physical_activity",
    "stroke",
    "kidney_disease",
    "copd",
    "difficulty_walking",
]

for column in binary_columns:
    df[column] = df[column].map({
        1: 1,
        2: 0,
    })


# ============================================================
# DIABETES
# ============================================================

# 1 = diabetes
# 2 = diabetes only during pregnancy
# 3 = no diabetes
# 4 = prediabetes
# 7 / 9 = unknown/refused

df["diabetes"] = df["diabetes"].where(
    df["diabetes"].isin([1, 2, 3, 4]),
    pd.NA
)


# ============================================================
# GENERAL HEALTH
# ============================================================

# 1 = excellent
# 2 = very good
# 3 = good
# 4 = fair
# 5 = poor
# 7 / 9 = unknown/refused

df["general_health"] = df["general_health"].where(
    df["general_health"].isin([1, 2, 3, 4, 5]),
    pd.NA
)


# ============================================================
# PHYSICAL AND MENTAL HEALTH DAYS
# ============================================================

health_days_columns = [
    "poor_physical_health_days",
    "poor_mental_health_days",
]

for column in health_days_columns:

    # 88 actually means 0 days
    # 77 = Don't know
    # 99 = Refused

    df[column] = df[column].replace({
        88: 0,
        77: pd.NA,
        99: pd.NA,
    })

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # Only 0-30 days are possible
    df.loc[
        ~df[column].between(0, 30),
        column
    ] = pd.NA


# ============================================================
# EDUCATION
# ============================================================

# Valid categories: 1-6
# 9 = refused

df["education"] = df["education"].where(
    df["education"].isin([1, 2, 3, 4, 5, 6]),
    pd.NA
)


# ============================================================
# INCOME
# ============================================================

# Valid categories: 1-11
# 77 = don't know
# 99 = refused

df["income"] = df["income"].where(
    df["income"].isin(
        range(1, 12)
    ),
    pd.NA
)


# ============================================================
# MARITAL STATUS
# ============================================================

# Valid categories: 1-6
# 9 = refused

df["marital_status"] = df["marital_status"].where(
    df["marital_status"].isin(
        range(1, 7)
    ),
    pd.NA
)


# ============================================================
# EMPLOYMENT STATUS
# ============================================================

# Valid categories: 1-8
# 9 = refused

df["employment_status"] = df["employment_status"].where(
    df["employment_status"].isin(
        range(1, 9)
    ),
    pd.NA
)


# ============================================================
# DATA TYPES
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

for column in integer_columns:
    df[column] = df[column].astype("Int64")

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

for column in integer_columns:
    df[column] = df[column].astype("Int64")

df["bmi"] = df["bmi"].astype("Float64")
# ============================================================
# SAVE
# ============================================================

parquet_path = (
    PROCESSED_DIR
    / "heart_disease_clean.parquet"
)

csv_path = (
    PROCESSED_DIR
    / "heart_disease_clean.csv"
)

df.to_parquet(
    parquet_path,
    index=False
)

df.to_csv(
    csv_path,
    index=False
)
