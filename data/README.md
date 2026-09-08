# Data Pipeline

This directory contains the complete data preparation pipeline for the Heart Disease Prediction project.

The project uses the 2024 Behavioral Risk Factor Surveillance System (BRFSS) dataset published by the U.S. Centers for Disease Control and Prevention (CDC). The goal of the pipeline is to transform the original survey data into clean and reproducible training, validation, and test datasets for binary heart disease classification.

## Project Goal

The model predicts whether a respondent reports a history of heart disease using demographic, behavioral, and general health-related features.

The target is binary:

| Target | Meaning |
|---:|---|
| `0` | No reported heart disease |
| `1` | Reported heart disease |

Heart disease is defined using two BRFSS variables:

- `CVDINFR4` — ever diagnosed with myocardial infarction / heart attack
- `CVDCRHD4` — ever diagnosed with coronary heart disease or angina

Target construction:

```text
CVDINFR4 = Yes OR CVDCRHD4 = Yes
    -> target = 1

CVDINFR4 = No AND CVDCRHD4 = No
    -> target = 0

Unknown / refused / missing response
    -> row is removed
```

The original disease variables are removed after target creation to prevent target leakage.

## Directory Structure

```text
data/
│
├── clean/
│   └── clean_data.py
│
├── processed/
│   ├── train_ready.csv
│   ├── validation_ready.csv
│   └── test_ready.csv
│
├── raw/
│   └── download_data.py
│
├── impute_data.py
├── split_data.py
└── README.md
```

Large raw and intermediate data files are intentionally excluded from Git and can be reproduced using the scripts in this directory.

## Data Source

The original dataset is BRFSS 2024, a large-scale U.S. health survey containing self-reported information about health conditions, lifestyle, demographics, and health-related behaviors.

The original data are distributed in SAS Transport (`.XPT`) format. Raw files are not stored in the repository because of their size.

To download and prepare the raw data:

```bash
python data/raw/download_data.py
```

Locally generated raw files may include:

```text
data/raw/brfss_2024/
data/raw/brfss_2024_raw.csv
data/raw/brfss_2024_raw.parquet
```

These files are excluded from version control.

## Pipeline Overview

```text
BRFSS 2024 raw data
        ↓
download_data.py
        ↓
clean_data.py
        ↓
split_data.py
        ↓
impute_data.py
        ↓
train_ready.csv
validation_ready.csv
test_ready.csv
```

## 1. Raw Data Download

File:

```text
data/raw/download_data.py
```

This script downloads the original BRFSS 2024 dataset and converts it into formats that can be processed efficiently with pandas.

## 2. Data Cleaning

File:

```text
data/clean/clean_data.py
```

The script selects the required BRFSS variables, renames them, creates the target, cleans special survey codes, validates ranges, handles invalid values, and removes duplicate respondents.

### Selected Features

| Original BRFSS variable | Project feature |
|---|---|
| `CVDINFR4` | `myocardial_infarction` |
| `CVDCRHD4` | `coronary_heart_disease` |
| `_AGE80` | `age` |
| `_SEX` | `sex` |
| `_BMI5` | `bmi` |
| `_SMOKER3` | `smoking_status` |
| `EXERANY2` | `physical_activity` |
| `CVDSTRK3` | `stroke` |
| `DIABETE4` | `diabetes` |
| `GENHLTH` | `general_health` |
| `PHYSHLTH` | `poor_physical_health_days` |
| `MENTHLTH` | `poor_mental_health_days` |
| `CHCKDNY2` | `kidney_disease` |
| `CHCCOPD3` | `copd` |
| `DIFFWALK` | `difficulty_walking` |
| `EDUCA` | `education` |
| `INCOME3` | `income` |
| `MARITAL` | `marital_status` |
| `EMPLOY1` | `employment_status` |

### Target Construction

```python
if myocardial_infarction == 1 or coronary_heart_disease == 1:
    target = 1
elif myocardial_infarction == 2 and coronary_heart_disease == 2:
    target = 0
```

Unknown, refused, and missing responses are excluded. The original heart-disease variables are dropped after target creation.

### Age

`_AGE80` is converted to an integer age feature.

```text
18-79 = exact age
80    = 80 years old or older
```

Invalid ages are treated as missing. Rows without a valid age are removed.

### Sex

Original BRFSS encoding:

```text
1 = Male
2 = Female
```

Project encoding:

```text
0 = Male
1 = Female
```

### BMI

`_BMI5` stores BMI multiplied by 100.

```text
2745 -> 27.45
3150 -> 31.50
```

Therefore:

```python
bmi = bmi / 100
```

Invalid BMI values are converted to missing values. The accepted range is:

```text
12 <= BMI < 100
```

BMI remains a continuous floating-point feature.

### Smoking Status

| Value | Meaning |
|---:|---|
| `1` | Current smoker, every day |
| `2` | Current smoker, some days |
| `3` | Former smoker |
| `4` | Never smoked |

Invalid, refused, and unknown responses are converted to missing values.

### Binary Health Features

The following features are converted to binary form:

```text
physical_activity
stroke
kidney_disease
copd
difficulty_walking
```

Original BRFSS encoding:

```text
1 = Yes
2 = No
```

Project encoding:

```text
1 = Yes
0 = No
```

Unknown and refused responses become missing values.

### Diabetes

| Value | Meaning |
|---:|---|
| `1` | Diabetes |
| `2` | Diabetes only during pregnancy |
| `3` | No diabetes |
| `4` | Prediabetes / borderline diabetes |

Unknown and refused responses are converted to missing values.

### General Health

| Value | Meaning |
|---:|---|
| `1` | Excellent |
| `2` | Very good |
| `3` | Good |
| `4` | Fair |
| `5` | Poor |

Invalid responses are converted to missing values.

### Physical and Mental Health Days

The following features represent the number of unhealthy days during the previous 30 days:

```text
poor_physical_health_days
poor_mental_health_days
```

Special BRFSS codes:

```text
1-30 = number of unhealthy days
88   = zero unhealthy days
77   = Don't know
99   = Refused
```

During preprocessing:

```text
88 -> 0
77 -> missing
99 -> missing
```

Only values from `0` to `30` are retained.

### Education

Education is stored as an ordered categorical feature with valid values from `1` to `6`. Invalid and refused responses become missing values.

### Income

`income` represents annual household income as an ordered category.

```text
1  -> lowest income group
...
11 -> highest income group
```

Special values such as `77` (Don't know) and `99` (Refused) are converted to missing values.

### Marital Status

Valid categories range from `1` to `6` and represent married, divorced, widowed, separated, never married, and unmarried couple categories.

Invalid and refused responses become missing values.

### Employment Status

Valid categories range from `1` to `8` and include employed for wages, self-employed, unemployed, homemaker, student, retired, and unable to work categories.

Invalid responses become missing values.

## Duplicate Handling

Duplicate respondents are removed before feature selection.

When respondent identifiers are available, duplicates are checked using:

```text
_STATE
SEQNO
```

Rows are not deduplicated only by model features because two different respondents may legitimately provide identical survey answers.

## 3. Train / Validation / Test Split

File:

```text
data/split_data.py
```

The cleaned dataset is divided into:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

The split is reproducible using:

```python
random_state=42
```

Stratified sampling is used:

```python
stratify=df["target"]
```

This keeps approximately the same target distribution in all three subsets.

The split is performed in two stages:

```text
100%
│
├── 70% Training
│
└── 30% Temporary
        │
        ├── 15% Validation
        └── 15% Test
```

## 4. Missing Value Imputation

File:

```text
data/impute_data.py
```

Missing values are handled only after the train/validation/test split.

This prevents data leakage because preprocessing statistics are calculated only from the training set.

### Numerical Features

Numerical missing values are filled using training-set medians.

Examples:

```text
bmi
poor_physical_health_days
poor_mental_health_days
```

The same training median is then applied to validation and test data.

Example:

```python
median_value = train_df["bmi"].median()

train_df["bmi"] = train_df["bmi"].fillna(median_value)
validation_df["bmi"] = validation_df["bmi"].fillna(median_value)
test_df["bmi"] = test_df["bmi"].fillna(median_value)
```

### Categorical Features

Categorical missing values are filled using statistics calculated only from the training dataset.

For nominal categorical variables, the most frequent training category is used and then applied unchanged to validation and test datasets.

## Final Datasets

The final version-controlled datasets are:

```text
data/processed/train_ready.csv
data/processed/validation_ready.csv
data/processed/test_ready.csv
```

The target column is:

```text
target
```

All other columns are model input features.

## Dataset Roles

### Training Set

```text
train_ready.csv
```

Used to fit model parameters and calculate preprocessing statistics.

### Validation Set

```text
validation_ready.csv
```

Used for model comparison, hyperparameter tuning, threshold selection, and feature selection.

It is not used to fit the final model directly.

### Test Set

```text
test_ready.csv
```

Used only for final unbiased model evaluation.

The test set must not be used for feature selection, model selection, hyperparameter tuning, imputation statistics, or threshold tuning.

## Reproducing the Dataset

Activate the project's virtual environment:

```bash
source .venv/bin/activate
```

Install required dependencies:

```bash
python -m pip install pandas pyarrow scikit-learn
```

Run the pipeline in order.

### Step 1 — Download raw data

```bash
python data/raw/download_data.py
```

### Step 2 — Clean data

```bash
python data/clean/clean_data.py
```

### Step 3 — Split dataset

```bash
python data/split_data.py
```

### Step 4 — Impute missing values

```bash
python data/impute_data.py
```

The final files are generated in:

```text
data/processed/
```

## Version Control Policy

The repository stores:

```text
Python data-processing scripts
train_ready.csv
validation_ready.csv
test_ready.csv
```

The repository does not store:

```text
original BRFSS XPT files
ZIP archives
raw CSV files
raw Parquet files
intermediate Parquet files
temporary processed datasets
```

This keeps the repository lightweight while preserving reproducibility through the provided Python scripts.

## Data Leakage Prevention

The pipeline includes several safeguards against leakage:

- variables used to construct `target` are removed from the feature set;
- train, validation, and test subsets are created before statistical imputation;
- medians, modes, and other preprocessing statistics are calculated only from the training set;
- the validation set is reserved for model development;
- the test set is reserved exclusively for final evaluation.

## Next Stage

After this pipeline, the next stage is model-specific preprocessing and training.

Possible steps include:

```text
categorical feature encoding
feature scaling
class imbalance analysis
baseline model training
hyperparameter tuning
model evaluation
```

The exact preprocessing strategy depends on the selected machine learning algorithm.
