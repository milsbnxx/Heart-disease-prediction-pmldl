# 1. Problem Statement, Dataset, and Preprocessing

## Problem Statement

The project solves a binary classification task: predicting whether a survey respondent reports a history of heart disease. The target variable is `target`, where `0` means no reported heart disease and `1` means reported heart disease.

This is a clinically relevant screening-style prediction problem with a strong class imbalance. In the prepared data, the positive class represents about 9.36% of the observations.

## Dataset Source

The dataset is based on the 2024 Behavioral Risk Factor Surveillance System (BRFSS), published by the U.S. Centers for Disease Control and Prevention (CDC). BRFSS is a large-scale U.S. health survey containing self-reported demographic, behavioral, and health-related variables.

The raw BRFSS data are distributed in SAS Transport (`.XPT`) format. The project download script is `data/raw/download_data.py`. Raw files are not version-controlled because of their size; the project stores reproducible scripts and processed datasets instead.

## Target Construction

The target is derived from two BRFSS variables:

- `CVDINFR4`: ever diagnosed with myocardial infarction / heart attack;
- `CVDCRHD4`: ever diagnosed with coronary heart disease or angina.

Rows where either variable indicates a positive diagnosis are assigned `target = 1`. Rows where both variables indicate no diagnosis are assigned `target = 0`. Unknown, refused, or missing target responses are removed.

The original heart-disease variables are dropped after target construction to avoid target leakage.

## Prepared Dataset

The final modeling datasets are:

| Split | Objects | Columns | Features without target |
|---|---:|---:|---:|
| Train | 316724 | 18 | 17 |
| Validation | 67870 | 18 | 17 |
| Test | 67870 | 18 | 17 |

The split is stratified by the target variable, so the class balance is stable across train, validation, and test:

| Split | Class 0 | Class 1 | Class 1 share |
|---|---:|---:|---:|
| Train | 287087 | 29637 | 9.36% |
| Validation | 61519 | 6351 | 9.36% |
| Test | 61520 | 6350 | 9.36% |

After preprocessing and imputation, the prepared datasets contain no missing values.

## Feature Groups

Numerical features:

- `age`
- `bmi`
- `poor_physical_health_days`
- `poor_mental_health_days`

Binary features:

- `sex`
- `physical_activity`
- `stroke`
- `kidney_disease`
- `copd`
- `difficulty_walking`

Categorical features:

- `smoking_status`
- `diabetes`
- `general_health`
- `education`
- `income`
- `marital_status`
- `employment_status`

## Preprocessing

The cleaning pipeline validates special BRFSS survey codes, converts invalid or refused answers to missing values, creates the binary target, removes target-construction variables, and removes duplicate respondents before feature selection when respondent identifiers are available.

Missing values are imputed only after the train/validation/test split:

- numerical features are imputed with medians calculated on the training set;
- categorical features are imputed with modes calculated on the training set;
- the same training-set statistics are then applied to validation and test.

The shared model preprocessing is implemented in `src/preprocessing.py`:

- numerical features are scaled with `StandardScaler`;
- binary features are passed through unchanged;
- categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")`.

The preprocessing transformer must be fitted only on the training split. Validation and test data must only be transformed with the already fitted transformer.

## EDA

The EDA notebook is `notebooks/01_eda.ipynb`. It includes:

- dataset source description;
- split sizes and number of features;
- data types;
- missing-value checks;
- duplicate-row checks;
- class-balance tables and plots;
- distributions of numerical, binary, and categorical features;
- correlation matrix on the training split;
- preprocessing smoke check.

Exact duplicate modeling rows can appear after feature selection because different survey respondents may have the same values for all selected features. Respondent-level duplicates are handled earlier in the cleaning script using `_STATE` and `SEQNO` when those columns are available.

## Leakage Prevention

The test set is reserved only for final unbiased evaluation. It must not be used for:

- model selection;
- hyperparameter tuning;
- feature selection;
- threshold selection;
- fitting imputers, scalers, encoders, or any other preprocessing statistics.

All modeling decisions should be made using the training and validation splits only.
