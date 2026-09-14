# Heart Disease Prediction

This repository contains the source code and reproducibility instructions for the Heart Disease Prediction project. The project solves a binary classification task to predict whether a survey respondent reports a history of heart disease using the 2024 Behavioral Risk Factor Surveillance System (BRFSS) dataset.

## Project structure & deliverables

*   **Final Report:** The comprehensive project report, covering all stages from data preprocessing to final evaluation, is available in `project.pdf`.
*   **Evaluation Code:** Implemented in `src/evaluation.py`. For detailed file descriptions, see `results/evaluation/README.md`.
*   **Saved Artifacts:** The saved evaluation results include validation and final test metrics for all five models, evaluated both at a default 0.5 threshold and at optimal thresholds selected on the validation split. Outputs also include ROC/PR curves, confusion matrices, training curves, error analysis, and 95% paired bootstrap intervals. 
*   *Note:* Re-running the evaluation pipeline utilizes these saved predictions and does not require retraining the models.

## Environment setup

The project requires Python 3.11. Install all required dependencies using the provided requirements file:

```bash
python3 -m pip install -r requirements.txt
```

## Data access
Raw BRFSS datasets are not tracked in version control due to size constraints. Download the raw BRFSS data by running:

```bash
python3 data/raw/download_data.py
```
The processed train_ready.csv, validation_ready.csv, and test_ready.csv datasets used by the modeling pipeline are stored directly in the project repository.

## Execution pipeline
Execute the pipeline sequentially from the project root to reproduce all metrics, model checkpoints, and evaluation results:

### 1. Evaluate baselines:

```bash
python3 src/baselines.py
```

### 2. Train proposed model:

```bash
python3 src/mlp.py
```

### 3. Run hyperparameter tuning:

```bash
python3 src/tuning.py
python3 src/tuning.py --stage extra
python3 src/tuning.py --export-only
```

### 4. Analyze thresholds:

```bash
python3 src/threshold_analysis.py
```

### 5. Final test evaluation:

```bash
python3 src/predict_test.py
python3 src/evaluation.py --bootstrap 1000 --seed 42
python3 -m unittest discover -s tests -v
```

## Isolated evaluation (Optional)
To independently reproduce inference in a new directory without overwriting the frozen experiment logs, run:

```bash
python3 src/predict_test.py --output-dir results/evaluation_runs/recheck
python3 src/evaluation.py --output-dir results/evaluation_runs/recheck --bootstrap 1000 --seed 42
```

The separate evaluation report will be generated at `results/evaluation_runs/recheck/05_results.md`. The main report and original results are preserved and will not be overwritten.
