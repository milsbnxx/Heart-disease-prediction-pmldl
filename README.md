# Heart Disease Prediction

This repository contains the source code and reproducibility instructions for the Heart Disease Prediction project. The project solves a binary classification task to predict whether a survey respondent reports a history of heart disease using the 2024 Behavioral Risk Factor Surveillance System (BRFSS) dataset.

## Project structure & deliverables

*   **Final Report:** The comprehensive project report, covering all stages from data preprocessing to final evaluation, is available in `project.pdf`.
*   **Evaluation Code:** Implemented in `src/evaluation.py`. For detailed file descriptions, see `results/evaluation/README.md`.
*   **Saved Artifacts:** The saved evaluation results include validation and final test metrics for all five models, evaluated both at a default 0.5 threshold and at optimal thresholds selected on the validation split. Outputs also include ROC/PR curves, confusion matrices, training curves, error analysis, and 95% paired bootstrap intervals. 
*   *Note:* `src/evaluation.py` analyzes an already frozen evaluation directory. It does not train models or regenerate predictions. `src/predict_test.py` creates a new, separate frozen evaluation directory and refuses to overwrite an existing one.

## Environment setup

The project requires Python 3.11. Install all training and evaluation dependencies using the provided requirements file:

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` includes the training and evaluation dependencies.
`requirements-evaluation.txt` lists the smaller dependency set needed to regenerate
only the Stage-5 evaluation tables and figures.

## Data access
Raw BRFSS datasets are not tracked in version control due to size constraints. Download the raw BRFSS data by running:

```bash
python3 data/raw/download_data.py
```
The processed train_ready.csv, validation_ready.csv, and test_ready.csv datasets used by the modeling pipeline are stored directly in the project repository.

## Execution pipeline

The complete workflow can be launched with one command from the repository root:

```bash
python3 src/run_pipeline.py --bootstrap 1000 --seed 42
```

By default, the command writes a new frozen evaluation to
`results/evaluation_runs/reproduction/`. If that directory already contains a
`manifest.json`, choose another directory with `--output-dir`. The script stops
on the first failed step and runs the unit tests after evaluation.

The full workflow retrains the models and runs the tuning experiments. It can take
considerably longer than analyzing the already saved Stage-5 results.

### Individual stages
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
python3 src/predict_test.py --output-dir results/evaluation_runs/reproduction
python3 src/evaluation.py --output-dir results/evaluation_runs/reproduction --bootstrap 1000 --seed 42
python3 -m unittest discover -s tests -v
```

## Isolated evaluation (Optional)
To independently reproduce inference in a new directory without overwriting the frozen experiment logs, run:

```bash
python3 src/predict_test.py --output-dir results/evaluation_runs/recheck
python3 src/evaluation.py --output-dir results/evaluation_runs/recheck --bootstrap 1000 --seed 42
```

The separate evaluation report will be generated at `results/evaluation_runs/recheck/05_results.md`. The main report and original frozen results are preserved and will not be overwritten. The prediction step requires the trained model weights and parameter files produced by the preceding training/tuning stages.
