"""Stage 5: reproducible tables, figures, paired bootstrap and Results report.

Run predict_test.py once first. This command reads the frozen predictions and
never trains models, selects thresholds, or changes the selected model.
"""
from __future__ import annotations
import argparse
import json
import shlex
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'pmldl-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, f1_score,
)
from src.evaluation_common import ROOT, OUT, MODELS, sha256, scores, validate_predictions

LABELS = dict(zip(MODELS, ['Logistic regression', 'Decision tree', 'Random forest', 'MLP', 'Tuned MLP']))
FIGURES = OUT / 'figures'
COLORS = dict(zip(MODELS, ['#2563eb', '#64748b', '#d97706', '#7c3aed', '#059669']))


def save_figure(fig, filename):
    fig.savefig(FIGURES / f'{filename}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(frames, thresholds, split):
    for mode in ('fixed_0.5', 'validation_selected'):
        for normalize in (None, 'true'):
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
            for ax, name in zip(axes.flat, MODELS):
                frame = frames[name]
                threshold = 0.5 if mode == 'fixed_0.5' else thresholds[name]
                ConfusionMatrixDisplay.from_predictions(
                    frame.y_true, frame.y_proba >= threshold, labels=[0, 1],
                    display_labels=['No disease', 'Disease'], normalize=normalize,
                    values_format='.1%' if normalize else 'd', cmap='Blues',
                    colorbar=False, ax=ax,
                    im_kw={'vmin': 0, 'vmax': 1 if normalize else len(frame)},
                )
                ax.set_title(f'{LABELS[name]} | threshold {threshold:.2f}', fontsize=11)
            axes.flat[-1].axis('off')
            suffix = 'normalized' if normalize else 'counts'
            mode_label = 'fixed threshold 0.5' if mode == 'fixed_0.5' else 'validation-selected thresholds'
            fig.suptitle(f'{split.title()} confusion matrices | {mode_label} | {suffix}', fontsize=15)
            save_figure(fig, f'{split}_confusion_{mode}_{suffix}')
    for kind in ('roc', 'pr'):
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        for name, frame in frames.items():
            if kind == 'roc':
                x, y, _ = roc_curve(frame.y_true, frame.y_proba)
                value = roc_auc_score(frame.y_true, frame.y_proba)
                metric = 'AUC'
            else:
                y, x, _ = precision_recall_curve(frame.y_true, frame.y_proba)
                value = average_precision_score(frame.y_true, frame.y_proba)
                metric = 'AP'
            if kind == 'pr':
                ax.step(x, y, where='post', label=f'{LABELS[name]} ({metric}={value:.4f})', color=COLORS[name])
            else:
                ax.plot(x, y, label=f'{LABELS[name]} ({metric}={value:.4f})', color=COLORS[name])
        if kind == 'roc':
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Random ranking')
            ax.set(xlabel='False positive rate', ylabel='True positive rate')
        else:
            ax.axhline(frame.y_true.mean(), ls='--', color='gray', label='Positive prevalence')
            ax.set(xlabel='Recall', ylabel='Precision')
        ax.set(xlim=(0, 1), ylim=(0, 1.02), title=f'{split.title()} {kind.upper()} curves')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        save_figure(fig, f'{split}_{kind}_curves')


def plot_training():
    history = pd.read_csv(ROOT / 'results/tuning_histories.csv')
    tuned_params = json.loads((ROOT / 'results/models/mlp_tuned_params.json').read_text())
    experiments = [('exp01_baseline', 'Baseline configuration (tuning rerun)'),
                   (tuned_params['source_experiment'], 'Selected tuned configuration')]
    saved = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for experiment, label in experiments:
        subset = history.loc[history.experiment == experiment].sort_values('epoch')
        if subset.empty or subset.epoch.duplicated().any():
            raise ValueError(f'Missing or ambiguous training history: {experiment}')
        saved.append(subset)
        for ax, column, title in zip(axes, ['train_loss', 'val_f1', 'val_roc_auc'],
                                     ['Training loss', 'Validation F1 at 0.5', 'Validation ROC-AUC']):
            ax.plot(subset.epoch, subset[column], label=label)
            ax.set(xlabel='Epoch', ylabel=title, title=title)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Recorded tuning histories (not the original Stage 3 training run)')
    save_figure(fig, 'training_curves')
    pd.concat(saved).to_csv(OUT / 'training_curves_source.csv', index=False)
    # The original runs did not log val_loss. Never fabricate retrospective loss.
    return experiments


def analyze_errors(frames, dataset, thresholds, split):
    summaries, groups, examples = [], [], []
    for name, frame in frames.items():
        threshold = thresholds[name]
        row = scores(frame.y_true, frame.y_proba, threshold)
        summaries.append({'model': name, 'split': split, **row})
        joined = dataset.copy().reset_index(drop=True)
        joined.insert(0, 'row_index', frame.row_index.to_numpy())
        joined['y_true'] = frame.y_true.to_numpy()
        joined['y_proba'] = frame.y_proba.to_numpy()
        joined['y_pred'] = (joined.y_proba >= threshold).astype(int)
        joined['error_type'] = np.select(
            [(joined.y_true == 1) & (joined.y_pred == 0),
             (joined.y_true == 0) & (joined.y_pred == 1)], ['FN', 'FP'], default='correct')
        for error, ascending in [('FN', True), ('FP', False)]:
            subset = joined.loc[joined.error_type == error].sort_values('y_proba', ascending=ascending).head(10).copy()
            subset.insert(0, 'model', name)
            subset.insert(1, 'split', split)
            examples.append(subset)
        subgroup_values = {
            'sex': joined.sex.map({0: 'Male', 1: 'Female'}).fillna('Unknown'),
            'age': pd.cut(joined.age, [-np.inf, 45, 65, np.inf], right=False,
                          labels=['under 45', '45-64', '65+']).astype(str),
        }
        for feature, values in subgroup_values.items():
            for group in sorted(values.unique()):
                subset = joined.loc[values == group]
                groups.append({'model': name, 'split': split, 'feature': feature, 'group': group,
                               **scores(subset.y_true, subset.y_proba, threshold)})
    pd.DataFrame(summaries).to_csv(OUT / f'{split}_error_summary.csv', index=False)
    pd.DataFrame(groups).to_csv(OUT / f'{split}_subgroup_metrics.csv', index=False)
    pd.concat(examples, ignore_index=True).to_csv(OUT / f'{split}_error_examples.csv', index=False)
    return pd.DataFrame(summaries), pd.DataFrame(groups)


def bootstrap(frames, thresholds, repeats, seed):
    """Paired IID row bootstrap, conditional on fitted models and fixed thresholds.

    Identical indices across models permit paired differences; never re-optimize
    thresholds. It does not account for survey clustering or training variability.
    """
    rng = np.random.default_rng(seed)
    y = frames[MODELS[0]].y_true.to_numpy()
    probabilities = {m: frames[m].y_proba.to_numpy() for m in MODELS}
    predictions = {m: probabilities[m] >= thresholds[m] for m in MODELS}
    samples = np.empty((repeats, len(MODELS), 3))
    for i in range(repeats):
        index = rng.integers(0, len(y), size=len(y))
        ys = y[index]
        if len(np.unique(ys)) != 2:
            raise ValueError('Bootstrap draw has one class; use stratification for very small datasets.')
        for j, name in enumerate(MODELS):
            samples[i, j] = [f1_score(ys, predictions[name][index], zero_division=0),
                             roc_auc_score(ys, probabilities[name][index]),
                             average_precision_score(ys, probabilities[name][index])]
        if (i + 1) % 100 == 0:
            print(f'Paired bootstrap: {i + 1}/{repeats}', flush=True)
    metrics = ['f1', 'roc_auc', 'average_precision']
    intervals, differences = [], []
    point = {m: scores(y, probabilities[m], thresholds[m]) for m in MODELS}
    selected = MODELS.index('mlp_tuned')
    for j, name in enumerate(MODELS):
        for k, metric in enumerate(metrics):
            lo, hi = np.quantile(samples[:, j, k], [0.025, 0.975])
            intervals.append(dict(model=name, metric=metric, estimate=point[name][metric],
                                  lower_95=lo, upper_95=hi, bootstrap_repeats=repeats, seed=seed))
            if name != 'mlp_tuned':
                lo, hi = np.quantile(samples[:, selected, k] - samples[:, j, k], [0.025, 0.975])
                differences.append(dict(comparison=f'mlp_tuned - {name}', metric=metric,
                                        estimate=point['mlp_tuned'][metric] - point[name][metric],
                                        lower_95=lo, upper_95=hi, bootstrap_repeats=repeats, seed=seed))
    ci, delta = pd.DataFrame(intervals), pd.DataFrame(differences)
    ci.to_csv(OUT / 'test_confidence_intervals.csv', index=False)
    delta.to_csv(OUT / 'test_paired_differences.csv', index=False)
    return ci, delta


def markdown_table(frame, columns, digits=4):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.{digits}f}' if np.isfinite(value) else 'NA'
        return str(value)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |' for _, row in frame.iterrows()]
    return '\n'.join(lines)


def write_report(tables, ci, differences, errors, groups, manifest, protocol, repeats, seed):
    val = tables['validation'].query("threshold_mode == 'validation_selected'").set_index('model')
    test = tables['test'].query("threshold_mode == 'validation_selected'").set_index('model')
    tuned = test.loc['mlp_tuned']
    threshold_table = pd.DataFrame([{'model': m, 'threshold': protocol['thresholds'][m]} for m in MODELS])
    audit = pd.read_csv(OUT / 'validation_replay_audit.csv')
    columns = ['model', 'threshold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    comparison = test.reset_index()[['model', 'f1', 'roc_auc', 'average_precision']].copy()
    for metric in ['f1', 'roc_auc', 'average_precision']:
        comparison[f'{metric}_validation'] = comparison.model.map(val[metric])
        comparison[f'{metric}_test_minus_validation'] = comparison[metric] - comparison[f'{metric}_validation']
    comparison.to_csv(OUT / 'validation_test_comparison.csv', index=False)
    f1_delta = differences.loc[(differences.comparison == 'mlp_tuned - mlp') & (differences.metric == 'f1')].iloc[0]
    evidence = ('includes zero, so these data do not establish a clear difference' if
                f1_delta.lower_95 <= 0 <= f1_delta.upper_95 else
                'excludes zero under the stated fixed-model IID bootstrap assumptions')
    tuned_groups = groups.query("model == 'mlp_tuned'")
    age_groups = tuned_groups.query("feature == 'age'").set_index('group')
    sex_groups = tuned_groups.query("feature == 'sex'").set_index('group')
    history = pd.read_csv(ROOT / 'results/tuning_histories.csv')
    history_rows = []
    tuned_experiment = json.loads((ROOT / 'results/models/mlp_tuned_params.json').read_text())['source_experiment']
    for experiment in ['exp01_baseline', tuned_experiment]:
        subset = history.loc[history.experiment == experiment]
        best = subset.loc[subset.val_f1.idxmax()]
        history_rows.append({'experiment': experiment, 'best_epoch_by_val_f1_at_0.5': int(best.epoch),
                             'recorded_epochs': len(subset), 'best_val_f1_at_0.5': best.val_f1})
    text = f'''# 5. Results

## 5.1 Evaluation protocol

The task is binary classification of self-reported heart-disease history, not prospective
prediction of a future diagnosis. The supplied processed splits contain {manifest['train_n']:,}
training, {manifest['validation_n']:,} validation and {manifest['test_n']:,} test rows.
The positive prevalence is {val.loc['mlp_tuned', 'positive_rate']:.2%} on validation and
{tuned.positive_rate:.2%} on test. All five models are evaluated on the same checked row IDs
and targets. Preprocessing is fitted on train only. The selected model is **tuned MLP**,
carried forward from Stage 4; test results do not change this selection.

Thresholds are selected by maximum validation F1 over 0.01–0.99 in steps of 0.01;
ties choose the smallest threshold. Predictions use `probability >= threshold`.
The protocol is written to `results/evaluation/frozen_protocol.json` before test is loaded.
A second comparison fixes all thresholds at 0.5. Threshold selection and reporting on
validation make validation scores optimistic development estimates, not final test scores.
ROC-AUC and average precision (AP) use scores, not binary predictions. AP is not the
trapezoidal integral of the PR curve. Precision/recall/F1 use the positive class (1).
Undefined precision/F1 are set to zero; undefined subgroup AUC is reported as NA.

{markdown_table(threshold_table, ['model', 'threshold'])}

Legacy baseline pipelines were not saved, so their fixed configurations were refitted on
train in the recorded environment. MLP weights were loaded without retraining; their
train-reconstructed preprocessing was checked by replaying validation predictions with
absolute/relative tolerance 2e-5. All Stage 5 validation and test tables use the resulting
same model instances; the legacy Stage 2–4 CSVs are preserved. The audit below quantifies
reconstruction differences rather than assuming exact reproduction across environments.
The logistic-regression legacy L2 defaults are mapped to equivalent scikit-learn 1.8 defaults.

{markdown_table(audit, ['model', 'matches_legacy_at_2e-5', 'validation_max_probability_difference', 'validation_mean_probability_difference', 'legacy_label_disagreements_at_0.5'], digits=8)}

The last column compares legacy `y_pred` with `y_proba >= 0.5`; for tuned MLP it reflects
the intentionally different saved threshold. For random forest, equality at 0.5 and CSV
rounding must be distinguished from changes caused by model reconstruction. Stage 5 always
uses the documented common threshold rule and its own full-precision prediction export.

## 5.2 Model comparison

**Final test results with validation-selected thresholds**:

{markdown_table(test.reset_index(), columns)}

**Final test results at the common threshold 0.5**:

{markdown_table(tables['test'].query("threshold_mode == 'fixed_0.5'"), columns)}

Always-negative predictions would achieve accuracy {1-tuned.positive_rate:.4f} and positive-class
F1/recall zero on this test set. This explains why accuracy alone is an inadequate comparison.

**Validation results with validation-selected thresholds** (development estimates):

{markdown_table(val.reset_index(), columns)}

Tuned MLP achieves test F1 **{tuned.f1:.4f}**, ROC-AUC **{tuned.roc_auc:.4f}** and
AP **{tuned.average_precision:.4f}**. Its F1 difference from original MLP is
{tuned.f1-test.loc['mlp', 'f1']:+.4f}; from logistic regression it is
{tuned.f1-test.loc['logistic_regression', 'f1']:+.4f}. Comparing the tuned MLP at its selected
threshold with baselines only at 0.5 would confound threshold changes with model changes.
The validation-to-test F1 change for tuned MLP is {tuned.f1-val.loc['mlp_tuned', 'f1']:+.4f}.
See `validation_test_comparison.csv` for all model/metric differences.

## 5.3 Uncertainty and paired comparisons

The following 95% percentile intervals use {repeats} IID row-bootstrap resamples (seed {seed})
on test. Each resample uses the same row indices for every model. Models and thresholds remain
fixed; these intervals describe test-sample uncertainty conditional on the fitted models.
They do not include training-seed or hyperparameter-selection variability, nor BRFSS survey
clustering/weights. Pairwise intervals are exploratory and are not multiplicity-adjusted.

{markdown_table(ci, ['model', 'metric', 'estimate', 'lower_95', 'upper_95'])}

{markdown_table(differences, ['comparison', 'metric', 'estimate', 'lower_95', 'upper_95'], digits=6)}

For tuned versus original MLP, the paired F1 difference interval
[{f1_delta.lower_95:.4f}, {f1_delta.upper_95:.4f}] {evidence}.
Stage 4 seed experiments additionally show training variability; a small observed advantage
must not be described as a general architectural improvement from a single split/seed.

## 5.4 ROC, PR and confusion matrices

![Test ROC curves](../results/evaluation/figures/test_roc_curves.png)

![Test precision-recall curves](../results/evaluation/figures/test_pr_curves.png)

The ROC diagonal represents random ranking; the PR horizontal reference is the test positive
prevalence. The PR plot makes the precision–recall tradeoff visible for the minority class.

![Test confusion counts](../results/evaluation/figures/test_confusion_validation_selected_counts.png)

![Test confusion proportions](../results/evaluation/figures/test_confusion_validation_selected_normalized.png)

Rows are true labels and columns are predicted labels. Normalization is within each true class,
so the positive-class off-diagonal entry is the false-negative rate. Fixed-0.5 matrices and
validation plots are also saved under `results/evaluation/figures/`.

## 5.5 Training behaviour

![Recorded training curves](../results/evaluation/figures/training_curves.png)

These curves compare `exp01_baseline` (a tuning rerun of the base configuration) with
`exp_hidden_256` (the selected configuration). They are not the original Stage 3 training
history. The original logs contain training loss and validation F1/ROC-AUC, but no validation
loss. Consequently no historical validation-loss curve is fabricated and loss-based
overfitting claims cannot be established here. The training code now logs validation loss
and exports preprocessing/history for future runs. Validation F1 in the epoch logs uses 0.5,
which differs from the final selected operating threshold. Training loss is the recorded
mean of batch losses in training mode, including dropout; it is not evaluation-mode train loss.

{markdown_table(pd.DataFrame(history_rows), ['experiment', 'best_epoch_by_val_f1_at_0.5', 'recorded_epochs', 'best_val_f1_at_0.5'])}

Training loss decreases while validation F1 fluctuates and validation ROC-AUC remains close
to 0.837. Later epochs do not consistently improve validation performance, supporting the
use of the saved best epoch rather than the last epoch. The ROC-AUC panel has a narrow
vertical range; visually noticeable fluctuations correspond to small absolute changes.

## 5.6 Error analysis

{markdown_table(errors, ['model', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall', 'false_positive_rate', 'false_negative_rate'])}

At threshold {tuned.threshold:.2f}, tuned MLP misses **{int(tuned.fn):,}** of
**{int(tuned.fn+tuned.tp):,}** positive test cases ({tuned.false_negative_rate:.1%}) and produces
**{int(tuned.fp):,}** false positives. Precision is {tuned.precision:.1%}, so a positive model
prediction should not be interpreted as a confirmed diagnosis. The max-F1 operating point
optimizes an empirical classification metric, not an established clinical utility function.

For tuned MLP, descriptive subgroup results are:

{markdown_table(tuned_groups, ['feature', 'group', 'n_samples', 'tp', 'fn', 'fp', 'tn', 'recall', 'false_positive_rate', 'precision'])}

Age groups are under 45, 45–64 and 65+; the dataset codes sex as 0=male, 1=female.
Rates must be read with their denominators, especially positive counts for recall.
These subgroup comparisons are exploratory; they do not establish causal explanations or
formal fairness conclusions. `test_subgroup_metrics.csv` includes all five models.
For tuned MLP, recall is {age_groups.loc['under 45', 'recall']:.1%} below age 45 versus
{age_groups.loc['65+', 'recall']:.1%} at age 65+, while the corresponding false-positive rates
are {age_groups.loc['under 45', 'false_positive_rate']:.1%} and
{age_groups.loc['65+', 'false_positive_rate']:.1%}. Thus the overall recall hides substantially
different error patterns by age. Recall is {sex_groups.loc['Female', 'recall']:.1%} for female
respondents versus {sex_groups.loc['Male', 'recall']:.1%} for male respondents at the same threshold.
These differences warrant further validation; age distributions, prevalence and feature
composition may differ across groups and have not been controlled in this descriptive analysis.
`test_error_examples.csv` contains the ten lowest-score false negatives and ten highest-score
false positives per model, with checked corresponding input features. These are deliberately
extreme examples, not a representative sample of all errors.

## 5.7 Limitations and reproducibility

The data describe self-reported history and may contain label noise. Evaluation uses one
random within-dataset split, without temporal/external validation or survey weighting.
Repeated validation use introduces selection optimism. Weighted training scores are not
shown to be calibrated disease probabilities. The processed data retain no respondent IDs,
so evaluation verifies row alignment but cannot independently audit respondent-level overlap
from these files alone. This work evaluates a dataset classifier and does not establish
clinical deployment suitability.

Environment: Python {manifest['python']}, NumPy {manifest['numpy']}, pandas {manifest['pandas']},
scikit-learn {manifest['scikit_learn']}, PyTorch {manifest['torch']}; CPU inference.
Source and artifact SHA-256 hashes are recorded in `results/evaluation/manifest.json`.

From the repository root:

```bash
python3 -m pip install -r requirements-evaluation.txt
# Once, if frozen evaluation artifacts do not exist:
python3 src/predict_test.py
# Rebuild tables, plots and this report from frozen predictions:
python3 src/evaluation.py --bootstrap {repeats} --seed {seed}
python3 -m unittest discover -s tests -v
```

`predict_test.py` refuses to overwrite a completed frozen evaluation. Analysis can be rerun
without training or new test inference. Legacy predictions and Stage 2–4 reports are inputs,
not overwritten outputs. Full fitted baseline pipelines are local reproducibility artifacts
excluded from Git for size; they are reconstructed by `predict_test.py` when needed.
'''
    report = report_path()
    output_label = os.path.relpath(OUT, ROOT)
    image_prefix = Path(os.path.relpath(OUT / 'figures', report.parent)).as_posix() + '/'
    text = text.replace('../results/evaluation/figures/', image_prefix)
    text = text.replace('results/evaluation/', output_label + '/')
    if OUT.resolve() != (ROOT / 'results/evaluation').resolve():
        option = ' --output-dir ' + shlex.quote(str(OUT))
        text = text.replace('python3 src/predict_test.py', 'python3 src/predict_test.py' + option)
        text = text.replace('python3 src/evaluation.py --bootstrap',
                            'python3 src/evaluation.py' + option + ' --bootstrap')
    text += """

### Reproduce inference in a separate directory

To reconstruct the fixed baselines and replay the original MLP weights without replacing
this frozen experiment, choose a new directory for both commands:

```bash
python3 src/predict_test.py --output-dir results/evaluation_runs/recheck
python3 src/evaluation.py --output-dir results/evaluation_runs/recheck --bootstrap 1000 --seed 42
```

The separate report is `results/evaluation_runs/recheck/05_results.md`; its figures and
predictions are stored alongside it. A completed inference run cannot be overwritten.
Choose another unused directory for a further replay. This is a reproducibility check of
fixed models and thresholds selected on validation, not additional tuning on test.
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)


def report_path():
    if OUT.resolve() == (ROOT / 'results/evaluation').resolve():
        return ROOT / 'reports/05_results.md'
    return OUT / '05_results.md'


def verify_manifest(manifest):
    for relative, expected in manifest['sources_sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Source changed since protocol freeze: {relative}')
    modern = manifest.get('artifacts_relative_to') == 'output_dir'
    base = OUT if modern else ROOT
    for relative, expected in manifest['artifacts_sha256'].items():
        path = base / relative
        model_prefix = 'models/' if modern else 'results/evaluation/models/'
        if relative.startswith(model_prefix) and not path.exists():
            continue  # Serialized pipelines are optional for analysis of frozen CSVs.
        if sha256(path) != expected:
            raise ValueError(f'Artifact changed since protocol freeze: {relative}')


def write_analysis_manifest(repeats, seed):
    sources = [ROOT / 'src/evaluation.py', ROOT / 'src/evaluation_common.py',
               ROOT / 'results/tuning_histories.csv', ROOT / 'requirements-evaluation.txt']
    (OUT / 'analysis_manifest.json').write_text(json.dumps({
        'bootstrap_repeats': repeats, 'seed': seed,
        'matplotlib': matplotlib.__version__,
        'sources_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
    }, indent=2) + '\n')


def main():
    global OUT, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Directory containing frozen inference artifacts (relative to cwd).')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    FIGURES = OUT / 'figures'
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least 100 (1000 recommended for final output)')
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    verify_manifest(manifest)
    protocol = json.loads((OUT / 'frozen_protocol.json').read_text())
    thresholds = protocol['thresholds']
    tables = {}
    for split in ['validation', 'test']:
        dataset = pd.read_csv(ROOT / f'data/processed/{split}_ready.csv')
        frames = {m: validate_predictions(pd.read_csv(OUT / f'predictions/{m}_{split}.csv'), dataset.target)
                  for m in MODELS}
        rows = []
        for name, frame in frames.items():
            for mode, threshold in [('fixed_0.5', 0.5), ('validation_selected', thresholds[name])]:
                rows.append({'model': name, 'split': split, 'threshold_mode': mode,
                             **scores(frame.y_true, frame.y_proba, threshold)})
        tables[split] = pd.DataFrame(rows)
        tables[split].to_csv(OUT / f'{split}_metrics.csv', index=False)
        plot_comparison(frames, thresholds, split)
        errors, groups = analyze_errors(frames, dataset, thresholds, split)
        print(f'Finished {split} tables, curves and error analysis.', flush=True)
    plot_training()
    # frames/errors/groups refer to test, the final iteration above.
    ci, differences = bootstrap(frames, thresholds, args.bootstrap, args.seed)
    write_report(tables, ci, differences, errors, groups, manifest, protocol, args.bootstrap, args.seed)
    write_analysis_manifest(args.bootstrap, args.seed)
    print(f'Saved {report_path()} and {OUT}', flush=True)


if __name__ == '__main__':
    main()
