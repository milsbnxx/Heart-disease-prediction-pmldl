# Heart-disease-prediction-pmldl 
Project 2026

## Stage 5: Evaluation and Results

Готовый раздел отчёта: [reports/05_results.md](reports/05_results.md).
Код оценки: [src/evaluation.py](src/evaluation.py).
Инструкция и описание файлов: [results/evaluation/README.md](results/evaluation/README.md).

```bash
python3 -m pip install -r requirements-evaluation.txt
python3 src/evaluation.py --bootstrap 1000 --seed 42
python3 -m unittest discover -s tests -v
```

Сохранённые результаты включают validation и final test для пяти моделей, метрики при
пороге 0.5 и при порогах, выбранных на validation, ROC/PR curves, confusion matrices,
доступные training curves, анализ ошибок и 95% парные bootstrap-интервалы.
Повторный запуск evaluation использует сохранённые предсказания без переобучения.

Для независимого воспроизведения inference в новом каталоге:

```bash
python3 src/predict_test.py --output-dir results/evaluation_runs/recheck
python3 src/evaluation.py --output-dir results/evaluation_runs/recheck --bootstrap 1000 --seed 42
```

Отдельный отчёт: `results/evaluation_runs/recheck/05_results.md`.
Основной отчёт и исходные результаты не перезаписываются.
