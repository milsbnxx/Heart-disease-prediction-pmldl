# Этап 5: Evaluation / Results

Основной текст для включения в общий project.pdf: [05_results.md](../../reports/05_results.md).
Отчёт написан на английском, как остальные части проекта. Markdown можно открыть через
Preview в IDE; пути к изображениям относительные.

## Что сделано

- Проверены row_index и target для всех предсказаний.
- Восстановлены три baseline-pipeline на train с исходными параметрами. Их validation
  вероятности совпали с прежними CSV с точностью до численной погрешности.
- Загружены исходные веса MLP и tuned MLP, восстановлен preprocessing на train и проверено
  совпадение validation-предсказаний. MLP не переобучались.
- До загрузки test зафиксированы пороги: LR 0.72, DT 0.96, RF 0.20, MLP 0.72, tuned MLP 0.70.
- Выполнен inference на test, построены таблицы и графики для пяти моделей.
- Посчитаны 95% интервалы для F1, ROC-AUC, AP и парных разностей моделей: 1000 bootstrap-выборок,
  seed 42. Пороги внутри bootstrap не подбираются.
- Сохранены ошибки FP/FN, крайние ошибочные примеры и метрики по полу и возрасту.

Test теперь использован для финальной оценки: не следует подбирать по нему новые пороги,
архитектуру или признаки и продолжать называть его независимым финальным test.

## Запуск из корня репозитория

Окружение проверено с Python 3.11. Для установки можно использовать собственное venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-evaluation.txt
python3 src/evaluation.py --bootstrap 1000 --seed 42
python3 -m unittest discover -s tests -v
```

`evaluation.py` пересоздаёт таблицы, графики и `reports/05_results.md` из frozen CSV.
Ручные правки сгенерированного отчёта будут заменены; для постоянных правок изменяйте
`write_report()` в коде или сделайте отдельную редакторскую копию отчёта.

`python3 src/predict_test.py` нужен только для первоначального получения артефактов, когда
`manifest.json` ещё отсутствует. Он отказывается перезаписывать уже завершённый эксперимент.
На текущем checkout всё уже получено. Для независимого воспроизведения inference с нуля
используйте новый каталог (пути относительно текущей рабочей папки):

```bash
python3 src/predict_test.py --output-dir results/evaluation_runs/recheck
python3 src/evaluation.py --output-dir results/evaluation_runs/recheck --bootstrap 1000 --seed 42
```

Отдельный отчёт появится в `results/evaluation_runs/recheck/05_results.md`.
Исходные `reports/05_results.md` и `results/evaluation/` остаются на месте.
Можно указать абсолютный путь вне репозитория. Оба скрипта должны получать один и тот же
`--output-dir`. Для ещё одного повторения выберите новый каталог: завершённый inference
защищён от перезаписи. В новом формате manifest пути артефактов относительны выходному
каталогу; прежний manifest также поддерживается.

Это повторение фиксированного эксперимента, а не новый подбор параметров по test.

## Файлы

| Файл / папка | Содержание |
|---|---|
| `validation_metrics.csv`, `test_metrics.csv` | По две строки на модель: fixed_0.5 и validation_selected |
| `validation_test_comparison.csv` | Разницы F1, ROC-AUC, AP между validation и test |
| `test_confidence_intervals.csv` | 95% интервалы метрик на test |
| `test_paired_differences.csv` | 95% интервалы разностей tuned MLP минус другая модель |
| `*_error_summary.csv` | TP, TN, FP, FN и частоты ошибок при выбранных порогах |
| `*_subgroup_metrics.csv` | Метрики каждой модели по полу и возрасту, с размерами групп |
| `*_error_examples.csv` | До 10 крайних FN и FP на модель с входными признаками |
| `predictions/` | Предсказания Stage 5 на validation и test, без замены старых CSV |
| `figures/` | 13 PNG: confusion matrices, ROC, PR и training curves |
| `training_curves_source.csv` | Использованные исходные строки истории tuning |
| `frozen_protocol.json` | Пороги, критерий выбора и выбранная до test модель |
| `validation_replay_audit.csv` | Расхождения восстановленных и исходных предсказаний |
| `manifest.json` | Версии окружения, размеры выборок, SHA-256 входов и артефактов inference |
| `analysis_manifest.json` | Параметры bootstrap, версия matplotlib и хеши источников анализа |
| `models/` | Локальные fitted baseline-pipeline и восстановленный MLP preprocessing |

Крупные pipeline в `models/` исключены из Git. Для пересчёта графиков и метрик они не нужны:
`evaluation.py` проверяет их хеши, если файлы доступны, но может работать без них. Сохранённые
предсказания, веса MLP, исходные CSV и протокол проверяются по хешам обязательно.

## Ограничения training curves

Старые логи не содержат validation loss, а история исходного запуска Stage 3 не сохранена.
Показаны train loss, validation F1 и ROC-AUC для `exp01_baseline` и `exp_hidden_256` из tuning.
Они подписаны как tuning-runs. Исторические значения не выдумывались; для будущих запусков
в `src/mlp.py` добавлено сохранение validation loss, истории и preprocessing. Переобучение
для восстановления отсутствующего графика не требуется для воспроизведения этих результатов.

## Главный результат

На test tuned MLP имеет F1 0.4136 против 0.4125 у обычной MLP. 95% парный интервал разницы
F1: [-0.0022, 0.0043], поэтому убедительного преимущества tuned MLP по F1 не установлено.
По сравнению с Logistic Regression разница F1 составляет +0.0107, интервал [0.0053, 0.0159]
при условии фиксированных моделей и IID bootstrap. Выводы ограничены одним split и не
учитывают дизайн обследования BRFSS или изменчивость обучения.
