# Экспериментальная модель v9

Версия `v9` обучена на датасете из 300 пар: 210 примеров использовались для
обучения и 90 для validation. Исходной точкой была локальная базовая модель
`models/rut5-base-absum`.

## Команда обучения

```powershell
python train_model.py --train-file data/train.json --validation-file data/validation.json --text-column text --summary-column summary --model-name models/rut5-base-absum --output-dir checkpoints/rut5-absum-finetuned-v9 --epochs 4 --learning-rate 5e-5 --label-smoothing-factor 0 --early-stopping-patience 2
```

Обучение на CPU заняло 21 минуту 35 секунд. Лучший validation-loss: `2.411`.

## Итоговые метрики

- Среднее сокращение: `72.48%`.
- Средний `reference_token_f1`: `0.1361`.
- Средняя доля копирования исходника: `0.2993`.
- Постоянных тестовых текстов: `5`.

## Вывод

По сравнению с `v8` новая версия сократила среднее копирование с `0.3265` до
`0.2993`, но `reference_token_f1` уменьшился с `0.1437` до `0.1361`. Ручная
проверка выявила подмену причин, неудачные отрицания и проблемы со связностью,
особенно в классической прозе. Поэтому версия сохранена для анализа, но не
назначена основной.

Полные результаты находятся в `test_results.json`. Файл весов не включён в
Git, потому что `model.safetensors` занимает около 932 МБ и превышает лимит
обычного репозитория GitHub.
