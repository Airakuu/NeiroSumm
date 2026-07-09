# Summar

Учебный проект на Python для суммаризации текста и дообучения русскоязычной модели.

Сейчас в проекте оставлена одна основная модель:

- `rut5_base_sum_gazeta`

Именно её лучше использовать как базовую и именно её имеет смысл дообучать на своих примерах.

## Структура проекта

- `main.py` — запуск суммаризации из консоли
- `run_api.py` — запуск API на FastAPI
- `train_model.py` — запуск дообучения
- `src/summar/model.py` — логика суммаризации
- `src/summar/train.py` — логика обучения
- `data/train.json` — стартовые обучающие примеры
- `data/validation.json` — стартовые примеры для проверки
- `data/dataset_notes.md` — пояснения по формату датасета

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Быстрый запуск

Если у тебя уже скачана модель:

```bash
python main.py --input-file text.txt
```

Если модель нужно скачать отдельно:

```bash
python download_model.py
```

## Что показывает вывод

После суммаризации проект выводит:

- само краткое содержание;
- количество символов в исходном тексте;
- количество символов в сокращённом тексте;
- процент сокращения.

## Дообучение на своих примерах

В проект уже добавлен стартовый датасет в папке `data/`.

Формат одной записи:

```json
{
  "text": "Полный текст",
  "summary": "Краткое содержание"
}
```

### Запуск дообучения на локальных файлах

```bash
python train_model.py ^
  --train-file data/train.json ^
  --validation-file data/validation.json ^
  --text-column text ^
  --summary-column summary ^
  --model-name models/rut5_base_sum_gazeta ^
  --output-dir checkpoints/rut5-finetuned ^
  --epochs 3
```

После обучения новая версия модели сохранится в `checkpoints/rut5-finetuned`.

## Проверка дообученной модели

После обучения можно запускать уже новый чекпоинт:

```bash
python main.py --model-name checkpoints/rut5-finetuned --input-file text.txt
```

## Запуск API

```bash
python run_api.py
```

После этого будет доступен `POST /summarize`.

## Тесты

```bash
python -m unittest discover -s tests -v
```

## Что делать дальше

1. Добавить больше своих примеров в `data/train.json`.
2. Сделать summaries в одном стиле.
3. Дообучить `rut5_base_sum_gazeta` на своём наборе.
4. После этого вынести модель в приложение или сайт.
>>>>>>> 9672065 (Initial commit)
