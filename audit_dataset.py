from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from summar.evaluation import source_copy_ratio
from summar.dataset_quality import max_shared_ngram_frequency


COPY_LIMIT = 0.4


def load_rows(paths: list[Path]) -> list[tuple[Path, int, dict[str, str]]]:
    rows: list[tuple[Path, int, dict[str, str]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            items = json.load(file)
        rows.extend((path, index, item) for index, item in enumerate(items))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка качества обучающего датасета")
    parser.add_argument(
        "files",
        type=Path,
        nargs="*",
        default=[Path("data/train.json"), Path("data/validation.json")],
    )
    args = parser.parse_args()
    rows = load_rows(args.files)
    copy_ratios = [source_copy_ratio(item["summary"], item["text"]) for _, _, item in rows]
    retention = [len(item["summary"]) / len(item["text"]) for _, _, item in rows]
    max_source_repetition = max_shared_ngram_frequency(
        [item["text"] for _, _, item in rows]
    )
    violations = [
        (path, index, ratio, item["summary"])
        for (path, index, item), ratio in zip(rows, copy_ratios)
        if ratio > COPY_LIMIT
    ]

    print(f"Примеров: {len(rows)}")
    print(f"Средняя доля summary: {statistics.mean(retention):.1%}")
    print(f"Среднее копирование: {statistics.mean(copy_ratios):.1%}")
    print(f"Максимальное копирование: {max(copy_ratios):.1%}")
    print(f"Максимальный повтор фрагмента исходника: {max_source_repetition}")

    for path, index, ratio, summary in violations:
        print(f"{path}:{index} — копирование {ratio:.1%}: {summary}")
    if violations:
        print(f"Нарушений порога {COPY_LIMIT:.0%}: {len(violations)}")
        return 1
    if max_source_repetition > 2:
        print("Найдены исходники с повторяющимся шаблоном из восьми слов.")
        return 1

    print("Нарушений качества не найдено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
