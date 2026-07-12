from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from summar.config import SummarizerConfig
from summar.model import SummarizationModel
from summar.preprocessing import normalize_text


WORD_PATTERN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def load_test_cases(test_dir: Path) -> list[dict[str, str]]:
    manifest_path = test_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest.json должен содержать непустой список")

    required_fields = {"id", "genre", "file", "reference"}
    seen_ids: set[str] = set()
    cases: list[dict[str, str]] = []
    for index, item in enumerate(manifest, start=1):
        if not isinstance(item, dict) or set(item) != required_fields:
            raise ValueError(f"Некорректная запись теста #{index}")
        if not all(isinstance(item[field], str) and item[field].strip() for field in required_fields):
            raise ValueError(f"Пустое поле в тесте #{index}")
        if item["id"] in seen_ids:
            raise ValueError(f"Повторяющийся id теста: {item['id']}")

        text_path = (test_dir / item["file"]).resolve()
        if test_dir.resolve() not in text_path.parents:
            raise ValueError(f"Файл теста находится вне test_texts: {item['file']}")
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Пустой тестовый текст: {item['file']}")

        cases.append({**item, "text": text})
        seen_ids.add(item["id"])
    return cases


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_words(prediction))
    expected = Counter(_words(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected or overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def source_copy_ratio(summary: str, source: str, n: int = 3) -> float:
    summary_words = _words(summary)
    source_words = _words(source)
    if len(summary_words) < n:
        return 0.0
    source_ngrams = {tuple(source_words[index : index + n]) for index in range(len(source_words) - n + 1)}
    summary_ngrams = [
        tuple(summary_words[index : index + n])
        for index in range(len(summary_words) - n + 1)
    ]
    copied = sum(ngram in source_ngrams for ngram in summary_ngrams)
    return copied / len(summary_ngrams)


def evaluate_summary(summary: str, source: str, reference: str) -> dict[str, float | int]:
    source_length = len(normalize_text(source))
    summary_length = len(normalize_text(summary))
    retention = summary_length / source_length if source_length else 0.0
    return {
        "source_characters": source_length,
        "summary_characters": summary_length,
        "retention_percent": round(retention * 100, 1),
        "reduction_percent": round((1 - retention) * 100, 1),
        "reference_token_f1": round(token_f1(summary, reference), 4),
        "source_copy_ratio": round(source_copy_ratio(summary, source), 4),
    }


def run_test_suite(
    model_name: str,
    test_dir: Path,
    output_path: Path,
    device: str = "cpu",
) -> dict:
    cases = load_test_cases(test_dir)
    model = SummarizationModel(SummarizerConfig(model_name=model_name, device=device))
    results: list[dict] = []

    for index, case in enumerate(cases, start=1):
        print(f"Тест {index}/{len(cases)}: {case['genre']} ({case['id']})", flush=True)
        summary = model.summarize(case["text"])
        metrics = evaluate_summary(summary, case["text"], case["reference"])
        results.append(
            {
                "id": case["id"],
                "genre": case["genre"],
                "file": case["file"],
                "summary": summary,
                "reference": case["reference"],
                "metrics": metrics,
            }
        )
        print(
            f"  сокращение={metrics['reduction_percent']}%, "
            f"эталон={metrics['reference_token_f1']}, "
            f"копирование={metrics['source_copy_ratio']}",
            flush=True,
        )

    report = {
        "model": model_name,
        "test_count": len(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report
