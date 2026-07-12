from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summar.evaluation import load_test_cases, source_copy_ratio, token_f1
from summar.preprocessing import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_permanent_suite_contains_five_genres(self) -> None:
        cases = load_test_cases(PROJECT_ROOT / "test_texts")
        self.assertEqual(len(cases), 5)
        self.assertEqual(len({case["genre"] for case in cases}), 5)

    def test_test_texts_are_not_in_training_or_validation_data(self) -> None:
        cases = load_test_cases(PROJECT_ROOT / "test_texts")
        dataset_texts: set[str] = set()
        for filename in ("train.json", "validation.json"):
            with (PROJECT_ROOT / "data" / filename).open("r", encoding="utf-8") as file:
                dataset_texts.update(normalize_text(item["text"]).lower() for item in json.load(file))
        for case in cases:
            self.assertNotIn(normalize_text(case["text"]).lower(), dataset_texts)

    def test_manifest_rejects_path_outside_test_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            test_dir = Path(directory)
            (test_dir / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "unsafe",
                            "genre": "test",
                            "file": "../outside.txt",
                            "reference": "summary",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_test_cases(test_dir)

    def test_token_f1_is_one_for_equal_texts(self) -> None:
        self.assertEqual(token_f1("Краткий русский текст", "Краткий русский текст"), 1.0)

    def test_source_copy_ratio_detects_copied_trigram(self) -> None:
        self.assertEqual(source_copy_ratio("два три четыре", "один два три четыре пять"), 1.0)


if __name__ == "__main__":
    unittest.main()
