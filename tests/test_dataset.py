from __future__ import annotations

import json
import unittest
from pathlib import Path

from summar.evaluation import source_copy_ratio
from summar.dataset_quality import max_shared_ngram_frequency


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_split(filename: str) -> list[dict[str, str]]:
    with (PROJECT_ROOT / "data" / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


class DatasetTests(unittest.TestCase):
    def test_dataset_uses_seventy_thirty_split(self) -> None:
        train = load_split("train.json")
        validation = load_split("validation.json")

        self.assertEqual(len(train), 210)
        self.assertEqual(len(validation), 90)
        self.assertEqual(len(validation) / (len(train) + len(validation)), 0.3)

    def test_splits_have_valid_schema_and_do_not_overlap(self) -> None:
        train = load_split("train.json")
        validation = load_split("validation.json")

        for item in train + validation:
            self.assertEqual(set(item), {"text", "summary"})
            self.assertTrue(item["text"].strip())
            self.assertTrue(item["summary"].strip())

        train_texts = {item["text"].strip().lower() for item in train}
        validation_texts = {item["text"].strip().lower() for item in validation}
        self.assertEqual(len(train_texts), len(train))
        self.assertEqual(len(validation_texts), len(validation))
        self.assertTrue(train_texts.isdisjoint(validation_texts))

    def test_summaries_are_abstractive_and_unique(self) -> None:
        rows = load_split("train.json") + load_split("validation.json")
        summaries = [item["summary"].strip().lower() for item in rows]

        self.assertEqual(len(summaries), len(set(summaries)))
        for item in rows:
            self.assertLess(len(item["summary"]), len(item["text"]))
            self.assertLessEqual(
                source_copy_ratio(item["summary"], item["text"]),
                0.4,
                msg=f"Summary слишком дословно повторяет исходник: {item['summary']}",
            )

    def test_sources_do_not_share_repeated_templates(self) -> None:
        rows = load_split("train.json") + load_split("validation.json")
        self.assertLessEqual(
            max_shared_ngram_frequency([item["text"] for item in rows]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
