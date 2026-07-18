from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_split(filename: str) -> list[dict[str, str]]:
    with (PROJECT_ROOT / "data" / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


class DatasetTests(unittest.TestCase):
    def test_dataset_uses_seventy_thirty_split(self) -> None:
        train = load_split("train.json")
        validation = load_split("validation.json")

        self.assertEqual(len(train), 175)
        self.assertEqual(len(validation), 75)
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


if __name__ == "__main__":
    unittest.main()
