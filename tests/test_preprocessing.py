from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from summar.preprocessing import chunk_sentences, normalize_text, prepare_text_chunks, split_sentences


class PreprocessingTests(unittest.TestCase):
    def test_normalize_text_removes_extra_spaces(self) -> None:
        source = "  Это   тестовый \n\n текст.  "
        self.assertEqual(normalize_text(source), "Это тестовый текст.")

    def test_split_sentences_keeps_sentence_order(self) -> None:
        source = "Первая фраза. Вторая фраза! Третья?"
        self.assertEqual(split_sentences(source), ["Первая фраза.", "Вторая фраза!", "Третья?"])

    def test_chunk_sentences_builds_overlap(self) -> None:
        sentences = ["a", "b", "c", "d", "e"]
        self.assertEqual(
            chunk_sentences(sentences, chunk_size=3, overlap=1),
            ["a b c", "c d e"],
        )

    def test_prepare_text_chunks_handles_empty_text(self) -> None:
        self.assertEqual(prepare_text_chunks("   "), [])


if __name__ == "__main__":
    unittest.main()
