from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from summar.config import SummarizerConfig
from summar.model import SummarizationModel


class ModelLengthTests(unittest.TestCase):
    def test_prepare_chunks_keeps_text_that_fits_token_limit(self) -> None:
        class StubTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1] * 20

        model = SummarizationModel(SummarizerConfig(max_input_tokens=32))
        model._tokenizer = StubTokenizer()
        text = "Первое. Второе. Третье. Четвертое. Пятое. Шестое. Седьмое. Восьмое. Девятое."
        self.assertEqual(model._prepare_chunks(text), [text])

    def test_required_summary_length_uses_min_retention_ratio(self) -> None:
        model = SummarizationModel(SummarizerConfig(min_retention_ratio=0.3))
        self.assertEqual(model._required_summary_length("a" * 100), 30)

    def test_fit_summary_length_keeps_generated_summary(self) -> None:
        model = SummarizationModel(SummarizerConfig(min_retention_ratio=0.3))
        source_text = "a" * 100
        summary = "Short summary."
        self.assertEqual(model._fit_summary_length(summary, source_text), summary)

    def test_generation_token_limits_target_requested_ratio(self) -> None:
        model = SummarizationModel(
            SummarizerConfig(
                min_retention_ratio=0.3,
                min_summary_tokens=8,
                max_summary_tokens=256,
            )
        )
        self.assertEqual(model._generation_token_limits(100), (25, 65))

    def test_generation_token_limits_respect_configured_bounds(self) -> None:
        model = SummarizationModel(
            SummarizerConfig(
                min_retention_ratio=0.3,
                min_summary_tokens=24,
                max_summary_tokens=40,
            )
        )
        self.assertEqual(model._generation_token_limits(200), (32, 40))

    def test_absum_model_uses_its_own_token_limits(self) -> None:
        model = SummarizationModel(
            SummarizerConfig(
                model_name="models/rut5-base-absum",
                min_retention_ratio=0.33,
                min_summary_tokens=24,
                max_summary_tokens=256,
            )
        )
        self.assertEqual(model._absum_token_limits(341), (92, 187))

    def test_clean_generated_summary_adds_final_punctuation(self) -> None:
        model = SummarizationModel()
        self.assertEqual(
            model._clean_generated_summary("краткий пересказ без точки"),
            "Краткий пересказ без точки.",
        )

    def test_clean_generated_summary_adds_space_after_punctuation(self) -> None:
        model = SummarizationModel()
        self.assertEqual(
            model._clean_generated_summary("первое предложение.Второе,третье"),
            "Первое предложение. Второе, третье.",
        )


if __name__ == "__main__":
    unittest.main()
