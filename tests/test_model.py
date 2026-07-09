from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from summar.config import SummarizerConfig
from summar.model import SummarizationModel


class ModelLengthTests(unittest.TestCase):
    def test_required_summary_length_uses_min_retention_ratio(self) -> None:
        model = SummarizationModel(SummarizerConfig(min_retention_ratio=0.3))
        self.assertEqual(model._required_summary_length("a" * 100), 30)

    def test_fit_summary_length_keeps_long_enough_summary(self) -> None:
        model = SummarizationModel(SummarizerConfig(min_retention_ratio=0.3))
        source_text = "a" * 100
        summary = "This summary is definitely longer than thirty characters."
        self.assertEqual(model._fit_summary_length(summary, source_text), summary)

    def test_fit_summary_length_returns_draft_if_refinement_is_shorter(self) -> None:
        class StubModel(SummarizationModel):
            def _generate_more_detailed_summary(self, source_text: str, draft_summary: str) -> str:
                return "short"

        model = StubModel(SummarizerConfig(min_retention_ratio=0.3))
        source_text = "a" * 100
        summary = "Draft summary text."
        self.assertEqual(model._fit_summary_length(summary, source_text), summary)

    def test_fit_summary_length_uses_refinement_if_it_is_longer(self) -> None:
        class StubModel(SummarizationModel):
            def _generate_more_detailed_summary(self, source_text: str, draft_summary: str) -> str:
                return "This is a longer refined summary that should be preferred."

        model = StubModel(SummarizerConfig(min_retention_ratio=0.3))
        source_text = "a" * 100
        summary = "Short draft."
        self.assertEqual(
            model._fit_summary_length(summary, source_text),
            "This is a longer refined summary that should be preferred.",
        )


if __name__ == "__main__":
    unittest.main()
