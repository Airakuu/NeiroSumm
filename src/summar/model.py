from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

from summar.config import DEFAULT_LOCAL_MODEL_DIR, DEFAULT_MODEL_NAME, SummarizerConfig
from summar.preprocessing import normalize_text, prepare_text_chunks


class SummarizationModel:
    def __init__(self, config: SummarizerConfig | None = None) -> None:
        self.config = config or SummarizerConfig()
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Neural summarization dependencies are missing. "
                "Install the packages from requirements.txt."
            ) from exc

        model_source = self._resolve_model_source()
        self._tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
        self._model.to(self.config.device)

    def _resolve_model_source(self) -> str:
        configured_path = Path(self.config.model_name)
        if configured_path.exists():
            return str(configured_path)

        if self.config.model_name == DEFAULT_MODEL_NAME and DEFAULT_LOCAL_MODEL_DIR.exists():
            return str(DEFAULT_LOCAL_MODEL_DIR)

        return self.config.model_name

    def summarize(self, text: str) -> str:
        source_text = normalize_text(text)
        if not source_text:
            return ""

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        chunks = prepare_text_chunks(source_text, chunk_size=8, overlap=self.config.chunk_overlap)
        if not chunks:
            chunks = [source_text]

        partial_summaries = [
            self._summarize_chunk(
                chunk,
                prompt_prefix="Кратко перескажи текст, сохранив ключевые события и смысл: ",
            )
            for chunk in chunks
        ]
        if len(partial_summaries) == 1:
            return self._fit_summary_length(partial_summaries[0], source_text)

        joined = " ".join(partial_summaries)
        second_pass = replace(
            self.config,
            max_input_tokens=min(self.config.max_input_tokens, 420),
            max_summary_tokens=min(self.config.max_summary_tokens, 220),
            min_summary_tokens=min(self.config.min_summary_tokens, 80),
        )
        final_summary = SummarizationModel(second_pass)._summarize_chunk(
            joined,
            tokenizer=self._tokenizer,
            model=self._model,
            prompt_prefix="Сделай связное краткое содержание без копирования фраз из исходного текста: ",
        )
        return self._fit_summary_length(final_summary, source_text)

    def _fit_summary_length(self, summary: str, source_text: str) -> str:
        normalized_summary = self._clean_generated_summary(summary)
        if not normalized_summary:
            return ""

        min_length = self._required_summary_length(source_text)
        if len(normalized_summary) >= min_length:
            return normalized_summary

        expanded_summary = self._generate_more_detailed_summary(source_text, normalized_summary)
        expanded_summary = self._clean_generated_summary(expanded_summary)
        if len(expanded_summary) >= len(normalized_summary):
            return expanded_summary
        return normalized_summary

    def _required_summary_length(self, source_text: str) -> int:
        ratio = min(max(self.config.min_retention_ratio, 0.1), 1.0)
        return max(1, int(len(source_text) * ratio))

    def _generate_more_detailed_summary(self, source_text: str, draft_summary: str) -> str:
        detailed_config = replace(
            self.config,
            max_summary_tokens=min(max(self.config.max_summary_tokens, 320), 384),
            min_summary_tokens=min(max(self.config.min_summary_tokens, 120), 220),
            num_beams=max(self.config.num_beams, 5),
        )

        refinement_source = source_text

        return SummarizationModel(detailed_config)._summarize_chunk(
            refinement_source,
            tokenizer=self._tokenizer,
            model=self._model,
            prompt_prefix=(
                "Подробно перескажи текст своими словами. "
                "Сохрани ключевые события, причинно-следственные связи и общий смысл: "
            ),
        )

    def _clean_generated_summary(self, summary: str) -> str:
        normalized = normalize_text(summary)
        if not normalized:
            return ""

        banned_fragments = (
            "подробно перескажи текст",
            "краткое содержание текста",
            "не копируй",
            "сохрани ключевые события",
        )
        sentences = [
            sentence
            for sentence in self._split_summary_sentences(normalized)
            if not any(fragment in sentence.lower() for fragment in banned_fragments)
        ]
        if sentences:
            return normalize_text(" ".join(sentences))
        return normalized

    def _split_summary_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _summarize_chunk(
        self,
        text: str,
        tokenizer=None,
        model=None,
        prompt_prefix: str = "",
    ) -> str:
        tokenizer = tokenizer or self._tokenizer
        model = model or self._model
        assert tokenizer is not None
        assert model is not None

        prompt_text = f"{prompt_prefix}{text}".strip()
        encoded = tokenizer(
            prompt_text,
            max_length=self.config.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}

        input_tokens = int(encoded["input_ids"].shape[1])
        dynamic_min_tokens = min(
            self.config.max_summary_tokens - 8,
            max(self.config.min_summary_tokens, int(input_tokens * self.config.min_retention_ratio * 0.85)),
        )

        output = model.generate(
            **encoded,
            max_length=self.config.max_summary_tokens,
            min_length=dynamic_min_tokens,
            num_beams=self.config.num_beams,
            length_penalty=1.35,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()
