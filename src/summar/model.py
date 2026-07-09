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

        chunks = self._prepare_chunks(source_text)

        partial_summaries = [
            self._summarize_chunk(chunk)
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
        )
        return self._fit_summary_length(final_summary, source_text)

    def _prepare_chunks(self, source_text: str) -> list[str]:
        assert self._tokenizer is not None
        token_ids = self._tokenizer.encode(source_text, add_special_tokens=True)
        if len(token_ids) <= self.config.max_input_tokens:
            return [source_text]

        chunks = prepare_text_chunks(
            source_text,
            chunk_size=8,
            overlap=self.config.chunk_overlap,
        )
        return chunks or [source_text]

    def _fit_summary_length(self, summary: str, source_text: str) -> str:
        # Do not pad a correct short summary: forced retries made the model invent details.
        return self._clean_generated_summary(summary)

    def _required_summary_length(self, source_text: str) -> int:
        ratio = min(max(self.config.min_retention_ratio, 0.1), 1.0)
        return max(1, int(len(source_text) * ratio))

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
        cleaned = normalize_text(" ".join(sentences)) if sentences else normalized
        return self._finalize_summary_text(cleaned)

    def _split_summary_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _finalize_summary_text(self, text: str) -> str:
        cleaned = normalize_text(text)
        if not cleaned:
            return ""

        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([,.;:!?]){2,}", r"\1", cleaned)
        cleaned = re.sub(r"([,.;:!?])(?=[A-Za-zА-Яа-яЁё])", r"\1 ", cleaned)
        cleaned = cleaned.strip(" ,;:")

        if cleaned and len(cleaned.split()) >= 3 and cleaned[-1] not in ".!?":
            cleaned += "."

        if cleaned and cleaned[0].isalpha():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _summarize_chunk(
        self,
        text: str,
        tokenizer=None,
        model=None,
    ) -> str:
        tokenizer = tokenizer or self._tokenizer
        model = model or self._model
        assert tokenizer is not None
        assert model is not None

        encoded = tokenizer(
            text,
            max_length=self.config.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}

        input_tokens = int(encoded["input_ids"].shape[1])
        if self._uses_absum_generation():
            min_tokens, max_tokens = self._absum_token_limits(input_tokens)
            output = model.generate(
                **encoded,
                min_length=min_tokens,
                max_length=max_tokens,
                num_beams=3,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            return tokenizer.decode(output[0], skip_special_tokens=True).strip()

        dynamic_min_tokens, dynamic_max_tokens = self._generation_token_limits(input_tokens)

        output = model.generate(
            **encoded,
            max_length=dynamic_max_tokens,
            min_length=dynamic_min_tokens,
            num_beams=self.config.num_beams,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _uses_absum_generation(self) -> bool:
        return "rut5-base-absum" in str(self.config.model_name).lower()

    def _absum_token_limits(self, input_tokens: int) -> tuple[int, int]:
        retention = min(max(self.config.min_retention_ratio, 0.1), 0.8)
        min_ratio = retention * 0.82
        max_ratio = min(max(retention * 1.67, 0.55), 0.9)
        configured_max = max(self.config.max_summary_tokens, 16)

        min_tokens = max(self.config.min_summary_tokens, int(input_tokens * min_ratio))
        min_tokens = min(min_tokens, configured_max - 8)
        max_tokens = max(min_tokens + 8, int(input_tokens * max_ratio))
        max_tokens = min(max_tokens, configured_max)
        return min_tokens, max_tokens

    def _generation_token_limits(self, input_tokens: int) -> tuple[int, int]:
        retention = min(max(self.config.min_retention_ratio, 0.1), 0.8)
        min_ratio = retention * 0.85
        # The upper bound is only a safety limit. A wider range lets the model
        # finish its current sentence instead of cutting a valid summary midway.
        max_ratio = min(max(retention * 2.0, 0.65), 0.9)
        configured_max = max(self.config.max_summary_tokens, 16)

        min_tokens = max(self.config.min_summary_tokens, int(input_tokens * min_ratio))
        min_tokens = min(min_tokens, configured_max - 8)
        max_tokens = max(min_tokens + 8, int(input_tokens * max_ratio))
        max_tokens = min(max_tokens, configured_max)
        return min_tokens, max_tokens
