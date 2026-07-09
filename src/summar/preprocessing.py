from __future__ import annotations

import re
from typing import Iterable


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    cleaned = WHITESPACE_PATTERN.sub(" ", text).strip()
    return cleaned


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(normalized) if part.strip()]


def chunk_sentences(
    sentences: Iterable[str],
    chunk_size: int = 8,
    overlap: int = 1,
) -> list[str]:
    items = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not items:
        return []
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(items), step):
        chunk = items[start : start + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(items):
            break
    return chunks


def prepare_text_chunks(text: str, chunk_size: int = 8, overlap: int = 1) -> list[str]:
    return chunk_sentences(split_sentences(text), chunk_size=chunk_size, overlap=overlap)
