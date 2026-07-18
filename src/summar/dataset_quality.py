from __future__ import annotations

import re
from collections import Counter


WORD_PATTERN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def max_shared_ngram_frequency(texts: list[str], n: int = 8) -> int:
    frequencies: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        words = WORD_PATTERN.findall(text.lower())
        document_ngrams = {
            tuple(words[index : index + n])
            for index in range(len(words) - n + 1)
        }
        frequencies.update(document_ngrams)
    return max(frequencies.values(), default=0)
