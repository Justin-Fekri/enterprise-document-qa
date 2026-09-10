"""Okapi BM25 over the chunk texts.

Written out rather than pulled in as a dependency: it is ~40 lines, and the
retrieval comparison in `evaluation/` needs to be able to explain exactly what
the sparse retriever is doing.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

# Longest first, so "ations" is stripped before "s". Without this a query for
# "password rotation" never matches a policy that says "passwords must be
# rotated" - the single most common way keyword search fails on these
# documents, where questions and clauses use different word forms.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ational", "at"),
    ("ations", "at"),
    ("ation", "at"),
    ("sses", "ss"),
    ("ings", ""),
    ("ing", ""),
    ("edly", ""),
    ("ed", ""),
    ("ies", "i"),
    ("es", ""),
    ("s", ""),
)

MIN_STEM_LENGTH = 3


def stem(token: str) -> str:
    """A deliberately small suffix stripper - not Porter, just enough to make
    question wording and document wording meet.

    `rotation` and `rotated` both reduce to `rotat`; `access` is left alone
    because a trailing `ss` is not a plural.
    """
    for suffix, replacement in _SUFFIX_RULES:
        if not token.endswith(suffix):
            continue
        if suffix in {"s", "es"} and token.endswith("ss"):
            continue  # "access", "process" - not plurals
        stemmed = token[: -len(suffix)] + replacement
        if len(stemmed) >= MIN_STEM_LENGTH:
            return stemmed
    return token


def tokenize(text: str) -> list[str]:
    return [stem(token) for token in _TOKEN.findall(text.lower())]


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.corpus]
        self.avg_length = (sum(self.doc_lengths) / len(self.corpus)) if self.corpus else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.corpus]

        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())

        total = len(self.corpus)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms or not self.corpus:
            return []

        scores = [0.0] * len(self.corpus)
        for index, frequencies in enumerate(self.term_frequencies):
            length_norm = self.k1 * (
                1 - self.b + self.b * self.doc_lengths[index] / (self.avg_length or 1)
            )
            total = 0.0
            for term in terms:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                total += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / (
                    frequency + length_norm
                )
            scores[index] = total

        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [pair for pair in ranked[:top_k] if pair[1] > 0]
