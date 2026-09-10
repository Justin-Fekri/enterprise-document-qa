"""Embedding backends.

`SentenceTransformerEmbedder` is the real one. `HashingEmbedder` is a
deterministic, dependency-free stand-in so tests and CI can exercise the whole
pipeline without downloading model weights.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dimension: int
    # Cosine scores are not comparable across embedding models, so each
    # backend carries the floor below which a passage is not evidence.
    relevance_floor: float

    def encode(self, texts: list[str]) -> np.ndarray: ...


# Without this the hashed embedder scores any two English sentences as
# similar purely on "the", "is", "of" - which defeats the relevance floor.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or
    that the this to was what when where which who will with within must may
    our we you your not no all any each every""".split()  # noqa: SIM905
)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model; no data leaves the machine."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        # Calibrated by evaluation/chunking_experiment.py: 0.25 let an
        # off-topic question through, 0.30 refused every unanswerable
        # question without costing any recall.
        self.relevance_floor = 0.30
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """Hashed bag-of-words vectors. Weak, but fast, offline and deterministic."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = f"hashing-{dimension}"
        # Hashed bag-of-words similarity is far more compressed than a
        # trained encoder's, so the same floor would reject everything.
        self.relevance_floor = 0.05

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if token in STOPWORDS:
                    continue
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16)
                matrix[row, bucket % self.dimension] += 1.0
        return _l2_normalize(matrix)


def build_embedder(model_name: str) -> Embedder:
    """Fall back to hashing if sentence-transformers is unavailable."""
    if model_name.startswith("hashing"):
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception:  # pragma: no cover - depends on the local environment
        return HashingEmbedder()
