"""Dense, sparse and hybrid retrieval over the store, with access filtering.

Permissions are applied to the *candidate set*, not to the final answer: a
passage a user may not read never reaches the model, so it can never leak into
an answer or a citation.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.bm25 import BM25Index
from app.retrieval.store import VectorStore
from app.schemas import Chunk, RetrievedChunk, Sensitivity, User

SENSITIVITY_ORDER = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.CONFIDENTIAL: 2,
    Sensitivity.RESTRICTED: 3,
}


def user_can_read(chunk: Chunk, user: User | None) -> bool:
    if user is None:
        return chunk.sensitivity == Sensitivity.PUBLIC
    if SENSITIVITY_ORDER[chunk.sensitivity] > SENSITIVITY_ORDER[user.max_sensitivity]:
        return False
    return not (user.allowed_categories and chunk.category not in user.allowed_categories)


class Retriever:
    def __init__(self, store: VectorStore, *, rrf_k: int = 60):
        self.store = store
        self.rrf_k = rrf_k
        self._bm25: BM25Index | None = None
        self._bm25_size = -1

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None or self._bm25_size != len(self.store.chunks):
            self._bm25 = BM25Index([chunk.text for chunk in self.store.chunks])
            self._bm25_size = len(self.store.chunks)
        return self._bm25

    def invalidate(self) -> None:
        self._bm25 = None
        self._bm25_size = -1

    # --- individual retrievers -----------------------------------------
    def _dense(self, query: str, allowed: list[int], k: int) -> list[tuple[int, float]]:
        if not allowed:
            return []
        query_vector = self.store.embedder.encode([query])[0]
        similarities = self.store.embeddings[allowed] @ query_vector
        order = np.argsort(-similarities)[:k]
        return [(allowed[int(i)], float(similarities[int(i)])) for i in order]

    def _sparse(self, query: str, allowed: list[int], k: int) -> list[tuple[int, float]]:
        allowed_set = set(allowed)
        hits = self._bm25_index().search(query, top_k=k * 4)
        return [(index, score) for index, score in hits if index in allowed_set][:k]

    def _fuse(
        self,
        dense: list[tuple[int, float]],
        sparse: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Reciprocal rank fusion - rank-based, so the two score scales never
        have to be normalized against each other."""
        fused: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, (index, _score) in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

    # --- public API -----------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        user: User | None = None,
        mode: str = "hybrid",
        top_k: int = 6,
        candidate_k: int = 20,
    ) -> list[RetrievedChunk]:
        if not self.store.chunks:
            return []

        allowed = [
            index
            for index, chunk in enumerate(self.store.chunks)
            if user_can_read(chunk, user)
        ]
        if not allowed:
            return []

        if mode == "dense":
            ranked = self._dense(query, allowed, top_k)
        elif mode == "sparse":
            ranked = self._sparse(query, allowed, top_k)
        elif mode == "hybrid":
            ranked = self._fuse(
                self._dense(query, allowed, candidate_k),
                self._sparse(query, allowed, candidate_k),
            )[:top_k]
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        # Always report a comparable relevance number, whatever the mode, so
        # the abstention threshold means the same thing everywhere.
        results: list[RetrievedChunk] = []
        if ranked:
            query_vector = self.store.embedder.encode([query])[0]
            for rank, (index, _fused_score) in enumerate(ranked, start=1):
                cosine = float(self.store.embeddings[index] @ query_vector)
                results.append(
                    RetrievedChunk(
                        chunk=self.store.chunks[index],
                        score=cosine,
                        rank=rank,
                        retriever=mode,
                    )
                )
        return results
