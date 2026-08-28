"""BM25, TF-IDF, and hybrid (RRF) retrieval over role-filtered chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from enterprise_qa.models import Chunk, RetrievedPassage

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """
    a an the and or of to for in on at by from with without into over after
    before about as is are was were be been being this that these those it its
    we you they i he she them our your their what which who whom how when where
    why do does did done can could should would will may might must not no nor
    if then than so such also just only more most other into per
    """.split()
)
_WEAK = frozenset(
    """
    time times page pages document documents company information please tell
    explain describe northstar holdings year years quarter current
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN.findall(text.lower()) if tok not in _STOP and len(tok) > 1]


def content_terms(text: str) -> set[str]:
    """Distinctive terms used for grounding / abstention (drops years and weak words)."""
    terms: set[str] = set()
    for tok in tokenize(text):
        if tok in _WEAK or (tok.isdigit() and len(tok) == 4):
            continue
        terms.add(tok)
    return terms


def _stems(tok: str) -> set[str]:
    out = {tok}
    if len(tok) <= 4:
        return out
    if tok.endswith("ies"):
        out.add(tok[:-3] + "y")
    if tok.endswith("es"):
        out.add(tok[:-2])
    if tok.endswith("s"):
        out.add(tok[:-1])
    if tok.endswith("ed"):
        out.add(tok[:-2])
        out.add(tok[:-1])
    if tok.endswith("ing"):
        out.add(tok[:-3])
    return out


def _term_in_doc(term: str, doc_terms: set[str]) -> bool:
    variants = _stems(term)
    expanded = set()
    for doc_term in doc_terms:
        expanded.update(_stems(doc_term))
    if variants & expanded:
        return True
    for doc_term in doc_terms:
        if min(len(doc_term), len(term)) >= 4 and (doc_term.startswith(term) or term.startswith(doc_term)):
            return True
    return False


def matched_content(question: str, text: str) -> set[str]:
    doc_terms = content_terms(text)
    return {term for term in content_terms(question) if _term_in_doc(term, doc_terms)}


class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._tokens = [tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokens) if chunks else None
        self._tfidf = None
        self._matrix = None
        if chunks:
            self._tfidf = TfidfVectorizer(
                tokenizer=tokenize,
                token_pattern=None,
                min_df=1,
                ngram_range=(1, 2),
            )
            self._matrix = self._tfidf.fit_transform(chunk.text for chunk in chunks)

    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        if not self.chunks:
            return []
        method = method.lower()
        if method == "bm25":
            scores = self._bm25_scores(query)
        elif method == "tfidf":
            scores = self._tfidf_scores(query)
        elif method == "hybrid":
            scores = self._hybrid_scores(query)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        ranked = np.argsort(scores)[::-1]
        results: list[RetrievedPassage] = []
        for rank, idx in enumerate(ranked[:top_k], start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(RetrievedPassage(chunk=self.chunks[idx], score=score, rank=rank))
        return results

    def _bm25_scores(self, query: str) -> np.ndarray:
        raw = np.array(self._bm25.get_scores(tokenize(query)), dtype=float)
        return _minmax(raw)

    def _tfidf_scores(self, query: str) -> np.ndarray:
        vector = self._tfidf.transform([query])
        raw = (self._matrix @ vector.T).toarray().ravel()
        return _minmax(raw)

    def _hybrid_scores(self, query: str, k: int = 60) -> np.ndarray:
        """Reciprocal rank fusion of BM25 and TF-IDF."""
        bm25 = self._bm25_scores(query)
        tfidf = self._tfidf_scores(query)
        fused = np.zeros(len(self.chunks), dtype=float)
        for scores in (bm25, tfidf):
            order = np.argsort(scores)[::-1]
            for rank, idx in enumerate(order, start=1):
                fused[idx] += 1.0 / (k + rank)
        return _minmax(fused)


def _minmax(values: np.ndarray) -> np.ndarray:
    peak = float(values.max()) if len(values) else 0.0
    if peak <= 0:
        return np.zeros_like(values, dtype=float)
    floor = float(values.min())
    if math.isclose(peak, floor):
        return np.ones_like(values, dtype=float) if peak > 0 else np.zeros_like(values)
    return (values - floor) / (peak - floor)


def question_coverage(question: str, passages: list[RetrievedPassage], *, top_n: int = 3) -> float:
    q_terms = content_terms(question)
    if not q_terms:
        return 0.0
    joined = " ".join(p.chunk.text for p in passages[:top_n])
    hits = matched_content(question, joined)
    return len(hits) / len(q_terms)


def lexical_overlap(a: str, b: str) -> float:
    left, right = Counter(tokenize(a)), Counter(tokenize(b))
    if not left or not right:
        return 0.0
    shared = sum((left & right).values())
    return shared / max(1, sum(left.values()))
