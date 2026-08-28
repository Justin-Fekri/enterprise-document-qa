"""Shared data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentMeta:
    doc_id: str
    title: str
    filename: str
    classification: str
    domain: str
    source_path: str
    page_count: int
    uploaded_by: str = "system"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    filename: str
    classification: str
    domain: str
    page: int
    chunk_index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedPassage:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "filename": self.chunk.filename,
            "classification": self.chunk.classification,
            "domain": self.chunk.domain,
            "page": self.chunk.page,
            "score": round(self.score, 4),
            "rank": self.rank,
            "text": self.chunk.text,
        }


@dataclass
class Citation:
    title: str
    filename: str
    page: int
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResult:
    question: str
    answer: str
    abstained: bool
    confidence: float
    retrieval_method: str
    chunk_size: int
    generator: str
    citations: list[Citation] = field(default_factory=list)
    passages: list[RetrievedPassage] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "confidence": round(self.confidence, 4),
            "retrieval_method": self.retrieval_method,
            "chunk_size": self.chunk_size,
            "generator": self.generator,
            "reason": self.reason,
            "citations": [c.to_dict() for c in self.citations],
            "passages": [p.to_dict() for p in self.passages],
        }
