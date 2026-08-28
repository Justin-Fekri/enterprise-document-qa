"""On-disk corpus of documents and chunks for a chosen chunk size."""

from __future__ import annotations

import json
from pathlib import Path

from enterprise_qa.config import CORPUS_DIR, DEFAULT_CHUNK_SIZE
from enterprise_qa.models import Chunk, DocumentMeta


def corpus_path(chunk_size: int) -> Path:
    return CORPUS_DIR / f"chunks_{chunk_size}.json"


class Corpus:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self.chunk_size = chunk_size
        self.documents: dict[str, DocumentMeta] = {}
        self.chunks: list[Chunk] = []

    def load(self) -> "Corpus":
        path = corpus_path(self.chunk_size)
        if not path.exists():
            return self
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.documents = {
            item["doc_id"]: DocumentMeta(**item) for item in payload.get("documents", [])
        }
        self.chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        return self

    def save(self) -> None:
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_size": self.chunk_size,
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        corpus_path(self.chunk_size).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def replace_document(self, meta: DocumentMeta, chunks: list[Chunk]) -> None:
        self.chunks = [c for c in self.chunks if c.doc_id != meta.doc_id]
        self.documents[meta.doc_id] = meta
        self.chunks.extend(chunks)
        self.save()

    def documents_for_role(self, role: str) -> list[DocumentMeta]:
        from enterprise_qa.auth import can_access

        return [doc for doc in self.documents.values() if can_access(role, doc.classification)]

    def chunks_for_role(self, role: str) -> list[Chunk]:
        from enterprise_qa.auth import can_access

        return [chunk for chunk in self.chunks if can_access(role, chunk.classification)]
