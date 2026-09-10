"""A small persisted vector + keyword store.

Chunks, their embeddings and their document metadata live in `index_dir` as
`chunks.jsonl`, `embeddings.npy` and `documents.json`. Everything is loaded
into memory on startup - fine for the corpus sizes this assistant targets,
and it keeps the project runnable with no external database.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.retrieval.embeddings import Embedder
from app.schemas import Chunk, DocumentMeta

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, index_dir: Path, embedder: Embedder):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder

        self.chunks: list[Chunk] = []
        self.documents: dict[str, DocumentMeta] = {}
        self.embeddings: np.ndarray = np.zeros(
            (0, embedder.dimension), dtype=np.float32
        )

        self.load()

    # --- paths ---------------------------------------------------------
    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def _embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    @property
    def _documents_path(self) -> Path:
        return self.index_dir / "documents.json"

    @property
    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    # --- persistence ---------------------------------------------------
    def load(self) -> None:
        if self._chunks_path.exists():
            with self._chunks_path.open() as fh:
                self.chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        if self._embeddings_path.exists():
            self.embeddings = np.load(self._embeddings_path)
        if self._documents_path.exists():
            raw = json.loads(self._documents_path.read_text())
            self.documents = {
                key: DocumentMeta.model_validate(value) for key, value in raw.items()
            }

        # Vectors from a different embedding model live in a different space,
        # so querying them returns confident nonsense. Rebuild instead.
        if self.chunks and self._indexed_model() not in (None, self.embedder.model_name):
            logger.warning(
                "Index was built with %s but %s is configured - discarding it. "
                "Re-run the ingestion to rebuild.",
                self._indexed_model(),
                self.embedder.model_name,
            )
            self.chunks, self.documents = [], {}
            self.embeddings = np.zeros((0, self.embedder.dimension), dtype=np.float32)
            return

        # A half-written index is worse than an empty one.
        if len(self.chunks) != self.embeddings.shape[0]:
            self.chunks, self.documents = [], {}
            self.embeddings = np.zeros((0, self.embedder.dimension), dtype=np.float32)

    def _indexed_model(self) -> str | None:
        if not self._manifest_path.exists():
            return None
        return json.loads(self._manifest_path.read_text()).get("embedding_model")

    def save(self) -> None:
        with self._chunks_path.open("w") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        np.save(self._embeddings_path, self.embeddings)
        self._manifest_path.write_text(
            json.dumps(
                {
                    "embedding_model": self.embedder.model_name,
                    "dimension": self.embedder.dimension,
                    "chunks": len(self.chunks),
                },
                indent=2,
            )
        )
        serialised = {
            key: json.loads(value.model_dump_json())
            for key, value in self.documents.items()
        }
        self._documents_path.write_text(json.dumps(serialised, indent=2))

    # --- mutation ------------------------------------------------------
    def add(self, meta: DocumentMeta, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        self.remove_document(meta.doc_id, save=False)

        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        self.chunks.extend(chunks)
        self.embeddings = (
            vectors if self.embeddings.size == 0 else np.vstack([self.embeddings, vectors])
        )

        meta.chunk_count = len(chunks)
        self.documents[meta.doc_id] = meta
        self.save()
        return len(chunks)

    def remove_document(self, doc_id: str, *, save: bool = True) -> int:
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = (
                self.embeddings[keep] if keep else
                np.zeros((0, self.embedder.dimension), dtype=np.float32)
            )
        self.documents.pop(doc_id, None)
        if save:
            self.save()
        return removed

    def clear(self) -> None:
        self.chunks = []
        self.documents = {}
        self.embeddings = np.zeros((0, self.embedder.dimension), dtype=np.float32)
        self.save()

    def __len__(self) -> int:
        return len(self.chunks)
