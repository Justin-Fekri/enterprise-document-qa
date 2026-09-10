"""Shared fixtures. Everything here runs offline: no API key, no model download."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force the offline embedder before any application module reads settings.
os.environ["EMBEDDING_MODEL"] = "hashing"
os.environ.pop("ANTHROPIC_API_KEY", None)

from app.config import Settings  # noqa: E402
from app.ingestion.loaders import PageBlock  # noqa: E402
from app.retrieval.embeddings import HashingEmbedder  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.retrieval.store import VectorStore  # noqa: E402
from app.schemas import Chunk, DocumentCategory, DocumentMeta, Sensitivity  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        document_dir=tmp_path / "documents",
        index_dir=tmp_path / "index",
        embedding_model="hashing",
        anthropic_api_key=None,
        jwt_secret="test-secret",
        chunk_size=200,
        chunk_overlap=40,
        # Leave the floor unset: it comes from the embedder, whose cosine
        # scale differs from a trained encoder's.
        min_relevance_score=None,
    )


@pytest.fixture
def store(settings: Settings) -> VectorStore:
    settings.ensure_dirs()
    return VectorStore(settings.index_dir, HashingEmbedder())


CORPUS = [
    (
        "policy.pdf",
        DocumentCategory.POLICY,
        Sensitivity.INTERNAL,
        1,
        "Privileged and administrative account passwords must be rotated every "
        "90 days. Standard user passwords must be rotated every 180 days.",
    ),
    (
        "risk.docx",
        DocumentCategory.RISK,
        Sensitivity.CONFIDENTIAL,
        3,
        "The Board has set an aggregate residual risk appetite threshold of 60 "
        "on the 100-point enterprise scale.",
    ),
    (
        "access.pdf",
        DocumentCategory.SECURITY,
        Sensitivity.RESTRICTED,
        2,
        "Break-glass credentials are held in sealed escrow and any use triggers "
        "an immediate page to the on-call security engineer.",
    ),
]


@pytest.fixture
def seeded_store(store: VectorStore) -> VectorStore:
    for index, (filename, category, sensitivity, page, text) in enumerate(CORPUS):
        doc_id = f"doc{index}"
        chunk = Chunk(
            chunk_id=f"chunk{index}",
            doc_id=doc_id,
            filename=filename,
            category=category,
            sensitivity=sensitivity,
            page=page,
            section="Test section",
            text=text,
            token_count=len(text) // 4,
        )
        store.add(
            DocumentMeta(
                doc_id=doc_id,
                filename=filename,
                file_type=filename.split(".")[-1],
                category=category,
                sensitivity=sensitivity,
                page_count=page,
            ),
            [chunk],
        )
    return store


@pytest.fixture
def retriever(seeded_store: VectorStore) -> Retriever:
    return Retriever(seeded_store)


@pytest.fixture
def blocks() -> list[PageBlock]:
    return [
        PageBlock(
            page=1,
            section="Access Control",
            text=" ".join(
                f"Control statement number {n} requires quarterly review." for n in range(40)
            ),
        ),
        PageBlock(page=2, section="Incident Response", text="Report incidents within one hour. " * 20),
    ]
