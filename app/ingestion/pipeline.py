"""Ingest a file end to end: load -> chunk -> embed -> index."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from app.config import Settings
from app.ingestion.chunking import chunk_blocks
from app.ingestion.loaders import (
    SUPPORTED_SUFFIXES,
    UnsupportedDocument,
    checksum,
    load_document,
)
from app.retrieval.store import VectorStore
from app.schemas import DocumentCategory, DocumentMeta, Sensitivity

logger = logging.getLogger(__name__)

# Magic bytes, checked in addition to the extension: an attacker renaming a
# payload to `.pdf` should not get it parsed.
MAGIC_BYTES = {".pdf": b"%PDF-", ".docx": b"PK\x03\x04"}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class InvalidUpload(ValueError):
    pass


def safe_filename(filename: str) -> str:
    """Strip any directory component and unusual characters."""
    name = Path(filename).name
    name = _UNSAFE.sub("_", name).lstrip(".")
    if not name:
        raise InvalidUpload("Filename is empty after sanitisation.")
    return name[:120]


def validate_upload(filename: str, content: bytes, *, max_mb: int) -> str:
    name = safe_filename(filename)
    suffix = Path(name).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise InvalidUpload(
            f"Only {', '.join(sorted(SUPPORTED_SUFFIXES))} files are accepted."
        )
    if len(content) > max_mb * 1024 * 1024:
        raise InvalidUpload(f"File exceeds the {max_mb} MB limit.")
    if not content.startswith(MAGIC_BYTES[suffix]):
        raise InvalidUpload(
            f"File content does not look like a real {suffix[1:].upper()} file."
        )
    return name


def ingest_path(
    path: Path,
    store: VectorStore,
    settings: Settings,
    *,
    category: DocumentCategory = DocumentCategory.GENERAL,
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
    doc_id: str | None = None,
) -> DocumentMeta:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(f"{path.name}: unsupported file type")

    blocks = load_document(path)
    if not blocks:
        raise UnsupportedDocument(
            f"{path.name}: no extractable text (a scanned document needs OCR first)"
        )

    doc_id = doc_id or uuid.uuid4().hex[:12]
    chunks = chunk_blocks(
        blocks,
        doc_id=doc_id,
        filename=path.name,
        category=category,
        sensitivity=sensitivity,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    meta = DocumentMeta(
        doc_id=doc_id,
        filename=path.name,
        file_type=path.suffix.lower().lstrip("."),
        category=category,
        sensitivity=sensitivity,
        page_count=max(block.page for block in blocks),
        checksum=checksum(path),
    )
    store.add(meta, chunks)
    logger.info("Ingested %s: %d pages, %d chunks", path.name, meta.page_count, len(chunks))
    return meta
