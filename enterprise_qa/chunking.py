"""Configurable character chunking with overlap, preserving page numbers."""

from __future__ import annotations

from enterprise_qa.models import Chunk, DocumentMeta


def chunk_pages(
    pages: list[tuple[int, str]],
    meta: DocumentMeta,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    if chunk_size < 64:
        raise ValueError("chunk_size must be at least 64 characters.")
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[Chunk] = []
    index = 0
    for page, text in pages:
        for piece in _split_text(text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.doc_id}-{index}",
                    doc_id=meta.doc_id,
                    title=meta.title,
                    filename=meta.filename,
                    classification=meta.classification,
                    domain=meta.domain,
                    page=page,
                    chunk_index=index,
                    text=piece,
                )
            )
            index += 1
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            window = text[start:end]
            break_at = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if break_at >= chunk_size // 3:
                end = start + break_at + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return pieces
