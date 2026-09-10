"""Split page blocks into overlapping, token-bounded passages.

Chunk size is the single biggest lever on retrieval quality here, so it stays
configurable and is what `evaluation/chunking_experiment.py` sweeps over.
"""

from __future__ import annotations

import hashlib
import re

from app.ingestion.loaders import PageBlock
from app.schemas import Chunk, DocumentCategory, Sensitivity

# Roughly 4 characters per token for English prose. Good enough for sizing
# chunks; exact token counts are not needed and would cost a model call.
CHARS_PER_TOKEN = 4

# Chunks shorter than this are page furniture - titles, footers, a stray line
# of boilerplate. They cannot support an answer and only add retrieval noise.
MIN_CHUNK_TOKENS = 20

_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+|\n{2,}")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_END.split(text)]
    return [part for part in parts if part]


def _chunk_ids(doc_id: str, page: int, index: int) -> str:
    raw = f"{doc_id}:{page}:{index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def chunk_blocks(
    blocks: list[PageBlock],
    *,
    doc_id: str,
    filename: str,
    category: DocumentCategory = DocumentCategory.GENERAL,
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Pack sentences into chunks of ~`chunk_size` tokens with overlap.

    Chunks never span pages: a citation that points at two pages at once is
    not a citation a reviewer can check.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []

    for block in blocks:
        sentences = split_sentences(block.text)
        if not sentences:
            continue

        window: list[str] = []
        window_tokens = 0
        index = 0

        def emit(sentences_in_window: list[str], block: PageBlock = block) -> None:
            nonlocal index
            text = " ".join(sentences_in_window).strip()
            if not text:
                return

            # A stub shorter than MIN_CHUNK_TOKENS is usually page furniture,
            # but it may also be a genuine trailing sentence. Merge it into
            # the previous chunk on the same page rather than lose it; drop it
            # only when the whole page is that short.
            if estimate_tokens(text) < MIN_CHUNK_TOKENS:
                if index and chunks:
                    previous = chunks[-1]
                    merged = f"{previous.text} {text}"
                    chunks[-1] = previous.model_copy(
                        update={"text": merged, "token_count": estimate_tokens(merged)}
                    )
                return

            chunks.append(
                Chunk(
                    chunk_id=_chunk_ids(doc_id, block.page, index),
                    doc_id=doc_id,
                    filename=filename,
                    category=category,
                    sensitivity=sensitivity,
                    page=block.page,
                    section=block.section,
                    text=text,
                    token_count=estimate_tokens(text),
                )
            )
            index += 1

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)

            if window and window_tokens + sentence_tokens > chunk_size:
                emit(window)
                # Carry the tail of the window forward so a fact split across
                # the boundary is still retrievable from one chunk.
                carry: list[str] = []
                carry_tokens = 0
                for previous in reversed(window):
                    previous_tokens = estimate_tokens(previous)
                    if carry_tokens + previous_tokens > chunk_overlap:
                        break
                    carry.insert(0, previous)
                    carry_tokens += previous_tokens
                window, window_tokens = carry, carry_tokens

            window.append(sentence)
            window_tokens += sentence_tokens

        emit(window)

    return chunks
