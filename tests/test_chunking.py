import pytest

from app.ingestion.chunking import (
    MIN_CHUNK_TOKENS,
    chunk_blocks,
    estimate_tokens,
    split_sentences,
)
from app.ingestion.loaders import PageBlock
from app.schemas import DocumentCategory, Sensitivity


def make(blocks, **kwargs):
    return chunk_blocks(
        blocks,
        doc_id="doc1",
        filename="test.pdf",
        category=DocumentCategory.POLICY,
        sensitivity=Sensitivity.INTERNAL,
        **kwargs,
    )


def test_chunks_respect_the_size_budget(blocks):
    chunks = make(blocks, chunk_size=100, chunk_overlap=20)
    assert chunks
    # A single oversized sentence can exceed the budget; nothing else may.
    for chunk in chunks:
        assert chunk.token_count <= 100 + estimate_tokens(
            max(split_sentences(chunk.text), key=len)
        )


def test_chunks_never_span_pages(blocks):
    """A citation that points at two pages at once is not checkable."""
    chunks = make(blocks, chunk_size=100, chunk_overlap=20)
    pages = {chunk.page for chunk in chunks}
    assert pages == {1, 2}
    for chunk in chunks:
        source = next(block for block in blocks if block.page == chunk.page)
        first_sentence = split_sentences(chunk.text)[0]
        assert first_sentence in " ".join(split_sentences(source.text))


def test_overlap_carries_context_forward(blocks):
    with_overlap = make(blocks[:1], chunk_size=80, chunk_overlap=40)
    without_overlap = make(blocks[:1], chunk_size=80, chunk_overlap=0)
    assert len(with_overlap) >= len(without_overlap)


def test_smaller_chunks_produce_more_of_them(blocks):
    assert len(make(blocks, chunk_size=60, chunk_overlap=0)) > len(
        make(blocks, chunk_size=400, chunk_overlap=0)
    )


def test_short_page_furniture_is_dropped():
    chunks = make([PageBlock(page=1, section="", text="Page 3")])
    assert chunks == []


def test_short_tail_is_merged_not_lost():
    text = " ".join(f"Requirement {n} applies to all systems." for n in range(30))
    chunks = make([PageBlock(page=1, section="", text=text + " Final note.")], chunk_size=60, chunk_overlap=0)
    assert "Final note." in " ".join(chunk.text for chunk in chunks)
    assert all(chunk.token_count >= MIN_CHUNK_TOKENS for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size(blocks):
    with pytest.raises(ValueError):
        make(blocks, chunk_size=100, chunk_overlap=100)


def test_chunk_ids_are_unique(blocks):
    chunks = make(blocks, chunk_size=60, chunk_overlap=10)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
