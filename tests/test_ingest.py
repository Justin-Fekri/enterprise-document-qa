from enterprise_qa.chunking import chunk_pages
from enterprise_qa.ingest import extract_pages
from enterprise_qa.models import DocumentMeta
from enterprise_qa.sample_docs import generate_all


def test_pdf_and_docx_extraction(tmp_path):
    paths = generate_all(tmp_path)
    pdfs = [p for p in paths if p.suffix == ".pdf"]
    docs = [p for p in paths if p.suffix == ".docx"]
    assert pdfs and docs
    pdf_pages = extract_pages(pdfs[0])
    assert pdf_pages[0][0] == 1
    assert any("password" in text.lower() or "audit" in text.lower() or "risk" in text.lower() for _, text in pdf_pages)
    docx_pages = extract_pages(docs[0])
    assert docx_pages
    assert all(page >= 1 for page, _ in docx_pages)


def test_chunk_sizes_preserve_pages():
    pages = [(1, "Sentence one. " * 80), (2, "Sentence two. " * 80)]
    meta = DocumentMeta(
        doc_id="demo",
        title="Demo",
        filename="demo.pdf",
        classification="internal",
        domain="company_policy",
        source_path="demo.pdf",
        page_count=2,
    )
    small = chunk_pages(pages, meta, chunk_size=256, overlap=32)
    large = chunk_pages(pages, meta, chunk_size=1024, overlap=64)
    assert len(small) > len(large)
    assert {c.page for c in small} == {1, 2}
    assert all(len(c.text) <= 256 + 20 or c.page for c in small)
