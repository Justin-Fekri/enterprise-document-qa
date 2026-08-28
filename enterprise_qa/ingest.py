"""PDF and Word ingestion with page-level text."""

from __future__ import annotations

import hashlib
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from enterprise_qa.models import DocumentMeta


SUPPORTED_SUFFIXES = {".pdf", ".docx"}

DOMAIN_HINTS = {
    "policy": "company_policy",
    "handbook": "company_policy",
    "audit": "audit_procedure",
    "risk": "risk_report",
    "cyber": "cybersecurity",
    "incident": "cybersecurity",
    "security": "cybersecurity",
}


def infer_domain(filename: str, fallback: str = "company_policy") -> str:
    lower = filename.lower()
    for needle, domain in DOMAIN_HINTS.items():
        if needle in lower:
            return domain
    return fallback


def document_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}-{digest}"


def extract_pages(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}. Upload a PDF or .docx file.")


def _extract_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((index, _normalize(text)))
    if not pages:
        raise ValueError(f"No extractable text in {path.name}.")
    return pages


def _extract_docx(path: Path) -> list[tuple[int, str]]:
    """Approximate pages by grouping paragraphs (~450 words ≈ 1 page)."""
    document = DocxDocument(str(path))
    paragraphs = [_normalize(p.text) for p in document.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError(f"No extractable text in {path.name}.")

    pages: list[tuple[int, str]] = []
    buffer: list[str] = []
    words = 0
    page_no = 1
    for para in paragraphs:
        buffer.append(para)
        words += len(para.split())
        if words >= 450:
            pages.append((page_no, "\n\n".join(buffer)))
            buffer, words, page_no = [], 0, page_no + 1
    if buffer:
        pages.append((page_no, "\n\n".join(buffer)))
    return pages


def _normalize(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def build_meta(
    path: Path,
    *,
    title: str,
    classification: str,
    domain: str,
    page_count: int,
    uploaded_by: str,
) -> DocumentMeta:
    return DocumentMeta(
        doc_id=document_id(path),
        title=title,
        filename=path.name,
        classification=classification,
        domain=domain,
        source_path=str(path),
        page_count=page_count,
        uploaded_by=uploaded_by,
    )
