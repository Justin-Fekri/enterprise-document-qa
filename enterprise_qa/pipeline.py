"""End-to-end RAG pipeline: ingest → chunk → retrieve → grounded answer."""

from __future__ import annotations

from pathlib import Path

from enterprise_qa.auth import can_access
from enterprise_qa.chunking import chunk_pages
from enterprise_qa.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL,
    DEFAULT_TOP_K,
    SAMPLE_DOCS_DIR,
    UPLOAD_DIR,
)
from enterprise_qa.corpus import Corpus
from enterprise_qa.generate import generate_answer
from enterprise_qa.ingest import build_meta, extract_pages, infer_domain
from enterprise_qa.models import AnswerResult, DocumentMeta
from enterprise_qa.retrieve import Retriever

# Seed catalog used when the sample library is generated.
SAMPLE_CATALOG: list[dict[str, str]] = [
    {
        "filename": "information_security_policy.pdf",
        "title": "Information Security Policy",
        "classification": "confidential",
        "domain": "cybersecurity",
    },
    {
        "filename": "internal_audit_procedure.pdf",
        "title": "Internal Audit Procedure",
        "classification": "confidential",
        "domain": "audit_procedure",
    },
    {
        "filename": "enterprise_risk_report.pdf",
        "title": "Enterprise Risk Report Q1 2026",
        "classification": "confidential",
        "domain": "risk_report",
    },
    {
        "filename": "employee_handbook.docx",
        "title": "Employee Handbook",
        "classification": "internal",
        "domain": "company_policy",
    },
    {
        "filename": "incident_response_plan.docx",
        "title": "Cybersecurity Incident Response Plan",
        "classification": "restricted",
        "domain": "cybersecurity",
    },
]


class RAGPipeline:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self.chunk_size = chunk_size
        self.corpus = Corpus(chunk_size).load()
        self._retrievers: dict[str, Retriever] = {}

    def ensure_seeded(self) -> None:
        if self.corpus.chunks:
            return
        from enterprise_qa.sample_docs import generate_all

        generate_all()
        self.ingest_directory(SAMPLE_DOCS_DIR)

    def ingest_directory(self, directory: Path) -> list[DocumentMeta]:
        ingested: list[DocumentMeta] = []
        lookup = {item["filename"]: item for item in SAMPLE_CATALOG}
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() not in {".pdf", ".docx"}:
                continue
            spec = lookup.get(path.name, {})
            ingested.append(
                self.ingest_file(
                    path,
                    title=spec.get("title") or path.stem.replace("_", " ").title(),
                    classification=spec.get("classification", "internal"),
                    domain=spec.get("domain") or infer_domain(path.name),
                    uploaded_by="system",
                )
            )
        return ingested

    def ingest_file(
        self,
        path: Path,
        *,
        title: str,
        classification: str,
        domain: str,
        uploaded_by: str,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> DocumentMeta:
        pages = extract_pages(path)
        meta = build_meta(
            path,
            title=title,
            classification=classification,
            domain=domain,
            page_count=len(pages),
            uploaded_by=uploaded_by,
        )
        chunks = chunk_pages(pages, meta, chunk_size=self.chunk_size, overlap=overlap)
        self.corpus.replace_document(meta, chunks)
        self._retrievers.clear()
        return meta

    def save_upload(self, filename: str, data: bytes) -> Path:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe = Path(filename).name
        dest = UPLOAD_DIR / safe
        dest.write_bytes(data)
        return dest

    def ask(
        self,
        question: str,
        role: str,
        *,
        method: str = DEFAULT_RETRIEVAL,
        top_k: int = DEFAULT_TOP_K,
    ) -> AnswerResult:
        retriever = self._retriever_for(role)
        passages = retriever.search(question, method=method, top_k=top_k)
        return generate_answer(
            question,
            passages,
            retrieval_method=method,
            chunk_size=self.chunk_size,
        )

    def documents_for(self, role: str) -> list[DocumentMeta]:
        return sorted(self.corpus.documents_for_role(role), key=lambda d: d.title)

    def stats(self, role: str) -> dict:
        docs = self.documents_for(role)
        chunks = self.corpus.chunks_for_role(role)
        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "pages": sum(doc.page_count for doc in docs),
        }

    def _retriever_for(self, role: str) -> Retriever:
        if role not in self._retrievers:
            chunks = [
                chunk
                for chunk in self.corpus.chunks
                if can_access(role, chunk.classification)
            ]
            self._retrievers[role] = Retriever(chunks)
        return self._retrievers[role]


_pipelines: dict[int, RAGPipeline] = {}


def get_pipeline(chunk_size: int = DEFAULT_CHUNK_SIZE) -> RAGPipeline:
    if chunk_size not in _pipelines:
        pipe = RAGPipeline(chunk_size)
        pipe.ensure_seeded()
        _pipelines[chunk_size] = pipe
    return _pipelines[chunk_size]
