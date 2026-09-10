"""Pydantic models shared by the ingestion, retrieval and API layers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentCategory(StrEnum):
    """Business document categories handled by the assistant."""

    POLICY = "company_policy"
    AUDIT = "audit_procedure"
    RISK = "risk_report"
    SECURITY = "cybersecurity"
    GENERAL = "general"


class Sensitivity(StrEnum):
    """Access tier of a document. Roles are granted one or more tiers."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class Role(StrEnum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    ADMIN = "admin"


class DocumentMeta(BaseModel):
    """Everything known about an ingested source file."""

    doc_id: str
    filename: str
    file_type: str = Field(description="pdf | docx")
    category: DocumentCategory = DocumentCategory.GENERAL
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    page_count: int = 0
    chunk_count: int = 0
    checksum: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A retrievable passage with enough metadata to cite it."""

    chunk_id: str
    doc_id: str
    filename: str
    category: DocumentCategory
    sensitivity: Sensitivity
    page: int = Field(description="1-indexed page (PDF) or block (DOCX).")
    section: str = ""
    text: str
    token_count: int = 0


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    retriever: str = Field(default="hybrid", description="dense | sparse | hybrid")


class Citation(BaseModel):
    filename: str
    page: int
    section: str = ""
    quote: str = Field(default="", description="Supporting span from the passage.")


class Answer(BaseModel):
    """Final response. `abstained=True` means the corpus lacked evidence."""

    question: str
    answer: str
    abstained: bool = False
    reason: str = ""
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    model: str = ""
    latency_ms: int = 0


class User(BaseModel):
    username: str
    role: Role
    allowed_categories: list[DocumentCategory] = Field(default_factory=list)
    max_sensitivity: Sensitivity = Sensitivity.INTERNAL
