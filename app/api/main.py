"""FastAPI service: authentication, document management, and grounded Q&A."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app import __version__
from app.api.deps import (
    admin_user,
    current_user,
    get_answerer,
    get_retriever,
    get_store,
    get_user_store,
)
from app.config import Settings, get_settings
from app.generation.answerer import GenerationError
from app.ingestion.loaders import UnsupportedDocument
from app.ingestion.pipeline import InvalidUpload, ingest_path, validate_upload
from app.retrieval.retriever import user_can_read
from app.schemas import (
    Answer,
    DocumentCategory,
    DocumentMeta,
    Sensitivity,
    User,
)
from app.security import create_access_token

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("app.api")

app = FastAPI(
    title="Enterprise Document QA Assistant",
    version=__version__,
    description=(
        "Retrieval-augmented question answering over PDF and Word business "
        "documents, with page-level citations, evidence-based abstention and "
        "role-based access control."
    ),
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    # Deliberately not "*": the UI is the only browser client.
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# --- request/response models ------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: str | None = Field(default=None, description="dense | sparse | hybrid")
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchHit(BaseModel):
    filename: str
    page: int
    section: str
    score: float
    rank: int
    excerpt: str


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    answerer: str


# --- endpoints ---------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    store = get_store()
    return HealthResponse(
        status="ok",
        version=__version__,
        documents=len(store.documents),
        chunks=len(store),
        answerer=get_answerer().name,
    )


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    user = get_user_store().authenticate(form.username, form.password)
    if user is None:
        logger.warning("Failed login for %r", form.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(user, settings)
    logger.info("Login: %s (%s)", user.username, user.role.value)
    return TokenResponse(access_token=token, expires_in=expires_in, role=user.role.value)


@app.get("/me", response_model=User, tags=["auth"])
def me(user: User = Depends(current_user)) -> User:
    return user


@app.get("/documents", response_model=list[DocumentMeta], tags=["documents"])
def list_documents(user: User = Depends(current_user)) -> list[DocumentMeta]:
    """Only documents the caller is cleared to read."""
    store = get_store()
    visible = []
    for meta in store.documents.values():
        probe = next(
            (chunk for chunk in store.chunks if chunk.doc_id == meta.doc_id), None
        )
        if probe is not None and user_can_read(probe, user):
            visible.append(meta)
    return sorted(visible, key=lambda meta: meta.filename)


@app.post(
    "/documents",
    response_model=DocumentMeta,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    category: DocumentCategory = Form(DocumentCategory.GENERAL),
    sensitivity: Sensitivity = Form(Sensitivity.INTERNAL),
    user: User = Depends(admin_user),
    settings: Settings = Depends(get_settings),
) -> DocumentMeta:
    content = await file.read()
    try:
        filename = validate_upload(
            file.filename or "", content, max_mb=settings.max_upload_mb
        )
    except InvalidUpload as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error

    destination = settings.document_dir / filename
    destination.write_bytes(content)

    try:
        meta = ingest_path(
            destination,
            get_store(),
            settings,
            category=category,
            sensitivity=sensitivity,
            doc_id=uuid.uuid4().hex[:12],
        )
    except UnsupportedDocument as error:
        destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    get_retriever().invalidate()
    logger.info("%s uploaded %s (%s)", user.username, filename, sensitivity.value)
    return meta


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def delete_document(
    doc_id: str,
    user: User = Depends(admin_user),
    settings: Settings = Depends(get_settings),
) -> None:
    store = get_store()
    meta = store.documents.get(doc_id)
    if meta is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown document id")

    store.remove_document(doc_id)
    get_retriever().invalidate()
    (settings.document_dir / Path(meta.filename).name).unlink(missing_ok=True)
    logger.info("%s deleted %s", user.username, meta.filename)


@app.post("/search", response_model=list[SearchHit], tags=["qa"])
def search(
    request: QueryRequest,
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> list[SearchHit]:
    """Retrieval only - useful for debugging grounding and for evaluation."""
    results = get_retriever().retrieve(
        request.question,
        user=user,
        mode=request.mode or settings.retrieval_mode,
        top_k=request.top_k or settings.top_k,
        candidate_k=settings.candidate_k,
    )
    return [
        SearchHit(
            filename=item.chunk.filename,
            page=item.chunk.page,
            section=item.chunk.section,
            score=round(item.score, 4),
            rank=item.rank,
            excerpt=item.chunk.text[:300],
        )
        for item in results
    ]


@app.post("/query", response_model=Answer, tags=["qa"])
def query(
    request: QueryRequest,
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> Answer:
    """Retrieve, then answer with citations - or abstain."""
    results = get_retriever().retrieve(
        request.question,
        user=user,
        mode=request.mode or settings.retrieval_mode,
        top_k=request.top_k or settings.top_k,
        candidate_k=settings.candidate_k,
    )
    try:
        answer = get_answerer().answer(request.question, results)
    except GenerationError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error

    logger.info(
        "query user=%s abstained=%s citations=%d latency=%dms",
        user.username,
        answer.abstained,
        len(answer.citations),
        answer.latency_ms,
    )
    return answer
