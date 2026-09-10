"""Shared, lazily-built singletons and FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.config import Settings, get_settings
from app.generation.answerer import AnswerGenerator, build_answerer
from app.retrieval.embeddings import build_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.schemas import User
from app.security import InvalidToken, UserStore, can_write, decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


@lru_cache
def get_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(settings.index_dir, build_embedder(settings.embedding_model))


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_store(), rrf_k=get_settings().rrf_k)


@lru_cache
def get_user_store() -> UserStore:
    return UserStore(get_settings().index_dir / "users.json")


@lru_cache
def get_answerer() -> AnswerGenerator:
    return build_answerer(get_settings(), get_store().embedder.relevance_floor)


def reset_caches() -> None:
    """Drop the singletons - used by tests and after re-indexing."""
    for cached in (get_store, get_retriever, get_user_store, get_answerer):
        cached.cache_clear()


def current_user(
    token: str | None = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
) -> User:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(token, settings)
    except InvalidToken as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


def admin_user(user: User = Depends(current_user)) -> User:
    if not can_write(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires the admin role.",
        )
    return user
