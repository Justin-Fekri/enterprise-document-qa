"""Application settings, loaded from the environment or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Storage -------------------------------------------------------
    document_dir: Path = Field(default=PROJECT_ROOT / "data" / "documents")
    index_dir: Path = Field(default=PROJECT_ROOT / "data" / "index")

    # --- Chunking ------------------------------------------------------
    chunk_size: int = Field(default=512, description="Target chunk size in tokens.")
    chunk_overlap: int = Field(default=64, description="Overlap in tokens.")

    # --- Retrieval -----------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_mode: str = Field(
        default="hybrid", description="dense | sparse | hybrid"
    )
    top_k: int = Field(default=6, description="Passages passed to the generator.")
    candidate_k: int = Field(default=20, description="Candidates before fusion.")
    rrf_k: int = Field(default=60, description="Reciprocal-rank-fusion constant.")

    # --- Abstention ----------------------------------------------------
    min_relevance_score: float | None = Field(
        default=None,
        description=(
            "Cosine floor: below this, nothing is treated as evidence. "
            "Defaults to the configured embedding model's own calibrated "
            "floor, since cosine scales are not comparable across models."
        ),
    )

    # --- Generation ----------------------------------------------------
    anthropic_api_key: str | None = None
    answer_model: str = "claude-opus-5"
    max_tokens: int = 4000
    effort: str = Field(default="medium", description="low | medium | high | xhigh | max")

    # --- Auth ----------------------------------------------------------
    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 480

    # --- API -----------------------------------------------------------
    api_url: str = "http://localhost:8000"
    max_upload_mb: int = 25

    def ensure_dirs(self) -> None:
        self.document_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
