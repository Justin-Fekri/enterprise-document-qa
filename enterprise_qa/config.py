"""Runtime configuration. Secrets are optional; the app runs fully locally."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
UPLOAD_DIR = DATA_DIR / "uploads"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DIR = DATA_DIR / "eval"

JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 12 * 60

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8765")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.18"))
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL = "hybrid"

CHUNK_SIZE_OPTIONS = (256, 512, 1024)
RETRIEVAL_METHODS = ("bm25", "tfidf", "hybrid")

for path in (DATA_DIR, SAMPLE_DOCS_DIR, UPLOAD_DIR, CORPUS_DIR, EVAL_DIR):
    path.mkdir(parents=True, exist_ok=True)
