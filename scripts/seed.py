"""Seed the demo: generate documents, create users, build the index.

Run:  python scripts/seed.py [--reset]

Demo passwords are generated at seed time and printed once. They are written
to `data/index/users.json` as PBKDF2 hashes and never committed.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingestion.pipeline import ingest_path  # noqa: E402
from app.retrieval.embeddings import build_embedder  # noqa: E402
from app.retrieval.store import VectorStore  # noqa: E402
from app.schemas import DocumentCategory, Role, Sensitivity  # noqa: E402
from app.security import UserStore  # noqa: E402

# filename -> (category, sensitivity)
CORPUS_LABELS = {
    "information_security_policy.pdf": (
        DocumentCategory.POLICY,
        Sensitivity.INTERNAL,
    ),
    "access_control_standard.pdf": (
        DocumentCategory.SECURITY,
        Sensitivity.RESTRICTED,
    ),
    "internal_audit_procedure.docx": (
        DocumentCategory.AUDIT,
        Sensitivity.CONFIDENTIAL,
    ),
    "q3_risk_report.docx": (DocumentCategory.RISK, Sensitivity.CONFIDENTIAL),
}

DEMO_USERS = [Role.ADMIN, Role.AUDITOR, Role.ANALYST, Role.VIEWER]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Rebuild the index from scratch.")
    parser.add_argument(
        "--password",
        default=None,
        help="Use one password for every demo user (default: generate per user).",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not any(settings.document_dir.glob("*")):
        print("No documents found - generating the sample corpus.")
        from scripts.generate_sample_docs import main as generate

        generate()

    store = VectorStore(settings.index_dir, build_embedder(settings.embedding_model))
    if args.reset:
        store.clear()

    print(f"\nIndexing from {settings.document_dir}")
    for path in sorted(settings.document_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        category, sensitivity = CORPUS_LABELS.get(
            path.name, (DocumentCategory.GENERAL, Sensitivity.INTERNAL)
        )
        meta = ingest_path(
            path, store, settings, category=category, sensitivity=sensitivity
        )
        print(
            f"  {meta.filename:<38} {meta.page_count:>2} pages  "
            f"{meta.chunk_count:>3} chunks  [{sensitivity.value}]"
        )

    users = UserStore(settings.index_dir / "users.json")
    print("\nDemo accounts (save these - they are shown once):")
    for role in DEMO_USERS:
        password = args.password or secrets.token_urlsafe(12)
        users.add(role.value, password, role)
        print(f"  {role.value:<9} {password}")

    print(f"\nIndex ready: {len(store.documents)} documents, {len(store)} chunks.")
    print("Start the API with `make api` and the UI with `make ui`.")


if __name__ == "__main__":
    main()
