"""End-to-end API tests against a temporary index, fully offline."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import Settings, get_settings
from app.schemas import Role


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(
        document_dir=tmp_path / "documents",
        index_dir=tmp_path / "index",
        embedding_model="hashing",
        anthropic_api_key=None,
        jwt_secret="test-secret",
    )
    settings.ensure_dirs()

    get_settings.cache_clear()
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)
    deps.reset_caches()

    from app.api.main import app

    app.dependency_overrides[get_settings] = lambda: settings

    users = deps.get_user_store()
    for role in (Role.ADMIN, Role.VIEWER):
        users.add(role.value, f"{role.value}-password", role)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    deps.reset_caches()
    get_settings.cache_clear()


def token_for(client: TestClient, username: str) -> str:
    response = client.post(
        "/auth/token", data={"username": username, "password": f"{username}-password"}
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def auth(client: TestClient, username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for(client, username)}"}


def sample_pdf_bytes() -> bytes:
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    import io

    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=LETTER).build(
        [
            Paragraph(
                "Privileged account passwords must be rotated every 90 days and "
                "multi-factor authentication is mandatory for remote access.",
                getSampleStyleSheet()["BodyText"],
            )
        ]
    )
    return buffer.getvalue()


# --- auth --------------------------------------------------------------


def test_health_is_public(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["answerer"] == "extractive-baseline"


def test_query_requires_authentication(client):
    assert client.post("/query", json={"question": "anything at all"}).status_code == 401


def test_bad_credentials_are_rejected(client):
    response = client.post("/auth/token", data={"username": "admin", "password": "nope"})
    assert response.status_code == 401


def test_invalid_token_is_rejected(client):
    response = client.get("/me", headers={"Authorization": "Bearer not-a-token"})
    assert response.status_code == 401


def test_me_returns_effective_permissions(client):
    body = client.get("/me", headers=auth(client, "viewer")).json()
    assert body["role"] == "viewer"
    assert body["max_sensitivity"] == "internal"


# --- uploads -----------------------------------------------------------


def test_upload_requires_admin(client):
    response = client.post(
        "/documents",
        headers=auth(client, "viewer"),
        files={"file": ("policy.pdf", sample_pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 403


def test_upload_rejects_a_disguised_executable(client):
    response = client.post(
        "/documents",
        headers=auth(client, "admin"),
        files={"file": ("payload.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")},
    )
    assert response.status_code == 400
    assert "does not look like" in response.json()["detail"]


def test_upload_rejects_unsupported_extensions(client):
    response = client.post(
        "/documents",
        headers=auth(client, "admin"),
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_index_query_delete_round_trip(client):
    headers = auth(client, "admin")

    created = client.post(
        "/documents",
        headers=headers,
        data={"category": "company_policy", "sensitivity": "internal"},
        files={"file": ("policy.pdf", sample_pdf_bytes(), "application/pdf")},
    )
    assert created.status_code == 201, created.text
    meta = created.json()
    assert meta["chunk_count"] >= 1
    assert meta["file_type"] == "pdf"

    listed = client.get("/documents", headers=headers).json()
    assert [item["filename"] for item in listed] == ["policy.pdf"]

    answer = client.post(
        "/query",
        headers=headers,
        json={"question": "How often must privileged passwords be rotated?"},
    ).json()
    assert not answer["abstained"]
    assert answer["citations"][0]["filename"] == "policy.pdf"
    assert answer["citations"][0]["page"] == 1

    hits = client.post(
        "/search", headers=headers, json={"question": "password rotation"}
    ).json()
    assert hits and hits[0]["rank"] == 1

    assert client.delete(f"/documents/{meta['doc_id']}", headers=headers).status_code == 204
    assert client.get("/documents", headers=headers).json() == []


def test_query_abstains_on_an_empty_index(client):
    answer = client.post(
        "/query", headers=auth(client, "admin"), json={"question": "anything at all"}
    ).json()
    assert answer["abstained"]


def test_path_traversal_filenames_are_sanitised(client):
    created = client.post(
        "/documents",
        headers=auth(client, "admin"),
        files={"file": ("../../etc/policy.pdf", sample_pdf_bytes(), "application/pdf")},
    )
    assert created.status_code == 201
    assert "/" not in created.json()["filename"]


def test_deleting_an_unknown_document_is_a_404(client):
    assert client.delete("/documents/nope", headers=auth(client, "admin")).status_code == 404


def test_question_length_is_validated(client):
    response = client.post("/query", headers=auth(client, "admin"), json={"question": "hi"})
    assert response.status_code == 422
