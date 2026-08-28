from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from enterprise_qa.pipeline import get_pipeline
from enterprise_qa.sample_docs import generate_all


@pytest.fixture(scope="session", autouse=True)
def seed_library() -> None:
    generate_all()
    for size in (256, 512, 1024):
        get_pipeline(size)


@pytest.fixture()
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


@pytest.fixture()
def admin_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def employee_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/auth/login", json={"username": "employee", "password": "emp123"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
