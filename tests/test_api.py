from fastapi.testclient import TestClient


def test_health(client: TestClient):
    assert client.get("/health").json()["status"] == "ok"


def test_login_and_ask(client: TestClient, admin_headers: dict[str, str]):
    bad = client.post("/auth/login", json={"username": "admin", "password": "nope"})
    assert bad.status_code == 401
    asked = client.post(
        "/ask",
        headers=admin_headers,
        json={"question": "What is the minimum password length?", "retrieval_method": "hybrid"},
    )
    assert asked.status_code == 200
    body = asked.json()
    assert body["abstained"] is False
    assert body["citations"]
    assert body["citations"][0]["page"] >= 1


def test_employee_library_hides_confidential(client: TestClient, employee_headers: dict[str, str], admin_headers: dict[str, str]):
    employee_docs = client.get("/documents", headers=employee_headers).json()["documents"]
    admin_docs = client.get("/documents", headers=admin_headers).json()["documents"]
    emp_files = {d["filename"] for d in employee_docs}
    admin_files = {d["filename"] for d in admin_docs}
    assert "employee_handbook.docx" in emp_files
    assert "incident_response_plan.docx" not in emp_files
    assert "enterprise_risk_report.pdf" not in emp_files
    assert "incident_response_plan.docx" in admin_files
    assert len(admin_files) > len(emp_files)


def test_employee_cannot_evaluate_or_upload(client: TestClient, employee_headers: dict[str, str]):
    assert client.post("/evaluate", headers=employee_headers).status_code == 403
    assert client.post(
        "/documents/upload",
        headers=employee_headers,
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"title": "x", "classification": "internal", "domain": "company_policy"},
    ).status_code == 403


def test_ask_requires_auth(client: TestClient):
    response = client.post("/ask", json={"question": "What is a password?"})
    assert response.status_code == 401
