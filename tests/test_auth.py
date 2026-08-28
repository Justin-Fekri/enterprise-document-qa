from enterprise_qa.auth import authenticate, can_access, issue_token, decode_token, visible_classifications


def test_demo_logins():
    assert authenticate("admin", "admin123").role == "admin"
    assert authenticate("employee", "emp123").role == "employee"
    assert authenticate("admin", "wrong") is None


def test_rbac_matrix():
    assert can_access("employee", "internal")
    assert not can_access("employee", "confidential")
    assert not can_access("employee", "restricted")
    assert can_access("auditor", "confidential")
    assert not can_access("auditor", "restricted")
    assert can_access("compliance", "restricted")
    assert "restricted" in visible_classifications("admin")
    assert "restricted" not in visible_classifications("employee")


def test_jwt_roundtrip():
    user = authenticate("auditor", "audit123")
    payload = decode_token(issue_token(user))
    assert payload["sub"] == "auditor"
    assert payload["role"] == "auditor"
