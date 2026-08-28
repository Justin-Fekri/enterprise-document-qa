from enterprise_qa.pipeline import get_pipeline


def test_password_policy_is_grounded_for_employee():
    result = get_pipeline(512).ask("How often must employees rotate their passwords?", "employee")
    assert not result.abstained
    assert result.citations
    assert any(c.page >= 1 for c in result.citations)
    assert "90" in result.answer


def test_employee_cannot_see_restricted_incident_plan():
    result = get_pipeline(512).ask("What is the P1 incident response time?", "employee")
    assert result.abstained


def test_compliance_can_see_incident_plan():
    result = get_pipeline(512).ask("What is the P1 incident response time?", "compliance")
    assert not result.abstained
    assert "15" in result.answer
    assert result.citations


def test_abstain_on_unknown_topic():
    result = get_pipeline(512).ask("What is the CEO salary for 2026?", "admin")
    assert result.abstained


def test_hybrid_retrieves_audit_sla():
    result = get_pipeline(512).ask(
        "What is the SLA for remediating critical audit findings?",
        "auditor",
        method="hybrid",
    )
    assert not result.abstained
    assert "15" in result.answer
