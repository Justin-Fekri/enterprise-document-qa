"""Abstention is the feature: the assistant must decline rather than guess."""

from app.generation.answerer import (
    NO_EVIDENCE,
    ClaudeAnswerer,
    ExtractiveAnswerer,
    resolve_citations,
)
from app.retrieval.embeddings import HashingEmbedder
from app.schemas import Role
from app.security import build_user


def answer_for(retriever, settings, question, role=Role.ADMIN):
    user = build_user(role.value, role)
    results = retriever.retrieve(question, user=user, top_k=5)
    return ExtractiveAnswerer(settings, HashingEmbedder().relevance_floor).answer(
        question, results
    )


def test_off_topic_question_abstains(retriever, settings):
    answer = answer_for(retriever, settings, "What is the 2027 revenue forecast?")
    assert answer.abstained
    assert answer.answer == NO_EVIDENCE
    assert answer.reason


def test_on_topic_question_does_not_abstain(retriever, settings):
    answer = answer_for(retriever, settings, "password rotation for privileged accounts")
    assert not answer.abstained
    assert answer.citations
    assert answer.citations[0].filename == "policy.pdf"


def test_answer_carries_a_page_level_citation(retriever, settings):
    answer = answer_for(retriever, settings, "password rotation for privileged accounts")
    citation = answer.citations[0]
    assert citation.page == 1
    assert citation.quote


def test_a_user_without_clearance_abstains(retriever, settings):
    """Retrieval finds nothing readable, so the answer is an abstention -
    not a leak and not a guess."""
    answer = answer_for(
        retriever, settings, "break-glass escrow credentials", role=Role.VIEWER
    )
    assert answer.abstained


def test_empty_retrieval_abstains(settings):
    answer = ExtractiveAnswerer(settings).answer("anything at all", [])
    assert answer.abstained
    assert answer.latency_ms >= 0


def test_relevance_floor_is_enforced(retriever, settings):
    strict = settings.model_copy(update={"min_relevance_score": 0.99})
    user = build_user("admin", Role.ADMIN)
    results = retriever.retrieve("password rotation", user=user, top_k=5)
    assert ExtractiveAnswerer(strict).answer("password rotation", results).abstained


# --- citation validation ----------------------------------------------


def _passages(retriever):
    return retriever.retrieve(
        "password rotation", user=build_user("admin", Role.ADMIN), top_k=3
    )


def test_citations_resolve_to_real_passages(retriever):
    passages = _passages(retriever)
    resolved = resolve_citations([{"passage_id": 1, "quote": "every 90 days"}], passages)
    assert len(resolved) == 1
    assert resolved[0].filename == passages[0].chunk.filename
    assert resolved[0].page == passages[0].chunk.page


def test_hallucinated_passage_numbers_are_dropped(retriever):
    passages = _passages(retriever)
    resolved = resolve_citations(
        [
            {"passage_id": 99, "quote": "invented"},
            {"passage_id": 0, "quote": "also invented"},
            {"passage_id": "abc", "quote": "not a number"},
        ],
        passages,
    )
    assert resolved == []


def test_duplicate_citations_are_collapsed(retriever):
    passages = _passages(retriever)
    resolved = resolve_citations(
        [{"passage_id": 1, "quote": "a"}, {"passage_id": 1, "quote": "b"}], passages
    )
    assert len(resolved) == 1


def test_claude_answerer_abstains_when_the_model_reports_no_evidence(retriever, settings):
    """Exercise the payload path without touching the network."""
    answerer = ClaudeAnswerer.__new__(ClaudeAnswerer)
    answerer.settings = settings
    answerer.relevance_floor = 0.05
    answerer.name = "test-model"

    payload = {
        "sufficient_evidence": False,
        "answer": "",
        "reason": "The passages do not mention a revenue forecast.",
        "citations": [],
    }
    answer = answerer._to_answer("q", payload, _passages(retriever), 0.0)
    assert answer.abstained
    assert "revenue forecast" in answer.reason


def test_claude_answerer_abstains_when_no_citation_resolves(retriever, settings):
    """An answer that cannot be traced to a passage is exactly what this
    system exists to suppress."""
    answerer = ClaudeAnswerer.__new__(ClaudeAnswerer)
    answerer.settings = settings
    answerer.relevance_floor = 0.05
    answerer.name = "test-model"

    payload = {
        "sufficient_evidence": True,
        "answer": "Passwords rotate every 30 days.",
        "reason": "",
        "citations": [{"passage_id": 42, "quote": "made up"}],
    }
    answer = answerer._to_answer("q", payload, _passages(retriever), 0.0)
    assert answer.abstained
    assert "traced back" in answer.reason
