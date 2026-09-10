from app.retrieval.bm25 import BM25Index
from app.schemas import Role
from app.security import build_user


def test_bm25_ranks_the_matching_document_first():
    index = BM25Index(
        [
            "password rotation every ninety days for privileged accounts",
            "the cafeteria menu changes weekly",
            "incident response requires reporting within one hour",
        ]
    )
    ranked = index.search("privileged password rotation")
    assert ranked and ranked[0][0] == 0


def test_bm25_returns_nothing_for_unmatched_terms():
    assert BM25Index(["alpha beta gamma"]).search("zebra quokka") == []


def test_dense_retrieval_finds_the_right_document(retriever):
    admin = build_user("admin", Role.ADMIN)
    results = retriever.retrieve(
        "risk appetite threshold", user=admin, mode="dense", top_k=3
    )
    assert results[0].chunk.filename == "risk.docx"


def test_hybrid_retrieval_returns_ranked_results(retriever):
    admin = build_user("admin", Role.ADMIN)
    results = retriever.retrieve(
        "password rotation", user=admin, mode="hybrid", top_k=3
    )
    assert results
    assert [item.rank for item in results] == list(range(1, len(results) + 1))
    assert results[0].chunk.filename == "policy.pdf"


def test_all_modes_report_a_comparable_cosine_score(retriever):
    admin = build_user("admin", Role.ADMIN)
    for mode in ("dense", "sparse", "hybrid"):
        results = retriever.retrieve("password rotation", user=admin, mode=mode, top_k=3)
        assert results
        assert all(-1.0 <= item.score <= 1.0 for item in results)
        assert all(item.retriever == mode for item in results)


def test_unknown_mode_is_rejected(retriever):
    import pytest

    with pytest.raises(ValueError):
        retriever.retrieve("anything", user=build_user("a", Role.ADMIN), mode="magic")


def test_empty_index_returns_nothing(store):
    from app.retrieval.retriever import Retriever

    assert Retriever(store).retrieve("anything", user=build_user("a", Role.ADMIN)) == []


def test_stemming_bridges_question_and_document_wording():
    """'password rotation' must match 'passwords must be rotated'."""
    from app.retrieval.bm25 import stem

    assert stem("rotation") == stem("rotated")
    assert stem("passwords") == stem("password")
    assert stem("accounts") == stem("account")
    assert stem("access") == stem("accesses")
    # A trailing "ss" is not a plural.
    assert stem("access") == "access"

    index = BM25Index(["Privileged account passwords must be rotated every 90 days."])
    assert index.search("password rotation")


def test_index_built_with_another_embedder_is_discarded(store, seeded_store):
    """Vectors from a different model live in a different space - reusing them
    would return confident nonsense."""
    from app.retrieval.embeddings import HashingEmbedder
    from app.retrieval.store import VectorStore

    assert len(seeded_store) > 0

    other = HashingEmbedder(dimension=128)
    other.model_name = "some-other-model"
    reopened = VectorStore(seeded_store.index_dir, other)
    assert len(reopened) == 0
