from enterprise_qa.evaluation import compare_configurations, evaluate_run


def test_evaluate_current_config():
    run = evaluate_run(chunk_size=512, method="hybrid")
    summary = run["summary"]
    assert summary["n_questions"] >= 8
    assert summary["abstention_accuracy"] >= 0.5
    assert summary["context_recall"] >= 0.5
    assert summary["citation_coverage"] >= 0.5


def test_compare_returns_ranked_runs():
    payload = compare_configurations()
    assert payload["best"]
    assert len(payload["runs"]) == 9
    scores = [row["grounding_score"] for row in payload["runs"]]
    assert scores == sorted(scores, reverse=True)
