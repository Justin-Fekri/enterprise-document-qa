"""Grounded evaluation: local lexical metrics plus optional RAGAS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enterprise_qa.config import (
    CHUNK_SIZE_OPTIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL,
    EVAL_DIR,
    OPENAI_API_KEY,
    RETRIEVAL_METHODS,
)
from enterprise_qa.models import AnswerResult
from enterprise_qa.pipeline import RAGPipeline, get_pipeline
from enterprise_qa.retrieve import lexical_overlap, tokenize

GOLD_PATH = EVAL_DIR / "gold_qa.json"


def load_gold() -> list[dict[str, Any]]:
    if not GOLD_PATH.exists():
        GOLD_PATH.write_text(json.dumps(default_gold(), indent=2), encoding="utf-8")
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def default_gold() -> list[dict[str, Any]]:
    return [
        {
            "id": "q1",
            "question": "How often must employees rotate their passwords?",
            "role": "employee",
            "ground_truth": "Passwords must be rotated every 90 days.",
            "should_answer": True,
        },
        {
            "id": "q2",
            "question": "What is the minimum password length?",
            "role": "employee",
            "ground_truth": "The minimum password length is 14 characters.",
            "should_answer": True,
        },
        {
            "id": "q3",
            "question": "How frequently are internal audits conducted?",
            "role": "auditor",
            "ground_truth": "Internal audits are conducted quarterly.",
            "should_answer": True,
        },
        {
            "id": "q4",
            "question": "What is the residual risk rating for ransomware?",
            "role": "auditor",
            "ground_truth": "Ransomware residual risk is rated High.",
            "should_answer": True,
        },
        {
            "id": "q5",
            "question": "What is the P1 incident response time?",
            "role": "compliance",
            "ground_truth": "P1 incidents require a 15-minute response.",
            "should_answer": True,
        },
        {
            "id": "q6",
            "question": "Who is the data protection officer?",
            "role": "compliance",
            "ground_truth": "The Data Protection Officer is Morgan Ellis.",
            "should_answer": True,
        },
        {
            "id": "q7",
            "question": "What is the CEO salary for 2026?",
            "role": "employee",
            "ground_truth": "",
            "should_answer": False,
        },
        {
            "id": "q8",
            "question": "Where is the Tokyo branch office located?",
            "role": "employee",
            "ground_truth": "",
            "should_answer": False,
        },
        {
            "id": "q9",
            "question": "What is the SLA for remediating critical audit findings?",
            "role": "auditor",
            "ground_truth": "Critical findings must be remediated within 15 calendar days.",
            "should_answer": True,
        },
        {
            "id": "q10",
            "question": "May employees share Restricted documents over personal email?",
            "role": "employee",
            "ground_truth": "Restricted and Confidential data must not be sent through personal email.",
            "should_answer": True,
        },
        {
            "id": "q11",
            "question": "What is the P1 incident response time?",
            "role": "employee",
            "ground_truth": "",
            "should_answer": False,
        },
    ]


def evaluate_run(
    items: list[dict[str, Any]] | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    method: str = DEFAULT_RETRIEVAL,
    pipeline: RAGPipeline | None = None,
) -> dict[str, Any]:
    items = items or load_gold()
    pipeline = pipeline or get_pipeline(chunk_size)
    rows: list[dict[str, Any]] = []

    for item in items:
        result = pipeline.ask(item["question"], item["role"], method=method)
        rows.append(_score_item(item, result))

    answerable = [row for row in rows if row["should_answer"]]
    unanswerable = [row for row in rows if not row["should_answer"]]
    answered = [row for row in answerable if not row["abstained"]]

    faithfulness = _mean(row["faithfulness"] for row in answered) if answered else 0.0
    relevancy = _mean(row["answer_relevancy"] for row in answered) if answered else 0.0
    recall = (
        sum(1 for row in answerable if not row["abstained"]) / len(answerable)
        if answerable
        else 0.0
    )
    abstention_acc = (
        sum(1 for row in unanswerable if row["abstained"]) / len(unanswerable)
        if unanswerable
        else 1.0
    )
    citation_rate = (
        sum(1 for row in answered if row["citation_count"] > 0) / len(answered)
        if answered
        else 0.0
    )

    summary = {
        "chunk_size": chunk_size,
        "retrieval_method": method,
        "n_questions": len(rows),
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "context_recall": round(recall, 4),
        "abstention_accuracy": round(abstention_acc, 4),
        "citation_coverage": round(citation_rate, 4),
        "grounding_score": round(0.5 * faithfulness + 0.3 * relevancy + 0.2 * citation_rate, 4),
    }
    return {"summary": summary, "rows": rows}


def compare_configurations(
    chunk_sizes: tuple[int, ...] = CHUNK_SIZE_OPTIONS,
    methods: tuple[str, ...] = RETRIEVAL_METHODS,
) -> dict[str, Any]:
    results = []
    for size in chunk_sizes:
        pipeline = get_pipeline(size)
        for method in methods:
            run = evaluate_run(chunk_size=size, method=method, pipeline=pipeline)
            results.append(run["summary"])
    ranked = sorted(results, key=lambda row: row["grounding_score"], reverse=True)
    return {"best": ranked[0] if ranked else None, "runs": ranked}


def maybe_ragas(run: dict[str, Any]) -> dict[str, Any]:
    """Run RAGAS when an OpenAI key is present; otherwise return local metrics."""
    if not OPENAI_API_KEY:
        return {
            "available": False,
            "reason": "Set OPENAI_API_KEY to run RAGAS LLM-as-judge metrics. Local grounding metrics were used instead.",
            "local": run["summary"],
        }
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        records = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        for row in run["rows"]:
            if not row["should_answer"]:
                continue
            records["question"].append(row["question"])
            records["answer"].append(row["answer"])
            records["contexts"].append(row["contexts"])
            records["ground_truth"].append(row["ground_truth"])
        dataset = Dataset.from_dict(records)
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return {"available": True, "ragas": dict(scores), "local": run["summary"]}
    except Exception as exc:
        return {
            "available": False,
            "reason": f"RAGAS evaluation failed ({exc}). Local metrics are reported instead.",
            "local": run["summary"],
        }


def _score_item(item: dict[str, Any], result: AnswerResult) -> dict[str, Any]:
    contexts = [p.chunk.text for p in result.passages]
    context_blob = " ".join(contexts)
    ground = item.get("ground_truth") or ""
    faithfulness = 0.0 if result.abstained else lexical_overlap(result.answer, context_blob)
    relevancy = 0.0 if result.abstained else lexical_overlap(result.answer, item["question"] + " " + ground)
    term_hit = 0.0
    if ground and not result.abstained:
        needed = set(tokenize(ground))
        term_hit = len(needed & set(tokenize(result.answer))) / max(1, len(needed))
    return {
        "id": item["id"],
        "question": item["question"],
        "role": item["role"],
        "should_answer": item["should_answer"],
        "ground_truth": ground,
        "answer": result.answer,
        "abstained": result.abstained,
        "confidence": result.confidence,
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "ground_truth_overlap": round(term_hit, 4),
        "citation_count": len(result.citations),
        "contexts": contexts,
        "citations": [c.to_dict() for c in result.citations],
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
