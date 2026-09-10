"""Score answer quality with RAGAS.

Four metrics, each answering a different question:

  faithfulness      - is the answer actually supported by the retrieved context?
  answer_relevancy  - does the answer address the question asked?
  context_precision - is the retrieved context mostly signal?
  context_recall    - did retrieval find everything the answer needed?

Requires `ANTHROPIC_API_KEY` (Claude is both the generator and the judge) and
the extras in requirements-dev.txt. Abstentions are excluded from faithfulness
and reported separately - "I don't know" is correct behaviour, not a low score.

Run:  python -m evaluation.ragas_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.config import get_settings
from app.generation.answerer import build_answerer
from app.retrieval.embeddings import build_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.schemas import Role
from app.security import build_user
from evaluation.questions import ANSWERABLE, UNANSWERABLE

RESULTS_DIR = Path(__file__).parent / "results"


def collect_samples():
    settings = get_settings()
    store = VectorStore(settings.index_dir, build_embedder(settings.embedding_model))
    if not len(store):
        sys.exit("Index is empty. Run `make seed` first.")

    retriever = Retriever(store, rrf_k=settings.rrf_k)
    answerer = build_answerer(settings, store.embedder.relevance_floor)
    admin = build_user("admin", Role.ADMIN)

    samples, abstentions = [], {"expected": 0, "unexpected": 0}

    for question in ANSWERABLE + UNANSWERABLE:
        retrieved = retriever.retrieve(
            question.question,
            user=admin,
            mode=settings.retrieval_mode,
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
        )
        answer = answerer.answer(question.question, retrieved)

        if answer.abstained:
            key = "expected" if not question.answerable else "unexpected"
            abstentions[key] += 1
            continue
        if not question.answerable:
            # Answered something it should have refused - the failure that
            # matters most. Kept in the set so it drags faithfulness down.
            pass

        samples.append(
            {
                "user_input": question.question,
                "response": answer.answer,
                "retrieved_contexts": answer.contexts,
                "reference": question.ground_truth or "Not answerable from the corpus.",
            }
        )

    return samples, abstentions


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        sys.exit(
            "RAGAS needs ANTHROPIC_API_KEY (Claude acts as generator and judge).\n"
            "Set it in .env, or run `make eval` for the retrieval comparison, "
            "which needs no key."
        )

    try:
        from datasets import Dataset
        from langchain_anthropic import ChatAnthropic
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        sys.exit("Install the evaluation extras: pip install -r requirements-dev.txt")

    samples, abstentions = collect_samples()
    print(
        f"{len(samples)} answered, "
        f"{abstentions['expected']} correctly abstained, "
        f"{abstentions['unexpected']} wrongly abstained"
    )
    if not samples:
        sys.exit("Nothing to score - the assistant abstained on every question.")

    judge = LangchainLLMWrapper(
        ChatAnthropic(
            model=settings.answer_model,
            api_key=settings.anthropic_api_key,
            max_tokens=2000,
        )
    )

    result = evaluate(
        Dataset.from_list(samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
    )

    scores = {key: round(float(value), 3) for key, value in result._repr_dict.items()}
    scores["correct_abstentions"] = abstentions["expected"]
    scores["false_abstentions"] = abstentions["unexpected"]
    scores["answered"] = len(samples)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ragas_scores.json").write_text(json.dumps(scores, indent=2))

    print("\nRAGAS scores")
    for key, value in scores.items():
        print(f"  {key:<20} {value}")
    print(f"\nWrote {RESULTS_DIR / 'ragas_scores.json'}")


if __name__ == "__main__":
    main()
