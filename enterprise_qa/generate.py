"""Grounded answer generation with page-level citations and abstention."""

from __future__ import annotations

import re

from enterprise_qa.config import ABSTAIN_THRESHOLD, OPENAI_API_KEY, OPENAI_MODEL
from enterprise_qa.models import AnswerResult, Citation, RetrievedPassage
from enterprise_qa.retrieve import (
    content_terms,
    lexical_overlap,
    matched_content,
    question_coverage,
)

ABSTAIN_MESSAGE = (
    "I do not have enough evidence in the documents you can access to answer "
    "this question. Try a more specific policy, audit, risk, or security topic, "
    "or ask an administrator to ingest the relevant file."
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def generate_answer(
    question: str,
    passages: list[RetrievedPassage],
    *,
    retrieval_method: str,
    chunk_size: int,
    threshold: float = ABSTAIN_THRESHOLD,
) -> AnswerResult:
    q_terms = content_terms(question)
    joined = " ".join(p.chunk.text for p in passages[:3])
    matched = matched_content(question, joined)
    coverage = question_coverage(question, passages)
    support = 0.0 if not q_terms else len(matched) / len(q_terms)
    # Min-max retrieval scores are relative; confidence is evidence overlap, not rank.
    confidence = round(0.55 * support + 0.45 * coverage, 4)
    enough_hits = len(matched) >= 2 or (len(q_terms) <= 1 and len(matched) == 1)
    if (
        not passages
        or not enough_hits
        or support < 0.5
        or coverage < 0.45
        or confidence < max(threshold, 0.4)
    ):
        return _abstain(question, passages, retrieval_method, chunk_size, confidence)

    sentences = _rank_sentences(question, passages)
    if not sentences or sentences[0][0] < 0.28:
        return _abstain(question, passages, retrieval_method, chunk_size, min(confidence, 0.35))

    if OPENAI_API_KEY:
        try:
            answer = _llm_answer(question, passages)
            generator = f"openai:{OPENAI_MODEL}"
        except Exception:
            answer = _format_extractive(sentences)
            generator = "extractive"
    else:
        answer = _format_extractive(sentences)
        generator = "extractive"

    return AnswerResult(
        question=question,
        answer=answer,
        abstained=False,
        confidence=confidence,
        retrieval_method=retrieval_method,
        chunk_size=chunk_size,
        generator=generator,
        citations=_citations(passages),
        passages=passages,
        reason="Grounded in retrieved passages with page-level citations.",
    )


def _abstain(question, passages, retrieval_method, chunk_size, confidence) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=ABSTAIN_MESSAGE,
        abstained=True,
        confidence=confidence,
        retrieval_method=retrieval_method,
        chunk_size=chunk_size,
        generator="abstain",
        citations=_citations(passages[:2]),
        passages=passages,
        reason="Insufficient supporting evidence in the accessible corpus.",
    )


def _rank_sentences(question: str, passages: list[RetrievedPassage]) -> list[tuple[float, str, RetrievedPassage]]:
    q_terms = content_terms(question)
    scored: list[tuple[float, str, RetrievedPassage]] = []
    seen: set[str] = set()
    for passage in passages:
        for sentence in _SENTENCE.split(passage.chunk.text):
            sentence = sentence.strip()
            if len(sentence) < 40:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            hits = matched_content(question, sentence)
            overlap = len(hits) / max(1, len(q_terms))
            if overlap < 0.34:
                continue
            density = lexical_overlap(question, sentence)
            score = 0.75 * overlap + 0.25 * density
            scored.append((score, sentence, passage))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:2]


def _format_extractive(chosen: list[tuple[float, str, RetrievedPassage]]) -> str:
    lines: list[str] = []
    for _, sentence, passage in chosen:
        cite = f"{passage.chunk.title}, p. {passage.chunk.page}"
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        lines.append(f"{sentence} ({cite})")
    return " ".join(lines)


def _llm_answer(question: str, passages: list[RetrievedPassage]) -> str:
    from openai import OpenAI

    context = "\n\n".join(
        f"[{p.rank}] {p.chunk.title} (page {p.chunk.page}): {p.chunk.text}"
        for p in passages
    )
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise document assistant. Answer only from the "
                    "provided passages. Cite title and page after each claim, e.g. "
                    "(Information Security Policy, p. 2). If the passages are not "
                    "enough, say you do not have enough evidence. Never invent policy."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPassages:\n{context}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _citations(passages: list[RetrievedPassage]) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for passage in passages:
        key = (passage.chunk.filename, passage.chunk.page)
        if key in seen:
            continue
        seen.add(key)
        snippet = passage.chunk.text.strip()
        if len(snippet) > 280:
            snippet = snippet[:277].rstrip() + "…"
        citations.append(
            Citation(
                title=passage.chunk.title,
                filename=passage.chunk.filename,
                page=passage.chunk.page,
                snippet=snippet,
                score=round(passage.score, 4),
            )
        )
    return citations
