"""Turn retrieved passages into a cited answer - or into an honest abstention.

There are three independent places this pipeline can decide it does not know:

1. Retrieval returned nothing the user is allowed to read.
2. Nothing retrieved clears the relevance floor (no API call is spent).
3. The model itself reports `sufficient_evidence: false`.

Any citation the model returns is validated back against the passages it was
actually given; anything that does not resolve is dropped rather than shown.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Protocol

from app.config import Settings
from app.generation.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from app.schemas import Answer, Citation, RetrievedChunk

logger = logging.getLogger(__name__)

NO_EVIDENCE = (
    "The document collection does not contain enough evidence to answer this "
    "question."
)


class AnswerGenerator(Protocol):
    name: str
    relevance_floor: float

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Answer: ...


def resolve_floor(settings: Settings, embedder_floor: float | None) -> float:
    """An explicit setting wins; otherwise use the embedder's calibrated floor."""
    if settings.min_relevance_score is not None:
        return settings.min_relevance_score
    return embedder_floor if embedder_floor is not None else 0.25


def _abstain(question: str, reason: str, model: str, started: float) -> Answer:
    return Answer(
        question=question,
        answer=NO_EVIDENCE,
        abstained=True,
        reason=reason,
        model=model,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _clip(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ClaudeAnswerer:
    """Grounded answering with Claude, using structured output for the contract."""

    def __init__(self, settings: Settings, embedder_floor: float | None = None):
        import anthropic

        self.settings = settings
        self.relevance_floor = resolve_floor(settings, embedder_floor)
        self.name = settings.answer_model
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        started = time.perf_counter()

        if not retrieved:
            return _abstain(
                question,
                "No passages in the collection are both relevant and readable by this user.",
                self.name,
                started,
            )

        # Cheap abstention: if the best passage is not even topically close,
        # there is no point paying for a generation to be told the same thing.
        usable = [
            item for item in retrieved if item.score >= self.relevance_floor
        ]
        if not usable:
            best = max(item.score for item in retrieved)
            return _abstain(
                question,
                (
                    f"Best passage scored {best:.2f}, below the "
                    f"{self.relevance_floor:.2f} relevance floor."
                ),
                self.name,
                started,
            )

        try:
            response = self._client.messages.create(
                model=self.settings.answer_model,
                max_tokens=self.settings.max_tokens,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.settings.effort,
                    "format": ANSWER_SCHEMA,
                },
                messages=[
                    {"role": "user", "content": build_user_prompt(question, usable)}
                ],
            )
        except self._anthropic.RateLimitError as error:
            raise GenerationError(
                "The answering model is rate limited; retry shortly."
            ) from error
        except self._anthropic.APIConnectionError as error:
            raise GenerationError("Could not reach the answering model.") from error
        except self._anthropic.APIStatusError as error:
            raise GenerationError(
                f"Answering model error ({error.status_code})."
            ) from error

        if response.stop_reason == "refusal":
            return _abstain(
                question, "The model declined to answer this request.", self.name, started
            )

        payload = self._parse(response)
        return self._to_answer(question, payload, usable, started)

    def _parse(self, response) -> dict:
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Model returned non-JSON output; treating as abstention.")
            return {
                "sufficient_evidence": False,
                "answer": "",
                "reason": "The model response could not be parsed.",
                "citations": [],
            }

    def _to_answer(
        self,
        question: str,
        payload: dict,
        passages: list[RetrievedChunk],
        started: float,
    ) -> Answer:
        latency = int((time.perf_counter() - started) * 1000)
        contexts = [item.chunk.text for item in passages]

        if not payload.get("sufficient_evidence") or not payload.get("answer", "").strip():
            return Answer(
                question=question,
                answer=NO_EVIDENCE,
                abstained=True,
                reason=payload.get("reason", "") or "Insufficient evidence in the passages.",
                contexts=contexts,
                model=self.name,
                latency_ms=latency,
            )

        citations = resolve_citations(payload.get("citations", []), passages)
        if not citations:
            # An answer with no resolvable citation is exactly the ungrounded
            # output this system exists to prevent.
            return Answer(
                question=question,
                answer=NO_EVIDENCE,
                abstained=True,
                reason="The answer could not be traced back to any retrieved passage.",
                contexts=contexts,
                model=self.name,
                latency_ms=latency,
            )

        return Answer(
            question=question,
            answer=payload["answer"].strip(),
            abstained=False,
            citations=citations,
            contexts=contexts,
            model=self.name,
            latency_ms=latency,
        )


class ExtractiveAnswerer:
    """No-API fallback: quote the best passage rather than compose an answer.

    This keeps the repository fully runnable - and the test suite fully
    offline - without an API key. It abstains on exactly the same conditions
    as the Claude path.
    """

    name = "extractive-baseline"

    def __init__(self, settings: Settings, embedder_floor: float | None = None):
        self.settings = settings
        self.relevance_floor = resolve_floor(settings, embedder_floor)

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        started = time.perf_counter()

        if not retrieved:
            return _abstain(question, "Nothing retrieved for this user.", self.name, started)

        usable = [
            item for item in retrieved if item.score >= self.relevance_floor
        ]
        if not usable:
            best = max(item.score for item in retrieved)
            return _abstain(
                question,
                f"Best passage scored {best:.2f}, below the relevance floor.",
                self.name,
                started,
            )

        top = usable[0].chunk
        return Answer(
            question=question,
            answer=(
                f"Based on {top.filename} (page {top.page}): {_clip(top.text, 700)}"
            ),
            abstained=False,
            citations=[
                Citation(
                    filename=top.filename,
                    page=top.page,
                    section=top.section,
                    quote=_clip(top.text, 240),
                )
            ],
            contexts=[item.chunk.text for item in usable],
            model=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class GenerationError(RuntimeError):
    """Raised when the answering model is unreachable or failing."""


def resolve_citations(
    raw_citations: list[dict], passages: list[RetrievedChunk]
) -> list[Citation]:
    """Map model-reported passage numbers back to real chunks.

    Citations pointing at a passage number that was never provided are
    hallucinated and get dropped.
    """
    resolved: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for entry in raw_citations:
        try:
            passage_id = int(entry.get("passage_id", 0))
        except (TypeError, ValueError):
            continue
        if not 1 <= passage_id <= len(passages):
            logger.warning("Dropping citation to unknown passage %s", passage_id)
            continue

        chunk = passages[passage_id - 1].chunk
        key = (chunk.filename, chunk.page)
        if key in seen:
            continue
        seen.add(key)

        resolved.append(
            Citation(
                filename=chunk.filename,
                page=chunk.page,
                section=chunk.section,
                quote=_clip(str(entry.get("quote", "")), 300),
            )
        )

    return resolved


def build_answerer(
    settings: Settings, embedder_floor: float | None = None
) -> AnswerGenerator:
    """Use Claude when a key is configured; otherwise stay runnable offline."""
    if not settings.anthropic_api_key:
        logger.info("ANTHROPIC_API_KEY not set - using the extractive baseline.")
        return ExtractiveAnswerer(settings, embedder_floor)
    try:
        return ClaudeAnswerer(settings, embedder_floor)
    except Exception as error:  # pragma: no cover - environment dependent
        logger.warning("Falling back to the extractive baseline: %s", error)
        return ExtractiveAnswerer(settings, embedder_floor)
