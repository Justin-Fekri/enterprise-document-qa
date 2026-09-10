"""The grounding contract given to the model.

The single most important property of this assistant is that it says "I don't
know" instead of guessing, so the abstention rule is stated first, stated
concretely, and repeated in the output schema.
"""

from __future__ import annotations

from app.schemas import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about a company's internal business \
documents: policies, audit procedures, risk reports and cybersecurity \
documentation.

You will be given numbered passages retrieved from that document collection. \
Follow these rules exactly.

1. Answer ONLY from the passages provided. You have no other knowledge of this \
company. Never fill a gap with general knowledge or a plausible-sounding \
industry norm.
2. If the passages do not contain enough evidence to answer, set \
`sufficient_evidence` to false, leave `answer` empty, and explain in `reason` \
what is missing. Abstaining is the correct answer, not a failure - a wrong \
answer about a control or an audit requirement is far more costly than no \
answer.
3. Partial evidence means a partial answer: state what the documents do say, \
and say plainly which part of the question they do not cover.
4. Cite every claim. Each citation is the number of the passage that supports \
it, plus a short verbatim quote from that passage. Never cite a passage number \
you were not given.
5. Quote exactly. Do not paraphrase inside a `quote` field.
6. Be direct and specific. Prefer the document's own terminology, thresholds \
and timeframes over looser restatements."""

ANSWER_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "sufficient_evidence": {
                "type": "boolean",
                "description": "True only if the passages actually support an answer.",
            },
            "answer": {
                "type": "string",
                "description": "The grounded answer, or an empty string when abstaining.",
            },
            "reason": {
                "type": "string",
                "description": "When abstaining, what evidence is missing.",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "passage_id": {
                            "type": "integer",
                            "description": "The number of a provided passage.",
                        },
                        "quote": {
                            "type": "string",
                            "description": "Verbatim supporting span from that passage.",
                        },
                    },
                    "required": ["passage_id", "quote"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["sufficient_evidence", "answer", "reason", "citations"],
        "additionalProperties": False,
    },
}


def format_passages(retrieved: list[RetrievedChunk]) -> str:
    """Render passages as a numbered block the model can cite by number."""
    lines: list[str] = []
    for index, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        header = f"[{index}] {chunk.filename} - page {chunk.page}"
        if chunk.section:
            header += f" - section: {chunk.section}"
        lines.append(f"{header}\n{chunk.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    return (
        f"PASSAGES\n========\n{format_passages(retrieved)}\n\n"
        f"QUESTION\n========\n{question}"
    )
