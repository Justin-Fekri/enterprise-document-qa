"""Compare chunk sizes and retrieval methods on the labelled question set.

For each (chunk_size, overlap, retrieval mode) configuration the corpus is
re-chunked and re-indexed from scratch, then scored on:

  Recall@k  - did the correct source document/page reach the top k?
  MRR       - how high did the first correct passage rank?
  Abstain   - fraction of unanswerable questions correctly refused, and the
              fraction of answerable ones wrongly refused.

Run:  python -m evaluation.chunking_experiment
Writes evaluation/results/retrieval_comparison.md and .png
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.generation.answerer import resolve_floor
from app.ingestion.pipeline import ingest_path
from app.retrieval.embeddings import build_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.schemas import Role
from app.security import build_user
from evaluation.questions import ANSWERABLE, UNANSWERABLE
from scripts.seed import CORPUS_LABELS

RESULTS_DIR = Path(__file__).parent / "results"

# Sized for the sample corpus, whose pages run 80-200 tokens. On a denser
# corpus the same sweep should be run at larger sizes - the point is the
# method, and that the configured default is whatever the sweep picks.
CHUNK_SIZES = (128, 256, 512, 1024)
OVERLAPS = (0, 64)
MODES = ("dense", "sparse", "hybrid")


@dataclass
class Result:
    chunk_size: int
    overlap: int
    mode: str
    recall_at_k: float
    mrr: float
    abstain_recall: float      # unanswerable questions correctly refused
    false_abstain: float       # answerable questions wrongly refused
    chunks: int

    @property
    def f1(self) -> float:
        """Single number balancing "finds the answer" against "knows when not to"."""
        answered = 1.0 - self.false_abstain
        if answered + self.abstain_recall == 0:
            return 0.0
        return 2 * answered * self.abstain_recall / (answered + self.abstain_recall)


def build_index(index_dir: Path, chunk_size: int, overlap: int) -> VectorStore:
    settings = get_settings().model_copy(
        update={"chunk_size": chunk_size, "chunk_overlap": overlap}
    )
    store = VectorStore(index_dir, build_embedder(settings.embedding_model))
    store.clear()
    for path in sorted(settings.document_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        category, sensitivity = CORPUS_LABELS.get(path.name, (None, None))
        if category is None:
            continue
        ingest_path(
            path, store, settings, category=category, sensitivity=sensitivity
        )
    return store


def evaluate(store: VectorStore, mode: str, top_k: int, floor: float) -> Result:
    retriever = Retriever(store)
    admin = build_user("admin", Role.ADMIN)

    hits = 0
    reciprocal_ranks = 0.0
    false_abstentions = 0

    for question in ANSWERABLE:
        results = retriever.retrieve(
            question.question, user=admin, mode=mode, top_k=top_k, candidate_k=20
        )
        matched_rank = next(
            (
                item.rank
                for item in results
                if item.chunk.filename == question.source_file
                and item.chunk.page in question.source_pages
            ),
            None,
        )
        if matched_rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / matched_rank
        if not any(item.score >= floor for item in results):
            false_abstentions += 1

    correct_abstentions = 0
    for question in UNANSWERABLE:
        results = retriever.retrieve(
            question.question, user=admin, mode=mode, top_k=top_k, candidate_k=20
        )
        if not any(item.score >= floor for item in results):
            correct_abstentions += 1

    return Result(
        chunk_size=0,
        overlap=0,
        mode=mode,
        recall_at_k=hits / len(ANSWERABLE),
        mrr=reciprocal_ranks / len(ANSWERABLE),
        abstain_recall=correct_abstentions / len(UNANSWERABLE),
        false_abstain=false_abstentions / len(ANSWERABLE),
        chunks=len(store),
    )


def run(top_k: int, floor: float) -> list[Result]:
    results: list[Result] = []
    workdir = Path(tempfile.mkdtemp(prefix="eval-index-"))
    try:
        for chunk_size, overlap in itertools.product(CHUNK_SIZES, OVERLAPS):
            if overlap >= chunk_size:
                continue
            store = build_index(workdir / f"{chunk_size}-{overlap}", chunk_size, overlap)
            for mode in MODES:
                result = evaluate(store, mode, top_k, floor)
                result.chunk_size, result.overlap = chunk_size, overlap
                results.append(result)
                print(
                    f"  size={chunk_size:<5} overlap={overlap:<4} {mode:<7} "
                    f"recall@{top_k}={result.recall_at_k:.2f} mrr={result.mrr:.2f} "
                    f"abstain={result.abstain_recall:.2f}"
                )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return results


def write_report(results: list[Result], top_k: int, floor: float) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best = max(results, key=lambda item: (item.recall_at_k, item.mrr))
    # A tie across every configuration means the corpus cannot tell them
    # apart. Saying so is more useful than crowning an arbitrary winner.
    tied = [
        item
        for item in results
        if (round(item.recall_at_k, 3), round(item.mrr, 3))
        == (round(best.recall_at_k, 3), round(best.mrr, 3))
    ]
    inconclusive = len(tied) == len(results)

    lines = [
        "# Retrieval comparison",
        "",
        f"Generated by `python -m evaluation.chunking_experiment` "
        f"(top_k={top_k}, relevance floor={floor}).",
        "",
        f"- **{len(ANSWERABLE)}** answerable questions with a known source page",
        f"- **{len(UNANSWERABLE)}** unanswerable questions that must be refused",
        "",
        "| Chunk size | Overlap | Mode | Chunks | Recall@k | MRR "
        "| Abstain recall | False abstain |",
        "|---:|---:|:--|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(
        results, key=lambda r: (-r.recall_at_k, -r.mrr, r.chunk_size)
    ):
        lines.append(
            f"| {item.chunk_size} | {item.overlap} | {item.mode} | {item.chunks} | "
            f"{item.recall_at_k:.2f} | {item.mrr:.2f} | "
            f"{item.abstain_recall:.2f} | {item.false_abstain:.2f} |"
        )

    lines += ["", "## Verdict", ""]
    if inconclusive:
        lines += [
            f"**Inconclusive on this corpus.** All {len(results)} configurations "
            f"scored identically (Recall@{top_k} {best.recall_at_k:.2f}, "
            f"MRR {best.mrr:.2f}, abstain recall {best.abstain_recall:.2f}).",
            "",
            f"The sample corpus is {len(set(q.source_file for q in ANSWERABLE))} "
            "short documents, and each question's answer sits in a passage no "
            "other passage resembles - so every chunk size and every retrieval "
            "method finds it. This is a property of the fixture, not evidence "
            "that the choices do not matter.",
            "",
            "The harness is the deliverable: point `DOCUMENT_DIR` at a real "
            "corpus, extend `evaluation/questions.py`, and the same sweep will "
            "separate the configurations. The defaults ship as "
            "`chunk_size=512`, `chunk_overlap=64`, `retrieval_mode=hybrid` - "
            "chosen on reasoning rather than on this tie: hybrid is the only "
            "mode that degrades gracefully when a question is phrased in the "
            "document's vocabulary (BM25 wins) or in the user's (embeddings win).",
        ]
    else:
        lines += [
            f"`chunk_size={best.chunk_size}`, `chunk_overlap={best.overlap}`, "
            f"`retrieval_mode={best.mode}` - Recall@{top_k} {best.recall_at_k:.2f}, "
            f"MRR {best.mrr:.2f}.",
        ]
    lines += [
        "",
        "## How to read this",
        "",
        "- **Recall@k** - the passage containing the answer reached the top k.",
        "- **MRR** - how near the top it landed; higher means less noise in the "
        "context window.",
        "- **Abstain recall** - unanswerable questions correctly refused. This is "
        "the metric that keeps the assistant honest.",
        "- **False abstain** - answerable questions wrongly refused. Raising the "
        "relevance floor trades this against abstain recall.",
    ]

    report = RESULTS_DIR / "retrieval_comparison.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def write_chart(results: list[Result]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for mode in MODES:
        subset = sorted(
            (item for item in results if item.mode == mode and item.overlap),
            key=lambda item: item.chunk_size,
        )
        if subset:
            axis.plot(
                [item.chunk_size for item in subset],
                [item.recall_at_k for item in subset],
                marker="o",
                label=mode,
            )
    axis.set_xlabel("Chunk size (tokens)")
    axis.set_ylabel("Recall@k")
    axis.set_title("Retrieval quality by chunk size and method")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    path = RESULTS_DIR / "retrieval_comparison.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--floor", type=float, default=None)
    args = parser.parse_args()

    settings = get_settings()
    floor = args.floor
    if floor is None:
        floor = resolve_floor(
            settings, build_embedder(settings.embedding_model).relevance_floor
        )

    print("Sweeping chunk sizes and retrieval methods…")
    results = run(args.top_k, floor)

    report = write_report(results, args.top_k, floor)
    chart = write_chart(results)
    print(f"\nWrote {report}")
    if chart:
        print(f"Wrote {chart}")


if __name__ == "__main__":
    main()
