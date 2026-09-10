# Enterprise Document Q&A

A retrieval-augmented question answering service for internal PDF and Word documents. It answers
from your corpus with **page-level citations**, **abstains when the evidence is thin** instead of
guessing, and enforces **role-based access control during retrieval** — so a passage a user may
not read is never retrieved, never sent to the model, and never citable.

`Python` · `FastAPI` · `RAG` · `Hybrid retrieval (BM25 + dense)` · `Streamlit` · `Docker`

[![CI](https://github.com/Justin-Fekri/enterprise-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Justin-Fekri/enterprise-document-qa/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The three problems this is built around

Most RAG demos retrieve a few chunks, paste them into a prompt, and print whatever comes back.
That fails in an enterprise setting for three specific reasons, and each one has a deliberate
answer here.

**1. An answer without a source is unusable.** Every response carries citations down to the
filename, page number and section, with the supporting quote. A reader can check the claim without
trusting the model.

**2. A confident wrong answer is worse than no answer.** The pipeline can decide it does not know
in three independent places, and says so plainly:

| Where | What happened | Cost |
|---|---|---|
| Retrieval | Nothing the user is allowed to read matched | No API call |
| Relevance floor | Nothing retrieved cleared the score threshold | No API call |
| Generation | The model reported `sufficient_evidence: false` | One API call |

Two of the three abstain *before* spending a model call.

**3. Permissions must hold inside the retrieval layer.** Filtering results after generation is too
late — the text has already reached the model. Here, permissions are applied to the candidate set
in `app/retrieval/retriever.py`, which is the single place the role table is evaluated.

## Access model

Roles map to a set of document categories and a sensitivity ceiling, declared once in
`app/security.py`:

| Role | Categories | Max sensitivity |
|---|---|---|
| `viewer` | policy, general | internal |
| `analyst` | policy, risk, general | confidential |
| `auditor` | policy, audit, risk, security, general | confidential |
| `admin` | all | restricted |

Only `admin` may upload or delete. Authentication is JWT; passwords are stored as
PBKDF2-HMAC-SHA256.

## Quickstart

```bash
make setup   # virtualenv + dependencies + .env
make seed    # generate a sample corpus, create demo users, build the index
make api     # FastAPI on :8000  (docs at /docs)
make ui      # Streamlit on :8501
```

Or with Docker:

```bash
docker compose up --build
```

**No API key required to try it.** Without `ANTHROPIC_API_KEY` the service falls back to an
extractive baseline and the whole pipeline — ingestion, retrieval, permissions, citations,
abstention — still runs.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/token` | Exchange credentials for a JWT |
| `GET` | `/me` | Current user and effective permissions |
| `GET` | `/documents` | List documents the caller may read |
| `POST` | `/documents` | Upload and ingest a PDF or DOCX (admin) |
| `DELETE` | `/documents/{doc_id}` | Remove a document and its chunks (admin) |
| `POST` | `/search` | Retrieval only — inspect what would be sent to the model |
| `POST` | `/query` | Grounded answer with citations, or an abstention |

`/search` exists so retrieval can be debugged on its own, without generation in the way.

## How retrieval works

Hybrid by default: **BM25** for lexical matches on names, IDs and exact phrases, plus **dense
embeddings** (`all-MiniLM-L6-v2`) for paraphrase, fused by reciprocal rank. `RETRIEVAL_MODE` can
be set to `dense`, `sparse` or `hybrid` to compare them.

Two implementation notes worth calling out:

- **BM25 is written out rather than imported.** It is about 40 lines, and the retrieval comparison
  in `evaluation/` needs to be able to explain exactly what the sparse retriever does.
- **A `HashingEmbedder` stands in for the real model** in tests and CI. It is deterministic and
  dependency-free, so the full pipeline is exercised without downloading model weights or holding
  an API key.

The store persists chunks, embeddings and metadata as `chunks.jsonl`, `embeddings.npy` and
`documents.json`, loaded into memory at startup. No external database — appropriate for the corpus
sizes this targets, and it keeps the project runnable in one command.

## Evaluation

```bash
make eval    # sweep chunk sizes and retrieval modes
make ragas   # answer-quality scoring (needs ANTHROPIC_API_KEY)
```

`evaluation/chunking_experiment.py` re-chunks and re-indexes the corpus for every combination of
chunk size `{128, 256, 512, 1024}` × overlap `{0, 64}` × mode `{dense, sparse, hybrid}`, scoring
recall@k, MRR, and how many *unanswerable* questions were correctly refused.
`evaluation/ragas_eval.py` scores faithfulness and answer relevance with RAGAS.

### Results on the sample corpus

Over 13 answerable questions with a known source page and 4 that must be refused:

| Metric | Result |
|---|---|
| Recall@3 | 1.00 |
| MRR | 0.92 |
| Unanswerable questions correctly refused | 4 / 4 |
| Answerable questions wrongly refused | 0 / 13 |

**This is not a comparison result, and the report says so.** All 24 configurations scored
identically. The sample corpus is four short documents in which every answer sits in a passage
nothing else resembles, so every chunk size and every retrieval method finds it — a property of
the fixture, not evidence that the choices do not matter. Point `DOCUMENT_DIR` at a real corpus
and extend `evaluation/questions.py`, and the same sweep will separate them.

The shipped defaults (`chunk_size=512`, `chunk_overlap=64`, `hybrid`) were therefore chosen on
reasoning, not on this tie. The one threshold the corpus *did* settle is the abstention floor: at
0.25 an off-topic question slipped through on a loose lexical match; 0.30 refused all four without
costing any recall, which is why the embedder ships with `relevance_floor = 0.30`.

Full output: [`evaluation/results/retrieval_comparison.md`](evaluation/results/retrieval_comparison.md).

## Tests

```bash
make test
```

65 tests, no network and no API key needed:

| File | Covers |
|---|---|
| `test_permissions.py` | Role table, JWT signing/rejection, unreadable passages never retrieved |
| `test_abstention.py` | All three abstention paths, including the two that skip the API call |
| `test_api.py` | Every endpoint, auth failures, upload validation |
| `test_retriever.py` | Dense, sparse and hybrid ranking |
| `test_chunking.py` | Boundaries, overlap, token counts |
| `test_loaders.py` | PDF and DOCX extraction, page attribution |

## Project layout

```
app/
├── api/          FastAPI routes and dependencies
├── ingestion/    PDF/DOCX loaders, chunking, ingest pipeline
├── retrieval/    BM25, embeddings, hybrid retriever, persisted store
├── generation/   Cited answering, abstention logic, prompts
├── config.py     Environment-driven settings
├── schemas.py    Pydantic models
└── security.py   Roles, permissions, JWT, password hashing
evaluation/       Chunking sweep + RAGAS scoring
scripts/          Sample corpus generator, seeding
tests/            65 tests
ui/               Streamlit interface
```

## Configuration

All settings are environment-driven; see `.env.example`. The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Unset falls back to the extractive baseline |
| `RETRIEVAL_MODE` | `hybrid` | `dense` \| `sparse` \| `hybrid` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | Tuned via `make eval` |
| `TOP_K` | 6 | Passages sent to the model |
| `MIN_RELEVANCE_SCORE` | model floor | The pre-generation abstention threshold |
| `JWT_SECRET` | — | **Must** be changed outside local use |

## Limitations

- The index loads fully into memory; it is not intended for corpora beyond the low hundreds of
  thousands of chunks. A production deployment would swap `app/retrieval/store.py` for a real
  vector database — the interface is small enough to make that a contained change.
- Users and documents are seeded from scripts, not managed through an admin UI.
- Citations point to the passage the model used. They confirm the answer is *grounded*; they do
  not prove it is *correct*, and a reader should still open the page.
- The demo credentials in `scripts/seed.py` are exactly that. Do not deploy them.

## Licence

MIT — see [LICENSE](LICENSE).
