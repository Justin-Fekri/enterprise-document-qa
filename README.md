# Enterprise Document Question-Answering Assistant

I built this retrieval-augmented generation (RAG) assistant for enterprise business documents: company policies, audit procedures, risk reports, and cybersecurity documentation.

The stack is **Python**, **FastAPI**, **Streamlit**, and **RAGAS**. The system ingests **PDF** and **Word (.docx)** files, retrieves supporting passages, and returns answers with **page-level citations**. If the documents I can access do not contain enough evidence, the assistant **abstains** instead of guessing. Role-based permissions control who can read, ingest, and evaluate each file.

This is a 2026 portfolio project. No API key is required to run it. Without a key, answers are extractive (quoted from retrieved passages). Set `OPENAI_API_KEY` for generative answers and RAGAS LLM-as-judge scores.

**Author:** [Justin-Fekri](https://github.com/Justin-Fekri)

## What I implemented

- Multi-format ingest for PDF and Word
- Configurable chunk sizes (`256 / 512 / 1024`) and retrievers (`BM25`, `TF-IDF`, hybrid RRF)
- Page-level citations on grounded answers
- Abstention when retrieval support is weak
- RBAC for admin, compliance, auditor, and employee roles
- Local grounding metrics plus optional RAGAS evaluation
- A Streamlit UI and a FastAPI backend

## Architecture

```
Streamlit UI  ──HTTP──►  FastAPI
                           │
                           ▼
              ingest → chunk → retrieve → generate
                           │
                           ▼
                    evaluation (local / RAGAS)
```

| Role | Can search | Can ingest | Can evaluate |
| --- | --- | --- | --- |
| Administrator | All classifications | Yes | Yes |
| Compliance officer | Public → Restricted | Yes | Yes |
| Internal auditor | Public → Confidential (not Restricted) | No | Yes |
| Employee | Public and Internal only | No | No |

Local demo accounts: `admin/admin123`, `compliance/comp123`, `auditor/audit123`, `employee/emp123`. Change these before any shared deployment.

## Quick start

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=.
python -m scripts.generate_sample_docs
python -m scripts.seed_corpus

# Terminal 1 — API
uvicorn backend.main:app --host 0.0.0.0 --port 8765

# Terminal 2 — UI
export BACKEND_URL=http://127.0.0.1:8765
streamlit run frontend/app.py --server.port 8502
```

Or run both with `bash scripts/dev.sh`, then open [http://127.0.0.1:8502](http://127.0.0.1:8502).

API docs: [http://127.0.0.1:8765/docs](http://127.0.0.1:8765/docs).

## Example questions

Grounded (signed in as admin or compliance):

- How often must employees rotate their passwords?
- What is the SLA for remediating critical audit findings?
- What is the residual risk rating for ransomware?
- What is the P1 incident response time?

Should abstain:

- What is the CEO salary for 2026?
- Where is the Tokyo branch office located?

An employee asking about P1 response time also abstains: the incident response plan is Restricted.

## Evaluation

From the **Evaluate** tab, or:

```bash
export PYTHONPATH=.
python -c "from enterprise_qa.evaluation import compare_configurations; import json; print(json.dumps(compare_configurations(), indent=2))"
```

Local metrics (no LLM): faithfulness (answer ∩ retrieved context), answer relevancy, context recall on the gold set, abstention accuracy, citation coverage.

RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) runs when `OPENAI_API_KEY` is set. Copy `.env.example` to `.env` to configure it.

## Tests

```bash
export PYTHONPATH=.
pytest -q
```

## Project layout

```
backend/main.py              FastAPI routes
frontend/app.py             Streamlit UI
enterprise_qa/              RAG, RBAC, ingest, evaluation
scripts/                    sample docs, seed, dev runner
data/sample_docs/           generated PDF / Word corpus
data/eval/gold_qa.json      gold questions
tests/
```

## License

MIT License. See [LICENSE](LICENSE).
