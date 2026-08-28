#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8765}"

python -m scripts.generate_sample_docs
python -m scripts.seed_corpus

uvicorn backend.main:app --host 0.0.0.0 --port 8765 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import time, urllib.request
for _ in range(40):
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=1)
        break
    except Exception:
        time.sleep(0.25)
else:
    raise SystemExit("API failed to start")
PY

streamlit run frontend/app.py --server.port 8502 --server.address 0.0.0.0 --server.headless true
