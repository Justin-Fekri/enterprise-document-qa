.PHONY: help setup seed api ui test lint eval clean docker

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	@test -f .env || cp .env.example .env

seed:  ## Generate the sample corpus, create demo users, build the index
	$(BIN)/python scripts/seed.py --reset

api:  ## Run the FastAPI service on :8000
	$(BIN)/uvicorn app.api.main:app --reload --port 8000

ui:  ## Run the Streamlit interface on :8501
	$(BIN)/streamlit run ui/streamlit_app.py

test:  ## Run the test suite
	$(BIN)/python -m pytest -q

lint:  ## Lint and format-check
	$(BIN)/ruff check app tests evaluation scripts ui
	$(BIN)/ruff format --check app tests evaluation scripts ui

eval:  ## Sweep chunk sizes and retrieval methods
	$(BIN)/python -m evaluation.chunking_experiment

ragas:  ## Score answer quality with RAGAS (needs ANTHROPIC_API_KEY)
	$(BIN)/python -m evaluation.ragas_eval

docker:  ## Build and run the API + UI with Docker Compose
	docker compose up --build

clean:  ## Remove the index and caches
	rm -rf data/index .pytest_cache .ruff_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
