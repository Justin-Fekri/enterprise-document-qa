# Contributing

## Setup

```bash
make setup && make seed
```

## Before opening a pull request

```bash
make lint && make test
```

Both must pass. The test suite runs fully offline — if a change makes a test
require an API key or a model download, that is a bug in the test.

## Conventions

- Permissions are evaluated in exactly one place, `user_can_read` in
  `app/retrieval/retriever.py`. Do not add a second copy of the rule.
- Anything that can produce an answer must also be able to abstain.
- New retrieval or chunking behaviour should come with a question in
  `evaluation/questions.py` that would catch a regression.
