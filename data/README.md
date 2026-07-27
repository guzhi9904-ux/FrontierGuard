# Data

Benchmark data and model-generated traces are not committed.

Expected source records use JSONL:

```json
{"id": "example-1", "problem": "What is 2+2?", "answer": "4"}
```

Use `scripts/01_prepare_data.py` to create deterministic disjoint manifests.
Every downloaded dataset should be recorded in `data/manifests/datasets.yaml`
with its revision and local hash before a paper run.
