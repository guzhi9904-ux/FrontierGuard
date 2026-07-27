"""Create deterministic, disjoint QCal/Profile/Validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from frontierguard.io import read_jsonl, write_json


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def problem_hash(text: str) -> str:
    return hashlib.sha256(normalize_problem(text).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="candidate JSONL")
    parser.add_argument("--exclude", nargs="*", default=[], help="evaluation JSONL files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qcal", type=int, default=128)
    parser.add_argument("--profile", type=int, default=96)
    parser.add_argument("--validation", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    excluded = {
        problem_hash(row["problem"])
        for path in args.exclude
        for row in read_jsonl(path)
    }
    unique: dict[str, dict] = {}
    for row in read_jsonl(args.input):
        digest = problem_hash(row["problem"])
        if digest not in excluded:
            unique.setdefault(digest, row)
    values = sorted(unique.items())
    random.Random(args.seed).shuffle(values)
    required = args.qcal + args.profile + args.validation
    if len(values) < required:
        raise RuntimeError(f"need {required} unique candidates; found {len(values)}")

    cursor = 0
    subsets = {}
    for name, size in (
        ("qcal", args.qcal),
        ("profile", args.profile),
        ("validation", args.validation),
    ):
        selected = values[cursor : cursor + size]
        cursor += size
        subsets[name] = [
            {"id": row.get("id", digest), "problem_hash": digest}
            for digest, row in selected
        ]
    manifest = {
        "source": str(Path(args.input).resolve()),
        "excluded_sources": [str(Path(path).resolve()) for path in args.exclude],
        "seed": args.seed,
        "normalization": "lowercase-strip-collapse-whitespace-sha256",
        "subsets": subsets,
    }
    write_json(args.output, manifest)
    print(json.dumps({name: len(rows) for name, rows in subsets.items()}, indent=2))


if __name__ == "__main__":
    main()
