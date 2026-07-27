"""Summarize generation JSONL files without treating seeds as new problems."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from frontierguard.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for path in args.results:
        for row in read_jsonl(path):
            by_condition[row["condition"]].append(row)
    summary = {}
    for condition, rows in sorted(by_condition.items()):
        by_problem: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_problem[str(row["problem_id"])].append(float(row["correct"]))
        problem_scores = np.asarray(
            [np.mean(values) for _, values in sorted(by_problem.items())]
        )
        summary[condition] = {
            "problems": len(by_problem),
            "generations": len(rows),
            "pass_at_1_estimate": float(problem_scores.mean()),
            "mean_output_tokens": float(np.mean([row["output_tokens"] for row in rows])),
            "truncation_rate": float(np.mean([row["truncated"] for row in rows])),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
