"""Summarize cross-problem and cross-model attribution stability."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from frontierguard import __version__
from frontierguard.attribution.stability import (
    aggregate_damage_rows,
    exact_patch_diagnostics,
    relative_depth_key,
    shared_rank_correlation,
    top_k_jaccard,
)
from frontierguard.io import read_jsonl, write_json


def _parse_input(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--scores must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--scores must use non-empty NAME=PATH")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        type=_parse_input,
        metavar="NAME=PATH",
        help="repeat for discovery, validation, or another model",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-field", default="predicted_nll_rescue")
    parser.add_argument("--depth-bins", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    names = [name for name, _path in args.scores]
    if len(set(names)) != len(names):
        raise ValueError("score input names must be unique")
    if args.top_k <= 0 or args.depth_bins <= 0:
        raise ValueError("--top-k and --depth-bins must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must lie in (0, 1)")

    split_rows = {
        name: list(read_jsonl(path)) for name, path in args.scores
    }
    for name, rows in split_rows.items():
        if not rows:
            raise RuntimeError(f"score input {name!r} is empty")
        missing = [
            field
            for field in (
                "problem_id",
                "module_name",
                "relative_depth",
                "family",
                "projection",
                args.score_field,
            )
            if field not in rows[0]
        ]
        if missing:
            raise ValueError(f"score input {name!r} misses fields: {missing}")

    summaries = {}
    for index, (name, rows) in enumerate(split_rows.items()):
        exact = aggregate_damage_rows(
            rows,
            key=lambda row: str(row["module_name"]),
            score_field=args.score_field,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed + index,
        )
        normalized = aggregate_damage_rows(
            rows,
            key=lambda row: relative_depth_key(row, bins=args.depth_bins),
            score_field=args.score_field,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed + index,
        )
        problems = len({str(row["problem_id"]) for row in rows})
        summaries[name] = {
            "source": str(Path(dict(args.scores)[name]).resolve()),
            "models": sorted({str(row.get("model")) for row in rows}),
            "architectures": sorted(
                {str(row.get("architecture")) for row in rows}
            ),
            "problems": problems,
            "rows": len(rows),
            "evidence_status": (
                "pilot_only"
                if problems < 20
                else "minimum_multi_problem_evidence"
            ),
            "exact_module_ranking": exact,
            "relative_depth_ranking": normalized,
            "exact_patch_diagnostics": exact_patch_diagnostics(rows),
        }

    comparisons = {}
    for left_name, right_name in combinations(names, 2):
        left = summaries[left_name]
        right = summaries[right_name]
        pair_name = f"{left_name}__{right_name}"
        comparisons[pair_name] = {
            "exact_top_k_jaccard": top_k_jaccard(
                left["exact_module_ranking"],
                right["exact_module_ranking"],
                k=args.top_k,
            ),
            "exact_shared_spearman": shared_rank_correlation(
                left["exact_module_ranking"],
                right["exact_module_ranking"],
            ),
            "relative_depth_top_k_jaccard": top_k_jaccard(
                left["relative_depth_ranking"],
                right["relative_depth_ranking"],
                k=args.top_k,
            ),
            "relative_depth_shared_spearman": shared_rank_correlation(
                left["relative_depth_ranking"],
                right["relative_depth_ranking"],
            ),
        }

    discovery_name = names[0]
    discovery_top = [
        item["key"]
        for item in summaries[discovery_name]["exact_module_ranking"][: args.top_k]
    ]
    heldout = {}
    for name in names[1:]:
        by_key = {
            item["key"]: item
            for item in summaries[name]["exact_module_ranking"]
        }
        heldout[name] = [
            {
                "discovery_rank": rank,
                "key": key,
                "heldout": by_key.get(key),
            }
            for rank, key in enumerate(discovery_top, start=1)
        ]

    output = {
        "frontierguard_version": __version__,
        "method": "problem_level_damage_stability",
        "score_field": args.score_field,
        "depth_bins": args.depth_bins,
        "top_k": args.top_k,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
        "discovery_split": discovery_name,
        "splits": summaries,
        "pairwise_stability": comparisons,
        "heldout_evaluation_of_discovery_top": heldout,
        "interpretation_guardrails": [
            "aggregate repeated trace seeds inside each problem before inference",
            "do not call a module stable from a one-problem confidence interval",
            "exact activation patching validates leverage, not unique causal storage",
            "cross-architecture claims use relative depth and projection family",
            "final claims still require prompt-only selective-rescue accuracy",
        ],
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2), flush=True)
    print(f"wrote {Path(args.output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
