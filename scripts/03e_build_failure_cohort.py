"""Build a multi-problem BF16-correct/quantized-failing frontier cohort."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from frontierguard import __version__
from frontierguard.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output-traces", required=True)
    parser.add_argument("--output-manifest")
    parser.add_argument("--bf16-condition", default="bf16")
    parser.add_argument("--quant-condition", required=True)
    parser.add_argument("--bf16-min-success", type=float, default=0.75)
    parser.add_argument("--quant-max-success", type=float, default=0.50)
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--minimum-problems", type=int, default=20)
    args = parser.parse_args()

    if not 0 <= args.quant_max_success < args.bf16_min_success <= 1:
        raise ValueError(
            "thresholds must satisfy 0 <= quant max < BF16 min <= 1"
        )
    if args.min_trials <= 0 or args.minimum_problems <= 0:
        raise ValueError("trial and problem minimums must be positive")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rows = list(read_jsonl(args.operating_points))
    available = sorted({str(row["condition"]) for row in rows})
    for row in rows:
        grouped[(str(row["problem_id"]), str(row["condition"]))].append(row)
    for condition in (args.bf16_condition, args.quant_condition):
        if condition not in available:
            raise ValueError(
                f"condition {condition!r} not found; available: {available}"
            )

    decisions = []
    selected_ids = set()
    problem_ids = sorted({problem_id for problem_id, _condition in grouped})
    for problem_id in problem_ids:
        bf16 = grouped.get((problem_id, args.bf16_condition), [])
        quant = grouped.get((problem_id, args.quant_condition), [])
        bf16_rate = (
            sum(bool(item["correct"]) for item in bf16) / len(bf16)
            if bf16
            else 0.0
        )
        quant_rate = (
            sum(bool(item["correct"]) for item in quant) / len(quant)
            if quant
            else 1.0
        )
        bf16_by_seed = {int(item["seed"]): bool(item["correct"]) for item in bf16}
        quant_by_seed = {int(item["seed"]): bool(item["correct"]) for item in quant}
        paired_seeds = sorted(bf16_by_seed.keys() & quant_by_seed.keys())
        paired_failures = sum(
            bf16_by_seed[seed] and not quant_by_seed[seed]
            for seed in paired_seeds
        )
        selected = bool(
            len(bf16) >= args.min_trials
            and len(quant) >= args.min_trials
            and bf16_rate >= args.bf16_min_success
            and quant_rate <= args.quant_max_success
            and paired_failures > 0
        )
        if selected:
            selected_ids.add(problem_id)
        decisions.append(
            {
                "problem_id": problem_id,
                "selected": selected,
                "bf16_trials": len(bf16),
                "bf16_success_rate": bf16_rate,
                "quant_trials": len(quant),
                "quant_success_rate": quant_rate,
                "paired_seeds": paired_seeds,
                "paired_bf16_correct_quant_wrong": paired_failures,
            }
        )

    traces = list(read_jsonl(args.traces))
    selected_traces = [
        {
            **row,
            "metadata": {
                **dict(row.get("metadata", {})),
                "failure_cohort": {
                    "source": str(Path(args.operating_points).resolve()),
                    "bf16_condition": args.bf16_condition,
                    "quant_condition": args.quant_condition,
                },
            },
        }
        for row in traces
        if str(row["problem_id"]) in selected_ids and row.get("correct", False)
    ]
    present_ids = {str(row["problem_id"]) for row in selected_traces}
    missing_trace_ids = sorted(selected_ids - present_ids)
    write_jsonl(args.output_traces, selected_traces)

    manifest = {
        "frontierguard_version": __version__,
        "method": "bf16_correct_quantized_failure_cohort",
        "operating_points": str(Path(args.operating_points).resolve()),
        "traces": str(Path(args.traces).resolve()),
        "bf16_condition": args.bf16_condition,
        "quant_condition": args.quant_condition,
        "bf16_min_success": args.bf16_min_success,
        "quant_max_success": args.quant_max_success,
        "min_trials": args.min_trials,
        "minimum_problems": args.minimum_problems,
        "selected_problems": len(selected_ids),
        "selected_trace_rows": len(selected_traces),
        "missing_trace_ids": missing_trace_ids,
        "evidence_status": (
            "ready_for_multi_problem_attribution"
            if len(selected_ids) >= args.minimum_problems and not missing_trace_ids
            else "insufficient_failure_cohort"
        ),
        "decisions": decisions,
    }
    manifest_path = Path(
        args.output_manifest or f"{args.output_traces}.manifest.json"
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"wrote {Path(args.output_traces).resolve()}", flush=True)
    print(f"wrote {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
