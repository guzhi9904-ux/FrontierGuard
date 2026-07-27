"""Complete shortlisted frontier scans with paired prefix rollouts."""

from __future__ import annotations

import argparse
from dataclasses import fields

from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import PrecisionAction, PrecisionMap, ReasoningStep, TraceRecord
from frontierguard.workflows import (
    complete_frontier,
    counterfactual_trace,
    scan_trace,
)


def _trace(row: dict) -> TraceRecord:
    row = dict(row)
    row["steps"] = [ReasoningStep(**value) for value in row["steps"]]
    allowed = {item.name for item in fields(TraceRecord)}
    return TraceRecord(**{key: value for key, value in row.items() if key in allowed})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--candidate-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()

    runner = HFRunner.from_pretrained(args.model)
    low = PrecisionAction(4, 4, 4)
    runner.controller = instrument_linear_layers(
        runner.model,
        PrecisionMap(default=low),
        materialize_weights=True,
    )
    runner.kv_bits = 4
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens)
    rows = []
    for raw in read_jsonl(args.traces):
        trace = _trace(raw)
        scan = scan_trace(runner, trace, shortlist_size=args.candidate_steps)
        counterfactual = counterfactual_trace(
            runner,
            trace,
            sampling,
            seeds=args.seeds,
            candidate_steps=scan.shortlist,
        )
        result = complete_frontier(scan, counterfactual)
        result.update(
            {
                "problem_id": trace.problem_id,
                "seed": trace.seed,
                "shortlist": scan.shortlist,
                "counterfactual": counterfactual,
            }
        )
        rows.append(result)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
