"""Teacher-forced BF16-vs-fake-quant frontier scan."""

from __future__ import annotations

import argparse
from dataclasses import fields

from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import PrecisionAction, PrecisionMap, ReasoningStep, TraceRecord
from frontierguard.workflows import scan_trace


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
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    runner = HFRunner.from_pretrained(args.model)
    action = PrecisionAction(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        kv_bits=args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    controller = instrument_linear_layers(
        runner.model, PrecisionMap(default=action), materialize_weights=True
    )
    runner.controller = controller
    runner.kv_bits = args.kv_bits
    rows = []
    for raw_trace in read_jsonl(args.traces):
        trace = _trace(raw_trace)
        scan = scan_trace(runner, trace)
        rows.append(
            {
                "problem_id": trace.problem_id,
                "seed": trace.seed,
                "step_jsd": scan.step_jsd,
                "step_margin_drop": scan.step_margin_drop,
                "step_nll_gap": scan.step_nll_gap,
                "shortlist": scan.shortlist,
            }
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
