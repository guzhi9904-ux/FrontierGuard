"""Complete shortlisted frontier scans with paired prefix rollouts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields

from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
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
    parser.add_argument("--revision")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=REFERENCE_BACKENDS, default="rtn")
    parser.add_argument("--calibration-scales")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--candidate-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--saturation-threshold", type=float, default=0.68)
    parser.add_argument("--max-saturation-fraction", type=float, default=0.8)
    parser.add_argument("--allow-saturated", action="store_true")
    args = parser.parse_args()

    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    low = PrecisionAction(
        args.weight_bits,
        args.activation_bits,
        args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    runner.controller = instrument_reference_backend(
        runner.model,
        PrecisionMap(default=low),
        backend=args.backend,
        calibration_scales=args.calibration_scales,
        materialize_weights=True,
    )
    runner.kv_bits = args.kv_bits
    runner.kv_group_size = args.group_size
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens)
    rows = []
    for raw in read_jsonl(args.traces):
        trace = _trace(raw)
        scan = scan_trace(runner, trace, shortlist_size=args.candidate_steps)
        saturation_fraction = sum(
            value >= args.saturation_threshold for value in scan.step_jsd
        ) / len(scan.step_jsd)
        if (
            saturation_fraction >= args.max_saturation_fraction
            and not args.allow_saturated
        ):
            raise RuntimeError(
                f"{trace.problem_id} is globally saturated "
                f"({saturation_fraction:.1%} steps with JSD >= "
                f"{args.saturation_threshold}); run 03a_precision_sweep.py and "
                "select a stronger backend before counterfactual rollouts"
            )
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
                "action": asdict(low),
                "quantization": runner.controller.metadata(),
                "shortlist": scan.shortlist,
                "jsd_saturation_threshold": args.saturation_threshold,
                "jsd_saturation_fraction": saturation_fraction,
                "counterfactual": counterfactual,
            }
        )
        rows.append(result)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
