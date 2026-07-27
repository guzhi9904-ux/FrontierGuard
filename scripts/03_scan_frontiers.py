"""Teacher-forced BF16-vs-fake-quant frontier scan."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields

from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
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
    parser.add_argument("--revision")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=REFERENCE_BACKENDS, default="rtn")
    parser.add_argument("--calibration-scales")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--saturation-threshold", type=float, default=0.68)
    args = parser.parse_args()

    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    action = PrecisionAction(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        kv_bits=args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    controller = instrument_reference_backend(
        runner.model,
        PrecisionMap(default=action),
        backend=args.backend,
        calibration_scales=args.calibration_scales,
        materialize_weights=True,
    )
    runner.controller = controller
    runner.kv_bits = args.kv_bits
    rows = []
    for raw_trace in read_jsonl(args.traces):
        trace = _trace(raw_trace)
        scan = scan_trace(runner, trace)
        saturation_fraction = sum(
            value >= args.saturation_threshold for value in scan.step_jsd
        ) / len(scan.step_jsd)
        rows.append(
            {
                "problem_id": trace.problem_id,
                "seed": trace.seed,
                "action": asdict(action),
                "quantization": controller.metadata(),
                "step_indices": scan.step_indices,
                "step_jsd": scan.step_jsd,
                "step_margin_drop": scan.step_margin_drop,
                "step_nll_gap": scan.step_nll_gap,
                "shortlist": scan.shortlist,
                "candidate_step_metadata": [
                    asdict(trace.steps[index]) for index in scan.shortlist
                ],
                "jsd_saturation_threshold": args.saturation_threshold,
                "jsd_saturation_fraction": saturation_fraction,
                "globally_saturated": saturation_fraction >= 0.8,
            }
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
