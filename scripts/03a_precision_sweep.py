"""Run the diagnostic precision ladder without reloading the model."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, fields
from pathlib import Path

from frontierguard.io import read_jsonl, write_json, write_jsonl
from frontierguard.models.hf_runner import HFRunner
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap, ReasoningStep, TraceRecord
from frontierguard.workflows import scan_trace


DEFAULT_CONDITIONS = (
    "bf16_identity:16:16",
    "w4a16:4:16",
    "w16a4:16:4",
    "w8a8:8:8",
    "w4a8:4:8",
    "w4a4:4:4",
)


def _trace(row: dict) -> TraceRecord:
    row = dict(row)
    row["steps"] = [ReasoningStep(**value) for value in row["steps"]]
    allowed = {item.name for item in fields(TraceRecord)}
    return TraceRecord(**{key: value for key, value in row.items() if key in allowed})


def _condition(value: str, group_size: int) -> tuple[str, PrecisionAction]:
    try:
        name, weight, activation = value.split(":")
        action = PrecisionAction(
            weight_bits=int(weight),
            activation_bits=int(activation),
            kv_bits=16,
            weight_group_size=group_size,
        )
        action.validate()
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f"condition must be NAME:WEIGHT_BITS:ACTIVATION_BITS; got {value!r}"
        ) from error
    return name, action


def _summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    conditions = {}
    for name, items in grouped.items():
        jsd = [value for item in items for value in item["step_jsd"]]
        margin = [value for item in items for value in item["step_margin_drop"]]
        nll = [value for item in items for value in item["step_nll_gap"]]
        conditions[name] = {
            "traces": len(items),
            "steps": len(jsd),
            "jsd_mean": statistics.fmean(jsd),
            "jsd_max": max(jsd),
            "jsd_saturation_fraction_ge_0p68": sum(value >= 0.68 for value in jsd)
            / len(jsd),
            "margin_drop_mean": statistics.fmean(margin),
            "nll_gap_mean": statistics.fmean(nll),
            "identity_pass": (
                max(abs(value) for value in jsd + margin + nll) <= 1e-6
                if name == "bf16_identity"
                else None
            ),
        }
        saturation_fraction = conditions[name][
            "jsd_saturation_fraction_ge_0p68"
        ]
        conditions[name]["status"] = (
            "identity_control"
            if name == "bf16_identity"
            else "global_collapse"
            if saturation_fraction >= 0.8
            else "counterfactual_candidate"
        )
    return {"conditions": conditions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--revision")
    parser.add_argument("--backend", choices=REFERENCE_BACKENDS, default="rtn")
    parser.add_argument("--calibration-scales")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="entries formatted as NAME:WEIGHT_BITS:ACTIVATION_BITS",
    )
    args = parser.parse_args()

    traces = [_trace(row) for row in read_jsonl(args.traces)]
    if args.limit is not None:
        traces = traces[: args.limit]
    if not traces:
        raise RuntimeError("trace input contains no rows")
    conditions = [_condition(value, args.group_size) for value in args.conditions]

    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    initial = PrecisionMap(default=conditions[0][1])
    controller = instrument_reference_backend(
        runner.model,
        initial,
        backend=args.backend,
        calibration_scales=args.calibration_scales,
        materialize_weights=True,
    )
    runner.controller = controller

    rows = []
    for condition_name, action in conditions:
        controller.set_precision_map(PrecisionMap(default=action))
        runner.kv_bits = 16
        for index, trace in enumerate(traces, start=1):
            scan = scan_trace(
                runner,
                trace,
                shortlist_size=args.shortlist_size,
            )
            rows.append(
                {
                    "condition": condition_name,
                    "problem_id": trace.problem_id,
                    "seed": trace.seed,
                    "action": asdict(action),
                    "quantization": controller.metadata(),
                    "step_indices": scan.step_indices,
                    "step_jsd": scan.step_jsd,
                    "step_margin_drop": scan.step_margin_drop,
                    "step_nll_gap": scan.step_nll_gap,
                    "shortlist": scan.shortlist,
                }
            )
            print(
                f"[{condition_name}] [{index}/{len(traces)}] {trace.problem_id}",
                flush=True,
            )

    write_jsonl(args.output, rows)
    summary = {
        "model": args.model,
        "model_revision": args.revision,
        "backend": controller.metadata(),
        **_summary(rows),
    }
    summary_path = args.summary_output or f"{args.output}.summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    print(f"wrote {Path(args.output).resolve()}")
    print(f"wrote {Path(summary_path).resolve()}")


if __name__ == "__main__":
    main()
