"""Measure local module rescue at detected frontier steps."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields

from frontierguard.attribution.measure import measure_local_nll_rescue
from frontierguard.attribution.rescue import RescueObservation
from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.adapters import infer_adapter
from frontierguard.models.hf_runner import HFRunner
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap, ReasoningStep, TraceRecord
from frontierguard.workflows import trace_input_ids


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
    parser.add_argument("--frontiers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=REFERENCE_BACKENDS, default="rtn")
    parser.add_argument("--calibration-scales")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--grouping",
        choices=["module", "projection", "layer", "block"],
        default="block",
    )
    parser.add_argument("--layers-per-block", type=int, default=4)
    parser.add_argument("--bf16-oracle", action="store_true")
    args = parser.parse_args()

    traces = {
        (trace.problem_id, trace.seed): trace
        for trace in (_trace(row) for row in read_jsonl(args.traces))
    }
    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    low = PrecisionAction(
        args.weight_bits,
        args.activation_bits,
        args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    high = PrecisionAction(
        args.high_bits,
        args.high_bits,
        args.high_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    adapter = infer_adapter(runner.model)
    descriptors = adapter.describe_modules(runner.model)
    if args.grouping == "module":
        groups = {item.name: [item.name] for item in descriptors}
    elif args.grouping == "projection":
        groups = {
            name: modules
            for name, modules in adapter.group_names(runner.model).items()
            if name.startswith("projection.")
        }
    elif args.grouping == "layer":
        groups = {
            name: modules
            for name, modules in adapter.group_names(runner.model).items()
            if name.startswith("layer_")
        }
    else:
        groups = {
            name: modules
            for name, modules in adapter.group_names(
                runner.model, layers_per_block=args.layers_per_block
            ).items()
            if name.startswith("block_")
        }
    controller = instrument_reference_backend(
        runner.model,
        PrecisionMap(default=low),
        backend=args.backend,
        calibration_scales=args.calibration_scales,
        materialize_weights=True,
    )
    runner.controller = controller

    observations = []
    for frontier in read_jsonl(args.frontiers):
        step_index = frontier.get("step_index")
        if step_index is None:
            continue
        trace = traces[(str(frontier["problem_id"]), int(frontier["seed"]))]
        input_ids, target_spans = trace_input_ids(runner, trace)
        start, end = target_spans[int(step_index)]
        for group_name, module_names in groups.items():
            rescue = measure_local_nll_rescue(
                runner.model,
                controller,
                input_ids,
                target_start=start + 1,
                target_end=end + 1,
                module_names=module_names,
                action=None if args.bf16_oracle else high,
                bf16_rescue=args.bf16_oracle,
            )
            observations.append(
                RescueObservation(
                    problem_id=trace.problem_id,
                    trace_id=f"{trace.problem_id}:{trace.seed}",
                    step_index=int(step_index),
                    module_name=group_name,
                    local_rescue=rescue,
                    outcome_rescue=0.0,
                    frontier_confidence=float(frontier.get("confidence", 1.0)),
                    metadata={
                        "quantization": controller.metadata(),
                        "low_action": asdict(low),
                        "high_action": None if args.bf16_oracle else asdict(high),
                        "bf16_oracle": args.bf16_oracle,
                    },
                )
            )
    write_jsonl(args.output, (asdict(item) for item in observations))


if __name__ == "__main__":
    main()
