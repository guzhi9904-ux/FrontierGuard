"""Measure local module rescue at detected frontier steps."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields

from tqdm.auto import tqdm

from frontierguard.attribution.measure import (
    component_rescue_action,
    measure_local_nll_rescue,
)
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
    parser.add_argument(
        "--rescue-component",
        choices=["weight", "activation", "weight_activation"],
        default="weight",
        help="raise only these components while preserving all other baseline bits",
    )
    parser.add_argument(
        "--frontier-field",
        choices=["step_index", "first_error_step", "recovery_frontier_step"],
        default="recovery_frontier_step",
    )
    parser.add_argument(
        "--target-scope",
        choices=["step", "window"],
        default="step",
    )
    parser.add_argument("--require-trustworthy", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
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
    low.validate()
    high = (
        low
        if args.bf16_oracle
        else component_rescue_action(
            low,
            high_bits=args.high_bits,
            component=args.rescue_component,
        )
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

    frontiers = list(read_jsonl(args.frontiers))
    observations = []
    progress = tqdm(
        total=len(frontiers) * len(groups),
        desc="module attribution",
        unit="intervention",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for frontier in frontiers:
        if args.require_trustworthy and not frontier.get("recovery_trustworthy", False):
            progress.update(len(groups))
            continue
        if args.target_scope == "window":
            window = frontier.get("frontier_window")
            if not window:
                progress.update(len(groups))
                continue
            start_step, end_step = (int(window[0]), int(window[1]))
            selected_step = int(frontier.get("recovery_frontier_step", end_step))
        else:
            value = frontier.get(args.frontier_field)
            if value is None:
                progress.update(len(groups))
                continue
            selected_step = int(value)
            start_step = selected_step
            end_step = selected_step
        trace = traces[(str(frontier["problem_id"]), int(frontier["seed"]))]
        input_ids, target_spans = trace_input_ids(runner, trace)
        start = target_spans[start_step][0]
        end = target_spans[end_step][1]
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
                    step_index=selected_step,
                    module_name=group_name,
                    local_rescue=rescue,
                    outcome_rescue=0.0,
                    frontier_confidence=1.0,
                    metadata={
                        "quantization": controller.metadata(),
                        "low_action": asdict(low),
                        "high_action": None if args.bf16_oracle else asdict(high),
                        "bf16_oracle": args.bf16_oracle,
                        "rescue_component": args.rescue_component,
                        "frontier_field": args.frontier_field,
                        "target_scope": args.target_scope,
                        "target_step_start": start_step,
                        "target_step_end": end_step,
                        "source_recovery_trustworthy": bool(
                            frontier.get("recovery_trustworthy", False)
                        ),
                    },
                )
            )
            write_jsonl(args.output, (asdict(item) for item in observations))
            progress.update(1)
            progress.set_postfix(
                problem=trace.problem_id,
                group=group_name,
                rescue=f"{rescue:.4f}",
                refresh=True,
            )
    progress.close()


if __name__ == "__main__":
    main()
