"""Attribute frontier-window loss to BF16-versus-quantized activation damage."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, fields
from pathlib import Path

from tqdm.auto import tqdm

from frontierguard import __version__
from frontierguard.attribution.patching import measure_compression_damage
from frontierguard.io import read_jsonl, write_json, write_jsonl
from frontierguard.models.adapters import infer_adapter
from frontierguard.models.hf_runner import HFRunner
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap, ReasoningStep, TraceRecord
from frontierguard.utils.provenance import (
    stable_fingerprint,
    validate_output_precision_label,
)
from frontierguard.workflows import trace_input_ids


def _trace(row: dict) -> TraceRecord:
    value = dict(row)
    value["steps"] = [ReasoningStep(**item) for item in value["steps"]]
    allowed = {item.name for item in fields(TraceRecord)}
    return TraceRecord(**{key: item for key, item in value.items() if key in allowed})


def _target_steps(
    frontier: dict,
    *,
    frontier_field: str,
    target_scope: str,
) -> tuple[int, int, int] | None:
    selected = frontier.get(frontier_field)
    if selected is None:
        return None
    selected_step = int(selected)
    if target_scope == "step":
        return selected_step, selected_step, selected_step
    window = frontier.get("frontier_window")
    if not window:
        return None
    return int(window[0]), int(window[1]), selected_step


def _cap_window(
    start: int,
    end: int,
    selected_start: int,
    selected_end: int,
    maximum_tokens: int,
) -> tuple[int, int, bool]:
    if maximum_tokens <= 0:
        raise ValueError("--max-window-tokens must be positive")
    if end - start <= maximum_tokens:
        return start, end, False
    center = (selected_start + selected_end) // 2
    capped_start = max(start, center - maximum_tokens // 2)
    capped_start = min(capped_start, end - maximum_tokens)
    capped_end = capped_start + maximum_tokens
    return capped_start, capped_end, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--frontiers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--backend", choices=REFERENCE_BACKENDS, default="rtn")
    parser.add_argument("--calibration-scales")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--kv-bits", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--frontier-field",
        choices=["step_index", "first_error_step", "recovery_frontier_step"],
        default="recovery_frontier_step",
    )
    parser.add_argument("--target-scope", choices=["step", "window"], default="window")
    parser.add_argument("--max-window-tokens", type=int, default=128)
    parser.add_argument(
        "--include-module-regex",
        help="only score projection names matching this regular expression",
    )
    parser.add_argument("--exact-top-k", type=int, default=5)
    parser.add_argument("--exact-random-k", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260728)
    parser.add_argument("--require-trustworthy", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.exact_top_k < 0 or args.exact_random_k < 0:
        raise ValueError("exact patch counts cannot be negative")
    low = PrecisionAction(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        kv_bits=args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    low.validate()
    validate_output_precision_label(args.output, low)

    traces = {
        (trace.problem_id, trace.seed): trace
        for trace in (_trace(row) for row in read_jsonl(args.traces))
    }
    frontiers = list(read_jsonl(args.frontiers))
    if args.require_trustworthy:
        frontiers = [
            item for item in frontiers if item.get("recovery_trustworthy", False)
        ]
    frontiers = [
        item
        for item in frontiers
        if _target_steps(
            item,
            frontier_field=args.frontier_field,
            target_scope=args.target_scope,
        )
        is not None
    ]
    if args.limit is not None:
        frontiers = frontiers[: args.limit]
    if not frontiers:
        raise RuntimeError("no frontier records satisfy the requested target policy")

    print(f"[setup] loading model {args.model}", flush=True)
    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    adapter = infer_adapter(runner.model)
    descriptors = adapter.describe_modules(runner.model)
    if args.include_module_regex:
        pattern = re.compile(args.include_module_regex)
        descriptors = [item for item in descriptors if pattern.search(item.name)]
    if not descriptors:
        raise RuntimeError("model adapter/module filter selected no projections")

    controller = instrument_reference_backend(
        runner.model,
        PrecisionMap(default=low),
        backend=args.backend,
        calibration_scales=args.calibration_scales,
        materialize_weights=True,
    )
    runner.controller = controller
    run_config = {
        "frontierguard_version": __version__,
        "method": "frontier_conditioned_compression_damage",
        "model": args.model,
        "revision": args.revision,
        "backend": args.backend,
        "calibration_scales": args.calibration_scales,
        "low_action": asdict(low),
        "traces": str(Path(args.traces).resolve()),
        "frontiers": str(Path(args.frontiers).resolve()),
        "frontier_field": args.frontier_field,
        "target_scope": args.target_scope,
        "max_window_tokens": args.max_window_tokens,
        "include_module_regex": args.include_module_regex,
        "module_names": [item.name for item in descriptors],
        "exact_top_k": args.exact_top_k,
        "exact_random_k": args.exact_random_k,
        "random_seed": args.random_seed,
    }
    run_fingerprint = stable_fingerprint(run_config)

    output_path = Path(args.output)
    records: list[dict] = []
    if output_path.exists() and not args.overwrite:
        records = list(read_jsonl(output_path))
        fingerprints = {item.get("run_fingerprint") for item in records}
        if fingerprints != {run_fingerprint}:
            raise ValueError(
                "existing output belongs to a different run; use --overwrite "
                "or choose a new output path"
            )
    expected_rows = len(descriptors)
    row_counts: dict[tuple[str, int], int] = {}
    for item in records:
        key = (str(item["problem_id"]), int(item["trace_seed"]))
        row_counts[key] = row_counts.get(key, 0) + 1
    completed = {
        key for key, count in row_counts.items() if count == expected_rows
    }
    if any(count != expected_rows for count in row_counts.values()):
        records = [
            item
            for item in records
            if (str(item["problem_id"]), int(item["trace_seed"])) in completed
        ]
        write_jsonl(output_path, records)

    progress = tqdm(
        total=len(frontiers),
        initial=sum(
            (str(item["problem_id"]), int(item["seed"])) in completed
            for item in frontiers
        ),
        desc="frontier damage",
        unit="trace",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    skipped_missing_trace = 0
    for frontier in frontiers:
        key = (str(frontier["problem_id"]), int(frontier["seed"]))
        if key in completed:
            continue
        trace = traces.get(key)
        if trace is None:
            skipped_missing_trace += 1
            progress.update(1)
            continue
        step_range = _target_steps(
            frontier,
            frontier_field=args.frontier_field,
            target_scope=args.target_scope,
        )
        if step_range is None:
            progress.update(1)
            continue
        start_step, end_step, selected_step = step_range
        input_ids, spans = trace_input_ids(runner, trace)
        if not (
            0 <= start_step <= end_step < len(spans)
            and 0 <= selected_step < len(spans)
        ):
            raise IndexError(
                f"frontier step outside trace for {trace.problem_id}: {step_range}"
            )
        start = spans[start_step][0]
        end = spans[end_step][1]
        target_start, target_end, capped = _cap_window(
            start,
            end,
            spans[selected_step][0],
            spans[selected_step][1],
            args.max_window_tokens,
        )
        trace_seed = int(
            stable_fingerprint(
                {
                    "problem_id": trace.problem_id,
                    "seed": trace.seed,
                    "random_seed": args.random_seed,
                },
                length=8,
            ),
            16,
        )
        progress.set_postfix(
            problem=trace.problem_id,
            tokens=target_end - target_start,
            refresh=True,
        )
        result = measure_compression_damage(
            runner.model,
            controller,
            input_ids,
            target_start,
            target_end,
            descriptors,
            exact_top_k=args.exact_top_k,
            exact_random_k=args.exact_random_k,
            random_seed=trace_seed,
        )
        shared = {
            "frontierguard_version": __version__,
            "run_fingerprint": run_fingerprint,
            "method": "frontier_conditioned_compression_damage",
            "problem_id": trace.problem_id,
            "trace_seed": trace.seed,
            "model": args.model,
            "model_revision": args.revision,
            "architecture": adapter.architecture,
            "frontier_field": args.frontier_field,
            "target_scope": args.target_scope,
            "selected_step": selected_step,
            "target_step_start": start_step,
            "target_step_end": end_step,
            "target_token_start": target_start,
            "target_token_end": target_end,
            "target_tokens": target_end - target_start,
            "window_capped": capped,
            "source_recovery_trustworthy": bool(
                frontier.get("recovery_trustworthy", False)
            ),
            "source_recovery_gain": frontier.get("recovery_gain"),
            "bf16_nll": result.bf16_nll,
            "quantized_nll": result.quantized_nll,
            "nll_gap": result.quantized_nll - result.bf16_nll,
            "bf16_margin": result.bf16_margin,
            "quantized_margin": result.quantized_margin,
            "margin_drop": result.bf16_margin - result.quantized_margin,
            "low_action": asdict(low),
            "quantization": controller.metadata(),
            "gradient_estimator": {
                "formula": "-grad(L_quant) dot (h_bf16 - h_quant)",
                "activation_fake_quant_gradient": "identity_ste",
                "forward_values_changed_by_ste": False,
            },
        }
        records.extend(
            {**shared, **measurement.to_dict()}
            for measurement in result.measurements
        )
        write_jsonl(output_path, records)
        completed.add(key)
        progress.update(1)
    progress.close()

    current = [
        item for item in records if item.get("run_fingerprint") == run_fingerprint
    ]
    exact = [item for item in current if item.get("exact_role") is not None]
    summary = {
        "frontierguard_version": __version__,
        "run_fingerprint": run_fingerprint,
        "model": args.model,
        "architecture": adapter.architecture,
        "backend": controller.metadata(),
        "low_action": asdict(low),
        "frontiers_requested": len(frontiers),
        "frontiers_completed": len(completed),
        "frontiers_missing_trace": skipped_missing_trace,
        "modules_per_trace": expected_rows,
        "score_rows": len(current),
        "exact_patch_rows": len(exact),
        "positive_predicted_fraction": (
            sum(float(item["predicted_nll_rescue"]) > 0 for item in current)
            / len(current)
            if current
            else 0.0
        ),
        "exact_top_mean_rescue": (
            sum(
                float(item["exact_nll_rescue"])
                for item in exact
                if item["exact_role"] == "predicted_top"
            )
            / max(
                1,
                sum(item["exact_role"] == "predicted_top" for item in exact),
            )
        ),
        "exact_random_mean_rescue": (
            sum(
                float(item["exact_nll_rescue"])
                for item in exact
                if item["exact_role"] == "matched_random"
            )
            / max(
                1,
                sum(item["exact_role"] == "matched_random" for item in exact),
            )
        ),
        "run_config": run_config,
    }
    summary_path = Path(args.summary_output or f"{args.output}.summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output_path.resolve()}", flush=True)
    print(f"wrote {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
