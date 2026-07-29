"""Complete shortlisted frontier scans with paired prefix rollouts."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, fields
from typing import Any

from tqdm.auto import tqdm

from frontierguard import __version__
from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap, ReasoningStep, TraceRecord
from frontierguard.utils.provenance import (
    precision_label,
    stable_fingerprint,
    validate_output_precision_label,
)
from frontierguard.workflows import (
    complete_frontier,
    counterfactual_trace,
    prepare_trace_steps,
    scan_trace,
)


class _CounterfactualProgress:
    def __init__(self, *, disabled: bool) -> None:
        self.disabled = disabled
        self.rollouts: tqdm | None = None
        self.tokens: tqdm | None = None

    def __call__(self, event: str, details: dict[str, Any]) -> None:
        if event == "plan":
            self.rollouts = tqdm(
                total=details["total_rollouts"],
                desc="counterfactual",
                unit="rollout",
                dynamic_ncols=True,
                mininterval=0.5,
                position=0,
                disable=self.disabled,
            )
            return
        if event == "rollout_start":
            self._close_tokens()
            description = (
                f"{details['condition']} "
                f"prefix={details['prefix_index']} "
                f"seed={details['seed']}"
            )
            self.tokens = tqdm(
                total=details["max_new_tokens"],
                desc=description,
                unit="tok",
                dynamic_ncols=True,
                mininterval=0.5,
                position=1,
                leave=False,
                disable=self.disabled,
            )
            return
        if event == "token" and self.tokens is not None:
            self.tokens.update(details["completed_tokens"] - self.tokens.n)
            return
        if event == "rollout_end":
            self._close_tokens()
            if self.rollouts is not None:
                self.rollouts.update(1)
                self.rollouts.set_postfix(
                    condition=details["condition"],
                    prefix=details["prefix_index"],
                    seed=details["seed"],
                    tokens=details["output_tokens"],
                    seconds=f"{details['latency_seconds']:.1f}",
                    truncated=details["truncated"],
                    correct=details["success"],
                    refresh=True,
                )

    def _close_tokens(self) -> None:
        if self.tokens is not None:
            self.tokens.close()
            self.tokens = None

    def close(self) -> None:
        self._close_tokens()
        if self.rollouts is not None:
            self.rollouts.close()
            self.rollouts = None


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
    parser.add_argument(
        "--exhaustive-step-threshold",
        type=int,
        default=16,
        help="evaluate every reasoning step when the trace has at most this many",
    )
    parser.add_argument(
        "--candidate-neighbor-radius",
        type=int,
        default=1,
        help="for long traces, include this many neighbors around screened candidates",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--saturation-threshold", type=float, default=0.68)
    parser.add_argument("--max-saturation-fraction", type=float, default=0.8)
    parser.add_argument("--allow-saturated", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-trustworthy-seeds", type=int, default=4)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable scan, rollout and token progress bars",
    )
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must be in (0, 1)")
    if args.min_trustworthy_seeds <= 0:
        raise ValueError("--min-trustworthy-seeds must be positive")
    if args.exhaustive_step_threshold < 0:
        raise ValueError("--exhaustive-step-threshold must be non-negative")
    if args.candidate_neighbor_radius < 0:
        raise ValueError("--candidate-neighbor-radius must be non-negative")
    if len(args.seeds) < args.min_trustworthy_seeds:
        print(
            f"[warning] {len(args.seeds)} seeds are enough for a smoke test but "
            f"fewer than --min-trustworthy-seeds={args.min_trustworthy_seeds}; "
            "frontiers will not be labeled trustworthy",
            flush=True,
        )

    raw_traces = list(read_jsonl(args.traces))
    if not raw_traces:
        raise RuntimeError("trace input contains no rows")
    print(f"[setup] loading model {args.model}", flush=True)
    started = time.perf_counter()
    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    print(f"[setup] model loaded in {time.perf_counter() - started:.1f}s", flush=True)
    low = PrecisionAction(
        args.weight_bits,
        args.activation_bits,
        args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    validate_output_precision_label(args.output, low)
    action_label = precision_label(low)
    print(
        f"[setup] instrumenting {args.backend} "
        f"W{args.weight_bits}A{args.activation_bits}KV{args.kv_bits}",
        flush=True,
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
    for trace_number, raw in enumerate(raw_traces, start=1):
        trace = _trace(raw)
        prepare_trace_steps(runner, trace)
        eligible_steps = [step for step in trace.steps if step.eligible]
        presentation_steps = [
            step for step in trace.steps if step.phase == "presentation"
        ]
        excluded_reasoning_steps = [
            step
            for step in trace.steps
            if step.phase == "reasoning" and not step.eligible
        ]
        print(
            f"[trace {trace_number}/{len(raw_traces)}] "
            f"{trace.problem_id}: {len(eligible_steps)} reasoning steps, "
            f"{len(presentation_steps)} presentation steps, "
            f"{len(excluded_reasoning_steps)} structural reasoning steps excluded",
            flush=True,
        )
        scan_progress = tqdm(
            total=2 * len(eligible_steps),
            desc=f"scan {trace.problem_id}",
            unit="window",
            dynamic_ncols=True,
            mininterval=0.5,
            disable=args.no_progress,
        )

        def update_scan(phase: str, index: int, total: int) -> None:
            scan_progress.set_postfix_str(
                f"{phase} step={index}/{total}",
                refresh=False,
            )
            scan_progress.update(1)

        try:
            scan = scan_trace(
                runner,
                trace,
                shortlist_size=args.candidate_steps,
                exhaustive_step_threshold=args.exhaustive_step_threshold,
                candidate_neighbor_radius=args.candidate_neighbor_radius,
                progress_callback=None if args.no_progress else update_scan,
            )
        finally:
            scan_progress.close()
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
        print(
            f"[trace {trace_number}/{len(raw_traces)}] "
            f"{trace.problem_id}: evaluated_steps={scan.shortlist}",
            flush=True,
        )
        rollout_progress = _CounterfactualProgress(disabled=args.no_progress)
        try:
            counterfactual = counterfactual_trace(
                runner,
                trace,
                sampling,
                seeds=args.seeds,
                candidate_steps=scan.shortlist,
                progress_callback=(
                    None if args.no_progress else rollout_progress
                ),
                bootstrap_samples=args.bootstrap_samples,
                confidence_level=args.confidence_level,
                min_trustworthy_seeds=args.min_trustworthy_seeds,
            )
        finally:
            rollout_progress.close()
        result = complete_frontier(scan, counterfactual)
        quant_successes = sum(
            outcome["successes"]
            for outcome in counterfactual["quant_outcomes"]
        )
        quant_trials = sum(
            outcome["trials"]
            for outcome in counterfactual["quant_outcomes"]
        )
        counterfactual_status = (
            "all_quantized_fail"
            if quant_successes == 0
            else "all_quantized_succeed"
            if quant_successes == quant_trials
            else "partially_recoverable"
        )
        result.update(
            {
                "frontierguard_version": __version__,
                "problem_id": trace.problem_id,
                "seed": trace.seed,
                "reference_answer": trace.reference_answer,
                "action": asdict(low),
                "precision_label": action_label,
                "quantization": runner.controller.metadata(),
                "shortlist": scan.shortlist,
                "candidate_step_metadata": [
                    asdict(trace.steps[index]) for index in scan.shortlist
                ],
                "segmentation": {
                    "reasoning_steps": len(eligible_steps),
                    "presentation_steps": len(presentation_steps),
                    "excluded_reasoning_steps": len(excluded_reasoning_steps),
                    "frontier_scope": "eligible_reasoning_only",
                    "candidate_policy": (
                        "all_steps"
                        if len(eligible_steps) <= args.exhaustive_step_threshold
                        else "multi_signal_with_neighbors"
                    ),
                    "exhaustive_step_threshold": args.exhaustive_step_threshold,
                    "candidate_neighbor_radius": args.candidate_neighbor_radius,
                },
                "jsd_saturation_threshold": args.saturation_threshold,
                "jsd_saturation_fraction": saturation_fraction,
                "counterfactual_status": counterfactual_status,
                "counterfactual": counterfactual,
                "run_fingerprint": stable_fingerprint(
                    {
                        "version": __version__,
                        "model": args.model,
                        "revision": args.revision,
                        "problem_id": trace.problem_id,
                        "action": asdict(low),
                        "backend": runner.controller.metadata(),
                        "seeds": args.seeds,
                        "max_new_tokens": args.max_new_tokens,
                    }
                ),
            }
        )
        rows.append(result)
        write_jsonl(args.output, rows)
        print(
            f"[trace {trace_number}/{len(raw_traces)}] "
            f"checkpointed {args.output}",
            flush=True,
        )
    print(f"wrote {len(rows)} frontier result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
