"""Complete shortlisted frontier scans with paired prefix rollouts."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, fields
from typing import Any

from tqdm.auto import tqdm

from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap, ReasoningStep, TraceRecord
from frontierguard.workflows import (
    complete_frontier,
    counterfactual_trace,
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
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--saturation-threshold", type=float, default=0.68)
    parser.add_argument("--max-saturation-fraction", type=float, default=0.8)
    parser.add_argument("--allow-saturated", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable scan, rollout and token progress bars",
    )
    args = parser.parse_args()

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
        print(
            f"[trace {trace_number}/{len(raw_traces)}] "
            f"{trace.problem_id}: teacher-forcing scan",
            flush=True,
        )
        scan_progress = tqdm(
            total=2 * len(trace.steps),
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
            f"{trace.problem_id}: shortlist={scan.shortlist}",
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
            )
        finally:
            rollout_progress.close()
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
        print(
            f"[trace {trace_number}/{len(raw_traces)}] "
            f"checkpointed {args.output}",
            flush=True,
        )
    print(f"wrote {len(rows)} frontier result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
