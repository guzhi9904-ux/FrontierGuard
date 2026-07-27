"""Screen prompt-only accuracy before expensive counterfactual rollouts."""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from tqdm.auto import tqdm

from frontierguard import __version__
from frontierguard.evaluation.statistics import binomial_wilson
from frontierguard.io import read_jsonl, write_json, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap
from frontierguard.traces.verify import extract_final_answer, verify_math_answer


DEFAULT_CONDITIONS = (
    "bf16:16:16:16",
    "w4a16kv16:4:16:16",
    "w4a8kv16:4:8:16",
    "w4a4kv16:4:4:16",
    "w4a4kv8:4:4:8",
    "w4a4kv4:4:4:4",
)


def _condition(value: str, group_size: int) -> tuple[str, PrecisionAction]:
    try:
        name, weight, activation, kv = value.split(":")
        action = PrecisionAction(
            weight_bits=int(weight),
            activation_bits=int(activation),
            kv_bits=int(kv),
            weight_group_size=group_size,
            kv_group_size=group_size,
        )
        action.validate()
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"condition must be NAME:WEIGHT_BITS:ACTIVATION_BITS:KV_BITS; got {value!r}"
        ) from error
    return name, action


def _unique_correct_traces(path: str, limit: int | None) -> list[dict]:
    by_problem = {}
    for row in read_jsonl(path):
        if not row.get("correct", False):
            continue
        by_problem.setdefault(str(row["problem_id"]), row)
    rows = list(by_problem.values())
    return rows if limit is None else rows[:limit]


def _status(
    accuracy: float,
    *,
    minimum: float,
    maximum: float,
) -> str:
    if accuracy < minimum:
        return "degenerate_fail"
    if accuracy > maximum:
        return "degenerate_success"
    return "counterfactual_candidate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--traces", required=True, help="verified BF16 trace JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--backend", choices=REFERENCE_BACKENDS, default="smoothquant")
    parser.add_argument("--calibration-scales")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--candidate-min-accuracy", type=float, default=0.2)
    parser.add_argument("--candidate-max-accuracy", type=float, default=0.8)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.candidate_min_accuracy < args.candidate_max_accuracy <= 1.0:
        raise ValueError("candidate accuracy bounds must satisfy 0 <= min < max <= 1")
    traces = _unique_correct_traces(args.traces, args.limit)
    if not traces:
        raise RuntimeError("trace input contains no verified BF16-correct problems")
    conditions = [_condition(value, args.group_size) for value in args.conditions]

    print(f"[setup] loading model {args.model}", flush=True)
    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    controller = instrument_reference_backend(
        runner.model,
        PrecisionMap(default=conditions[0][1]),
        backend=args.backend,
        calibration_scales=args.calibration_scales,
        materialize_weights=True,
    )
    runner.controller = controller
    records = []
    total = len(conditions) * len(traces) * len(args.seeds)
    progress = tqdm(
        total=total,
        desc="operating-point sweep",
        unit="generation",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    try:
        for condition_name, action in conditions:
            controller.set_precision_map(PrecisionMap(default=action))
            runner.kv_bits = action.kv_bits
            runner.kv_group_size = action.kv_group_size
            runner.kv_symmetric = action.symmetric_kv
            is_bf16 = (
                action.weight_bits >= 16
                and action.activation_bits >= 16
                and action.kv_bits >= 16
            )
            manager = runner.full_precision() if is_bf16 else contextlib.nullcontext()
            with manager:
                for trace in traces:
                    prompt_ids = runner.encode_chat(trace["problem"])
                    for seed in args.seeds:
                        token_progress = tqdm(
                            total=args.max_new_tokens,
                            desc=(
                                f"{condition_name} "
                                f"problem={trace['problem_id']} seed={seed}"
                            ),
                            unit="tok",
                            dynamic_ncols=True,
                            position=1,
                            leave=False,
                            disable=args.no_progress,
                        )
                        try:
                            generated = runner.generate(
                                prompt_ids,
                                SamplingConfig(
                                    max_new_tokens=args.max_new_tokens,
                                    seed=seed,
                                ),
                                progress_callback=(
                                    None
                                    if args.no_progress
                                    else lambda completed, _total: token_progress.update(
                                        completed - token_progress.n
                                    )
                                ),
                            )
                        finally:
                            token_progress.close()
                        extracted = extract_final_answer(generated["text"])
                        correct = verify_math_answer(
                            extracted,
                            str(trace["reference_answer"]),
                        )
                        records.append(
                            {
                                "frontierguard_version": __version__,
                                "condition": condition_name,
                                "problem_id": str(trace["problem_id"]),
                                "seed": seed,
                                "action": asdict(action),
                                "correct": correct,
                                "output": generated["text"],
                                "extracted_answer": extracted,
                                "output_tokens": generated["output_tokens"],
                                "truncated": generated["truncated"],
                                "latency_seconds": generated["latency_seconds"],
                                "quantization": (
                                    {"backend": "bf16"}
                                    if is_bf16
                                    else controller.metadata()
                                ),
                            }
                        )
                        write_jsonl(args.output, records)
                        progress.update(1)
                        progress.set_postfix(
                            condition=condition_name,
                            problem=trace["problem_id"],
                            seed=seed,
                            correct=correct,
                            refresh=True,
                        )
    finally:
        progress.close()

    summaries = {}
    for condition_name, action in conditions:
        items = [item for item in records if item["condition"] == condition_name]
        successes = sum(item["correct"] for item in items)
        trials = len(items)
        accuracy = successes / trials
        lower, upper = binomial_wilson(successes, trials)
        summaries[condition_name] = {
            "problems": len({item["problem_id"] for item in items}),
            "trials": trials,
            "successes": successes,
            "accuracy": accuracy,
            "accuracy_wilson_95": [lower, upper],
            "truncation_fraction": sum(item["truncated"] for item in items) / trials,
            "mean_output_tokens": statistics.fmean(
                item["output_tokens"] for item in items
            ),
            "status": (
                "bf16_control"
                if (
                    action.weight_bits >= 16
                    and action.activation_bits >= 16
                    and action.kv_bits >= 16
                )
                else _status(
                    accuracy,
                    minimum=args.candidate_min_accuracy,
                    maximum=args.candidate_max_accuracy,
                )
            ),
        }
    summary = {
        "frontierguard_version": __version__,
        "model": args.model,
        "model_revision": args.revision,
        "backend": controller.metadata(),
        "input": str(Path(args.traces).resolve()),
        "verified_problems": len(traces),
        "seeds": args.seeds,
        "candidate_accuracy_range": [
            args.candidate_min_accuracy,
            args.candidate_max_accuracy,
        ],
        "conditions": summaries,
    }
    summary_path = args.summary_output or f"{args.output}.summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {Path(args.output).resolve()}", flush=True)
    print(f"wrote {Path(summary_path).resolve()}", flush=True)


if __name__ == "__main__":
    main()
