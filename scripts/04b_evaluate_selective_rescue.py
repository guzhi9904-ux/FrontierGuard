"""Evaluate candidate mixed-precision rescues with prompt-only generation."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from frontierguard import __version__
from frontierguard.evaluation.selective import (
    ModuleRescueSpec,
    build_module_precision_map,
    build_precision_map,
    matched_random_module_specs,
    paired_success_lift,
    parse_rescue_spec,
    rank_damage_modules,
    random_layer_specs,
    summarize_generation_condition,
)
from frontierguard.io import read_jsonl, write_json, write_jsonl
from frontierguard.models.adapters import infer_adapter
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.factory import REFERENCE_BACKENDS, instrument_reference_backend
from frontierguard.schemas import PrecisionAction, PrecisionMap
from frontierguard.traces.verify import classify_generation, extract_answer_details
from frontierguard.utils.provenance import precision_label, stable_fingerprint


DEFAULT_RESCUES = (
    "layer13=13",
    "layer16=16",
    "top2=13,16",
    "early2=1,2",
)


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    precision_map: PrecisionMap
    full_precision: bool
    metadata: dict[str, Any]


def _unique_correct_traces(path: str, limit: int | None) -> list[dict]:
    by_problem = {}
    for row in read_jsonl(path):
        if not row.get("correct", False):
            continue
        by_problem.setdefault(str(row["problem_id"]), row)
    rows = list(by_problem.values())
    return rows if limit is None else rows[:limit]


def _uniform_condition(
    name: str,
    kind: str,
    action: PrecisionAction,
    *,
    module_count: int,
    parameter_count: int,
    high_precision: bool,
    full_precision: bool = False,
) -> Condition:
    selected_modules = module_count if high_precision else 0
    selected_parameters = parameter_count if high_precision else 0
    return Condition(
        name=name,
        kind=kind,
        precision_map=PrecisionMap(default=action),
        full_precision=full_precision,
        metadata={
            "selected_module_count": selected_modules,
            "selected_parameter_count": selected_parameters,
            "instrumented_parameter_count": parameter_count,
            "high_precision_parameter_fraction": 1.0 if high_precision else 0.0,
            "effective_weight_bits": action.weight_bits,
            "rescue_spec": None,
        },
    )


def _condition_payload(condition: Condition) -> dict:
    return {
        "name": condition.name,
        "kind": condition.kind,
        "full_precision": condition.full_precision,
        "precision_map": condition.precision_map.to_dict(),
        "metadata": condition.metadata,
    }


def _summary(
    records: list[dict],
    conditions: list[Condition],
    *,
    run_fingerprint: str,
    model: str,
    revision: str | None,
    backend: dict,
    traces: str,
    seeds: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    low_name: str,
    high_name: str,
    selection_provenance: dict | None,
) -> dict:
    by_condition = {
        condition.name: [
            row for row in records if row["condition"] == condition.name
        ]
        for condition in conditions
    }
    summaries = {}
    baseline_rows = by_condition[low_name]
    low_accuracy = summarize_generation_condition(baseline_rows)["accuracy"]
    high_accuracy = summarize_generation_condition(by_condition[high_name])["accuracy"]
    denominator = high_accuracy - low_accuracy
    for condition in conditions:
        rows = by_condition[condition.name]
        value = summarize_generation_condition(rows)
        if condition.name != low_name:
            value["paired_lift_vs_low"] = paired_success_lift(
                baseline_rows,
                rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            value["recovery_ratio_vs_uniform_high"] = (
                (value["accuracy"] - low_accuracy) / denominator
                if denominator > 0
                else None
            )
        else:
            value["paired_lift_vs_low"] = None
            value["recovery_ratio_vs_uniform_high"] = 0.0 if denominator > 0 else None
        value.update(condition.metadata)
        value["kind"] = condition.kind
        summaries[condition.name] = value

    random_items = [
        summaries[item.name] for item in conditions if item.kind == "random_rescue"
    ]
    random_control = None
    if random_items:
        accuracies = [item["accuracy"] for item in random_items]
        random_control = {
            "maps": len(random_items),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
        }
        random_budget = {
            item["selected_parameter_count"] for item in random_items
        }
        for condition in conditions:
            value = summaries[condition.name]
            if (
                condition.kind == "candidate_rescue"
                and value["selected_parameter_count"] in random_budget
            ):
                value["accuracy_minus_random_mean"] = (
                    value["accuracy"] - random_control["mean_accuracy"]
                )

    ranked_random_controls = {}
    ranked_candidates = [
        item for item in conditions if item.kind == "ranked_module_rescue"
    ]
    for candidate in ranked_candidates:
        controls = [
            summaries[item.name]
            for item in conditions
            if item.kind == "ranked_matched_random"
            and item.metadata.get("matched_condition") == candidate.name
        ]
        if not controls:
            continue
        accuracies = [item["accuracy"] for item in controls]
        control = {
            "maps": len(controls),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "condition_names": [
                item.name
                for item in conditions
                if item.kind == "ranked_matched_random"
                and item.metadata.get("matched_condition") == candidate.name
            ],
        }
        ranked_random_controls[candidate.name] = control
        summaries[candidate.name]["accuracy_minus_matched_random_mean"] = (
            summaries[candidate.name]["accuracy"] - control["mean_accuracy"]
        )

    trace_problem_ids = {str(row["problem_id"]) for row in records}
    selection_problem_ids = set(
        (selection_provenance or {}).get("selection_problem_ids", [])
    )
    overlap = sorted(trace_problem_ids & selection_problem_ids)
    return {
        "frontierguard_version": __version__,
        "run_fingerprint": run_fingerprint,
        "model": model,
        "model_revision": revision,
        "backend": backend,
        "input": str(Path(traces).resolve()),
        "seeds": seeds,
        "conditions": summaries,
        "random_control": random_control,
        "ranked_random_controls": ranked_random_controls,
        "selection_provenance": selection_provenance,
        "interpretation": {
            "primary_endpoint": "strict_eos_answer_accuracy",
            "local_nll_is_primary": False,
            "recovery_ratio_definition": (
                "(candidate_accuracy - uniform_low_accuracy) / "
                "(uniform_high_accuracy - uniform_low_accuracy)"
            ),
            "single_problem_is_pilot_only": (
                len({row["problem_id"] for row in records}) < 5
            ),
            "selection_evaluation_overlap_problems": len(overlap),
            "selection_evaluation_overlap_ids": overlap,
            "selection_evaluation_status": (
                "in_sample_exploratory"
                if overlap
                else "held_out"
                if selection_provenance is not None
                else "manual_candidates"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--traces", required=True, help="verified BF16 trace JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--backend", choices=REFERENCE_BACKENDS, default="smoothquant")
    parser.add_argument("--calibration-scales")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--kv-bits", type=int, default=16)
    parser.add_argument("--high-weight-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--rescue",
        action="append",
        help=(
            "candidate NAME=LAYER[.attention|.mlp][,...]; "
            "repeat for multiple conditions"
        ),
    )
    parser.add_argument(
        "--damage-scores",
        help=(
            "04c projection-score JSONL; enables cumulative exact-module "
            "ranked rescue conditions"
        ),
    )
    parser.add_argument("--damage-score-field", default="predicted_nll_rescue")
    parser.add_argument(
        "--rank-budgets",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="cumulative top-module budgets used with --damage-scores",
    )
    parser.add_argument("--minimum-module-problem-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-module-positive-fraction", type=float, default=0.5)
    parser.add_argument("--require-positive-module-ci", action="store_true")
    parser.add_argument("--ranked-random-maps", type=int, default=2)
    parser.add_argument("--random-maps", type=int, default=5)
    parser.add_argument("--random-budget", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.random_maps < 0:
        raise ValueError("--random-maps cannot be negative")
    if args.ranked_random_maps < 0:
        raise ValueError("--ranked-random-maps cannot be negative")
    if not args.rank_budgets or any(value <= 0 for value in args.rank_budgets):
        raise ValueError("--rank-budgets must contain positive values")
    if len(set(args.rank_budgets)) != len(args.rank_budgets):
        raise ValueError("--rank-budgets must be unique")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain unique values")
    traces = _unique_correct_traces(args.traces, args.limit)
    if not traces:
        raise RuntimeError("trace input contains no verified BF16-correct problems")

    low = PrecisionAction(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        kv_bits=args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    high = PrecisionAction(
        weight_bits=args.high_weight_bits,
        activation_bits=args.activation_bits,
        kv_bits=args.kv_bits,
        weight_group_size=args.group_size,
        kv_group_size=args.group_size,
    )
    low.validate()
    high.validate()
    if high.weight_bits <= low.weight_bits:
        raise ValueError("--high-weight-bits must exceed --weight-bits")

    print(f"[setup] loading model {args.model}", flush=True)
    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    adapter = infer_adapter(runner.model)
    descriptors = adapter.describe_modules(runner.model)
    if not descriptors:
        raise RuntimeError("model adapter found no supported transformer projections")
    total_parameters = sum(item.parameter_count for item in descriptors)

    requested_specs = [
        parse_rescue_spec(value)
        for value in (
            args.rescue
            if args.rescue is not None
            else ()
            if args.damage_scores
            else DEFAULT_RESCUES
        )
    ]
    legacy_random_maps = args.random_maps if requested_specs else 0
    random_specs = random_layer_specs(
        (item.layer_index for item in descriptors),
        budget=args.random_budget,
        count=legacy_random_maps,
        seed=args.random_seed,
    )
    ranked_specs: list[ModuleRescueSpec] = []
    ranked_random_specs: list[ModuleRescueSpec] = []
    selection_provenance = None
    if args.damage_scores:
        damage_rows = list(read_jsonl(args.damage_scores))
        ranking = rank_damage_modules(
            damage_rows,
            descriptors,
            score_field=args.damage_score_field,
            minimum_problem_fraction=args.minimum_module_problem_fraction,
            minimum_positive_fraction=args.minimum_module_positive_fraction,
            require_positive_ci=args.require_positive_module_ci,
            bootstrap_samples=args.bootstrap_samples,
            confidence=0.95,
            seed=args.bootstrap_seed,
        )
        largest_budget = max(args.rank_budgets)
        if len(ranking) < largest_budget:
            raise RuntimeError(
                f"only {len(ranking)} modules pass the damage filters, "
                f"but rank budget {largest_budget} was requested"
            )
        for budget in sorted(args.rank_budgets):
            selected_ranking = ranking[:budget]
            selected_names = tuple(item["key"] for item in selected_ranking)
            name = f"ranked_top{budget}"
            ranked_specs.append(
                ModuleRescueSpec(
                    name=name,
                    module_names=selected_names,
                    metadata={
                        "selection_method": "problem_aggregated_damage_ranking",
                        "damage_score_field": args.damage_score_field,
                        "damage_source": str(Path(args.damage_scores).resolve()),
                        "ranking": selected_ranking,
                    },
                )
            )
            ranked_random_specs.extend(
                matched_random_module_specs(
                    descriptors,
                    selected_names,
                    count=args.ranked_random_maps,
                    seed=args.random_seed + budget,
                    prefix=name,
                )
            )
        selection_provenance = {
            "damage_source": str(Path(args.damage_scores).resolve()),
            "damage_score_field": args.damage_score_field,
            "selection_problem_ids": sorted(
                {str(row["problem_id"]) for row in damage_rows}
            ),
            "selection_problems": len(
                {str(row["problem_id"]) for row in damage_rows}
            ),
            "ranked_modules_passing_filters": len(ranking),
            "rank_budgets": sorted(args.rank_budgets),
            "minimum_module_problem_fraction": args.minimum_module_problem_fraction,
            "minimum_module_positive_fraction": args.minimum_module_positive_fraction,
            "require_positive_module_ci": args.require_positive_module_ci,
            "top_ranking": ranking[:largest_budget],
        }
    reserved = {
        precision_label(low),
        f"{precision_label(high)}_full",
        "bf16",
    }
    all_names = [
        item.name
        for item in (
            list(requested_specs)
            + list(random_specs)
            + ranked_specs
            + ranked_random_specs
        )
    ]
    if len(set(all_names)) != len(all_names):
        raise ValueError("rescue condition names must be unique")
    if reserved.intersection(all_names):
        raise ValueError(f"rescue names collide with reserved conditions: {reserved}")

    low_name = precision_label(low)
    high_name = f"{precision_label(high)}_full"
    conditions = [
        _uniform_condition(
            low_name,
            "uniform_low",
            low,
            module_count=len(descriptors),
            parameter_count=total_parameters,
            high_precision=False,
        )
    ]
    for spec in requested_specs:
        precision_map, metadata = build_precision_map(
            descriptors,
            spec,
            low=low,
            high=high,
        )
        conditions.append(
            Condition(
                name=spec.name,
                kind="candidate_rescue",
                precision_map=precision_map,
                full_precision=False,
                metadata=metadata,
            )
        )
    for spec in random_specs:
        precision_map, metadata = build_precision_map(
            descriptors,
            spec,
            low=low,
            high=high,
        )
        conditions.append(
            Condition(
                name=spec.name,
                kind="random_rescue",
                precision_map=precision_map,
                full_precision=False,
                metadata=metadata,
            )
        )
    for spec in ranked_specs:
        precision_map, metadata = build_module_precision_map(
            descriptors,
            spec,
            low=low,
            high=high,
        )
        conditions.append(
            Condition(
                name=spec.name,
                kind="ranked_module_rescue",
                precision_map=precision_map,
                full_precision=False,
                metadata=metadata,
            )
        )
    for spec in ranked_random_specs:
        precision_map, metadata = build_module_precision_map(
            descriptors,
            spec,
            low=low,
            high=high,
        )
        conditions.append(
            Condition(
                name=spec.name,
                kind="ranked_matched_random",
                precision_map=precision_map,
                full_precision=False,
                metadata=metadata,
            )
        )
    conditions.append(
        _uniform_condition(
            high_name,
            "uniform_high",
            high,
            module_count=len(descriptors),
            parameter_count=total_parameters,
            high_precision=True,
        )
    )
    if not args.skip_bf16:
        conditions.append(
            _uniform_condition(
                "bf16",
                "bf16",
                PrecisionAction(16, 16, 16),
                module_count=len(descriptors),
                parameter_count=total_parameters,
                high_precision=True,
                full_precision=True,
            )
        )

    run_config = {
        "frontierguard_version": __version__,
        "model": args.model,
        "model_revision": args.revision,
        "backend": args.backend,
        "calibration_scales": args.calibration_scales,
        "traces": str(Path(args.traces).resolve()),
        "problem_ids": [str(item["problem_id"]) for item in traces],
        "seeds": args.seeds,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "conditions": [_condition_payload(item) for item in conditions],
        "selection_provenance": selection_provenance,
    }
    run_fingerprint = stable_fingerprint(run_config)

    output_path = Path(args.output)
    records = []
    if output_path.exists() and not args.overwrite:
        records = list(read_jsonl(output_path))
        fingerprints = {item.get("run_fingerprint") for item in records}
        if fingerprints != {run_fingerprint}:
            raise ValueError(
                "existing output belongs to a different run; use --overwrite "
                "or choose a new output path"
            )
    completed = {
        (str(item["condition"]), str(item["problem_id"]), int(item["seed"]))
        for item in records
    }
    if len(completed) != len(records):
        raise ValueError("existing output contains duplicate condition/problem/seed rows")

    controller = instrument_reference_backend(
        runner.model,
        PrecisionMap(default=low),
        backend=args.backend,
        calibration_scales=args.calibration_scales,
        materialize_weights=True,
    )
    runner.controller = controller
    total = len(conditions) * len(traces) * len(args.seeds)
    progress = tqdm(
        total=total,
        initial=len(completed),
        desc="selective rescue",
        unit="generation",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    try:
        for condition in conditions:
            active_map = (
                PrecisionMap(default=low)
                if condition.full_precision
                else condition.precision_map
            )
            controller.set_precision_map(active_map)
            runner.kv_bits = (
                16 if condition.full_precision else active_map.default.kv_bits
            )
            runner.kv_group_size = active_map.default.kv_group_size
            runner.kv_symmetric = active_map.default.symmetric_kv
            manager = (
                runner.full_precision()
                if condition.full_precision
                else contextlib.nullcontext()
            )
            with manager:
                for trace in traces:
                    prompt_ids = runner.encode_chat(str(trace["problem"]))
                    for seed in args.seeds:
                        key = (
                            condition.name,
                            str(trace["problem_id"]),
                            int(seed),
                        )
                        if key in completed:
                            continue
                        token_progress = tqdm(
                            total=args.max_new_tokens,
                            desc=(
                                f"{condition.name} "
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
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    max_new_tokens=args.max_new_tokens,
                                    seed=seed,
                                ),
                                progress_callback=(
                                    None
                                    if args.no_progress
                                    else lambda count, _total: token_progress.update(
                                        count - token_progress.n
                                    )
                                ),
                            )
                        finally:
                            token_progress.close()
                        extraction = extract_answer_details(generated["text"])
                        failure = classify_generation(
                            generated["text"],
                            extraction,
                            str(trace["reference_answer"]),
                            truncated=generated["truncated"],
                        )
                        record = {
                            "frontierguard_version": __version__,
                            "run_fingerprint": run_fingerprint,
                            "condition": condition.name,
                            "condition_kind": condition.kind,
                            "problem_id": str(trace["problem_id"]),
                            "seed": int(seed),
                            "correct": bool(failure["correct"]),
                            "output": generated["text"],
                            "extracted_answer": extraction.answer,
                            "extraction": extraction.to_dict(),
                            "failure": failure,
                            "prompt_tokens": int(prompt_ids.shape[-1]),
                            "output_tokens": generated["output_tokens"],
                            "truncated": generated["truncated"],
                            "eos_reached": generated["eos_reached"],
                            "latency_seconds": generated["latency_seconds"],
                            "low_action": asdict(low),
                            "high_action": (
                                asdict(PrecisionAction(16, 16, 16))
                                if condition.full_precision
                                else asdict(high)
                            ),
                            "precision_map": condition.precision_map.to_dict(),
                            "precision_budget": condition.metadata,
                            "quantization": (
                                {"backend": "bf16"}
                                if condition.full_precision
                                else controller.metadata()
                            ),
                        }
                        records.append(record)
                        completed.add(key)
                        write_jsonl(output_path, records)
                        progress.update(1)
                        progress.set_postfix(
                            condition=condition.name,
                            problem=trace["problem_id"],
                            seed=seed,
                            correct=bool(failure["correct"]),
                            refresh=True,
                        )
    finally:
        progress.close()

    current_records = [
        item for item in records if item.get("run_fingerprint") == run_fingerprint
    ]
    if len(current_records) != total:
        raise RuntimeError(
            f"run ended with {len(current_records)}/{total} generation records"
        )
    summary = _summary(
        current_records,
        conditions,
        run_fingerprint=run_fingerprint,
        model=args.model,
        revision=args.revision,
        backend=controller.metadata(),
        traces=args.traces,
        seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        low_name=low_name,
        high_name=high_name,
        selection_provenance=selection_provenance,
    )
    summary_path = args.summary_output or f"{args.output}.summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output_path.resolve()}", flush=True)
    print(f"wrote {Path(summary_path).resolve()}", flush=True)


if __name__ == "__main__":
    main()
