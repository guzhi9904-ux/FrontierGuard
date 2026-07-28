"""Helpers for auditable selective-precision rescue experiments."""

from __future__ import annotations

import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from frontierguard.evaluation.statistics import (
    PairedObservation,
    binomial_wilson,
    paired_problem_bootstrap,
)
from frontierguard.models.adapters import ModuleDescriptor
from frontierguard.schemas import PrecisionAction, PrecisionMap


_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SELECTOR = re.compile(r"^(\d+)(?:\.(attention|mlp))?$")


@dataclass(frozen=True)
class ModuleSelector:
    layer_index: int
    family: str | None = None

    @property
    def label(self) -> str:
        suffix = f".{self.family}" if self.family else ""
        return f"{self.layer_index}{suffix}"


@dataclass(frozen=True)
class RescueSpec:
    name: str
    selectors: tuple[ModuleSelector, ...]

    @property
    def label(self) -> str:
        return f"{self.name}={','.join(item.label for item in self.selectors)}"


def parse_rescue_spec(value: str) -> RescueSpec:
    """Parse ``NAME=LAYER[.FAMILY],...`` into a validated rescue spec."""

    separator = "=" if "=" in value else ":"
    name, found, selector_text = value.partition(separator)
    if not found or not name or not selector_text or not _NAME.fullmatch(name):
        raise ValueError(
            "rescue must be NAME=LAYER[.attention|.mlp][,...]; "
            f"got {value!r}"
        )
    selectors = []
    for token in selector_text.split(","):
        match = _SELECTOR.fullmatch(token.strip().lower())
        if match is None:
            raise ValueError(
                "rescue selectors must be LAYER, LAYER.attention or LAYER.mlp; "
                f"got {token!r}"
            )
        selectors.append(
            ModuleSelector(
                layer_index=int(match.group(1)),
                family=match.group(2),
            )
        )
    unique = tuple(dict.fromkeys(selectors))
    if len(unique) != len(selectors):
        raise ValueError(f"rescue {name!r} repeats a selector")
    return RescueSpec(name=name, selectors=unique)


def select_module_names(
    descriptors: Sequence[ModuleDescriptor],
    selectors: Sequence[ModuleSelector],
) -> tuple[str, ...]:
    """Expand layer/family selectors to exact linear-module names."""

    available_layers = {item.layer_index for item in descriptors}
    requested_layers = {item.layer_index for item in selectors}
    missing = sorted(requested_layers - available_layers)
    if missing:
        raise ValueError(f"rescue references unavailable layers: {missing}")
    selected = [
        item.name
        for item in descriptors
        if any(
            item.layer_index == selector.layer_index
            and (selector.family is None or item.family == selector.family)
            for selector in selectors
        )
    ]
    if not selected:
        raise ValueError("rescue selectors matched no quantized linear modules")
    return tuple(sorted(selected))


def build_precision_map(
    descriptors: Sequence[ModuleDescriptor],
    spec: RescueSpec,
    *,
    low: PrecisionAction,
    high: PrecisionAction,
) -> tuple[PrecisionMap, dict]:
    """Construct a module-exact precision map and parameter-budget metadata."""

    names = select_module_names(descriptors, spec.selectors)
    selected = set(names)
    total_parameters = sum(item.parameter_count for item in descriptors)
    selected_parameters = sum(
        item.parameter_count for item in descriptors if item.name in selected
    )
    fraction = selected_parameters / total_parameters if total_parameters else 0.0
    effective_bits = (
        (
            (total_parameters - selected_parameters) * low.weight_bits
            + selected_parameters * high.weight_bits
        )
        / total_parameters
        if total_parameters
        else 0.0
    )
    precision_map = PrecisionMap(
        default=low,
        modules={name: high for name in names},
        metadata={
            "rescue_spec": spec.label,
            "selected_module_count": len(names),
            "selected_parameter_count": selected_parameters,
            "instrumented_parameter_count": total_parameters,
            "high_precision_parameter_fraction": fraction,
            "effective_weight_bits": effective_bits,
        },
    )
    return precision_map, dict(precision_map.metadata)


def random_layer_specs(
    layer_indices: Iterable[int],
    *,
    budget: int,
    count: int,
    seed: int,
    prefix: str = "random",
) -> list[RescueSpec]:
    """Create deterministic, unique, equal-layer-budget random controls."""

    layers = sorted(set(layer_indices))
    if budget <= 0 or budget > len(layers):
        raise ValueError("random layer budget must lie within the model layer count")
    if count < 0:
        raise ValueError("random map count cannot be negative")
    maximum = 1
    for index in range(budget):
        maximum = maximum * (len(layers) - index) // (index + 1)
    if count > maximum:
        raise ValueError(
            f"requested {count} unique random maps but only {maximum} combinations exist"
        )
    rng = random.Random(seed)
    combinations: set[tuple[int, ...]] = set()
    while len(combinations) < count:
        combinations.add(tuple(sorted(rng.sample(layers, budget))))
    return [
        RescueSpec(
            name=f"{prefix}_{index:02d}",
            selectors=tuple(ModuleSelector(layer) for layer in combination),
        )
        for index, combination in enumerate(sorted(combinations))
    ]


def summarize_generation_condition(rows: Sequence[dict]) -> dict:
    """Summarize strict generation outcomes for one precision condition."""

    if not rows:
        raise ValueError("cannot summarize an empty generation condition")
    successes = sum(bool(row["correct"]) for row in rows)
    trials = len(rows)
    lower, upper = binomial_wilson(successes, trials)
    failure_types = Counter(row["failure"]["failure_type"] for row in rows)
    return {
        "problems": len({str(row["problem_id"]) for row in rows}),
        "trials": trials,
        "successes": successes,
        "accuracy": successes / trials,
        "accuracy_wilson_95": [lower, upper],
        "truncation_fraction": sum(bool(row["truncated"]) for row in rows) / trials,
        "mean_output_tokens": statistics.fmean(
            int(row["output_tokens"]) for row in rows
        ),
        "mean_repetition_fraction": statistics.fmean(
            float(row["failure"]["repetition_fraction"]) for row in rows
        ),
        "failure_types": dict(sorted(failure_types.items())),
    }


def paired_success_lift(
    baseline_rows: Sequence[dict],
    method_rows: Sequence[dict],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict:
    """Return paired success lift using identical problem/seed generations."""

    def keyed(rows: Sequence[dict]) -> dict[tuple[str, int], dict]:
        result = {}
        for row in rows:
            key = (str(row["problem_id"]), int(row["seed"]))
            if key in result:
                raise ValueError(f"duplicate generation key: {key}")
            result[key] = row
        return result

    baseline = keyed(baseline_rows)
    method = keyed(method_rows)
    if baseline.keys() != method.keys():
        missing_method = sorted(baseline.keys() - method.keys())
        missing_baseline = sorted(method.keys() - baseline.keys())
        raise ValueError(
            "paired conditions have different problem/seed coverage: "
            f"missing_method={missing_method[:3]}, "
            f"missing_baseline={missing_baseline[:3]}"
        )
    effects = []
    observations = []
    for problem_id, seed in sorted(baseline):
        low = float(bool(baseline[(problem_id, seed)]["correct"]))
        high = float(bool(method[(problem_id, seed)]["correct"]))
        effects.append(
            {
                "problem_id": problem_id,
                "seed": seed,
                "baseline": bool(low),
                "method": bool(high),
                "delta": high - low,
            }
        )
        observations.append(PairedObservation(problem_id, low, high))
    bootstrap = paired_problem_bootstrap(
        observations,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "estimate": bootstrap.estimate,
        "ci_95": [bootstrap.lower, bootstrap.upper],
        "bootstrap_samples": bootstrap.samples,
        "problems": bootstrap.problems,
        "paired_trials": len(effects),
        "improved": sum(item["delta"] > 0 for item in effects),
        "regressed": sum(item["delta"] < 0 for item in effects),
        "unchanged": sum(item["delta"] == 0 for item in effects),
        "seed_effects": effects,
    }
