"""Helpers for auditable selective-precision rescue experiments."""

from __future__ import annotations

import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from frontierguard.attribution.stability import aggregate_damage_rows
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


@dataclass(frozen=True)
class ModuleRescueSpec:
    """A static rescue condition defined by exact projection names."""

    name: str
    module_names: tuple[str, ...]
    metadata: dict


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
    return build_module_precision_map(
        descriptors,
        ModuleRescueSpec(
            name=spec.name,
            module_names=names,
            metadata={"rescue_spec": spec.label, "selection_method": "manual_layer"},
        ),
        low=low,
        high=high,
    )


def build_module_precision_map(
    descriptors: Sequence[ModuleDescriptor],
    spec: ModuleRescueSpec,
    *,
    low: PrecisionAction,
    high: PrecisionAction,
) -> tuple[PrecisionMap, dict]:
    """Construct a precision map from exact projection names."""

    if not spec.name or not _NAME.fullmatch(spec.name):
        raise ValueError(f"invalid module rescue name: {spec.name!r}")
    names = tuple(dict.fromkeys(spec.module_names))
    if not names:
        raise ValueError("module rescue must select at least one projection")
    available = {item.name for item in descriptors}
    unknown = sorted(set(names) - available)
    if unknown:
        raise ValueError(f"module rescue references unavailable modules: {unknown[:3]}")
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
            **spec.metadata,
            "rescue_name": spec.name,
            "selected_modules": list(names),
            "selected_module_count": len(names),
            "selected_parameter_count": selected_parameters,
            "instrumented_parameter_count": total_parameters,
            "high_precision_parameter_fraction": fraction,
            "effective_weight_bits": effective_bits,
        },
    )
    return precision_map, dict(precision_map.metadata)


def rank_damage_modules(
    rows: Sequence[dict],
    descriptors: Sequence[ModuleDescriptor],
    *,
    score_field: str = "predicted_nll_rescue",
    minimum_problem_fraction: float = 0.5,
    minimum_positive_fraction: float = 0.5,
    require_positive_ci: bool = False,
    bootstrap_samples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Rank projections by problem-level frontier damage.

    Repeated trace seeds are averaged within a problem before ranking. Coverage
    and sign-consistency filters prevent a module seen on only a few examples
    from becoming a global precision exception.
    """

    if not rows:
        raise ValueError("damage score input is empty")
    if not 0 < minimum_problem_fraction <= 1:
        raise ValueError("minimum problem fraction must lie in (0, 1]")
    if not 0 <= minimum_positive_fraction <= 1:
        raise ValueError("minimum positive fraction must lie in [0, 1]")
    available = {item.name for item in descriptors}
    observed = {str(row["module_name"]) for row in rows}
    unknown = sorted(observed - available)
    if unknown:
        raise ValueError(f"damage scores reference unavailable modules: {unknown[:3]}")
    usable = [
        row
        for row in rows
        if row.get(score_field) is not None
        and np.isfinite(float(row[score_field]))
    ]
    if not usable:
        raise ValueError(f"damage scores contain no finite {score_field!r} values")
    total_problems = len({str(row["problem_id"]) for row in usable})
    ranking = aggregate_damage_rows(
        usable,
        key=lambda row: str(row["module_name"]),
        score_field=score_field,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    by_name = {item.name: item for item in descriptors}
    selected = []
    for item in ranking:
        coverage = item["problems"] / total_problems
        if coverage < minimum_problem_fraction:
            continue
        if item["mean"] <= 0:
            continue
        if item["positive_fraction"] < minimum_positive_fraction:
            continue
        if require_positive_ci and item["ci_lower"] <= 0:
            continue
        descriptor = by_name[item["key"]]
        selected.append(
            {
                **item,
                "problem_coverage_fraction": coverage,
                "score_field": score_field,
                "family": descriptor.family,
                "projection": descriptor.projection,
                "layer_index": descriptor.layer_index,
                "parameter_count": descriptor.parameter_count,
            }
        )
    return selected


def matched_random_module_specs(
    descriptors: Sequence[ModuleDescriptor],
    selected_names: Sequence[str],
    *,
    count: int,
    seed: int,
    prefix: str,
) -> list[ModuleRescueSpec]:
    """Create deterministic random controls matched on projection families."""

    if count < 0:
        raise ValueError("random module map count cannot be negative")
    selected = tuple(dict.fromkeys(selected_names))
    if not selected:
        raise ValueError("matched random controls need selected modules")
    by_name = {item.name: item for item in descriptors}
    unknown = sorted(set(selected) - by_name.keys())
    if unknown:
        raise ValueError(f"selected modules are unavailable: {unknown[:3]}")
    excluded = set(selected)
    pools: dict[str, list[str]] = defaultdict(list)
    family_pools: dict[str, list[str]] = defaultdict(list)
    for item in descriptors:
        if item.name in excluded:
            continue
        pools[item.projection].append(item.name)
        family_pools[item.family].append(item.name)

    rng = random.Random(seed)
    combinations: set[tuple[str, ...]] = set()
    attempts = 0
    maximum_attempts = max(1000, count * 200)
    while len(combinations) < count and attempts < maximum_attempts:
        attempts += 1
        chosen: list[str] = []
        for name in selected:
            descriptor = by_name[name]
            candidates = [
                candidate
                for candidate in pools[descriptor.projection]
                if candidate not in chosen
            ]
            if not candidates:
                candidates = [
                    candidate
                    for candidate in family_pools[descriptor.family]
                    if candidate not in chosen
                ]
            if not candidates:
                break
            chosen.append(rng.choice(candidates))
        if len(chosen) == len(selected):
            combinations.add(tuple(sorted(chosen)))
    if len(combinations) < count:
        raise ValueError(
            f"could create only {len(combinations)} of {count} unique matched controls"
        )
    return [
        ModuleRescueSpec(
            name=f"{prefix}_random_{index:02d}",
            module_names=combination,
            metadata={
                "selection_method": "matched_random_module",
                "matched_condition": prefix,
                "matched_to": list(selected),
            },
        )
        for index, combination in enumerate(sorted(combinations))
    ]


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
