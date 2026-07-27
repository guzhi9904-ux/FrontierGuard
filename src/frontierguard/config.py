"""YAML configuration loading with explicit recursive overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {source}")
    return value


def load_experiment(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Load an experiment and its optional `includes` list.

    Includes are resolved relative to ``root`` when supplied, otherwise relative
    to the experiment file. Later includes and the experiment itself win.
    """

    source = Path(path).resolve()
    config = load_yaml(source)
    include_root = Path(root).resolve() if root else source.parent
    merged: dict[str, Any] = {}
    for include in config.pop("includes", []):
        include_path = include_root / include
        merged = deep_merge(merged, load_yaml(include_path))
    return deep_merge(merged, config)
