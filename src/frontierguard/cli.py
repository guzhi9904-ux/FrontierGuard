"""FrontierGuard command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from frontierguard.allocation.export import precision_map_from_selection
from frontierguard.allocation.greedy import additive_greedy
from frontierguard.config import load_experiment
from frontierguard.io import read_json, write_json
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import PrecisionAction, PrecisionMap
from frontierguard.utils.environment import audit_environment


def _audit(args: argparse.Namespace) -> int:
    value = audit_environment()
    if args.output:
        write_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def _show_config(args: argparse.Namespace) -> int:
    value = load_experiment(args.path, root=args.root)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _smoke_quant(_: argparse.Namespace) -> int:
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).eval()
    precision_map = PrecisionMap(default=PrecisionAction(4, 4, 16, weight_group_size=4))
    controller = instrument_linear_layers(model, precision_map, exclude=None)
    inputs = torch.randn(3, 8)
    quantized = model(inputs)
    with controller.disabled():
        full_precision = model(inputs)
    delta = float((quantized - full_precision).abs().mean().item())
    if not delta > 0:
        raise RuntimeError("fake quant smoke test did not perturb the model")
    print(json.dumps({"modules": controller.module_names, "mean_abs_delta": delta}, indent=2))
    return 0


def _solve_map(args: argparse.Namespace) -> int:
    payload = read_json(args.scores)
    scores = {name: float(value["score"]) for name, value in payload["modules"].items()}
    costs = {name: float(value["cost_bytes"]) for name, value in payload["modules"].items()}
    result = additive_greedy(scores, costs, args.budget_bytes)
    low = PrecisionAction(**payload.get("low_action", {}))
    high = PrecisionAction(**payload.get("high_action", {"weight_bits": 8, "activation_bits": 8}))
    precision_map = precision_map_from_selection(
        list(result.selected),
        low_action=low,
        high_action=high,
        metadata={"utility": result.utility, "cost_bytes": result.cost},
    )
    write_json(args.output, precision_map.to_dict())
    print(f"selected {len(result.selected)} modules; wrote {Path(args.output).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frontierguard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-env", help="record the software and GPU environment")
    audit.add_argument("--output")
    audit.set_defaults(function=_audit)

    show = subparsers.add_parser("show-config", help="resolve an experiment YAML")
    show.add_argument("path")
    show.add_argument("--root")
    show.set_defaults(function=_show_config)

    smoke = subparsers.add_parser("smoke-quant", help="run a CPU fake-quant smoke test")
    smoke.set_defaults(function=_smoke_quant)

    solve = subparsers.add_parser("solve-map", help="solve an additive precision-map budget")
    solve.add_argument("scores", help="JSON with module scores and byte costs")
    solve.add_argument("--budget-bytes", type=float, required=True)
    solve.add_argument("--output", required=True)
    solve.set_defaults(function=_solve_map)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
