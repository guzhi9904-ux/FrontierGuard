"""Small helpers for auditable experiment names and configuration fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from frontierguard.schemas import PrecisionAction


_PRECISION_IN_NAME = re.compile(r"w(\d+)a(\d+)kv(\d+)", re.IGNORECASE)


def precision_label(action: PrecisionAction) -> str:
    return (
        f"w{action.weight_bits}"
        f"a{action.activation_bits}"
        f"kv{action.kv_bits}"
    )


def validate_output_precision_label(path: str, action: PrecisionAction) -> None:
    """Reject misleading output names while allowing generic filenames."""

    match = _PRECISION_IN_NAME.search(Path(path).name)
    if match is None:
        return
    observed = tuple(int(item) for item in match.groups())
    expected = (action.weight_bits, action.activation_bits, action.kv_bits)
    if observed != expected:
        raise ValueError(
            f"output filename encodes W{observed[0]}A{observed[1]}KV{observed[2]} "
            f"but the requested action is W{expected[0]}A{expected[1]}KV{expected[2]}"
        )


def stable_fingerprint(value: Any, *, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
