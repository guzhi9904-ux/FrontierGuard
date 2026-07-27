"""Build deployable precision maps from allocator output."""

from __future__ import annotations

from frontierguard.schemas import PrecisionAction, PrecisionMap


def precision_map_from_selection(
    selected: list[str] | tuple[str, ...],
    *,
    low_action: PrecisionAction,
    high_action: PrecisionAction,
    metadata: dict | None = None,
) -> PrecisionMap:
    return PrecisionMap(
        default=low_action,
        modules={name: high_action for name in selected},
        metadata=metadata or {},
    )
