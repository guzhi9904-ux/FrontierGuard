"""Backend protocol for research and deployment quantizers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, Sequence

from frontierguard.schemas import PrecisionAction, PrecisionMap


class QuantBackend(Protocol):
    def prepare(self, model: Any, calibration_data: Any | None = None) -> Any: ...

    def set_precision_map(self, precision_map: PrecisionMap) -> None: ...

    def intervention(
        self,
        module_names: Sequence[str],
        action: PrecisionAction | None = None,
        *,
        disable_quantization: bool = False,
    ) -> AbstractContextManager[None]: ...

    def cost(self, precision_map: PrecisionMap, sequence_length: int) -> dict[str, float]: ...

    def metadata(self) -> dict[str, Any]: ...
