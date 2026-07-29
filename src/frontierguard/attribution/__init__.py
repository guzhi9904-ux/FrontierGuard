"""Module-level causal rescue measurement."""

from frontierguard.attribution.rescue import RescueObservation, aggregate_rescue

from frontierguard.attribution.patching import (
    DamageAttributionResult,
    DamageMeasurement,
    measure_compression_damage,
)

__all__ = [
    "DamageAttributionResult",
    "DamageMeasurement",
    "RescueObservation",
    "aggregate_rescue",
    "measure_compression_damage",
]
