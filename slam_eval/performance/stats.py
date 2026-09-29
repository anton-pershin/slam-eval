"""Extensible registry of aggregate statistics (spec 04, FR8).

A statistic is a function mapping a list of non-null numbers to a number.
Quantiles are parameterized: ``qN`` means the N-th percentile (``q5`` = 5th).
Adding a new non-quantile statistic requires one registry entry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Union

Number = Union[int, float]

StatsFunc = Callable[[list[Number]], Number]


def _median(values: list[Number]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


_STATS: dict[str, StatsFunc] = {
    "min": min,
    "max": max,
    "mean": lambda values: sum(values) / len(values),
    "median": _median,
}


def register_stat(name: str, func: StatsFunc) -> None:
    """Register a custom non-quantile statistic."""
    _STATS[name] = func


def resolve_stat(name: str) -> StatsFunc:
    """Resolve a configured statistic name (e.g. ``q95``) to a function."""
    if name in _STATS:
        return _STATS[name]
    match = re.fullmatch(r"q(\d+)", name)
    if match is None:
        raise ValueError(
            f"Unknown statistic {name!r}; expected min/max/mean/median or qN"
        )
    percentile = int(match.group(1))
    if not 0 <= percentile <= 100:
        raise ValueError(f"Quantile percentile must be in [0, 100], got {percentile}")

    def quantile(values: list[Number], p: float = percentile / 100.0) -> Number:
        ordered = sorted(values)
        if not ordered:
            raise ValueError("Cannot compute a quantile of an empty list")
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        frac = pos - lower
        return ordered[lower] * (1.0 - frac) + ordered[upper] * frac

    return quantile


def aggregate(
    values: list[Number | None], stats: list[str]
) -> dict[str, Number | int | None]:
    """Compute the configured statistics over non-null values.

    Each aggregate reports ``n`` -- the number of non-null values used. With
    no non-null value the statistic entries are None (the caller stores the
    block with ``n: 0`` per FR5 rather than omitting it).
    """
    non_null = [v for v in values if v is not None]
    result: dict[str, Number | int | None] = {"n": len(non_null)}
    for name in stats:
        if not non_null:
            result[name] = None
        else:
            result[name] = resolve_stat(name)(non_null)
    return result
