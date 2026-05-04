"""Stratified sampling helpers for PitchBench experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable


def stratified_sample(
    items: list[dict],
    n: int,
    key_fn: Callable[[dict], Any],
    seed: int = 42,
) -> list[dict]:
    """Draw n items from items, stratified by key_fn(item).

    Allocation per stratum: floor(n / num_strata), with the remainder
    distributed one-extra to strata in sorted-key order.

    Raises ValueError if any stratum has fewer items than its quota.
    Returns items unchanged if n >= len(items).
    """
    if n >= len(items):
        return list(items)

    groups: dict[Any, list[dict]] = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    strata = sorted(groups.keys(), key=str)
    k = len(strata)
    base = n // k
    remainder = n % k

    rng = random.Random(seed)
    selected: list[dict] = []
    for i, key in enumerate(strata):
        quota = base + (1 if i < remainder else 0)
        pool = groups[key]
        if len(pool) < quota:
            raise ValueError(
                f"Stratum {key!r} has only {len(pool)} items but needs {quota}; "
                f"reduce --sample-n or pick a different stratification."
            )
        selected.extend(rng.sample(pool, quota))

    return selected


def sampling_meta(
    total: int,
    stratified_by: str,
    sample_n: int | None,
    sample_seed: int,
) -> dict[str, Any]:
    """Return the sampling fields to embed in run metadata."""
    return {
        "sample_n":        sample_n,
        "sample_seed":     sample_seed,
        "total_available": total,
        "stratified_by":   stratified_by,
    }


def sampling_summary_lines(meta: dict[str, Any]) -> list[str]:
    """Human-readable sampling lines for the .txt summary."""
    if meta.get("sample_n") is None:
        return [f"  Sampling     : none (full set of {meta['total_available']})"]
    return [
        f"  Sampling     : {meta['sample_n']} of {meta['total_available']} "
        f"(stratified by {meta['stratified_by']!r}, seed={meta['sample_seed']})",
    ]
