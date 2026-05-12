"""Stratified sampling helpers for PitchBench experiments."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pitchbench.config as config


def stratified_sample(
    items: list[dict[str, Any]],
    n: int,
    key_fn: Callable[[dict[str, Any]], Any],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Draw n items from items, stratified by key_fn(item).

    Allocation per stratum: floor(n / num_strata), with the remainder
    distributed one-extra to strata in sorted-key order.

    Raises ValueError if any stratum has fewer items than its quota.
    Returns items unchanged if n >= len(items).
    """
    if n >= len(items):
        return list(items)

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    strata = sorted(groups.keys(), key=str)
    k = len(strata)
    base = n // k
    remainder = n % k

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
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
    """Human-readable sampling lines for the .txt summary.

    Tolerant of partial/empty ``meta`` dicts — callers (notably tests that
    bypass ``apply_default_sampling``) sometimes pass ``{}``.
    """
    total = meta.get("total_available", "?")
    if meta.get("sample_n") is None:
        return [f"  Sampling     : none (full set of {total})"]
    return [
        f"  Sampling     : {meta['sample_n']} of {total} "
        f"(stratified by {meta.get('stratified_by', '?')!r}, "
        f"seed={meta.get('sample_seed', '?')})",
    ]


def _anchor_filter(
    all_conds: list[dict[str, Any]],
    fixed_vars: tuple[str, ...],
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically pick one value for each fixed variable and return all rows
    that match, effectively "anchoring" those variables to a single value across
    all conditions.  The chosen anchor values are printed for reproducibility.
    """
    if not fixed_vars:
        return list(all_conds)
    rng = random.Random(seed)
    fixed_values: dict[str, Any] = {}
    for var in fixed_vars:
        unique_vals = sorted({c[var] for c in all_conds}, key=str)
        fixed_values[var] = rng.choice(unique_vals)
    filtered = [c for c in all_conds if all(c[v] == fixed_values[v] for v in fixed_vars)]
    print(f"  [sampling] anchor: {fixed_values}  → {len(filtered)} of {len(all_conds)} conditions")
    return filtered


def apply_default_sampling(
    exp_name: str,
    all_conds: list[dict[str, Any]],
    cli_sample_n: int | None,
    cli_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve sample size + strata key from ``config.EXPERIMENT_DEFAULTS`` and apply.

    Sizing model:
      - The config field ``"per_stratum"`` is samples *per stratum cell* — total
        drawn = per_stratum × num_distinct_strata_keys, computed from the
        actual condition list.
      - ``per_stratum=None`` → run the full grid.
      - CLI ``--sample-n N`` (cli_sample_n is not None) overrides as a *total*
        cap (legacy semantics, useful for quick pipeline checks).

    Strata key always comes from ``EXPERIMENT_DEFAULTS[exp_name]["strata"]`` —
    a tuple of condition-dict field names. The lambda is built from that spec
    so dev-mode and paper-mode share the same strata axis.
    Falls back to ``("source",)`` if the experiment is missing from the config.

    If the spec has a ``"fixed_vars"`` tuple, those variables are anchored to a
    single seed-deterministic value *before* stratified sampling, so all sampled
    conditions share the same value for those variables (e.g. same midi pitch and
    same source across all levels of the main independent variable).
    """
    spec   = config.EXPERIMENT_DEFAULTS.get(exp_name, {})
    fields: tuple[str, ...] = spec.get("strata") or ("source",)
    per_stratum = spec.get("per_stratum")

    # Anchor-filter: hold fixed_vars constant across sampled conditions.
    fixed_vars: tuple[str, ...] = tuple(spec.get("fixed_vars") or ())
    if fixed_vars:
        all_conds = _anchor_filter(all_conds, fixed_vars, cli_seed)

    key_fn: Callable[[dict[str, Any]], Any] = lambda c, fs=fields: tuple(c[f] for f in fs)
    strata_label = "(" + ", ".join(fields) + ")" if len(fields) > 1 else fields[0]

    if cli_sample_n is not None:
        n = cli_sample_n
    elif per_stratum is None:
        n = None
    else:
        num_strata = len({key_fn(c) for c in all_conds})
        n = per_stratum * num_strata

    if n is None or n >= len(all_conds):
        return all_conds, sampling_meta(len(all_conds), strata_label, None, cli_seed)

    sampled = stratified_sample(all_conds, n, key_fn, seed=cli_seed)
    return sampled, sampling_meta(len(all_conds), strata_label, n, cli_seed)


def export_sampled_conditions_csv(exp_name: str, conds: list[dict[str, Any]]) -> Path:
    """Write sampled condition rows to data/audio/<exp_name>/content_<exp_name>.csv."""
    out_dir = config.AUDIO_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"_content_{exp_name}.csv"

    headers = sorted({k for c in conds for k in c.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for c in conds:
            row: dict[str, Any] = {}
            for k in headers:
                v = c.get(k)
                if isinstance(v, (dict, list, tuple, set)):
                    row[k] = json.dumps(v, ensure_ascii=True, separators=(",", ":"))
                else:
                    row[k] = v
            writer.writerow(row)
    return out_path
