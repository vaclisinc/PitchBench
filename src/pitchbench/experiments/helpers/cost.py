"""
Process-global LLM usage / cost tracker.

OpenRouter calls (issued from ``helpers.api._openrouter_text``) record their
token counts and USD cost here, bucketed by model_name. ``results.save_results``
reads the per-model bucket and embeds it in every result file; ``make_run_dir``
calls :func:`reset` so each experiment run starts with a clean slate.

Local-server models do not record anything (no cost data available), so their
result files simply omit the LLM USAGE block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass
class _ModelUsage:
    calls:             int   = 0
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cached_tokens:     int   = 0
    cost_usd:          float = 0.0


_lock: Lock                              = Lock()
_per_model: dict[str, _ModelUsage]       = {}


def record(
    model_name: str,
    *,
    prompt_tokens:     int   = 0,
    completion_tokens: int   = 0,
    total_tokens:      int   = 0,
    cached_tokens:     int   = 0,
    cost_usd:          float = 0.0,
) -> None:
    """Add a single call's usage to the running totals for ``model_name``."""
    with _lock:
        u = _per_model.setdefault(model_name, _ModelUsage())
        u.calls             += 1
        u.prompt_tokens     += int(prompt_tokens or 0)
        u.completion_tokens += int(completion_tokens or 0)
        u.total_tokens      += int(total_tokens or (prompt_tokens + completion_tokens) or 0)
        u.cached_tokens     += int(cached_tokens or 0)
        u.cost_usd          += float(cost_usd or 0.0)


def get(model_name: str) -> dict[str, Any]:
    """Return cumulative usage for one model (zeroed dict if never recorded)."""
    with _lock:
        u = _per_model.get(model_name) or _ModelUsage()
        d = asdict(u)
    d["cost_usd"] = round(d["cost_usd"], 6)
    return d


def all_totals() -> dict[str, dict[str, Any]]:
    """Snapshot of every model's cumulative usage."""
    with _lock:
        items = list(_per_model.items())
    out: dict[str, dict[str, Any]] = {}
    for name, u in items:
        d = asdict(u)
        d["cost_usd"] = round(d["cost_usd"], 6)
        out[name] = d
    return out


def reset(model_name: str | None = None) -> None:
    """Clear all buckets (default) or just one model's bucket."""
    with _lock:
        if model_name is None:
            _per_model.clear()
        else:
            _per_model.pop(model_name, None)


def parse_openrouter_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise an OpenRouter ``usage`` block to the keys :func:`record` accepts.

    OpenRouter (with ``usage: {include: true}``) returns the OpenAI shape plus a
    ``cost`` field in USD. ``prompt_tokens_details.cached_tokens`` reports the
    portion of the prompt served from cache, when available.
    """
    if not isinstance(usage, dict):
        return {}
    details = usage.get("prompt_tokens_details") or {}
    cached  = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    return {
        "prompt_tokens":     usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "total_tokens":      usage.get("total_tokens", 0) or 0,
        "cached_tokens":     cached or 0,
        "cost_usd":          float(usage.get("cost", 0.0) or 0.0),
    }
