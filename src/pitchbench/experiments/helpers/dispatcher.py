"""
Bounded-concurrency dispatcher for ALM HTTP queries.

One experiment's stimulus loop submits work here; the dispatcher runs
``worker_fn`` in a thread pool whose size comes from
``config.concurrency_for(model_name)`` (env-overridable). Results are returned
in **input order** so result CSVs stay byte-identical to a sequential run.

Workers MUST NOT print — the dispatcher emits one ``[i/N]`` line per completion
so logs don't interleave. Pass ``verbose=False`` to ``query_four_formats`` etc.

Failures inside a worker are logged and recorded as ``None`` in the returned
list; callers filter them out (matching the ``[SKIP]`` semantics scripts
already use for engine.tone() failures).

Sub-experiments stay sequential: ``dispatch()`` blocks until every future
resolves before returning, so the next ``run_one_model()`` / next experiment
can't start until the current one is fully done.
"""

from __future__ import annotations

import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

import pitchbench.config as config

T = TypeVar("T")
R = TypeVar("R")


def dispatch(
    items: Sequence[T],
    worker_fn: Callable[[T], R],
    *,
    model_name: str,
    label_fn: Callable[[T], str] | None = None,
    result_label_fn: Callable[[T, R], str] | None = None,
    max_workers: int | None = None,
) -> list[R | None]:
    """Run ``worker_fn(item)`` for each item with bounded concurrency.

    Args:
        items:           sequence of work units (typically dicts describing a stimulus).
        worker_fn:       function called once per item; returns the per-item record.
                         Must not print; must not share mutable state.
        model_name:      used to size the pool via ``config.concurrency_for``.
        label_fn:        ``item -> short str`` for the per-completion log line.
        result_label_fn: ``(item, result) -> str`` — when set, replaces ``label_fn``
                         for the success log line so the per-stimulus audit
                         (gt/pred/correct) can be printed alongside ``[i/N]``.
        max_workers:     override the per-model cap (rare; e.g. tests).

    Returns:
        list of results in INPUT order. Items whose worker raised return ``None``;
        callers should filter these out before aggregation.
    """
    n = len(items)
    if n == 0:
        return []

    cap = max_workers if max_workers is not None else config.concurrency_for(model_name)
    cap = max(1, min(cap, n))

    print(f"    [dispatch] {n} items, max_workers={cap}")

    results: list[R | None] = [None] * n
    completed = 0

    with ThreadPoolExecutor(max_workers=cap) as pool:
        future_to_idx: dict[Future, int] = {
            pool.submit(worker_fn, item): i for i, item in enumerate(items)
        }
        # Iterate in completion order for live progress, but write into the
        # input-indexed slot so the returned list stays deterministic.
        from concurrent.futures import as_completed
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            base_label = label_fn(items[idx]) if label_fn else f"item {idx}"
            completed += 1
            try:
                result = fut.result()
                results[idx] = result
                line = (result_label_fn(items[idx], result)
                        if result_label_fn is not None else base_label)
                print(f"    [{completed:>4d}/{n}] {line}")
            except Exception as e:
                print(f"    [{completed:>4d}/{n}] {base_label}  *** FAILED: {e!r}")
                traceback.print_exc()
                results[idx] = None

    return results
