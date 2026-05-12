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


_TRACEBACK_SHOW_LIMIT = 3   # show full traceback for first N failures only


def dispatch(
    items: Sequence[T],
    worker_fn: Callable[[T], R],
    *,
    model_name: str,
    label_fn: Callable[[T], str] | None = None,
    result_label_fn: Callable[[T, R], str] | None = None,
    max_workers: int | None = None,
    bail_on_all_fail_after: int = 3,
) -> list[R | None]:
    """Run ``worker_fn(item)`` for each item with bounded concurrency.

    Args:
        items:                  sequence of work units.
        worker_fn:              called once per item; must not print.
        model_name:             used to size the pool via ``config.concurrency_for``.
        label_fn:               ``item -> str`` for the per-completion log line.
        result_label_fn:        ``(item, result) -> str`` — replaces ``label_fn``
                                on success for per-stimulus audit lines.
        max_workers:            override the per-model concurrency cap.
        bail_on_all_fail_after: abort if this many completed items have ALL failed
                                (systemic server failure). 0 or None disables.

    Returns:
        list of results in INPUT order. Items whose worker raised return ``None``.
    """
    n = len(items)
    if n == 0:
        return []

    cap = max_workers if max_workers is not None else config.concurrency_for(model_name)
    cap = max(1, min(cap, n))

    print(f"    [dispatch] {n} items, max_workers={cap}")

    results: list[R | None] = [None] * n
    completed  = 0
    failures   = 0
    aborted    = False

    with ThreadPoolExecutor(max_workers=cap) as pool:
        future_to_idx: dict[Future, int] = {
            pool.submit(worker_fn, item): i for i, item in enumerate(items)
        }
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
                failures += 1
                print(f"    [{completed:>4d}/{n}] {base_label}  *** FAILED: {e!r}")
                if failures <= _TRACEBACK_SHOW_LIMIT:
                    traceback.print_exc()
                elif failures == _TRACEBACK_SHOW_LIMIT + 1:
                    print("    (further tracebacks suppressed — same error pattern)")
                results[idx] = None

                # Systemic-failure bail-out: if every completed item so far has
                # failed and we've passed the sample threshold, the server is
                # almost certainly down.  Cancel pending futures and stop.
                if (bail_on_all_fail_after
                        and failures == completed >= bail_on_all_fail_after):
                    remaining = n - completed
                    print(
                        f"\n    [ABORT] All {completed} completed items failed — "
                        f"looks like a systemic server error.\n"
                        f"    Cancelling {remaining} remaining item(s).\n"
                        f"    Last error: {e}"
                    )
                    for f2 in future_to_idx:
                        f2.cancel()
                    aborted = True
                    break

    if aborted:
        print(f"    [dispatch] Aborted after {completed}/{n} items ({failures} failures).")

    return results
