"""Deterministic helpers for non-overlapping note timeline generation."""

from __future__ import annotations

import hashlib
import random


def stable_cell_seed(base_seed: int, *parts: object) -> int:
    """Return a deterministic 32-bit seed independent of PYTHONHASHSEED."""
    payload = "|".join([str(base_seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def cap_notes_for_total(requested_n: int, note_dur_ms: int, total_dur_ms: int) -> int:
    """Cap note count so notes can always fit in total duration."""
    if requested_n <= 0 or note_dur_ms <= 0 or total_dur_ms <= 0:
        return 0
    return min(requested_n, total_dur_ms // note_dur_ms)


def sample_adaptive_gaps(
    rng: random.Random,
    n_notes: int,
    note_dur_ms: int,
    total_dur_ms: int,
) -> list[int]:
    """Return n+1 nonnegative gaps whose sum makes the clip fit exactly.

    This ignores configured min/max gap bounds by design.
    """
    if n_notes <= 0:
        raise ValueError("n_notes must be positive")
    remaining = total_dur_ms - (n_notes * note_dur_ms)
    if remaining < 0:
        raise ValueError("Notes exceed total duration")

    slots = n_notes + 1
    gaps: list[int] = []
    rem = remaining
    for _ in range(slots - 1):
        g = rng.randint(0, rem)
        gaps.append(g)
        rem -= g
    gaps.append(rem)
    rng.shuffle(gaps)
    return gaps


def onsets_from_gaps(gaps: list[int], n_notes: int, note_dur_ms: int) -> list[int]:
    """Compute note onsets from gap schedule and fixed note duration."""
    onsets: list[int] = []
    cursor = gaps[0]
    for i in range(n_notes):
        onsets.append(cursor)
        cursor += note_dur_ms + gaps[i + 1]
    return onsets
