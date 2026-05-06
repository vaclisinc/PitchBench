"""
b5 — Onset/offset detection of every note in a sequence.

Given a sequence of N non-overlapping notes, the prompt asks for every
onset/offset in chronological order. The unified
:func:`cat_b.score_timestamps` is order-preserving and matches GT against
predicted indices: a stimulus is correct iff the model returns the right
COUNT of timestamps (2 × n_notes) AND each is within
``config.BENCHMARK_TIMESTAMP_TOLERANCE_MS`` of the corresponding GT.

Universal IVs: source, source_type, duration_ms.
Experiment-specific IVs: n_notes, rhythm, pitch_pattern.

Usage::
    pitchbench --id b5 --preview
    pitchbench --id b5 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_b import CatBSpec, run_cat_b_experiment
from pitchbench.experiments.helpers.timing_layout import (
    cap_notes_for_total,
    onsets_from_gaps,
    sample_adaptive_gaps,
    stable_cell_seed,
)

EXP_NAME       = Path(__file__).stem
SOURCES        = config.pitchbench_b5_SOURCES
PITCHES        = config.pitchbench_b5_PITCHES
DURATIONS_MS   = config.pitchbench_b5_DURATIONS_MS
N_NOTES_OPTS   = config.pitchbench_b5_N_NOTES_OPTS
RHYTHMS        = config.pitchbench_b5_RHYTHMS
PITCH_PATTERNS = config.pitchbench_b5_PITCH_PATTERNS
TOTAL_DUR_MS   = config.pitchbench_b5_TOTAL_DUR_MS
SEED           = config.pitchbench_b5_SEED

PROMPT = (
    "This audio contains a sequence of musical notes separated by silence. "
    "List the onset and offset of EVERY note in order, as comma-separated "
    "MM:SS.cc timestamps (onset, offset, onset, offset, …). "
    "Example for 2 notes: '0:01.20, 0:02.50, 0:03.10, 0:04.00'. "
    "Reply with the timestamp list only, ordered chronologically."
)


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for n_req in N_NOTES_OPTS:
            for rhythm in RHYTHMS:
                for pattern in PITCH_PATTERNS:
                    for dur in DURATIONS_MS:
                        n = cap_notes_for_total(n_req, dur, TOTAL_DUR_MS)
                        if n < 1:
                            continue

                        cell_seed = stable_cell_seed(SEED, src, n_req, n, rhythm, pattern, dur)
                        sub_rng   = random.Random(cell_seed)
                        if pattern == "fixed_pitch":
                            base  = sub_rng.choice(PITCHES)
                            midis = [base] * n
                        else:
                            if n <= len(PITCHES):
                                midis = sub_rng.sample(PITCHES, n)
                            else:
                                midis = [sub_rng.choice(PITCHES) for _ in range(n)]
                        if rhythm == "regular":
                            total_gap_budget = TOTAL_DUR_MS - dur * n
                            gap_size = total_gap_budget // (n + 1)
                            rem = total_gap_budget - gap_size * (n + 1)
                            gaps = [gap_size] * (n + 1)
                            for i in range(rem):
                                gaps[i] += 1
                        else:
                            gaps = sample_adaptive_gaps(
                                sub_rng,
                                n_notes=n,
                                note_dur_ms=dur,
                                total_dur_ms=TOTAL_DUR_MS,
                            )
                        onsets = onsets_from_gaps(gaps, n_notes=n, note_dur_ms=dur)
                        rows.append({
                            "source":        src,
                            "source_type":   "waveform" if src in config.WAVEFORMS else "instrument",
                            "n_notes":       n,
                            "rhythm":        rhythm,
                            "pitch_pattern": pattern,
                            "duration_ms":   dur,
                            "midi_seq":      midis,
                            "onsets_ms":     onsets,
                            "seed":          cell_seed,
                        })
    return rows


def wav_for(c: dict) -> Path:
    triples = [(m, c["onsets_ms"][i], c["duration_ms"]) for i, m in enumerate(c["midi_seq"])]
    path, _ = engine.clip_with_notes(triples, c["source"], TOTAL_DUR_MS, name_hint="b5")
    return path


def gt_timestamps_for(c: dict) -> list[float]:
    """Flatten every (onset, offset) pair into a single chronological list."""
    out: list[float] = []
    for on_ms in c["onsets_ms"]:
        out.append(on_ms / 1000.0)
        out.append((on_ms + c["duration_ms"]) / 1000.0)
    return out


SPEC = CatBSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="timing",
    prompt=PROMPT,
    gt_timestamps_fn=gt_timestamps_for,
    record_extras=("n_notes", "rhythm", "pitch_pattern", "duration_ms"),
)


def preview() -> None:
    run_cat_b_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_b_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
