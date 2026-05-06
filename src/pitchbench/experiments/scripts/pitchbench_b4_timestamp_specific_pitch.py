"""
b3 — Onset/offset detection of a *named* note among distractors.

Given a sequence of N+1 non-overlapping notes, the prompt names one
target by note name and asks for its onset and offset. Scoring is
:func:`cat_b.score_timestamps` against the target's GT pair.

Universal IVs: source, source_type, midi (target), duration_ms.
Experiment-specific IVs: n_distractors, target_pos.

Usage::
    pitchbench --id b3 --preview
    pitchbench --id b3 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_b import CatBSpec, run_cat_b_experiment
from pitchbench.experiments.helpers.music import midi_to_note
from pitchbench.experiments.helpers.timing_layout import (
    cap_notes_for_total,
    onsets_from_gaps,
    sample_adaptive_gaps,
    stable_cell_seed,
)

EXP_NAME        = Path(__file__).stem
SOURCES         = config.pitchbench_b4_SOURCES
PITCHES         = config.pitchbench_b4_PITCHES
DURATIONS_MS    = config.pitchbench_b4_DURATIONS_MS
N_DISTRACTORS   = config.pitchbench_b4_N_DISTRACTORS
TARGET_POS_OPTS = config.pitchbench_b4_TARGET_POS_OPTS
TOTAL_DUR_MS    = config.pitchbench_b4_TOTAL_DUR_MS
SEED            = config.pitchbench_b4_SEED


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for tgt in PITCHES:
            for nd in N_DISTRACTORS:
                for pos in TARGET_POS_OPTS:
                    for dur in DURATIONS_MS:
                        requested_n = nd + 1
                        n = cap_notes_for_total(requested_n, dur, TOTAL_DUR_MS)
                        if n < 1:
                            continue

                        cell_seed = stable_cell_seed(SEED, src, tgt, nd, pos, dur, n)
                        sub_rng   = random.Random(cell_seed)
                        candidates  = [p for p in PITCHES if p != tgt]
                        if n > len(candidates) + 1:
                            n = len(candidates) + 1
                        n_distractors = n - 1
                        distractors = sub_rng.sample(candidates, n_distractors)
                        notes = list(distractors)
                        if pos == "first":
                            notes.insert(0, tgt)
                        elif pos == "last":
                            notes.append(tgt)
                        else:
                            notes.insert(len(notes) // 2, tgt)
                        gaps = sample_adaptive_gaps(
                            sub_rng,
                            n_notes=len(notes),
                            note_dur_ms=dur,
                            total_dur_ms=TOTAL_DUR_MS,
                        )
                        onsets = onsets_from_gaps(gaps, n_notes=len(notes), note_dur_ms=dur)
                        target_idx = notes.index(tgt)
                        rows.append({
                            "source":           src,
                            "source_type":      "waveform" if src in config.WAVEFORMS else "instrument",
                            "midi":             tgt,
                            "duration_ms":      dur,
                            "n_distractors":    n_distractors,
                            "target_pos":       pos,
                            "midi_seq":         notes,
                            "onsets_ms":        onsets,
                            "target_onset_ms":  onsets[target_idx],
                            "target_offset_ms": onsets[target_idx] + dur,
                            "seed":             cell_seed,
                        })
    return rows


def wav_for(c: dict) -> Path:
    triples = [(m, c["onsets_ms"][i], c["duration_ms"]) for i, m in enumerate(c["midi_seq"])]
    path, _ = engine.clip_with_notes(triples, c["source"], TOTAL_DUR_MS, name_hint="b3")
    return path


def gt_timestamps_for(c: dict) -> list[float]:
    return [c["target_onset_ms"] / 1000.0, c["target_offset_ms"] / 1000.0]


def prompt_for(c: dict) -> str:
    note_name = midi_to_note(c["midi"])
    return (
        f"This audio contains a sequence of musical notes separated by silence. "
        f"Identify the onset and offset times of the note {note_name} "
        f"(MIDI {c['midi']}; it appears exactly once). Reply with ONLY two "
        f"timestamps in MM:SS.cc format separated by a comma, e.g. "
        f"'0:05.20, 0:08.50'. Nothing else. Output only the answer."
    )


SPEC = CatBSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="timing",
    prompt=prompt_for,
    gt_timestamps_fn=gt_timestamps_for,
    record_extras=("midi", "n_distractors", "target_pos", "duration_ms"),
)


def preview() -> None:
    run_cat_b_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_b_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
