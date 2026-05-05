"""
b4 — Pitch identification at a specific time within a sequence.

Given a sequence of N non-overlapping notes, the prompt asks for the pitch
sounding at a queried time (computed at the midpoint of one designated
target note). 4-format pitch scoring — same shape as cat-A.

Universal IVs: source, source_type, midi (the target pitch).
Experiment-specific IVs: n_notes, duration_ms, query_time_s.

Usage::
    pitchbench --id b4 --preview
    pitchbench --id b4 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_b import CatBSpec, run_cat_b_experiment
from pitchbench.experiments.helpers.music import (
    PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
)

EXP_NAME     = Path(__file__).stem
SOURCES      = config.pitchbench_b4_SOURCES
PITCHES      = config.pitchbench_b4_PITCHES
DURATIONS_MS = config.pitchbench_b4_DURATIONS_MS
N_NOTES_OPTS = config.pitchbench_b4_N_NOTES_OPTS
TOTAL_DUR_MS = config.pitchbench_b4_TOTAL_DUR_MS
GAP_MIN_MS   = config.pitchbench_b4_GAP_MIN_MS
GAP_MAX_MS   = config.pitchbench_b4_GAP_MAX_MS
SEED         = config.pitchbench_b4_SEED


def _query_str(secs: float) -> str:
    minutes = int(secs // 60)
    seconds = secs - minutes * 60
    return f"{minutes}:{seconds:05.2f}"


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for tgt in PITCHES:
            for n in N_NOTES_OPTS:
                for dur in DURATIONS_MS:
                    cell_seed = (SEED ^ hash((src, tgt, n, dur))) & 0xFFFFFFFF
                    sub_rng   = random.Random(cell_seed)
                    distractors = [p for p in PITCHES if p != tgt]
                    other_pitches = sub_rng.sample(distractors, n - 1)
                    midis_seq = list(other_pitches)
                    insert_at = sub_rng.randint(0, n - 1)
                    midis_seq.insert(insert_at, tgt)
                    gaps = [sub_rng.randint(GAP_MIN_MS, GAP_MAX_MS) for _ in range(n + 1)]
                    if sum(gaps) + dur * n > TOTAL_DUR_MS:
                        continue
                    onsets: list[int] = []
                    cursor = gaps[0]
                    for _ in range(n):
                        onsets.append(cursor)
                        cursor += dur + gaps[len(onsets)]
                    target_idx   = midis_seq.index(tgt)
                    query_time_s = round((onsets[target_idx] + dur / 2) / 1000.0, 3)
                    rows.append({
                        "source":       src,
                        "source_type":  "waveform" if src in config.WAVEFORMS else "instrument",
                        "midi":         tgt,
                        "duration_ms":  dur,
                        "n_notes":      n,
                        "midi_seq":     midis_seq,
                        "onsets_ms":    onsets,
                        "target_idx":   target_idx,
                        "query_time_s": query_time_s,
                        "seed":         cell_seed,
                    })
    return rows


def wav_for(c: dict) -> Path:
    triples = [(m, c["onsets_ms"][i], c["duration_ms"]) for i, m in enumerate(c["midi_seq"])]
    path, _ = engine.clip_with_notes(triples, c["source"], TOTAL_DUR_MS, name_hint="b4")
    return path


def prompts_for(c: dict) -> dict[str, str]:
    qs = _query_str(c["query_time_s"])
    prefix = (
        f"This audio contains a sequence of musical notes separated by silence. "
        f"Identify the pitch that is sounding at exactly {qs}. "
    )
    return {
        "midi":   prefix + PROMPT_MIDI,
        "spn":    prefix + PROMPT_SPN,
        "doremi": prefix + PROMPT_DOREMI,
        "hz":     prefix + PROMPT_HZ,
    }


SPEC = CatBSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="pitch",
    prompts_fn=prompts_for,
    record_extras=("n_notes", "duration_ms", "query_time_s"),
)


def preview() -> None:
    run_cat_b_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_b_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
