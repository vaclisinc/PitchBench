"""
c2 — Chord pitch count.

Question: can the ALM correctly count the number of distinct pitches in a
*simultaneously*-sounding chord?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    n:                {1..6}                    (chord size)
    same_instrument:  {True, False}             (single-source vs mixed-timbre)
    chord_quality:    {maj, min, dim, aug, dom7, maj7, min7, random_set}
    root_midi:        anchor MIDI used to build the chord
    trial:            for the random_set quality, multiple seeded trials

Usage::
    pitchbench --id c2 --preview
    pitchbench --id c2 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_c import CatCSpec, run_cat_c_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME             = Path(__file__).stem
CHORD_INTERVALS      = config.pitchbench_c1_CHORD_INTERVALS
QUALITIES_FIXED      = list(CHORD_INTERVALS.keys())
QUALITIES            = config.pitchbench_c1_QUALITIES
ROOT_MIDIS           = config.pitchbench_c1_ROOT_MIDIS
SAME_INSTRUMENT_OPTS = config.pitchbench_c1_SAME_INSTRUMENT_OPTS
RANDOM_TRIALS        = config.pitchbench_c1_RANDOM_TRIALS
SEED                 = config.pitchbench_c1_SEED
RANDOM_NS            = config.pitchbench_c1_RANDOM_NS
RANDOM_PITCH_RANGE   = config.pitchbench_c1_RANDOM_PITCH_RANGE
SOURCES              = config.pitchbench_c1_SOURCES
DURATIONS_MS         = config.pitchbench_c1_DURATIONS_MS

PROMPT = (
    "Listen to this audio. How many distinct musical pitches are sounding "
    "at the same time? Reply with ONLY a single integer. Nothing else. "
    "Output only the answer."
)


def _mixed_sources(n: int, src_idx_seed: int) -> list[str]:
    rng  = random.Random(src_idx_seed)
    pool = list(SOURCES)
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n)]


def build_conditions() -> list[dict]:
    rows: list[dict] = []

    def _add(n: int, midis: list[int], quality: str, root: int, src_or_list, dur: int, trial: int):
        same = isinstance(src_or_list, str)
        rows.append({
            "duration_ms":     dur,
            "n":               n,
            "chord_quality":   quality,
            "root_midi":       root,
            "midis":           midis,
            "source":          src_or_list if same else "+".join(src_or_list),
            "_source_arg":     src_or_list,
            "same_instrument": same,
            "trial":           trial,
        })

    for src in SOURCES:
        for dur in DURATIONS_MS:
            for quality in QUALITIES_FIXED:
                intervals = CHORD_INTERVALS[quality]
                n = len(intervals)
                for root in ROOT_MIDIS:
                    midis = [root + iv for iv in intervals]
                    if any(m > 96 for m in midis):
                        continue
                    for same in SAME_INSTRUMENT_OPTS:
                        if same:
                            _add(n, midis, quality, root, src, dur, 0)
                        else:
                            srcs = _mixed_sources(n, SEED + root + n * 1000)
                            _add(n, midis, quality, root, srcs, dur, 0)

            for n in RANDOM_NS:
                for trial in range(RANDOM_TRIALS):
                    sub_rng = random.Random(SEED + 7919 * (n * 100 + trial))
                    midis   = sorted(sub_rng.sample(range(*RANDOM_PITCH_RANGE), n))
                    for same in SAME_INSTRUMENT_OPTS:
                        if same:
                            _add(n, midis, "random_set", midis[0], src, dur, trial)
                        else:
                            srcs = _mixed_sources(n, SEED + 991 * (n * 100 + trial))
                            _add(n, midis, "random_set", midis[0], srcs, dur, trial)
    return rows


def wav_for(c: dict) -> Path:
    return engine.chord(c["midis"], c["_source_arg"], c["duration_ms"])


def prompts_for(_: dict) -> dict[str, str]:
    return {"main": PROMPT}


def _parse_count(text: str) -> int | None:
    m = re.search(r"\b(\d+)\b", (text or "").strip())
    return int(m.group(1)) if m else None


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw  = responses["main"]
    pred = _parse_count(raw)
    ok   = (pred == c["n"]) if pred is not None else False
    off  = abs(pred - c["n"]) if pred is not None else None
    return {
        "duration_ms":     c["duration_ms"],
        "source":          c["source"],
        "same_instrument": c["same_instrument"],
        "chord_quality":   c["chord_quality"],
        "root_midi":       c["root_midi"],
        "midi_set":        str(c["midis"]),
        "note_set":        ", ".join(midi_to_note(m) for m in c["midis"]),
        "n":               c["n"],
        "trial":           c["trial"],
        "raw_response":    raw,
        "count_pred":      pred,
        "count_correct":   int(ok),
        "off_by":          off,
    }


SPEC = CatCSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="count",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("count",),
    record_extras=(
        "duration_ms", "n", "chord_quality", "root_midi",
        "same_instrument", "trial",
    ),
    label_fn=lambda j: (
        f"n={j['cond']['n']} {j['cond']['chord_quality']:11s} "
        f"root={midi_to_note(j['cond']['root_midi']):4s} "
        f"same_instrumentation={str(j['cond']['same_instrument']):5s} "
        f"dur={j['cond']['duration_ms']:>5}ms"
    ),
)


def preview() -> None:
    run_cat_c_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_c_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
