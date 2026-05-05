"""
c1 — Dyad interval identification.

Question: given two simultaneous tones, can the ALM correctly name the
musical interval between them?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    same_instrument:  {True, False}              (single timbre vs mixed)
    root_midi:        ∈ DEFAULT_PITCHES
    interval_st:      {1..12}                    (m2..octave)

Scoring: ``extract_interval`` parses the semitone count from the response;
``interval_correct`` = predicted == ground truth.

Usage::
    pitchbench --id c1 --preview
    pitchbench --id c1 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_c import CatCSpec, run_cat_c_experiment
from pitchbench.experiments.helpers.music import (
    INTERVAL_NAMES, extract_interval, midi_to_note,
)

EXP_NAME             = Path(__file__).stem
INTERVALS_ST         = config.pitchbench_c1_INTERVALS_ST
SOURCES              = config.pitchbench_c1_SOURCES
SAME_INSTRUMENT_OPTS = config.pitchbench_c1_SAME_INSTRUMENT_OPTS
DURATIONS_MS         = config.pitchbench_c1_DURATIONS_MS
PITCHES              = config.pitchbench_c1_PITCHES

PROMPT = (
    "This audio contains two simultaneous musical notes. "
    "How many semitones apart are they? "
    "Reply with ONLY the integer number of semitones."
)


def _mixed_pair(seed_int: int) -> tuple[str, str]:
    pool = list(SOURCES)
    rng_idx = seed_int % len(pool)
    a = pool[rng_idx]
    b = pool[(rng_idx + 1 + (seed_int >> 8) % (len(pool) - 1)) % len(pool)]
    return a, b


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for root in PITCHES:
            for iv in INTERVALS_ST:
                pair = root + iv
                if pair > 96:
                    continue
                for dur in DURATIONS_MS:
                    for same in SAME_INSTRUMENT_OPTS:
                        if same:
                            src_arg, src_label = src, src
                        else:
                            srcs = list(_mixed_pair(hash((src, root, iv, dur))))
                            src_arg, src_label = srcs, "+".join(srcs)
                        rows.append({
                            "duration_ms":     dur,
                            "source":          src_label,
                            "_source_arg":     src_arg,
                            "same_instrument": same,
                            "root_midi":       root,
                            "interval_st":     iv,
                            "midis":           [root, pair],
                        })
    return rows


def wav_for(c: dict) -> Path:
    return engine.chord(c["midis"], c["_source_arg"], c["duration_ms"])


def prompts_for(_: dict) -> dict[str, str]:
    return {"main": PROMPT}


def _interval_label(iv: int) -> str:
    return INTERVAL_NAMES[iv][0] if iv in INTERVAL_NAMES else str(iv)


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw  = responses["main"]
    pred = extract_interval(raw)
    ok   = (pred == c["interval_st"]) if pred is not None else False
    return {
        "duration_ms":      c["duration_ms"],
        "source":           c["source"],
        "same_instrument":  c["same_instrument"],
        "root_midi":        c["root_midi"],
        "root_note":        midi_to_note(c["root_midi"]),
        "interval_st":      c["interval_st"],
        "interval_name":    _interval_label(c["interval_st"]),
        "midi_pair":        f"{c['midis'][0]}+{c['midis'][1]}",
        "raw_response":     raw,
        "interval_pred":    pred,
        "interval_correct": int(ok),
    }


SPEC = CatCSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="interval",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("interval",),
    record_extras=("duration_ms", "same_instrument", "root_midi", "interval_st"),
    label_fn=lambda j: (
        f"iv={j['cond']['interval_st']:>2}st "
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
