"""
c3 — Chord quality identification.

Question: can the ALM correctly identify the harmonic quality of a
simultaneously-sounding chord (major, minor, dim, aug, dom7, maj7, …)?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    same_instrument:  {True, False}
    chord_quality_gt: {major, minor, ...}
    root_midi:        12 root notes

Headline metric is ``quality_correct``.
"""

from __future__ import annotations
from pathlib import Path

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.experiments.helpers.cat_c import CatCSpec, run_cat_c_experiment
from pitchbench.experiments.helpers.music import extract_chord_quality, midi_to_note
from pitchbench.experiments.helpers.timing_layout import stable_cell_seed

EXP_NAME             = Path(__file__).stem
QUALITIES            = config.pitchbench_c3_QUALITIES
ROOT_MIDIS           = config.pitchbench_c3_ROOT_MIDIS
SAME_INSTRUMENT_OPTS = config.pitchbench_c3_SAME_INSTRUMENT_OPTS
SOURCES              = config.pitchbench_c3_SOURCES
DURATIONS_MS         = config.pitchbench_c3_DURATIONS_MS

PROMPT_QUALITY_ONLY = (
    "This audio contains a chord (multiple simultaneous notes). "
    "What is its harmonic quality? "
    f"Choose one of the following: {', '.join(QUALITIES.keys())}. "
    "Reply with ONLY the quality."
)

def _mixed_sources(n: int, seed_int: int) -> list[str]:
    pool  = list(SOURCES)
    start = seed_int % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(n)]


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for dur in DURATIONS_MS:
            for quality, (ivs, _) in QUALITIES.items():
                for root in ROOT_MIDIS:
                    midis = [root + iv for iv in ivs]
                    if any(m > 96 for m in midis):
                        continue
                    for same in SAME_INSTRUMENT_OPTS:
                        if same:
                            src_arg, src_label = src, src
                        else:
                            mix_seed = stable_cell_seed(0, src, root, quality, dur)
                            srcs = _mixed_sources(len(midis), mix_seed)
                            src_arg, src_label = srcs, "+".join(srcs)
                        rows.append({
                            "duration_ms":      dur,
                            "source":           src_label,
                            "_source_arg":      src_arg,
                            "same_instrument":  same,
                            "root_midi":        root,
                            "chord_quality_gt": quality,
                            "midis":            midis,
                        })
    return rows


def wav_for(c: dict) -> Path:
    return engine.chord(c["midis"], c["_source_arg"], c["duration_ms"])


def prompts_for(c: dict) -> dict[str, str]:
    return {"main": PROMPT_QUALITY_ONLY}


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw          = responses["main"]
    quality_pred = extract_chord_quality(raw)
    quality_ok   = (quality_pred == c["chord_quality_gt"])

    rec: dict = {
        "duration_ms":      c["duration_ms"],
        "source":           c["source"],
        "same_instrument":  c["same_instrument"],
        "root_midi":        c["root_midi"],
        "root_note":        midi_to_note(c["root_midi"]),
        "chord_quality_gt": c["chord_quality_gt"],
        "midi_set":         "+".join(str(m) for m in c["midis"]),
        "raw_response":     raw,
        "quality_pred":     quality_pred,
        "quality_correct":  int(quality_ok),
    }
    return rec


SPEC = CatCSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="quality",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("quality",),
    record_extras=(
        "duration_ms", "same_instrument", "root_midi",
        "chord_quality_gt",
    ),
    label_fn=lambda j: (
        f"{j['cond']['chord_quality_gt']:11s} "
        f"root={midi_to_note(j['cond']['root_midi']):4s} "
        f"same_instrumentation={str(j['cond']['same_instrument']):5s}"
    ),
)


def preview() -> None:
    run_cat_c_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_c_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
