"""
c4 — Chord quality identification.

Question: can the ALM correctly identify the harmonic quality of a
simultaneously-sounding chord (major, minor, dim, aug, dom7, maj7, …)?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    same_instrument:  {True, False}
    chord_quality_gt: {major, minor, ...}
    root_midi:        12 root notes
    task:             {quality_only, root_and_quality}

Headline metric is ``quality_correct`` (always defined). The
``root_and_quality`` task additionally produces ``root_correct`` and
``joint_correct`` columns; those average over the joint-task subset only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_c import CatCSpec, run_cat_c_experiment
from pitchbench.experiments.helpers.music import (
    FLAT_TO_SHARP, NOTE_NAMES, extract_chord_quality, midi_to_note, note_pc,
)

EXP_NAME             = Path(__file__).stem
QUALITIES            = config.pitchbench_c3_QUALITIES
ROOT_MIDIS           = config.pitchbench_c3_ROOT_MIDIS
TASKS                = config.pitchbench_c3_TASKS
SAME_INSTRUMENT_OPTS = config.pitchbench_c3_SAME_INSTRUMENT_OPTS
SOURCES              = config.pitchbench_c3_SOURCES
DURATIONS_MS         = config.pitchbench_c3_DURATIONS_MS

PROMPT_QUALITY_ONLY = (
    "This audio contains a chord (multiple simultaneous notes). "
    "What is its harmonic quality? "
    f"Choose one of the following: {', '.join(QUALITIES.keys())}. "
    "Reply with ONLY the quality."
)

PROMPT_ROOT_AND_QUALITY = (
    "This audio contains a chord (multiple simultaneous notes). "
    "Identify both the root note and the harmonic quality. "
    "Reply as <NOTE> <QUALITY>, e.g. 'C major', 'F# minor seventh', "
    "'B half diminished'. The note should be expressed in scientific pitch "
    "notation (C4, F#3, etc.) and the quality should be one of: "
    f"{', '.join(QUALITIES.keys())}. "
    "Reply with ONLY the chord name."
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
                            srcs = _mixed_sources(len(midis), hash((src, root, quality, dur)))
                            src_arg, src_label = srcs, "+".join(srcs)
                        for task in TASKS:
                            rows.append({
                                "duration_ms":      dur,
                                "source":           src_label,
                                "_source_arg":      src_arg,
                                "same_instrument":  same,
                                "root_midi":        root,
                                "chord_quality_gt": quality,
                                "midis":            midis,
                                "task":             task,
                            })
    return rows


def wav_for(c: dict) -> Path:
    return engine.chord(c["midis"], c["_source_arg"], c["duration_ms"])


def prompts_for(c: dict) -> dict[str, str]:
    p = PROMPT_QUALITY_ONLY if c["task"] == "quality_only" else PROMPT_ROOT_AND_QUALITY
    return {"main": p}


def _parse_root_letter(raw: str) -> str | None:
    m = re.match(r"\s*([A-Ga-g])\s*([#b♯♭]?)", raw or "")
    if not m:
        return None
    letter = m.group(1).upper()
    acc    = m.group(2).replace("♯", "#").replace("♭", "b")
    cand   = FLAT_TO_SHARP.get(letter + acc, letter + acc)
    return cand if cand in NOTE_NAMES else None


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
        "task":             c["task"],
        "midi_set":         "+".join(str(m) for m in c["midis"]),
        "raw_response":     raw,
        "quality_pred":     quality_pred,
        "quality_correct":  int(quality_ok),
    }
    if c["task"] == "root_and_quality":
        pred_letter = _parse_root_letter(raw)
        gt_letter, _ = note_pc(midi_to_note(c["root_midi"]))
        root_ok  = int(pred_letter is not None and pred_letter == gt_letter)
        rec["root_pred"]      = pred_letter
        rec["root_correct"]   = root_ok
        rec["joint_correct"]  = int(bool(quality_ok) and bool(root_ok))
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
        "chord_quality_gt", "task",
    ),
    label_fn=lambda j: (
        f"{j['cond']['chord_quality_gt']:11s} "
        f"root={midi_to_note(j['cond']['root_midi']):4s} "
        f"same_instrumentation={str(j['cond']['same_instrument']):5s} "
        f"{j['cond']['task']}"
    ),
)


def preview() -> None:
    run_cat_c_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_c_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
