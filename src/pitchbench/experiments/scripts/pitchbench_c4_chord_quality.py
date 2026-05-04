"""
c4 — Chord quality identification.

Question: can the ALM correctly identify the harmonic quality of a
simultaneously-sounding chord (major, minor, dim, aug, dom7, maj7, …)?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    same_instrument:  {True, False}
    chord_quality:    {maj, min, dim, aug, dom7, maj7, min7, m7b5, sus2, sus4}
    root_midi:        12 root notes (C3..B3)
    task:             {quality_only, root_and_quality}

Fixed conditions: root-position, equal level. The `task` IV exists in the
same script with a column flag so quality-only and joint root+quality
results sit side-by-side in one CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import (
    extract_chord_quality, extract_note, midi_to_note,
)
from pitchbench.experiments.helpers.results import (
    get_run_metadata, make_run_dir, save_comparison, save_results,
)

EXP_NAME = Path(__file__).stem

# Quality → (intervals over the root, canonical full-name)
QUALITIES: dict[str, tuple[tuple[int, ...], str]] = {
    "major":      ((0, 4, 7),     "major"),
    "minor":      ((0, 3, 7),     "minor"),
    "diminished": ((0, 3, 6),     "diminished"),
    "augmented":  ((0, 4, 8),     "augmented"),
    "dom7":       ((0, 4, 7, 10), "dominant seventh"),
    "maj7":       ((0, 4, 7, 11), "major seventh"),
    "min7":       ((0, 3, 7, 10), "minor seventh"),
    "m7b5":       ((0, 3, 6, 10), "half diminished"),
    "sus2":       ((0, 2, 7),     "sus2"),
    "sus4":       ((0, 5, 7),     "sus4"),
}

ROOT_MIDIS:   list[int]  = list(range(48, 60))
TASKS:        list[str]  = ["quality_only", "root_and_quality"]
SAME_INSTRUMENT_OPTS:  list[bool] = [True, False]
SOURCES:      list[str]  = list(config.WAVEFORMS) + list(config.GM_PROGRAMS_V1.keys())


PROMPT_QUALITY_ONLY = (
    "This audio contains a chord (multiple simultaneous notes). "
    "What is its harmonic quality? "
    "Examples: 'major', 'minor', 'diminished', 'augmented', "
    "'dominant seventh', 'major seventh', 'minor seventh', "
    "'half diminished', 'sus2', 'sus4'. "
    "Reply with ONLY the quality."
)

PROMPT_ROOT_AND_QUALITY = (
    "This audio contains a chord (multiple simultaneous notes). "
    "Identify both the root note and the harmonic quality. "
    "Reply as <NOTE> <QUALITY>, e.g. 'C major', 'F# minor seventh', 'B half diminished'. "
    "Reply with ONLY the chord name."
)


# ── Conditions ────────────────────────────────────────────────────────────────

def _mixed_sources(n: int, seed_int: int) -> list[str]:
    pool = list(SOURCES)
    start = seed_int % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(n)]


def build_conditions(durations_ms: list[int]) -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for dur in durations_ms:
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
                                "duration_ms":     dur,
                                "source":          src_label,
                                "_source_arg":     src_arg,
                                "same_instrument": same,
                                "root_midi":       root,
                                "chord_quality":   quality,
                                "midis":           midis,
                                "task":            task,
                            })
    return rows


def _wav_for(c: dict) -> Path:
    return engine.chord(c["midis"], c["_source_arg"], c["duration_ms"])


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        try:
            wav = str(_wav_for(c))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        prompt = PROMPT_QUALITY_ONLY if c["task"] == "quality_only" else PROMPT_ROOT_AND_QUALITY
        out    = query_alm(model_name, wav, prompt)
        raw    = (out["result"] or "").strip()

        quality_pred = extract_chord_quality(raw)
        quality_ok   = (quality_pred == c["chord_quality"])
        root_pred:  str | None = None
        root_ok:    int        = 0
        joint_ok:   int        = 0
        if c["task"] == "root_and_quality":
            # Parse the leading note letter + optional accidental (octave is
            # often missing in chord names — "C major" not "C4 major").
            import re as _re
            from pitchbench.experiments.helpers.music import (
                FLAT_TO_SHARP, NOTE_NAMES, note_pc,
            )
            m = _re.match(r"\s*([A-Ga-g])\s*([#b♯♭]?)", raw)
            pred_letter: str | None = None
            if m:
                letter = m.group(1).upper()
                acc    = m.group(2).replace("♯", "#").replace("♭", "b")
                cand   = letter + acc
                cand   = FLAT_TO_SHARP.get(cand, cand)
                if cand in NOTE_NAMES:
                    pred_letter = cand
            gt_letter, _ = note_pc(midi_to_note(c["root_midi"]))
            root_pred = pred_letter
            root_ok   = int(pred_letter is not None and pred_letter == gt_letter)
            joint_ok  = int(quality_ok and root_ok)

        records.append({
            "duration_ms":      c["duration_ms"],
            "source":           c["source"],
            "same_instrument":  c["same_instrument"],
            "root_midi":        c["root_midi"],
            "root_note":        midi_to_note(c["root_midi"]),
            "chord_quality_gt": c["chord_quality"],
            "task":             c["task"],
            "midi_set":         "+".join(str(m) for m in c["midis"]),
            "wav":              wav,
            "raw_response":     raw,
            "quality_pred":     quality_pred,
            "quality_correct":  int(quality_ok),
            "root_pred":        root_pred,
            "root_correct":     root_ok,
            "joint_correct":    joint_ok,
            "prompt":           prompt,
            "model_params":     out["model_params"],
        })

    n = len(records)
    qonly = [r for r in records if r["task"] == "quality_only"]
    rq    = [r for r in records if r["task"] == "root_and_quality"]
    summary = {
        "total":             n,
        "quality_only_acc":  round(sum(r["quality_correct"] for r in qonly) / max(1, len(qonly)), 4),
        "root_and_quality_quality_acc": round(sum(r["quality_correct"] for r in rq) / max(1, len(rq)), 4),
        "root_and_quality_root_acc":    round(sum(r["root_correct"]    for r in rq) / max(1, len(rq)), 4),
        "root_and_quality_joint_acc":   round(sum(r["joint_correct"]   for r in rq) / max(1, len(rq)), 4),
    }
    summary_lines = [
        f"  Stimuli                   : {n}  ({len(qonly)} quality_only / {len(rq)} root+quality)",
        f"  Quality-only accuracy     : {summary['quality_only_acc']:.1%}",
        f"  R+Q  quality accuracy     : {summary['root_and_quality_quality_acc']:.1%}",
        f"  R+Q  root accuracy        : {summary['root_and_quality_root_acc']:.1%}",
        f"  R+Q  joint (both) accuracy: {summary['root_and_quality_joint_acc']:.1%}",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, qualities=list(QUALITIES.keys()),
        roots=ROOT_MIDIS, tasks=TASKS,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        prompt_quality_only=PROMPT_QUALITY_ONLY,
        prompt_root_and_quality=PROMPT_ROOT_AND_QUALITY,
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("quality_correct", "root_correct", "joint_correct"),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    conds = build_conditions(config.DEFAULT_DURATIONS_MS)
    for c in conds:
        try: _wav_for(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {SOURCES}")
    print(f"Stimuli    : {len(conds)}")
    print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    target_models = args.models or list(config.MODELS)
    conds = build_conditions(config.DEFAULT_DURATIONS_MS)
    for c in conds:
        try: _wav_for(c)
        except ValueError: pass

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for m in target_models:
        all_summaries[m] = run_one_model(m, conds, run_dir)
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
