"""
Experiment 06 — Pitch recognition under loudness variation
Tests whether models can identify pitch when the stimulus amplitude varies
from near-silence (−30 dBFS) to full level (0 dBFS).

Stimuli: pure sine waves at 11 representative MIDI pitches (C3–C6)
         at 6 loudness levels: −30, −20, −12, −6, −3, 0 dBFS.
Three prompts per stimulus:
  MIDI:   integer note number (0–127)
  ABC:    note name + octave (e.g. "C4")
  Doremi: solfege syllable and accidental (if needed) (e.g. "do", "sol#")

Usage:
    python experiments/run.py exp_6_loudness
    python experiments/run.py exp_6_loudness --preview
    python experiments/run.py exp_6_loudness --models audio_flamingo_next_instruct
"""

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_three_formats
from pitchbench.experiments.helpers.music import (
    PROMPT_ABC, PROMPT_MIDI, PROMPT_DOREMI,
    midi_to_note,
    standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.plots import save_accuracy_plots
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results

EXP_NAME = Path(__file__).stem

# ── Test parameters ───────────────────────────────────────────────────────────

# Representative pitches spanning C3–C6
PITCHES: list[int] = config.DEFAULT_PITCHES

# dBFS levels; 0 = peak amplitude 0.9, each step ÷ ~3–4
LOUDNESS_DB: list[int] = [-30, -20, -12, -6, -3, 0]

TONE_DURATION = 2.0   # seconds

PROMPT_MIDI_FULL   = "Listen to this audio clip of a single musical note. " + PROMPT_MIDI
PROMPT_ABC_FULL    = "Listen to this audio clip of a single musical note. " + PROMPT_ABC
PROMPT_DOREMI_FULL = "Listen to this audio clip of a single musical note. " + PROMPT_DOREMI

SOURCES: list[str] = config.ALL_SOURCES
TONE_MS = int(TONE_DURATION * 1000)


def build_conditions() -> list[dict]:
    rows = []
    for src in SOURCES:
        for midi in PITCHES:
            note = midi_to_note(midi)
            for db in LOUDNESS_DB:
                rows.append({
                    "source":      src,
                    "midi":        midi,
                    "note":        note,
                    "loudness_db": db,
                })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        engine.tone_at_volume(c["midi"], c["source"], TONE_MS, c["loudness_db"])


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav = str(engine.tone_at_volume(c["midi"], c["source"], TONE_MS, c["loudness_db"]))

        print(f"    {c['note']:4s}  {c['loudness_db']:>4} dBFS  {c['source']}")
        raw_midi, raw_abc, raw_doremi = query_three_formats(
            model_name, wav,
            PROMPT_MIDI_FULL, PROMPT_ABC_FULL, PROMPT_DOREMI_FULL,
            verbose=True,
        )

        record = standard_pitch_record(
            wav=wav,
            source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["midi"],
            raw_midi=raw_midi,
            raw_abc=raw_abc,
            raw_doremi=raw_doremi,
            prompt_midi=PROMPT_MIDI_FULL,
            prompt_abc=PROMPT_ABC_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL,
            loudness_db=c["loudness_db"],
        )
        records.append(record)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_loudness: dict[int, dict] = {}
    for db in LOUDNESS_DB:
        sub = [r for r in records if r["loudness_db"] == db]
        per_loudness[db] = {
            "n":          len(sub),
            "midi_acc":   round(sum(r["midi_correct"]   for r in sub) / len(sub), 4) if sub else 0.0,
            "abc_acc":    round(sum(r["abc_correct"]    for r in sub) / len(sub), 4) if sub else 0.0,
            "doremi_acc": round(sum(r["doremi_correct"] for r in sub) / len(sub), 4) if sub else 0.0,
        }

    summary = {
        "total":          n,
        "midi_correct":   sum(r["midi_correct"]   for r in records),
        "midi_within_1":  sum(r["midi_within_1"]  for r in records),
        "abc_correct":    sum(r["abc_correct"]    for r in records),
        "doremi_correct": sum(r["doremi_correct"] for r in records),
        "per_loudness_db": per_loudness,
    }

    summary_lines = [
        f"  Stimuli : {n}  ({len(PITCHES)} pitches × {len(LOUDNESS_DB)} loudness levels)",
        f"",
        f"  [MIDI — integer]",
        f"    Exact match   : {summary['midi_correct']} / {n}  ({summary['midi_correct']/n:.1%})",
        f"    Within 1      : {summary['midi_within_1']} / {n}",
        f"",
        f"  [ABC — note name]",
        f"    Exact match   : {summary['abc_correct']} / {n}  ({summary['abc_correct']/n:.1%})",
        f"",
        f"  [Doremi — solfege]",
        f"    Exact match   : {summary['doremi_correct']} / {n}  ({summary['doremi_correct']/n:.1%})",
        f"",
        f"  {'dBFS':>6}  {'MIDI':>7}  {'ABC':>7}  {'Doremi':>8}",
        f"  {'─' * 34}",
    ]
    for db, d in per_loudness.items():
        summary_lines.append(
            f"  {db:>6}  {d['midi_acc']:>7.1%}  {d['abc_acc']:>7.1%}  {d['doremi_acc']:>8.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        pitches=PITCHES, loudness_db=LOUDNESS_DB,
        tone_duration=TONE_DURATION,
        prompt_midi=PROMPT_MIDI_FULL,
        prompt_abc=PROMPT_ABC_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL,
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    _save_plot(records, run_dir, model_name)

    long_records = wide_to_long_records(records)
    save_accuracy_plots(
        long_records, run_dir, model_name,
        instrument_key="source",
        pitch_key="midi_gt",
        prompt_key="prompt_variant",
        accuracy_key="exact_match",
    )
    return summary


def _save_plot(records: list[dict], run_dir: Path, model_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, key in [
        ("MIDI (integer)", "midi_correct"),
        ("ABC (note name)", "abc_correct"),
        ("Doremi (solfege)", "doremi_correct"),
    ]:
        accs = []
        for db in LOUDNESS_DB:
            sub = [r for r in records if r["loudness_db"] == db]
            accs.append(sum(r[key] for r in sub) / len(sub) * 100 if sub else float("nan"))
        ax.plot(LOUDNESS_DB, accs, "o-", linewidth=2, label=label)

    ax.set_xlabel("Loudness (dBFS)")
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_title(f"Pitch accuracy vs. loudness — {config.MODELS.get(model_name, model_name)}")
    ax.set_ylim(-5, 105)
    ax.set_xticks(LOUDNESS_DB)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = run_dir / f"loudness_accuracy_{model_name}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved  → {p}")


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    conds = build_conditions()
    generate_stimuli(conds)
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {SOURCES}")
    print(f"Pitches    : {len(PITCHES)}  (MIDI {PITCHES[0]}–{PITCHES[-1]})")
    print(f"Loudness   : {LOUDNESS_DB} dBFS")
    print(f"Stimuli    : {len(conds)}  → {config.AUDIO_DIR}")
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    conds = build_conditions()
    generate_stimuli(conds)
    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, conds, run_dir)
    save_comparison(run_dir, all_summaries, EXP_NAME)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
