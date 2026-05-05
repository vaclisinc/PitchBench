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
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.audit import pitch_record_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    PROMPT_ABC, PROMPT_HZ, PROMPT_MIDI, PROMPT_DOREMI,
    midi_to_note,
    standard_pitch_record,
)
from pitchbench.experiments.helpers.plots import (
    save_combined_iv_plot, save_per_format_iv_plots,
)
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

# Data-generation parameters (sourced from config.pitchbench_e1_*)
PITCHES     = config.pitchbench_e1_PITCHES
LOUDNESS_DB = config.pitchbench_e1_LOUDNESS_DB

PROMPT_MIDI_FULL   = "Listen to this audio clip of a single musical note. " + PROMPT_MIDI
PROMPT_ABC_FULL    = "Listen to this audio clip of a single musical note. " + PROMPT_ABC
PROMPT_DOREMI_FULL = "Listen to this audio clip of a single musical note. " + PROMPT_DOREMI
PROMPT_HZ_FULL     = "Listen to this audio clip of a single musical note. " + PROMPT_HZ

SOURCES = config.pitchbench_e1_SOURCES
TONE_MS = config.pitchbench_e1_TONE_MS


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

def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        wav = str(engine.tone_at_volume(c["midi"], c["source"], TONE_MS, c["loudness_db"]))
        jobs.append({"wav": wav, "cond": c})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict) -> dict:
        c = job["cond"]
        r_m, r_a, r_d, r_h = query_four_formats(
            model_name, job["wav"],
            PROMPT_MIDI_FULL, PROMPT_ABC_FULL, PROMPT_DOREMI_FULL, PROMPT_HZ_FULL,
            verbose=False,
        )
        return standard_pitch_record(
            wav=job["wav"],
            source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["midi"],
            raw_midi=r_m["result"],
            raw_abc=r_a["result"],
            raw_doremi=r_d["result"],
            raw_hz=r_h["result"],
            prompt_midi=PROMPT_MIDI_FULL,
            prompt_abc=PROMPT_ABC_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL,
            prompt_hz=PROMPT_HZ_FULL,
            loudness_db=c["loudness_db"],
        )

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{j['cond']['note']:4s}  {j['cond']['loudness_db']:>4} dBFS  {j['cond']['source']}",
        result_label_fn=lambda j, r: pitch_record_audit_str(r, label=f"{j['cond']['note']:4s}  {j['cond']['loudness_db']:>4} dBFS  {j['cond']['source']:10s}"),
    )
    records: list[dict] = [r for r in raw if r is not None]

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
            "hz_acc":     round(sum(r["hz_correct"]     for r in sub) / len(sub), 4) if sub else 0.0,
        }

    summary = {
        "total":          n,
        "midi_correct":   sum(r["midi_correct"]   for r in records),
        "midi_within_1":  sum(r["midi_within_1"]  for r in records),
        "abc_correct":    sum(r["abc_correct"]    for r in records),
        "doremi_correct": sum(r["doremi_correct"] for r in records),
        "hz_correct":     sum(r["hz_correct"]     for r in records),
        "per_loudness_db": per_loudness,
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
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
        f"  [Hz — frequency]",
        f"    Exact match (≤1 Hz)  : {summary['hz_correct']} / {n}  ({summary['hz_correct']/n:.1%})",
        f"",
        f"  {'dBFS':>6}  {'n':>5}  {'MIDI':>7}  {'ABC':>7}  {'Doremi':>8}  {'Hz':>6}",
        f"  {'─' * 50}",
    ]
    for db, d in per_loudness.items():
        summary_lines.append(
            f"  {db:>6}  {d['n']:>5}  {d['midi_acc']:>7.1%}  {d['abc_acc']:>7.1%}  "
            f"{d['doremi_acc']:>8.1%}  {d['hz_acc']:>6.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        pitches=PITCHES, loudness_db=LOUDNESS_DB,
        tone_duration=TONE_MS / 1000,
        prompt_midi=PROMPT_MIDI_FULL,
        prompt_abc=PROMPT_ABC_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL,
        prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    _save_plot(records, run_dir, model_name)

    plots_dir = run_dir / "plots"; plots_dir.mkdir(exist_ok=True)
    save_per_format_iv_plots(records, plots_dir, model_name, iv_key="loudness_db",
                             iv_label="Loudness (dB)", group_by_source=False)
    save_combined_iv_plot(records, plots_dir, model_name, iv_key="loudness_db",
                          iv_label="Loudness (dB)")
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
        ("Hz (frequency)", "hz_correct"),
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
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    all_conds = build_conditions()
    conds = all_conds  # rename: save the full list
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {SOURCES}")
    print(f"Pitches    : {len(PITCHES)}  (MIDI {PITCHES[0]}–{PITCHES[-1]})")
    print(f"Loudness   : {LOUDNESS_DB} dBFS")
    print(f"Stimuli    : {len(conds)}  → {config.AUDIO_DIR}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds = build_conditions()
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)
    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
