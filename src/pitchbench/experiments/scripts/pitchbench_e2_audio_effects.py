"""
Experiment 07 — Pitch recognition under audio effects
Tests whether models can identify pitch when audio is degraded with effects
(white noise, reverb, hard clipping, added harmonics) that preserve the
fundamental frequency but alter timbre, dynamics, or spectral content.

Stimuli: all waveforms × 11 representative pitches × 10 effect conditions.
Three prompts per stimulus:
  MIDI:   integer note number (0–127)
  ABC:    note name + octave (e.g. "C4")
  Doremi: solfege syllable (e.g. "do", "sol#")

All effects are applied deterministically (fixed seed per condition).

Usage:
    python experiments/run.py exp_7_effects
    python experiments/run.py exp_7_effects --preview
    python experiments/run.py exp_7_effects --models audio_flamingo_next_instruct
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

PITCHES: list[int] = [48, 52, 55, 60, 64, 67, 69, 72, 76, 79, 84]
#                     C3  E3  G3  C4  E4  G4  A4  C5  E5  G5  C6

TONE_DURATION = 2.0   # seconds
TONE_MS = int(TONE_DURATION * 1000)

# Effect definitions; "type" key selects the apply function in engine
# noise:    add white Gaussian noise at given SNR (dB)
# reverb:   recursive comb filter — delay_s seconds at given decay coefficient
# clip:     hard-clip at (threshold × peak), renormalise
# harmonic: add a sinusoidal partial at (ratio × f0) with relative amplitude level
EFFECTS: dict[str, dict] = {
    "clean":       {},
    "noise_30":    {"type": "noise",    "snr_db": 30},
    "noise_20":    {"type": "noise",    "snr_db": 20},
    "noise_10":    {"type": "noise",    "snr_db": 10},
    "noise_0":     {"type": "noise",    "snr_db":  0},
    "reverb_s":    {"type": "reverb",   "delay_s": 0.05, "decay": 0.30},
    "reverb_l":    {"type": "reverb",   "delay_s": 0.20, "decay": 0.70},
    "clip_50":     {"type": "clip",     "threshold": 0.50},
    "clip_25":     {"type": "clip",     "threshold": 0.25},
    # "harm_oct":    {"type": "harmonic", "ratio": 2.000, "level": 0.50},
    # "harm_fifth":  {"type": "harmonic", "ratio": 1.498, "level": 0.30},
}

PROMPT_MIDI_FULL   = "Listen to this audio clip of a single musical note. " + PROMPT_MIDI
PROMPT_ABC_FULL    = "Listen to this audio clip of a single musical note. " + PROMPT_ABC
PROMPT_DOREMI_FULL = "Listen to this audio clip of a single musical note. " + PROMPT_DOREMI

SOURCES: list[str] = list(config.WAVEFORMS) + list(config.GM_PROGRAMS_V1.keys())


def build_conditions() -> list[dict]:
    rows = []
    for src in SOURCES:
        for eff_idx, (eff_name, eff_params) in enumerate(EFFECTS.items()):
            for midi in PITCHES:
                note       = midi_to_note(midi)
                noise_seed = (eff_idx * 1000 + midi) % (2 ** 31)
                rows.append({
                    "source":        src,
                    "effect":        eff_name,
                    "effect_type":   eff_params.get("type", "clean"),
                    "effect_params": str(eff_params),
                    "midi":          midi,
                    "note":          note,
                    "noise_seed":    noise_seed,
                })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        engine.tone_with_effect(
            c["midi"], c["source"], TONE_MS,
            c["effect"], EFFECTS[c["effect"]], c["noise_seed"],
        )


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav = str(engine.tone_with_effect(
            c["midi"], c["source"], TONE_MS,
            c["effect"], EFFECTS[c["effect"]], c["noise_seed"],
        ))

        print(f"    {c['source']:10s}  {c['effect']:12s}  {c['note']:4s}")
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
            effect=c["effect"],
            effect_type=c["effect_type"],
            effect_params=c["effect_params"],
            noise_seed=c["noise_seed"],
        )
        records.append(record)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_effect: dict[str, dict] = {}
    for eff_name in EFFECTS:
        sub = [r for r in records if r["effect"] == eff_name]
        per_effect[eff_name] = {
            "n":          len(sub),
            "midi_acc":   round(sum(r["midi_correct"]   for r in sub) / len(sub), 4) if sub else 0.0,
            "abc_acc":    round(sum(r["abc_correct"]    for r in sub) / len(sub), 4) if sub else 0.0,
            "doremi_acc": round(sum(r["doremi_correct"] for r in sub) / len(sub), 4) if sub else 0.0,
        }

    per_source: dict[str, dict] = {}
    for src in SOURCES:
        sub = [r for r in records if r["source"] == src]
        per_source[src] = {
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
        "per_effect":     per_effect,
        "per_source":     per_source,
    }

    summary_lines = [
        f"  Sources : {SOURCES}",
        f"  Stimuli : {n}  ({len(SOURCES)} sources × {len(PITCHES)} pitches × {len(EFFECTS)} effects)",
        f"",
        f"  {'Effect':12s}  {'MIDI':>7}  {'ABC':>7}  {'Doremi':>8}",
        f"  {'─' * 38}",
    ]
    for eff_name, d in per_effect.items():
        summary_lines.append(
            f"  {eff_name:12s}  {d['midi_acc']:>7.1%}  {d['abc_acc']:>7.1%}  {d['doremi_acc']:>8.1%}"
        )
    summary_lines += ["", "  Per source (MIDI | ABC | Doremi):"]
    for src, d in per_source.items():
        summary_lines.append(
            f"    {src:12s}: {d['midi_acc']:.1%} | {d['abc_acc']:.1%} | {d['doremi_acc']:.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=PITCHES, effects=EFFECTS,
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

    eff_names = list(EFFECTS.keys())
    midi_accs, abc_accs, doremi_accs = [], [], []
    for eff_name in eff_names:
        sub = [r for r in records if r["effect"] == eff_name]
        midi_accs.append(sum(r["midi_correct"]   for r in sub) / len(sub) * 100 if sub else 0)
        abc_accs.append( sum(r["abc_correct"]    for r in sub) / len(sub) * 100 if sub else 0)
        doremi_accs.append(sum(r["doremi_correct"] for r in sub) / len(sub) * 100 if sub else 0)

    x = list(range(len(eff_names)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(eff_names) * 1.2), 4))
    ax.bar([xi - width for xi in x], midi_accs,   width, label="MIDI (integer)")
    ax.bar([xi          for xi in x], abc_accs,   width, label="ABC (note name)")
    ax.bar([xi + width  for xi in x], doremi_accs, width, label="Doremi (solfege)")
    ax.set_xticks(x)
    ax.set_xticklabels(eff_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_title(f"Pitch accuracy per effect — {config.MODELS.get(model_name, model_name)}")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = run_dir / f"effects_accuracy_{model_name}.png"
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
    print(f"Effects    : {list(EFFECTS.keys())}")
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
