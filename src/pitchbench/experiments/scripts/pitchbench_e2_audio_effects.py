"""
Experiment e2 — Pitch recognition under audio effects
Tests whether models can identify pitch when audio is processed with effects
that genuinely threaten pitch cues. Effects use plugin-quality DSP (pedalboard
+ scipy Butterworth) and every output is RMS-matched to the dry signal so
loudness does not leak into the experiment as effect strength increases.

Stimuli: all waveforms + GM instruments × 11 representative pitches × effect conditions.
Four prompts per stimulus:
  MIDI:   integer note number (0–127)
  ABC:    note name + octave (e.g. "C4")
  Doremi: solfege syllable and accidental (if needed) (e.g. "do", "sol#")
  Hz:     fundamental frequency

All effects are applied deterministically (fixed seed per condition).

Effects:
  clean             — no processing
  highpass_above_f0 — Butterworth HP at 1.5·f0 (order 6) — removes the fundamental,
                      probing whether the model can recover pitch from harmonic spacing.
                      Note: collapses to silence on pure-sine sources by design.
  lowpass_at_f0     — Butterworth LP at 1.2·f0 (order 6) — strips all harmonics,
                      leaving only the fundamental.
  bitcrush_4bit     — pedalboard.Bitcrush, 4-bit depth: quantization noise floor
                      competes with the fundamental.
  distortion_heavy  — pedalboard.Distortion @ 30 dB drive (oversampled 4×): fundamental
                      drops as energy redistributes into harmonics.
  reverb_long       — pedalboard.Reverb (room_size=0.9): algorithmic tail smears attack
                      and adds late energy.
  chorus_heavy      — pedalboard.Chorus (depth=0.9, rate=1.2 Hz): detuned modulated
                      copies near f0 — tests whether the model locks onto a single pitch.

Usage:
    pitchbench e2
    pitchbench e2 --preview
    pitchbench --id e2 --models audio_flamingo_next_instruct
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
    format_accuracy_dict,
    midi_to_note,
    standard_pitch_record,
)
from pitchbench.experiments.helpers.plots import (
    save_combined_iv_plot, save_per_format_iv_plots,
)
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

# Data-generation parameters (sourced from config.pitchbench_e2_*)
PITCHES = config.pitchbench_e2_PITCHES
TONE_MS = config.pitchbench_e2_TONE_MS
EFFECTS = config.pitchbench_e2_EFFECTS

FORMAT_METRICS = {
    "midi": "midi_correct",
    "abc": "abc_correct",
    "doremi": "doremi_correct",
    "hz": "hz_correct",
}

PROMPT_MIDI_FULL   = "Listen to this audio clip of a single musical note. " + PROMPT_MIDI
PROMPT_ABC_FULL    = "Listen to this audio clip of a single musical note. " + PROMPT_ABC
PROMPT_DOREMI_FULL = "Listen to this audio clip of a single musical note. " + PROMPT_DOREMI
PROMPT_HZ_FULL     = "Listen to this audio clip of a single musical note. " + PROMPT_HZ

SOURCES = config.pitchbench_e2_SOURCES


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

def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        wav = str(engine.tone_with_effect(
            c["midi"], c["source"], TONE_MS,
            c["effect"], EFFECTS[c["effect"]], c["noise_seed"],
        ))
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
            effect=c["effect"],
            effect_type=c["effect_type"],
            effect_params=c["effect_params"],
            noise_seed=c["noise_seed"],
        )

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{j['cond']['source']:10s}  {j['cond']['effect']:12s}  {j['cond']['note']:4s}",
        result_label_fn=lambda j, r: pitch_record_audit_str(r, label=f"{j['cond']['source']:10s}  {j['cond']['effect']:12s}  {j['cond']['note']:4s}"),
    )
    records: list[dict] = [r for r in raw if r is not None]

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_effect: dict[str, dict] = {}
    for eff_name in EFFECTS:
        sub = [r for r in records if r["effect"] == eff_name]
        per_effect[eff_name] = {
            "n":          len(sub),
            **format_accuracy_dict(sub, FORMAT_METRICS),
        }

    per_source: dict[str, dict] = {}
    for src in SOURCES:
        sub = [r for r in records if r["source"] == src]
        per_source[src] = {
            "n":          len(sub),
            **format_accuracy_dict(sub, FORMAT_METRICS),
        }

    summary = {
        "total":          n,
        "accuracy":   format_accuracy_dict(records, FORMAT_METRICS),
        "by_effect":     per_effect,
        "by_source":     per_source,
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources : {SOURCES}",
        f"  Stimuli : {n}  ({len(SOURCES)} sources × {len(PITCHES)} pitches × {len(EFFECTS)} effects)",
        f"",
        f"  {'Effect':12s}  {'n':>5}  {'MIDI':>7}  {'ABC':>7}  {'Doremi':>8}  {'Hz':>6}  {'All':>6}",
        f"  {'─' * 63}",
    ]
    for eff_name, d in per_effect.items():
        summary_lines.append(
            f"  {eff_name:12s}  {d['n']:>5}  {d['midi']:>7.1%}  {d['abc']:>7.1%}  "
            f"{d['doremi']:>8.1%}  {d['hz']:>6.1%}  {d['all']:>6.1%}"
        )
    summary_lines += ["", "  Per source (MIDI | ABC | Doremi | Hz | All):"]
    for src, d in per_source.items():
        summary_lines.append(
            f"    {src:12s}: {d['midi']:.1%} | {d['abc']:.1%} | "
            f"{d['doremi']:.1%} | {d['hz']:.1%} | {d['all']:.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=PITCHES, effects=EFFECTS,
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
    save_per_format_iv_plots(records, plots_dir, model_name, iv_key="effect",
                             iv_label="Effect", group_by_source=False)
    save_combined_iv_plot(records, plots_dir, model_name, iv_key="effect",
                          iv_label="Effect")
    return summary


def _save_plot(records: list[dict], run_dir: Path, model_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    eff_names = list(EFFECTS.keys())
    midi_accs, abc_accs, doremi_accs, hz_accs, all_accs = [], [], [], [], []
    for eff_name in eff_names:
        sub = [r for r in records if r["effect"] == eff_name]
        midi_accs.append(sum(r["midi_correct"]   for r in sub) / len(sub) * 100 if sub else 0)
        abc_accs.append( sum(r["abc_correct"]    for r in sub) / len(sub) * 100 if sub else 0)
        doremi_accs.append(sum(r["doremi_correct"] for r in sub) / len(sub) * 100 if sub else 0)
        hz_accs.append(  sum(r["hz_correct"]     for r in sub) / len(sub) * 100 if sub else 0)
        all_accs.append( sum(r["any_correct"]    for r in sub) / len(sub) * 100 if sub else 0)

    x = list(range(len(eff_names)))
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(10, len(eff_names) * 1.2), 4))
    ax.bar([xi - 2.0 * width for xi in x], midi_accs,   width, label="MIDI (integer)")
    ax.bar([xi - 1.0 * width for xi in x], abc_accs,    width, label="ABC (note name)")
    ax.bar([xi + 0.0 * width for xi in x], doremi_accs, width, label="Doremi (solfege)")
    ax.bar([xi + 1.0 * width for xi in x], hz_accs,     width, label="Hz (frequency)")
    ax.bar([xi + 2.0 * width for xi in x], all_accs,    width, label="Any format")
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
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    all_conds = build_conditions()
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {SOURCES}")
    print(f"Pitches    : {len(PITCHES)}  (MIDI {PITCHES[0]}–{PITCHES[-1]})")
    print(f"Effects    : {list(EFFECTS.keys())}")
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
