"""
Experiment e5 — Time stretching vs resampling: pitch robustness under tempo change.

Tests whether ALMs can correctly identify pitch when audio duration is altered
via two fundamentally different operations:

  resample  — simulates playback speed change; duration AND pitch both change
                (analogous to slowing/speeding a record or tape)
  stretch   — phase-vocoder time stretching; duration changes, pitch is preserved

Conditions:
  clean         — no modification (3 s)
  resample_0.5x — 2× speed (shorter, ~1.5 s), pitch RISES 12 st  (GT = midi + 12)
  resample_2x   — ½× speed (longer, ~6 s),    pitch DROPS 12 st  (GT = midi − 12)
  stretch_0.5x  — 2× speed (shorter, ~1.5 s), pitch UNCHANGED    (GT = midi)
  stretch_2x    — ½× speed (longer,  ~6 s),   pitch UNCHANGED    (GT = midi)

Pitches: restricted to MIDI 36–84 so ±12-semitone shifts remain in audible range.
Sources: all waveforms + GM instruments.
Four prompts per stimulus: MIDI, SPN, Doremi, Hz.

Usage:
    pitchbench e5
    pitchbench e5 --preview
    pitchbench --id e5 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.audit import pitch_record_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
    format_accuracy_dict,
    midi_to_note, standard_pitch_record,
)
from pitchbench.experiments.helpers.plots import save_combined_iv_plot, save_per_format_iv_plots
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME   = Path(__file__).stem

PITCHES    = config.pitchbench_e5_PITCHES
TONE_MS    = config.pitchbench_e5_TONE_MS
CONDITIONS = config.pitchbench_e5_CONDITIONS
SOURCES    = config.pitchbench_e5_SOURCES

FORMAT_METRICS = {
    "midi": "midi_correct",
    "spn": "spn_correct",
    "doremi": "doremi_correct",
    "hz": "hz_correct",
}

PROMPT_PREFIX      = (
    "Listen to this audio clip of a single musical note. The recording "
    "may have been sped up or slowed down. Identify the PITCH of the note "
    "as it sounds in the audio. "
)
PROMPT_MIDI_FULL   = PROMPT_PREFIX + PROMPT_MIDI
PROMPT_SPN_FULL    = PROMPT_PREFIX + PROMPT_SPN
PROMPT_DOREMI_FULL = PROMPT_PREFIX + PROMPT_DOREMI
PROMPT_HZ_FULL     = PROMPT_PREFIX + PROMPT_HZ


def _gt_midi(original_midi: int, mode: str, factor: float) -> int:
    """Ground-truth MIDI for a given condition.

    factor is a duration ratio (output/input):
        factor=2.0 → twice as long → half speed → pitch drops one octave (−12 st)
        factor=0.5 → half as long → double speed → pitch rises one octave (+12 st)
    For stretch/clean: pitch is always unchanged.
    """
    if mode == "resample" and factor != 1.0:
        return original_midi + round(-12.0 * math.log2(factor))
    return original_midi


def build_conditions() -> list[dict]:
    rows = []
    for src in SOURCES:
        for cond in CONDITIONS:
            for midi in PITCHES:
                gt = _gt_midi(midi, cond["mode"], cond["factor"])
                rows.append({
                    "source":        src,
                    "condition":     cond["name"],
                    "mode":          cond["mode"],
                    "factor":        cond["factor"],
                    "original_midi": midi,
                    "midi":          gt,          # used as ground-truth throughout
                    "note":          midi_to_note(gt),
                    "original_note": midi_to_note(midi),
                })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        engine.tone_time_modified(
            c["original_midi"], c["source"], TONE_MS,
            c["mode"], c["factor"],
        )


def run_one_model(
    model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    jobs: list[dict] = []
    for c in conds:
        wav = str(engine.tone_time_modified(
            c["original_midi"], c["source"], TONE_MS,
            c["mode"], c["factor"],
        ))
        jobs.append({"wav": wav, "cond": c})

    def _query_one(job: dict) -> dict:
        c = job["cond"]
        r_m, r_s, r_d, r_h = query_four_formats(
            model_name, job["wav"],
            PROMPT_MIDI_FULL, PROMPT_SPN_FULL, PROMPT_DOREMI_FULL, PROMPT_HZ_FULL,
            verbose=False,
        )
        return standard_pitch_record(
            wav=job["wav"],
            source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["midi"],
            raw_midi=r_m["result"],
            raw_spn=r_s["result"],
            raw_doremi=r_d["result"],
            raw_hz=r_h["result"],
            prompt_midi=PROMPT_MIDI_FULL,
            prompt_spn=PROMPT_SPN_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL,
            prompt_hz=PROMPT_HZ_FULL,
            condition=c["condition"],
            mode=c["mode"],
            speed_factor=c["factor"],
            original_midi=c["original_midi"],
            original_note=c["original_note"],
        )

    raw     = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: (
            f"{j['cond']['source']:10s}  {j['cond']['condition']:14s}  "
            f"{j['cond']['original_note']:4s}→{j['cond']['note']:4s}"
        ),
        result_label_fn=lambda j, r: pitch_record_audit_str(
            r, label=(
                f"{j['cond']['source']:10s}  {j['cond']['condition']:14s}  "
                f"{j['cond']['original_note']:4s}→{j['cond']['note']:4s}"
            )
        ),
    )
    records: list[dict] = [r for r in raw if r is not None]

    n            = len(records)
    cond_names   = [c["name"] for c in CONDITIONS]

    per_cond: dict[str, dict] = {}
    for cond_name in cond_names:
        sub = [r for r in records if r["condition"] == cond_name]
        per_cond[cond_name] = {
            "n":      len(sub),
            **format_accuracy_dict(sub, FORMAT_METRICS),
        }

    per_source: dict[str, dict] = {}
    for src in SOURCES:
        sub = [r for r in records if r["source"] == src]
        per_source[src] = format_accuracy_dict(sub, FORMAT_METRICS)

    summary = {
        "total":          n,
        "accuracy":   format_accuracy_dict(records, FORMAT_METRICS),
        "by_condition":  per_cond,
        "by_source":     per_source,
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources     : {SOURCES}",
        f"  Stimuli     : {n}  ({len(SOURCES)} sources × {len(PITCHES)} pitches × {len(CONDITIONS)} conditions)",
        f"  Base dur.   : {TONE_MS} ms  (output length varies with speed factor)",
        f"",
        f"  {'Condition':14s}  {'n':>5}  {'MIDI':>7}  {'SPN':>7}  {'Doremi':>8}  {'Hz':>6}  {'All':>6}",
        f"  {'─' * 67}",
    ]
    for cond_name, d in per_cond.items():
        summary_lines.append(
            f"  {cond_name:14s}  {d['n']:>5}  {d['midi']:>7.1%}  {d['spn']:>7.1%}  "
            f"{d['doremi']:>8.1%}  {d['hz']:>6.1%}  {d['all']:>6.1%}"
        )
    summary_lines += ["", "  Per source (MIDI | SPN | Doremi | Hz | All):"]
    for src, d in per_source.items():
        summary_lines.append(
            f"    {src:12s}: {d['midi']:.1%} | {d['spn']:.1%} | "
            f"{d['doremi']:.1%} | {d['hz']:.1%} | {d['all']:.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=PITCHES, conditions=CONDITIONS,
        tone_duration_ms=TONE_MS,
        prompt_midi=PROMPT_MIDI_FULL,
        prompt_spn=PROMPT_SPN_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL,
        prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    plots_dir = run_dir / "plots"; plots_dir.mkdir(exist_ok=True)
    save_per_format_iv_plots(records, plots_dir, model_name, iv_key="condition",
                             iv_label="Condition", group_by_source=False)
    save_combined_iv_plot(records, plots_dir, model_name, iv_key="condition",
                          iv_label="Condition")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sample-n",    type=int, default=None, metavar="N")
    parser.add_argument("--sample-seed", type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
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
    print(f"Conditions : {[c['name'] for c in CONDITIONS]}")
    print(f"Base dur.  : {TONE_MS} ms")
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


if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
