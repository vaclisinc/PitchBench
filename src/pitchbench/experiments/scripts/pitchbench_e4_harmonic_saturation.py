"""
Experiment e4 — Pitch recognition under harmonic saturation
Tests whether models can identify pitch when audio is processed with tanh
soft-clipping (harmonic saturation), which preserves the fundamental frequency
while enriching the harmonic spectrum.

Stimuli: all waveforms + GM instruments × 11 representative pitches × 4 saturation levels.
Four prompts per stimulus: MIDI, SPN, Doremi, Hz.

Saturation levels:
  clean      — no processing
  sat_light  — tanh drive=2  (mild harmonic enrichment)
  sat_medium — tanh drive=5  (moderate, analogous to tape saturation)
  sat_heavy  — tanh drive=20 (strong, approaches hard clipping)

Usage:
    pitchbench e4
    pitchbench e4 --preview
    pitchbench --id e4 --models audio_flamingo_next_instruct
"""

import argparse
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

EXP_NAME = Path(__file__).stem

PITCHES     = config.pitchbench_e4_PITCHES
TONE_MS     = config.pitchbench_e4_TONE_MS
SATURATIONS = config.pitchbench_e4_SATURATIONS
SOURCES     = config.pitchbench_e4_SOURCES

FORMAT_METRICS = {
    "midi": "midi_correct",
    "spn": "spn_correct",
    "doremi": "doremi_correct",
    "hz": "hz_correct",
}

PROMPT_PREFIX    = "Listen to this audio clip of a single musical note. "
PROMPT_MIDI_FULL = PROMPT_PREFIX + PROMPT_MIDI
PROMPT_SPN_FULL  = PROMPT_PREFIX + PROMPT_SPN
PROMPT_DOREMI_FULL = PROMPT_PREFIX + PROMPT_DOREMI
PROMPT_HZ_FULL   = PROMPT_PREFIX + PROMPT_HZ


def build_conditions() -> list[dict]:
    rows = []
    for src in SOURCES:
        for sat_idx, (sat_name, sat_params) in enumerate(SATURATIONS.items()):
            for midi in PITCHES:
                note       = midi_to_note(midi)
                noise_seed = (sat_idx * 1000 + midi) % (2 ** 31)
                rows.append({
                    "source":            src,
                    "saturation_level":  sat_name,
                    "saturation_params": str(sat_params),
                    "saturation_type":   sat_params.get("type", "clean"),
                    "midi":              midi,
                    "note":              note,
                    "noise_seed":        noise_seed,
                })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        engine.tone_with_effect(
            c["midi"], c["source"], TONE_MS,
            c["saturation_level"], SATURATIONS[c["saturation_level"]], c["noise_seed"],
        )


def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    jobs: list[dict] = []
    for c in conds:
        wav = str(engine.tone_with_effect(
            c["midi"], c["source"], TONE_MS,
            c["saturation_level"], SATURATIONS[c["saturation_level"]], c["noise_seed"],
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
            saturation_level=c["saturation_level"],
            saturation_type=c["saturation_type"],
            saturation_params=c["saturation_params"],
            noise_seed=c["noise_seed"],
        )

    raw     = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{j['cond']['source']:10s}  {j['cond']['saturation_level']:12s}  {j['cond']['note']:4s}",
        result_label_fn=lambda j, r: pitch_record_audit_str(
            r, label=f"{j['cond']['source']:10s}  {j['cond']['saturation_level']:12s}  {j['cond']['note']:4s}"
        ),
    )
    records: list[dict] = [r for r in raw if r is not None]

    n = len(records)
    per_sat: dict[str, dict] = {}
    for sat_name in SATURATIONS:
        sub = [r for r in records if r["saturation_level"] == sat_name]
        per_sat[sat_name] = {
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
        "by_saturation": per_sat,
        "by_source":     per_source,
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources     : {SOURCES}",
        f"  Stimuli     : {n}  ({len(SOURCES)} sources × {len(PITCHES)} pitches × {len(SATURATIONS)} levels)",
        f"",
        f"  {'Saturation':12s}  {'n':>5}  {'MIDI':>7}  {'SPN':>7}  {'Doremi':>8}  {'Hz':>6}  {'All':>6}",
        f"  {'─' * 63}",
    ]
    for sat_name, d in per_sat.items():
        summary_lines.append(
            f"  {sat_name:12s}  {d['n']:>5}  {d['midi']:>7.1%}  {d['spn']:>7.1%}  "
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
        sources=SOURCES, pitches=PITCHES, saturations=SATURATIONS,
        tone_duration=TONE_MS / 1000,
        prompt_midi=PROMPT_MIDI_FULL,
        prompt_spn=PROMPT_SPN_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL,
        prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    plots_dir = run_dir / "plots"; plots_dir.mkdir(exist_ok=True)
    save_per_format_iv_plots(records, plots_dir, model_name, iv_key="saturation_level",
                             iv_label="Saturation Level", group_by_source=False)
    save_combined_iv_plot(records, plots_dir, model_name, iv_key="saturation_level",
                          iv_label="Saturation Level")
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
    print(f"Saturation : {list(SATURATIONS.keys())}")
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
