"""
Experiment 10 — Pitch duration
Tests whether pitch identification accuracy depends on how long the tone lasts.
Sweeps 7 durations from 50 ms to 4 s across all waveforms and 11 representative
MIDI pitches.

Three prompt variants: MIDI, ABC, Doremi.
Plots: accuracy vs. duration curve per instrument (Plot A) + per pitch (Plot B).

Usage:
    python experiments/run.py exp_10_pitch_duration
    python experiments/run.py exp_10_pitch_duration --preview
    python experiments/run.py exp_10_pitch_duration --models audio_flamingo_next_instruct
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
    standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.plots import (
    save_accuracy_plots, save_combined_iv_plot, save_per_format_iv_plots,
)
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

PITCHES: list[int] = config.DEFAULT_SELECTION 

DURATIONS_MS: list[int] = [50, 100, 250, 500, 1_000, 2_000, 4_000, 5_000, 15_000, 60_000]

SOURCES: list[str] = config.ALL_SOURCES

PROMPT_MIDI_FULL   = "This audio contains a single musical pitch. " + PROMPT_MIDI
PROMPT_ABC_FULL    = "This audio contains a single musical pitch. " + PROMPT_ABC
PROMPT_DOREMI_FULL = "This audio contains a single musical pitch. " + PROMPT_DOREMI
PROMPT_HZ_FULL     = "This audio contains a single musical pitch. " + PROMPT_HZ


# ── Conditions / stimulus generation ─────────────────────────────────────────

def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for midi in PITCHES:
        note = midi_to_note(midi)
        for dur_ms in DURATIONS_MS:
            for src in SOURCES:
                rows.append({
                    "midi":        midi,
                    "note":        note,
                    "duration_ms": dur_ms,
                    "source":      src,
                })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        engine.tone(c["midi"], c["source"], c["duration_ms"])


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    run_dir: Path,
    sample_info: dict | None = None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        wav = str(engine.tone(c["midi"], c["source"], c["duration_ms"]))
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
            duration_ms=c["duration_ms"],
        )

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{j['cond']['note']:4s} {j['cond']['duration_ms']:>5}ms {j['cond']['source']:10s}",
        result_label_fn=lambda j, r: pitch_record_audit_str(r, label=f"{j['cond']['note']:4s} {j['cond']['duration_ms']:>5}ms {j['cond']['source']:10s}"),
    )
    records: list[dict] = [r for r in raw if r is not None]

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_duration: dict[int, dict] = {}
    for dur_ms in DURATIONS_MS:
        sub = [r for r in records if r["duration_ms"] == dur_ms]
        per_duration[dur_ms] = {
            "n":      len(sub),
            "midi":   round(sum(r["midi_correct"]   for r in sub) / max(1, len(sub)), 4),
            "abc":    round(sum(r["abc_correct"]    for r in sub) / max(1, len(sub)), 4),
            "doremi": round(sum(r["doremi_correct"] for r in sub) / max(1, len(sub)), 4),
            "hz":     round(sum(r["hz_correct"]     for r in sub) / max(1, len(sub)), 4),
        }

    per_source: dict[str, dict] = {}
    for src in SOURCES:
        sub = [r for r in records if r["source"] == src]
        per_source[src] = {
            "midi":   round(sum(r["midi_correct"]   for r in sub) / max(1, len(sub)), 4),
            "abc":    round(sum(r["abc_correct"]    for r in sub) / max(1, len(sub)), 4),
            "doremi": round(sum(r["doremi_correct"] for r in sub) / max(1, len(sub)), 4),
            "hz":     round(sum(r["hz_correct"]     for r in sub) / max(1, len(sub)), 4),
        }

    summary = {
        "total":        n,
        "midi_correct":   sum(r["midi_correct"]   for r in records),
        "midi_within_1":  sum(r["midi_within_1"]  for r in records),
        "abc_correct":    sum(r["abc_correct"]    for r in records),
        "doremi_correct": sum(r["doremi_correct"] for r in records),
        "hz_correct":     sum(r["hz_correct"]     for r in records),
        "per_duration": {str(k): v for k, v in per_duration.items()},
        "per_source":   per_source,
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources : {SOURCES}",
        f"  Stimuli : {n}  ({len(PITCHES)} pitches × {len(DURATIONS_MS)} durations × {len(SOURCES)} sources)",
        "",
        f"  {'Duration':>10}  {'n':>5}  {'MIDI%':>7}  {'ABC%':>7}  {'Doremi%':>9}  {'Hz%':>6}",
        f"  {'─' * 56}",
    ]
    for dur_ms, d in per_duration.items():
        summary_lines.append(
            f"  {dur_ms:>8}ms  {d['n']:>5}  {d['midi']:>7.1%}  "
            f"{d['abc']:>7.1%}  {d['doremi']:>9.1%}  {d['hz']:>6.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        pitches=PITCHES, durations_ms=DURATIONS_MS, sources=SOURCES,
        prompt_midi=PROMPT_MIDI_FULL,
        prompt_abc=PROMPT_ABC_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL,
        prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    long_records = wide_to_long_records(records)
    save_accuracy_plots(
        long_records, run_dir, model_name,
        instrument_key="source",
        pitch_key="midi_gt",
        prompt_key="prompt_variant",
        accuracy_key="exact_match",
    )
    save_per_format_iv_plots(records, run_dir, model_name, iv_key="duration_ms", iv_label="Duration (ms)")
    save_combined_iv_plot(records, run_dir, model_name, iv_key="duration_ms", iv_label="Duration (ms)")
    return summary


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
    print(f"Pitches    : {len(PITCHES)}")
    print(f"Durations  : {DURATIONS_MS} ms")
    print(f"Stimuli    : {len(conds)} × 4 variants = {len(conds)*4} queries/model")
    print(f"Audio dir  : {config.AUDIO_DIR}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds = build_conditions()
    conds = all_conds  # rename: save the full list
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)} × 4 variants")
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
