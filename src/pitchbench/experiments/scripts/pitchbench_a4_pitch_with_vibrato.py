"""
a4 — Pitch with vibrato.

Question: can the ALM correctly identify the nominal centre pitch of a tone
that is frequency-modulated (vibrato)?

Universal IVs: duration_ms, midi, source.
Experiment-specific IVs:
    vibrato_rate_hz:    {0, 3, 5, 7, 10}
    vibrato_depth_cents: {0, 25, 50, 100, 200}
                        (rate=0 OR depth=0 ⇒ flat-pitch control; flagged
                         in the ``is_control`` column)

Fixed conditions: waveforms only (FluidSynth pitch-bend would interact with
preset envelopes); equal level. Phase-accumulated FM in ``engine.tone_with_vibrato``.

Scoring: standard 4-format pitch-ID (MIDI / SPN / Doremi / Hz). Both
``midi_correct`` and ``midi_within_1`` are reported because depths > 100
cents can straddle a semitone boundary.

Usage::
    pitchbench --id a4 --preview
    pitchbench --id a4 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.audit import pitch_record_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
    midi_to_note, standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.plots import save_combined_iv_plot, save_per_format_iv_plots
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

# Data-generation parameters (sourced from config.pitchbench_a4_*)
VIBRATO_RATES_HZ     = config.pitchbench_a4_VIBRATO_RATES_HZ
VIBRATO_DEPTHS_CENTS = config.pitchbench_a4_VIBRATO_DEPTHS_CENTS
DURATIONS_MS         = config.pitchbench_a4_DURATIONS_MS
SOURCES              = config.pitchbench_a4_SOURCES
PITCHES              = config.pitchbench_a4_PITCHES

PROMPT_PREFIX = (
    "This audio contains a single sustained musical note that may have vibrato. "
    "Identify the nominal CENTRE pitch (ignore the vibrato modulation). "
)

PROMPT_MIDI_FULL   = PROMPT_PREFIX + PROMPT_MIDI
PROMPT_SPN_FULL    = PROMPT_PREFIX + PROMPT_SPN
PROMPT_DOREMI_FULL = PROMPT_PREFIX + PROMPT_DOREMI
PROMPT_HZ_FULL     = PROMPT_PREFIX + PROMPT_HZ


# ── Conditions ────────────────────────────────────────────────────────────────

def build_conditions(
    durations_ms: list[int],
    pitches:      list[int],
    sources:      list[str],
) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        for midi in pitches:
            for dur in durations_ms:
                for rate in VIBRATO_RATES_HZ:
                    for depth in VIBRATO_DEPTHS_CENTS:
                        is_control = (rate == 0) or (depth == 0)
                        rows.append({
                            "source":              src,
                            "duration_ms":         dur,
                            "midi":                midi,
                            "vibrato_rate_hz":     rate,
                            "vibrato_depth_cents": depth,
                            "is_control":          int(is_control),
                        })
    return rows


def _wav_for(c: dict) -> Path:
    return engine.tone_with_vibrato(
        c["midi"], c["source"], c["duration_ms"],
        c["vibrato_rate_hz"], c["vibrato_depth_cents"],
    )


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        wav = str(_wav_for(c))
        jobs.append({"wav": wav, "cond": c})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict) -> dict:
        c = job["cond"]
        r_m, r_s, r_d, r_h = query_four_formats(
            model_name, job["wav"],
            PROMPT_MIDI_FULL, PROMPT_SPN_FULL, PROMPT_DOREMI_FULL, PROMPT_HZ_FULL,
            verbose=False,
        )
        rec = standard_pitch_record(
            wav=job["wav"], source=c["source"], source_type="waveform",
            midi_gt=c["midi"],
            raw_midi=r_m["result"], raw_spn=r_s["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=PROMPT_MIDI_FULL, prompt_spn=PROMPT_SPN_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL, prompt_hz=PROMPT_HZ_FULL,
            duration_ms=c["duration_ms"],
            vibrato_rate_hz=c["vibrato_rate_hz"],
            vibrato_depth_cents=c["vibrato_depth_cents"],
            is_control=c["is_control"],
        )
        rec["model_params_midi"] = r_m["model_params"]
        return rec

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{midi_to_note(j['cond']['midi']):4s} {j['cond']['source']:8s} dur={j['cond']['duration_ms']:>4}ms vib={j['cond']['vibrato_rate_hz']:>4}hz/{j['cond']['vibrato_depth_cents']:>4}c",
        result_label_fn=lambda j, r: pitch_record_audit_str(r, label=f"{midi_to_note(j['cond']['midi']):4s} {j['cond']['source']:8s} vib={j['cond']['vibrato_rate_hz']:>4}hz/{j['cond']['vibrato_depth_cents']:>4}c"),
    )
    records: list[dict] = [r for r in raw if r is not None]

    n = len(records)
    summary: dict[str, float | int] = {"total": n}
    for fmt in ("midi", "spn", "doremi", "hz"):
        col = f"{fmt}_correct"
        summary[f"acc_{fmt}"] = round(sum(r[col] for r in records) / max(1, n), 4)
    summary["acc_midi_within_1"] = round(
        sum(r["midi_within_1"] for r in records) / max(1, n), 4
    )

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli : {n}",
        f"  Sources : {SOURCES}",
        "",
        f"  {'Format':>10}  {'n':>5}  {'Accuracy':>10}",
        f"  {'─' * 32}",
    ]
    for fmt in ("midi", "spn", "doremi", "hz"):
        summary_lines.append(f"  {fmt.upper():>10}  {n:>5}  {summary[f'acc_{fmt}']:>10.1%}")
    summary_lines.append(f"  {'MIDI±1':>10}  {n:>5}  {summary['acc_midi_within_1']:>10.1%}")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=PITCHES,
        durations_ms=DURATIONS_MS,
        vibrato_rates_hz=VIBRATO_RATES_HZ,
        vibrato_depths_cents=VIBRATO_DEPTHS_CENTS,
        prompt_midi=PROMPT_MIDI_FULL, prompt_spn=PROMPT_SPN_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL, prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    plots_dir = run_dir / "plots"; plots_dir.mkdir(exist_ok=True)
    save_per_format_iv_plots(records, plots_dir, model_name, iv_key="vibrato_depth_cents",
                             iv_label="Vibrato depth (cents)", group_by_source=False)
    save_combined_iv_plot(records, plots_dir, model_name, iv_key="vibrato_depth_cents",
                          iv_label="Vibrato depth (cents)")
    return summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    parser.add_argument("--sources", nargs="+", metavar="SRC", default=None)
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args    = _parse_args()
    sources = args.sources or SOURCES
    all_conds = build_conditions(DURATIONS_MS, PITCHES, sources)
    conds = all_conds  # rename: save the full list
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    for c in conds:
        _wav_for(c)
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {sources}")
    print(f"Stimuli    : {len(conds)}")
    print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = args.sources or SOURCES
    all_conds = build_conditions(DURATIONS_MS, PITCHES, sources)
    conds = all_conds  # rename: save the full list
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    for c in conds:
        _wav_for(c)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    all_records:   dict[str, list[dict]] = {}
    for m in target_models:
        s = run_one_model(m, conds, run_dir, s_meta)
        all_summaries[m] = s
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
