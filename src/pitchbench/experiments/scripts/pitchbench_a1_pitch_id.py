"""
Experiment 01 — Pitch recognition
Tests whether models can identify a single musical note across all sources
(4 programmatic waveforms + 15 GM instruments rendered with FluidSynth) and
all 49 MIDI pitches in the standard range (C2–C6, MIDI 36–84).

Three prompt variants per stimulus (one row per audio file in the CSV):
  MIDI    — "Reply with ONLY the integer (0–127)."
  ABC     — "Reply with ONLY the note name, e.g. C4, F#3."
  Doremi  — "Reply with ONLY the solfège syllable and accidental (if needed)."

Instruments require FluidSynth; the experiment runs on waveform sources only
if FluidSynth is unavailable.

Usage:
    python experiments/run.py exp_1_pitch
    python experiments/run.py exp_1_pitch --preview
    python experiments/run.py exp_1_pitch --models audio_flamingo_next_instruct
    python experiments/run.py exp_1_pitch --sources sine piano violin
"""

import argparse
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.audit import pitch_record_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    PROMPT_ABC, PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI,
    midi_to_note, standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.plots import (
    save_accuracy_plots, save_bar_plot_by_key, save_combined_iv_plot,
    save_cross_model_pitch_plots, save_per_format_iv_plots,
    save_pitch_prediction_plots,
)
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

# Data-generation parameters (sourced from config.pitchbench_a1_*)
PITCHES          = config.pitchbench_a1_PITCHES
TONE_DURATION_MS = config.pitchbench_a1_TONE_DURATION_MS
ALL_SOURCES      = config.pitchbench_a1_SOURCES
MIDI_MIN         = min(PITCHES)
MIDI_MAX         = max(PITCHES)

PROMPT_MIDI_FULL   = "This audio contains a single musical note. " + PROMPT_MIDI
PROMPT_ABC_FULL    = "This audio contains a single musical note. " + PROMPT_ABC
PROMPT_DOREMI_FULL = "This audio contains a single musical note. " + PROMPT_DOREMI
PROMPT_HZ_FULL     = "This audio contains a single musical note. " + PROMPT_HZ


def build_conditions(sources: list[str]) -> list[dict]:
    return [
        {
            "source":      src,
            "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
            "midi":        midi,
        }
        for src in sources
        for midi in PITCHES
    ]


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    run_dir: Path,
    sample_info: dict | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    print('a1')
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate all audio sequentially (engine.tone caches; FluidSynth
    # is not safe to fan out, and these calls are fast on a warm cache).
    jobs: list[dict[str, Any]] = []
    for c in conds:
        src         = c["source"]
        source_type = c["source_type"]
        midi        = c["midi"]
        try:
            wav = engine.tone(midi, src, TONE_DURATION_MS)
        except ValueError as exc:
            print(f"    [SKIP] {src} MIDI {midi}: {exc}")
            continue
        jobs.append({"source": src, "source_type": source_type, "midi": midi, "wav": wav})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict[str, Any]) -> dict[str, Any]:
        r_m, r_a, r_d, r_h = query_four_formats(
            model_name, job["wav"],
            PROMPT_MIDI_FULL, PROMPT_ABC_FULL, PROMPT_DOREMI_FULL, PROMPT_HZ_FULL,
            verbose=False,
        )
        return standard_pitch_record(
            wav=job["wav"], source=job["source"], source_type=job["source_type"],
            midi_gt=job["midi"],
            raw_midi=r_m["result"], raw_abc=r_a["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=PROMPT_MIDI_FULL,
            prompt_abc=PROMPT_ABC_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL,
            prompt_hz=PROMPT_HZ_FULL,
        )

    raw_results = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{midi_to_note(j['midi']):4s}  {j['source']}",
        result_label_fn=lambda j, r: pitch_record_audit_str(
            r, label=f"{midi_to_note(j['midi']):4s}  {j['source']:<10s}"
        ),
    )
    records: list[dict[str, Any]] = [r for r in raw_results if r is not None]

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_fmt: dict[str, float] = {
        fmt: round(sum(r[f"{fmt}_correct"] for r in records) / max(1, n), 4)
        for fmt in ("midi", "abc", "doremi", "hz")
    }

    sources_seen = sorted({c["source"] for c in conds})
    per_src: dict[str, dict[str, float]] = {}
    per_src_n: dict[str, int] = {}
    for src in sources_seen:
        sub = [r for r in records if r["source"] == src]
        if not sub:
            continue
        per_src_n[src] = len(sub)
        per_src[src] = {
            fmt: round(sum(r[f"{fmt}_correct"] for r in sub) / max(1, len(sub)), 4)
            for fmt in ("midi", "abc", "doremi", "hz")
        }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources: {sources_seen}",
        f"  Pitches: {MIDI_MIN}–{MIDI_MAX}  ({len(PITCHES)} notes)",
        f"  Stimuli: {n}",
        "",
        f"  {'Format':>8}  {'n':>5}  {'Accuracy':>9}",
        f"  {'─' * 28}",
    ]
    for fmt, acc in per_fmt.items():
        summary_lines.append(f"  {fmt.upper():>8}  {n:>5}  {acc:>9.1%}")
    for label, key in (("ABC", "abc"), ("MIDI", "midi"), ("doremi", "doremi"), ("Hz", "hz")):
        summary_lines += ["", f"  Per source ({label} accuracy):"]
        for src, d in per_src.items():
            summary_lines.append(
                f"    {src:16s}: n={per_src_n[src]:>4}  {d.get(key, 0):.1%}"
            )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    summary: dict[str, Any] = {"total": n, "per_format": per_fmt, "per_source": per_src}
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=sources_seen, pitches=PITCHES,
        tone_duration_ms=TONE_DURATION_MS,
        prompt_midi=PROMPT_MIDI_FULL, prompt_abc=PROMPT_ABC_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL, prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    # All plots + their JSON sidecars live under <run_dir>/plots/.
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    long_records = wide_to_long_records(records)
    save_accuracy_plots(
        long_records, plots_dir, model_name,
        instrument_key="source", pitch_key="midi_gt",
        prompt_key="prompt_variant", accuracy_key="exact_match",
    )
    save_pitch_prediction_plots(
        long_records, plots_dir, model_name,
        source_key="source", task_key="midi_gt",
    )
    save_per_format_iv_plots(records, plots_dir, model_name, iv_key="midi_gt", iv_label="Pitch (MIDI)")
    # Combined accuracy-vs-IV bar plots: one with pitch on the x-axis, one with source.
    save_combined_iv_plot(records, plots_dir, model_name, iv_key="midi_gt", iv_label="Pitch (MIDI)")
    save_combined_iv_plot(records, plots_dir, model_name, iv_key="source",  iv_label="Instrument")
    # MIDI-only single-format bar plots (instrument × accuracy and pitch × accuracy).
    save_bar_plot_by_key(
        records, group_key="source", score_key="midi_correct",
        score_label="MIDI accuracy (%)",
        title=f"MIDI accuracy by instrument — {model_name}",
        xlabel="Instrument",
        out_path=plots_dir / f"midi_accuracy_by_source_{model_name}.png",
    )
    try:
        from pitchbench.experiments.helpers.music import midi_to_note as _m2n
        _label = lambda v: f"{v}\n({_m2n(int(v))})" if isinstance(v, int) else str(v)
    except Exception:
        _label = None
    save_bar_plot_by_key(
        records, group_key="midi_gt", score_key="midi_correct",
        score_label="MIDI accuracy (%)",
        title=f"MIDI accuracy by pitch — {model_name}",
        xlabel="Pitch (MIDI)",
        out_path=plots_dir / f"midi_accuracy_by_pitch_{model_name}.png",
        label_formatter=_label,
    )
    return {f"acc_{fmt}": per_fmt[fmt] for fmt in per_fmt}, records


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",      action="store_true")
    parser.add_argument("--sources",      nargs="+", metavar="SRC", default=None,
                        help=f"Sources to run (default: all). Available: {ALL_SOURCES}")
    parser.add_argument("--models",       nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def _apply_sampling(all_conds: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    return apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    sources = args.sources or ALL_SOURCES
    all_conds = build_conditions(sources)
    conds, s_meta = _apply_sampling(all_conds, args)
    n = len(conds)
    print(f"Experiment   : {EXP_NAME}")
    print(f"Sources      : {sources}")
    print(f"Pitches      : {MIDI_MIN}–{MIDI_MAX}  ({len(PITCHES)} MIDI notes)")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print(f"Audio files  : {n}  (generating in {config.AUDIO_DIR})")
    print(f"Queries/model: {n * 4}  (MIDI + ABC + doremi + Hz)")
    print("Generating audio …")
    for c in conds:
        try:
            engine.tone(c["midi"], c["source"], TONE_DURATION_MS)
        except ValueError as exc:
            print(f"  [SKIP] {c['source']} MIDI {c['midi']}: {exc}")
    print("Done. Run without --preview to query the model(s).")


def run() -> dict:
    print(f"Experiment : {EXP_NAME} with MIDI pitches {MIDI_MIN}–{MIDI_MAX} ({len(PITCHES)} notes)")

    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = args.sources or ALL_SOURCES

    available: list[str] = []
    for src in sources:
        if src in config.WAVEFORMS:
            available.append(src)
        else:
            if find_spec("fluidsynth") is not None:
                available.append(src)
            else:
                print(f"  [SKIP] Instrument {src!r}: FluidSynth not installed")

    all_conds = build_conditions(available)
    conds, s_meta = _apply_sampling(all_conds, args)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Sources    : {available}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print(f"Pitches    : {len(PITCHES)}  |  Formats: MIDI + ABC + doremi + Hz")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict[str, float]] = {}
    all_records: dict[str, list[dict[str, Any]]] = {}
    for model_name in target_models:
        summary, records = run_one_model(model_name, conds, run_dir, s_meta)
        all_summaries[model_name] = summary
        all_records[model_name] = records
    save_comparison(run_dir, all_summaries, EXP_NAME)
    long_all = {m: wide_to_long_records(r) for m, r in all_records.items()}
    save_cross_model_pitch_plots(long_all, run_dir, source_key="source", task_key="midi_gt")
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))


if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
