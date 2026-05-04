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
from pitchbench.experiments.helpers.api import get_model_info, query_three_formats
from pitchbench.experiments.helpers.music import (
    PROMPT_ABC, PROMPT_DOREMI, PROMPT_MIDI,
    midi_to_note, standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.plots import save_accuracy_plots, save_cross_model_pitch_plots, save_pitch_prediction_plots
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample

EXP_NAME = Path(__file__).stem

MIDI_MIN = config.DEFAULT_MIDI_MIN
MIDI_MAX = config.DEFAULT_MIDI_MAX

PITCHES: list[int] = list(range(MIDI_MIN, MIDI_MAX + 1))



TONE_DURATION_MS = config.DEFAULT_DURATION_MS

ALL_SOURCES: list[str] = config.ALL_SOURCES

PROMPT_MIDI_FULL   = "This audio contains a single musical note. " + PROMPT_MIDI
PROMPT_ABC_FULL    = "This audio contains a single musical note. " + PROMPT_ABC
PROMPT_DOREMI_FULL = "This audio contains a single musical note. " + PROMPT_DOREMI


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

    records: list[dict[str, Any]] = []
    for c in conds:
        src         = c["source"]
        source_type = c["source_type"]
        midi        = c["midi"]
        try:
            wav = engine.tone(midi, src, TONE_DURATION_MS)
        except ValueError as exc:
            print(f"    [SKIP] {src} MIDI {midi}: {exc}")
            continue

        print(f"    {midi_to_note(midi):4s}  {src}")
        raw_midi, raw_abc, raw_doremi = query_three_formats(
            model_name, wav,
            PROMPT_MIDI_FULL, PROMPT_ABC_FULL, PROMPT_DOREMI_FULL,
        )
        rec = standard_pitch_record(
            wav=wav, source=src, source_type=source_type,
            midi_gt=midi,
            raw_midi=raw_midi, raw_abc=raw_abc, raw_doremi=raw_doremi,
            prompt_midi=PROMPT_MIDI_FULL,
            prompt_abc=PROMPT_ABC_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL,
        )
        records.append(rec)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_fmt: dict[str, float] = {
        fmt: round(sum(r[f"{fmt}_correct"] for r in records) / max(1, n), 4)
        for fmt in ("midi", "abc", "doremi")
    }

    sources_seen = sorted({c["source"] for c in conds})
    per_src: dict[str, dict[str, float]] = {}
    for src in sources_seen:
        sub = [r for r in records if r["source"] == src]
        if not sub:
            continue
        per_src[src] = {
            fmt: round(sum(r[f"{fmt}_correct"] for r in sub) / max(1, len(sub)), 4)
            for fmt in ("midi", "abc", "doremi")
        }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources: {sources_seen}",
        f"  Pitches: {MIDI_MIN}–{MIDI_MAX}  ({len(PITCHES)} notes)",
        f"  Stimuli: {n}",
        "",
        f"  {'Format':>8}  {'Accuracy':>9}",
        f"  {'─' * 20}",
    ]
    for fmt, acc in per_fmt.items():
        summary_lines.append(f"  {fmt.upper():>8}  {acc:>9.1%}")
    summary_lines += ["", "  Per source (ABC accuracy):"]
    for src, d in per_src.items():
        summary_lines.append(f"    {src:16s}: {d.get('abc', 0):.1%}")
    summary_lines += ["", "  Per source (MIDI accuracy):"]
    for src, d in per_src.items():
        summary_lines.append(f"    {src:16s}: {d.get('midi', 0):.1%}")
    summary_lines += ["", "  Per source (doremi accuracy):"]
    for src, d in per_src.items():
        summary_lines.append(f"    {src:16s}: {d.get('doremi', 0):.1%}")

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
        prompt_doremi=PROMPT_DOREMI_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    long_records = wide_to_long_records(records)
    save_accuracy_plots(
        long_records, run_dir, model_name,
        instrument_key="source", pitch_key="midi_gt",
        prompt_key="prompt_variant", accuracy_key="exact_match",
    )
    save_pitch_prediction_plots(
        long_records, run_dir, model_name,
        source_key="source", task_key="midi_gt",
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
    parser.add_argument("--sample-seed",  type=int, default=42,   metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def _apply_sampling(all_conds: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    if args.sample_n is not None:
        return stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed), s_meta
    return all_conds, s_meta


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
    print(f"Queries/model: {n * 3}  (MIDI + ABC + doremi)")
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
    print(f"Pitches    : {len(PITCHES)}  |  Formats: MIDI + ABC + doremi")

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
