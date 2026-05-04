"""
a5 — Pitch slightly off (nearest in-tune pitch).

Question: when a tone is slightly detuned, will the ALM snap to the nearest
in-tune pitch?

Universal IVs: duration_ms, midi (the in-tune target), source.
Experiment-specific IVs:
    detune_hz: per-pitch detune values, with |detune| < 50 % of distance to
               the nearest neighbouring semitone (so the answer is unambiguous).
               Each pitch gets the same number of detune levels symmetrically
               around 0; e.g. {-Δmax, -Δmax/2, 0, +Δmax/2, +Δmax}.

Fixed conditions: equal level. Prompt asks for the nearest in-tune pitch in
4 formats; we also record whether the model leaked the off-tune Hz value.

Scoring:
    nearest_correct: predicted MIDI == in-tune target
    midi_within_1, midi_correct, hz_correct, etc. — full 4-format scoring.

Usage::
    pitchbench --id a5 --preview
    pitchbench --id a5 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.music import (
    PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
    midi_to_freq, midi_to_note,
    standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

SOURCES: list[str] = config.ALL_SOURCES
N_DETUNE_LEVELS = 5
DETUNE_FRACTION = 0.40    # |detune| ≤ 40 % of half-distance-to-neighbour, well inside the basin

PROMPT_PREFIX = (
    "This audio contains a single sustained musical note that may be slightly "
    "out of tune. Identify the NEAREST in-tune pitch (the closest standard "
    "musical note). "
)
PROMPT_MIDI_FULL   = PROMPT_PREFIX + PROMPT_MIDI
PROMPT_SPN_FULL    = PROMPT_PREFIX + PROMPT_SPN
PROMPT_DOREMI_FULL = PROMPT_PREFIX + PROMPT_DOREMI
PROMPT_HZ_FULL     = PROMPT_PREFIX + PROMPT_HZ


# ── Conditions ────────────────────────────────────────────────────────────────

def _detune_grid(midi: int) -> list[float]:
    """Symmetric detune values bounded inside the basin of the target pitch."""
    f0 = midi_to_freq(midi)
    f_lo = midi_to_freq(midi - 1)
    f_hi = midi_to_freq(midi + 1)
    half_lo = (f0 - f_lo) / 2     # half-distance to neighbour below
    half_hi = (f_hi - f0) / 2     # half-distance to neighbour above
    max_neg = -half_lo * DETUNE_FRACTION
    max_pos =  half_hi * DETUNE_FRACTION
    if N_DETUNE_LEVELS == 1:
        return [0.0]
    out: list[float] = []
    half = N_DETUNE_LEVELS // 2
    for i in range(-half, half + 1):
        if i < 0:
            out.append(round(max_neg * (i / -half), 4))
        elif i == 0:
            out.append(0.0)
        else:
            out.append(round(max_pos * (i / half), 4))
    return out


def build_conditions(
    durations_ms: list[int],
    pitches:      list[int],
    sources:      list[str],
) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        for midi in pitches:
            for dur in durations_ms:
                for detune_hz in _detune_grid(midi):
                    rows.append({
                        "source":      src,
                        "duration_ms": dur,
                        "midi":        midi,
                        "detune_hz":   detune_hz,
                    })
    return rows


def _wav_for(c: dict) -> Path:
    f = midi_to_freq(c["midi"]) + c["detune_hz"]
    return engine.tone_hz(f, c["source"], c["duration_ms"])


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav = str(_wav_for(c))
        print(f"    {midi_to_note(c['midi']):4s} {c['source']:8s} dur={c['duration_ms']:>4}ms "
              f"detune={c['detune_hz']:+.2f}hz")
        r_m, r_s, r_d, r_h = query_four_formats(
            model_name, wav,
            PROMPT_MIDI_FULL, PROMPT_SPN_FULL, PROMPT_DOREMI_FULL, PROMPT_HZ_FULL,
        )
        rec = standard_pitch_record(
            wav=wav, source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["midi"],
            raw_midi=r_m["result"], raw_spn=r_s["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=PROMPT_MIDI_FULL, prompt_spn=PROMPT_SPN_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL, prompt_hz=PROMPT_HZ_FULL,
            duration_ms=c["duration_ms"],
            detune_hz=c["detune_hz"],
        )
        rec["nearest_correct"] = rec["midi_correct"]      # alias for clarity in plots
        rec["model_params_midi"] = r_m["model_params"]
        records.append(rec)

    n = len(records)
    summary: dict[str, float | int] = {"total": n}
    for fmt in ("midi", "spn", "doremi", "hz"):
        col = f"{fmt}_correct"
        summary[f"acc_{fmt}"] = round(sum(r[col] for r in records) / max(1, n), 4)
    summary["acc_nearest"]      = summary["acc_midi"]
    summary["acc_midi_within_1"] = round(
        sum(r["midi_within_1"] for r in records) / max(1, n), 4
    )

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli : {n}",
        f"  Sources : {SOURCES}",
        f"",
        f"  Nearest-pitch accuracy (4 formats):",
    ]
    for fmt in ("midi", "spn", "doremi", "hz"):
        summary_lines.append(f"    {fmt.upper():>6}  {summary[f'acc_{fmt}']:.1%}")
    summary_lines.append(f"    MIDI±1  {summary['acc_midi_within_1']:.1%}")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=config.DEFAULT_PITCHES,
        durations_ms=[config.DEFAULT_DURATION_MS],
        n_detune_levels=N_DETUNE_LEVELS, detune_fraction=DETUNE_FRACTION,
        prompt_midi=PROMPT_MIDI_FULL, prompt_spn=PROMPT_SPN_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL, prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
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
    all_conds = build_conditions([config.DEFAULT_DURATION_MS], config.DEFAULT_PITCHES, sources)
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
    args    = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = args.sources or SOURCES
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources)
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
    for m in target_models:
        all_summaries[m] = run_one_model(m, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
