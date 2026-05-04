"""
d3 — Sequential interval identification.

Question: given two notes played one after another, can the ALM correctly
report the interval between them — as a signed integer number of
semitones (positive = ascending, negative = descending)?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    base_midi:       ∈ DEFAULT_PITCHES
    interval_st:     {1..12}
    direction:       {ascending, descending}
    separation_ms:   {200, 500, 1000, 2000}

Fixed conditions: non-overlapping notes; equal level; pure waveforms or
instruments (the model should be timbre-agnostic for interval ID).

Scoring:
    interval_correct  → predicted signed semitones == ground truth
    interval_within_1 → |pred − gt| ≤ 1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import midi_to_note
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample


EXP_NAME = Path(__file__).stem

INTERVALS_ST:  list[int] = list(range(1, 13))
DIRECTIONS:    list[str] = ["ascending", "descending"]
SEPARATIONS_MS: list[int] = [200, 500, 1000, 2000]
SOURCES:       list[str] = config.ALL_SOURCES

PROMPT = (
    "Two musical notes play in sequence, separated by a brief silence. "
    "How many semitones apart are they? Reply with a SIGNED integer where "
    "positive = ascending (the second note is higher) and negative = "
    "descending. Examples: '7' for an ascending fifth, '-3' for a descending "
    "minor third. Reply with ONLY the integer."
)

_INT_RE = re.compile(r"-?\d+")


def _parse_signed_st(text: str) -> int | None:
    m = _INT_RE.search(text or "")
    if not m:
        return None
    v = int(m.group(0))
    return v if -36 <= v <= 36 else None


def build_conditions(durations_ms: list[int], pitches: list[int]) -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for base in pitches:
            for iv in INTERVALS_ST:
                for direction in DIRECTIONS:
                    other = base + iv if direction == "ascending" else base - iv
                    if other < 12 or other > 96:
                        continue
                    for dur in durations_ms:
                        for sep in SEPARATIONS_MS:
                            rows.append({
                                "source":        src,
                                "duration_ms":   dur,
                                "base_midi":     base,
                                "interval_st":   iv,
                                "direction":     direction,
                                "separation_ms": sep,
                                "midis":         [base, other],
                                "signed_st":     iv if direction == "ascending" else -iv,
                            })
    return rows


def _wav_for(c: dict) -> Path:
    return engine.sequence(c["midis"], c["source"], c["duration_ms"], c["separation_ms"])


def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        try:
            wav = str(_wav_for(c))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        out  = query_alm(model_name, wav, PROMPT)
        raw  = (out["result"] or "").strip()
        pred = _parse_signed_st(raw)
        ok        = (pred == c["signed_st"]) if pred is not None else False
        within_1  = (pred is not None and abs(pred - c["signed_st"]) <= 1)

        records.append({
            "duration_ms":       c["duration_ms"],
            "source":            c["source"],
            "source_type":       "waveform" if c["source"] in config.WAVEFORMS else "instrument",
            "base_midi":         c["base_midi"],
            "base_note":         midi_to_note(c["base_midi"]),
            "interval_st":       c["interval_st"],
            "direction":         c["direction"],
            "separation_ms":     c["separation_ms"],
            "midis":             "+".join(str(m) for m in c["midis"]),
            "wav":               wav,
            "raw_response":      raw,
            "interval_pred":     pred,
            "interval_correct":  int(ok),
            "interval_within_1": int(within_1),
            "prompt":            PROMPT,
            "model_params":      out["model_params"],
        })

    n = len(records)
    summary = {
        "total":             n,
        "interval_correct":  round(sum(r["interval_correct"]  for r in records) / max(1, n), 4),
        "interval_within_1": round(sum(r["interval_within_1"] for r in records) / max(1, n), 4),
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli       : {n}",
        f"  Exact accuracy: {summary['interval_correct']:.1%}",
        f"  ±1 semitone   : {summary['interval_within_1']:.1%}",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, intervals_st=INTERVALS_ST, directions=DIRECTIONS,
        separations_ms=SEPARATIONS_MS,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        prompt=PROMPT,
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("interval_correct", "interval_within_1"),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=42,   metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {SOURCES}")
    print(f"Stimuli    : {len(conds)}")
    print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError: pass

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
