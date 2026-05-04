"""
b2 — Onset/offset detection of a single note in silence (no pitch asked).

Question: can the ALM correctly identify the start and end times of a single
musical note inside a long silent clip?

Universal IVs: duration_ms (note length), midi, source.
Experiment-specific IVs:
    pos_ms: onset position inside a 30 s clip, ∈ {2000, 7000, 14000, 22000, 27000}.

Fixed conditions: 30 s clip; one tone per clip; constant level.

Prompt asks for ``MM:SS.cc, MM:SS.cc`` (onset, offset). Scoring uses
``timing_metrics`` (abs error, IoU, ±100/250/500 ms thresholds).

Usage::
    pitchbench --id b2 --preview
    pitchbench --id b2 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import midi_to_note, parse_mm_ss_cc, timing_metrics
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample

EXP_NAME = Path(__file__).stem

POSITIONS_MS: list[int] = [2_000, 7_000, 14_000, 22_000, 27_000]
TOTAL_DUR_MS  = 30_000
SOURCES: list[str] = config.ALL_SOURCES

PROMPT = (
    "This audio is a 30-second clip that contains exactly ONE sustained "
    "musical note inside silence. Identify the onset and offset times of "
    "the note. Reply with ONLY two timestamps in MM:SS.cc format separated "
    "by a comma, e.g. '0:05.20, 0:08.50'. Nothing else. Do not think."
)


def build_conditions(durations_ms: list[int], pitches: list[int], sources: list[str]) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        for midi in pitches:
            for dur in durations_ms:
                for pos in POSITIONS_MS:
                    if pos + dur > TOTAL_DUR_MS:
                        continue
                    rows.append({
                        "source":      src,
                        "midi":        midi,
                        "duration_ms": dur,
                        "pos_ms":      pos,
                    })
    return rows


def _wav_for(c: dict) -> tuple[Path, tuple[float, float]]:
    path, gt = engine.clip_with_notes(
        [(c["midi"], c["pos_ms"], c["duration_ms"])],
        c["source"], TOTAL_DUR_MS,
        name_hint="b2",
    )
    on_s, off_s, _ = gt[0]
    return path, (on_s, off_s)


def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        try:
            wav_path, (on_gt, off_gt) = _wav_for(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        wav  = str(wav_path)
        out  = query_alm(model_name, wav, PROMPT)
        raw  = (out["result"] or "").strip()
        ts   = parse_mm_ss_cc(raw)
        on_p, off_p = (ts[0], ts[1]) if len(ts) >= 2 else (None, None)
        m    = timing_metrics(on_gt, off_gt, on_p, off_p)

        records.append({
            "source":         c["source"],
            "source_type":    "waveform" if c["source"] in config.WAVEFORMS else "instrument",
            "midi":           c["midi"],
            "note":           midi_to_note(c["midi"]),
            "duration_ms":    c["duration_ms"],
            "pos_ms":         c["pos_ms"],
            "onset_s_gt":     round(on_gt, 4),
            "offset_s_gt":    round(off_gt, 4),
            "onset_s_pred":   on_p,
            "offset_s_pred": off_p,
            "wav":            wav,
            "raw_response":   raw,
            "prompt":         PROMPT,
            "iou":            m["iou"],
            "abs_error_on":   m["abs_error_on"],
            "abs_error_off":  m["abs_error_off"],
            "within_100ms_on":   m["within_100ms_on"],
            "within_100ms_off":  m["within_100ms_off"],
            "within_100ms_both": m["within_100ms_both"],
            "within_250ms_on":   m["within_250ms_on"],
            "within_250ms_off":  m["within_250ms_off"],
            "within_250ms_both": m["within_250ms_both"],
            "within_500ms_on":   m["within_500ms_on"],
            "within_500ms_off":  m["within_500ms_off"],
            "within_500ms_both": m["within_500ms_both"],
            "valid":          int(m["valid"]),
            "model_params":   out["model_params"],
        })
        if m["valid"]:
            print(f"    {midi_to_note(c['midi']):>4}/{c['source']:<10} pos={c['pos_ms']/1000:>5.1f}s "
                  f"IoU={m['iou']:.2f}  Δon={m['abs_error_on']:.2f}s  Δoff={m['abs_error_off']:.2f}s")
        else:
            print(f"    {midi_to_note(c['midi']):>4}/{c['source']:<10} INVALID  raw={raw[:40]!r}")

    n   = len(records)
    valid = [r for r in records if r["valid"]]
    summary = {
        "total":        n,
        "valid":        len(valid),
        "mean_iou":     sum(r["iou"] for r in valid) / max(1, n),
        "within_100ms_on":   round(sum(r["within_100ms_on"]   for r in records) / max(1, n), 4),
        "within_100ms_off":  round(sum(r["within_100ms_off"]  for r in records) / max(1, n), 4),
        "within_100ms_both": round(sum(r["within_100ms_both"] for r in records) / max(1, n), 4),
        "within_250ms_on":   round(sum(r["within_250ms_on"]   for r in records) / max(1, n), 4),
        "within_250ms_off":  round(sum(r["within_250ms_off"]  for r in records) / max(1, n), 4),
        "within_250ms_both": round(sum(r["within_250ms_both"] for r in records) / max(1, n), 4),
        "within_500ms_on":   round(sum(r["within_500ms_on"]   for r in records) / max(1, n), 4),
        "within_500ms_off":  round(sum(r["within_500ms_off"]  for r in records) / max(1, n), 4),
        "within_500ms_both": round(sum(r["within_500ms_both"] for r in records) / max(1, n), 4),
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli       : {n}  (parsable: {len(valid)})",
        f"  Mean IoU      : {summary['mean_iou']:.3f}",
        f"  ±100 ms onset : {summary['within_100ms_on']:.1%}",
        f"  ±100 ms offset: {summary['within_100ms_off']:.1%}",
        f"  ±100 ms both  : {summary['within_100ms_both']:.1%}",
        f"  ±250 ms onset : {summary['within_250ms_on']:.1%}",
        f"  ±250 ms offset: {summary['within_250ms_off']:.1%}",
        f"  ±250 ms both  : {summary['within_250ms_both']:.1%}",
        f"  ±500 ms onset : {summary['within_500ms_on']:.1%}",
        f"  ±500 ms offset: {summary['within_500ms_off']:.1%}",
        f"  ±500 ms both  : {summary['within_500ms_both']:.1%}",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=config.DEFAULT_PITCHES,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        positions_ms=POSITIONS_MS, total_dur_ms=TOTAL_DUR_MS,
        prompt=PROMPT,
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("iou",
                       "within_100ms_on", "within_100ms_off", "within_100ms_both",
                       "within_250ms_on", "within_250ms_off", "within_250ms_both",
                       "within_500ms_on", "within_500ms_off", "within_500ms_both"),
    )
    return summary


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
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    for c in conds:
        try:    _wav_for(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
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
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    for c in conds:
        try:    _wav_for(c)
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
