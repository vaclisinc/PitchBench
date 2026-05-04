"""
b3 — Onset/offset detection of a *specific* note among multiple notes.

Question: given several non-overlapping musical notes in a clip, can the ALM
identify the onset and offset of one named target note?

Universal IVs: duration_ms (per note), source.
Experiment-specific IVs:
    target_midi:    pitch of the target (drawn from DEFAULT_PITCHES)
    n_distractors:  {2, 4}
    target_pos:     {first, middle, last}        position of target in the sequence
    seed:           per-trial RNG seed for distractor pitches and gap layout

Fixed conditions: 30 s clip; non-overlapping notes; gaps drawn from
``[500, 2000]`` ms with a per-trial seed; equal level. The target pitch is
named in the prompt.

Scoring: ``timing_metrics`` on the target's onset/offset.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import midi_to_note, parse_mm_ss_cc, timing_metrics
from pitchbench.experiments.helpers.results import (
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample

EXP_NAME = Path(__file__).stem

N_DISTRACTORS:  list[int] = [2, 4]
TARGET_POS_OPTS: list[str] = ["first", "middle", "last"]
TOTAL_DUR_MS = 30_000
GAP_MIN_MS, GAP_MAX_MS = 500, 2000
DEFAULT_SEED = config.DEFAULT_SEED

SOURCES: list[str] = config.ALL_SOURCES


def _prompt_for(target_note: str) -> str:
    return (
        f"This audio contains a sequence of musical notes separated by silence. "
        f"Identify the onset and offset times of the note {target_note} "
        f"(it appears exactly once). Reply with ONLY two timestamps in MM:SS.cc "
        f"format separated by a comma, e.g. '0:05.20, 0:08.50'. Nothing else. Do not think."
    )


def build_conditions(durations_ms: list[int], pitches: list[int], sources: list[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for src in sources:
        for tgt in pitches:
            for nd in N_DISTRACTORS:
                for pos in TARGET_POS_OPTS:
                    for dur in durations_ms:
                        # distractor pitches: drawn from DEFAULT_PITCHES, excluding target
                        candidates = [p for p in pitches if p != tgt]
                        # deterministic per-cell seed
                        cell_seed = (seed
                                     ^ hash((src, tgt, nd, pos, dur))) & 0xFFFFFFFF
                        sub_rng   = random.Random(cell_seed)
                        distractors = sub_rng.sample(candidates, nd)
                        # build sequence of (n_distractors + 1) notes; place target at pos
                        notes = list(distractors)
                        if pos == "first":
                            notes.insert(0, tgt)
                        elif pos == "last":
                            notes.append(tgt)
                        else:                         # middle
                            notes.insert(len(notes) // 2, tgt)
                        # layout: gaps drawn from [GAP_MIN_MS, GAP_MAX_MS]
                        gaps = [sub_rng.randint(GAP_MIN_MS, GAP_MAX_MS)
                                for _ in range(len(notes) + 1)]   # leading + interior + trailing
                        # check fits in clip
                        total = sum(gaps) + dur * len(notes)
                        if total > TOTAL_DUR_MS:
                            continue
                        # absolute onsets
                        onsets: list[int] = []
                        cursor = gaps[0]
                        for i, _ in enumerate(notes):
                            onsets.append(cursor)
                            cursor += dur + gaps[i + 1]
                        target_idx = notes.index(tgt)
                        rows.append({
                            "source":       src,
                            "midi":         tgt,
                            "duration_ms":  dur,
                            "n_distractors": nd,
                            "target_pos":   pos,
                            "midi_seq":     notes,
                            "onsets_ms":    onsets,
                            "target_onset_ms":  onsets[target_idx],
                            "target_offset_ms": onsets[target_idx] + dur,
                            "seed":         cell_seed,
                        })
    return rows


def _wav_for(c: dict) -> Path:
    triples = [(m, c["onsets_ms"][i], c["duration_ms"]) for i, m in enumerate(c["midi_seq"])]
    path, _ = engine.clip_with_notes(triples, c["source"], TOTAL_DUR_MS, name_hint="b3")
    return path


def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav    = str(_wav_for(c))
        prompt = _prompt_for(midi_to_note(c["midi"]))
        out    = query_alm(model_name, wav, prompt)
        raw    = (out["result"] or "").strip()
        ts     = parse_mm_ss_cc(raw)
        on_p, off_p = (ts[0], ts[1]) if len(ts) >= 2 else (None, None)
        on_gt   = c["target_onset_ms"]  / 1000
        off_gt  = c["target_offset_ms"] / 1000
        m       = timing_metrics(on_gt, off_gt, on_p, off_p)

        records.append({
            "source":           c["source"],
            "source_type":      "waveform" if c["source"] in config.WAVEFORMS else "instrument",
            "midi":             c["midi"],
            "note":             midi_to_note(c["midi"]),
            "duration_ms":      c["duration_ms"],
            "n_distractors":    c["n_distractors"],
            "target_pos":       c["target_pos"],
            "midi_seq":         str(c["midi_seq"]),
            "onsets_ms":        str(c["onsets_ms"]),
            "onset_s_gt":       round(on_gt, 4),
            "offset_s_gt":      round(off_gt, 4),
            "onset_s_pred":     on_p,
            "offset_s_pred":    off_p,
            "wav":              wav,
            "raw_response":     raw,
            "prompt":           prompt,
            "iou":              m["iou"],
            "abs_error_on":     m["abs_error_on"],
            "abs_error_off":    m["abs_error_off"],
            "within_500ms_on":  m["within_500ms_on"],
            "within_500ms_off": m["within_500ms_off"],
            "within_250ms_on":  m["within_250ms_on"],
            "within_250ms_off": m["within_250ms_off"],
            "valid":            int(m["valid"]),
            "seed":             c["seed"],
            "model_params":     out["model_params"],
        })

    n     = len(records)
    valid = [r for r in records if r["valid"]]
    summary = {
        "total":            n,
        "valid":            len(valid),
        "mean_iou":         round(sum(r["iou"] for r in valid) / max(1, len(valid)), 4),
        "within_500ms_on":  round(sum(r["within_500ms_on"]  for r in records) / max(1, n), 4),
        "within_500ms_off": round(sum(r["within_500ms_off"] for r in records) / max(1, n), 4),
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli   : {n}  (valid: {len(valid)})",
        f"  Mean IoU  : {summary['mean_iou']:.3f}",
        f"  ±500 ms on/off: {summary['within_500ms_on']:.1%} / {summary['within_500ms_off']:.1%}",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=config.DEFAULT_PITCHES,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        n_distractors=N_DISTRACTORS, target_pos_opts=TARGET_POS_OPTS,
        total_dur_ms=TOTAL_DUR_MS,
        gap_range_ms=[GAP_MIN_MS, GAP_MAX_MS],
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("iou", "within_500ms_on", "within_500ms_off",
                       "within_250ms_on", "within_250ms_off"),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    parser.add_argument("--sources", nargs="+", metavar="SRC", default=None)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=42,   metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args    = _parse_args()
    sources = args.sources or SOURCES
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources, args.seed)
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


def run() -> None:
    engine.set_exp(EXP_NAME)
    args    = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = args.sources or SOURCES
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources, args.seed)
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


if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
