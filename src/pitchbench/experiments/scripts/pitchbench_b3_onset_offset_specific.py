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
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import midi_to_note, parse_mm_ss_cc, timing_metrics
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

N_DISTRACTORS:  list[int] = [5, 7]
TARGET_POS_OPTS: list[str] = ["first", "middle", "last"]
TOTAL_DUR_MS = config.DEFAULT_TOTAL_DUR_MS
GAP_MIN_MS, GAP_MAX_MS = 500, 2000
DEFAULT_SEED = config.DEFAULT_SEED

SOURCES: list[str] = config.ALL_SOURCES


def _prompt_for(target_note: str) -> str:
    return (
        f"This audio contains a sequence of musical notes separated by silence. "
        f"Identify the onset and offset times of the note {target_note} "
        f"(it appears exactly once). Reply with ONLY two timestamps in MM:SS.cc "
        f"format separated by a comma, e.g. '0:05.20, 0:08.50'. Nothing else. Output only the answer."
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
                        gaps = [sub_rng.randrange(GAP_MIN_MS, GAP_MAX_MS + 10, 10)
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

    # Phase 1: generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        wav    = str(_wav_for(c))
        prompt = _prompt_for(f"{midi_to_note(c['midi'])} (MIDI {c['midi']})")
        jobs.append({"wav": wav, "cond": c, "prompt": prompt})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        prompt  = job["prompt"]
        out     = query_alm(model_name, job["wav"], prompt)
        raw     = (out["result"] or "").strip()
        ts      = parse_mm_ss_cc(raw)
        on_p, off_p = (ts[0], ts[1]) if len(ts) >= 2 else (None, None)
        on_gt   = c["target_onset_ms"]  / 1000
        off_gt  = c["target_offset_ms"] / 1000
        m       = timing_metrics(on_gt, off_gt, on_p, off_p)
        return {
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
            "wav":              job["wav"],
            "raw_response":     raw,
            "prompt":           prompt,
            "iou":              m["iou"],
            "abs_error_on":     m["abs_error_on"],
            "abs_error_off":    m["abs_error_off"],
            "within_100ms_on":   m["within_100ms_on"],
            "within_100ms_off":  m["within_100ms_off"],
            "within_100ms_both": m["within_100ms_both"],
            "within_250ms_on":   m["within_250ms_on"],
            "within_250ms_off":  m["within_250ms_off"],
            "within_250ms_both": m["within_250ms_both"],
            "within_500ms_on":   m["within_500ms_on"],
            "within_500ms_off":  m["within_500ms_off"],
            "within_500ms_both": m["within_500ms_both"],
            "valid":            int(m["valid"]),
            "seed":             c["seed"],
            "model_params":     out["model_params"],
        }

    raw_results = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{midi_to_note(j['cond']['midi']):>4}/{j['cond']['source']:<10} nd={j['cond']['n_distractors']} pos={j['cond']['target_pos']}",
        result_label_fn=lambda j, r: audit_line(
            f"{midi_to_note(j['cond']['midi']):>4}/{j['cond']['source']:<10} nd={j['cond']['n_distractors']} pos={j['cond']['target_pos']}",
            gt=f"({r['onset_s_gt']},{r['offset_s_gt']})",
            pred=f"({r['onset_s_pred']},{r['offset_s_pred']})",
            score=r.get('iou'),
            correct=bool(r.get('within_100ms_both')),
        ),
    )
    records: list[dict] = [r for r in raw_results if r is not None]

    n     = len(records)
    valid = [r for r in records if r["valid"]]
    summary = {
        "total":            n,
        "valid":            len(valid),
        "mean_iou":          round(sum(r["iou"] for r in valid) / max(1, n), 4),
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
        f"  Stimuli   : {n}  (valid: {len(valid)})",
        f"  Mean IoU  : {summary['mean_iou']:.3f}",
        f"  ±100 ms on/off: {summary['within_100ms_on']:.1%} / {summary['within_100ms_off']:.1%}",
        f"  ±100 ms both  : {summary['within_100ms_both']:.1%}",
        f"  ±250 ms on/off: {summary['within_250ms_on']:.1%} / {summary['within_250ms_off']:.1%}",
        f"  ±250 ms both  : {summary['within_250ms_both']:.1%}",
        f"  ±500 ms on/off: {summary['within_500ms_on']:.1%} / {summary['within_500ms_off']:.1%}",
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
        n_distractors=N_DISTRACTORS, target_pos_opts=TARGET_POS_OPTS,
        total_dur_ms=TOTAL_DUR_MS,
        gap_range_ms=[GAP_MIN_MS, GAP_MAX_MS],
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
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args    = _parse_args()
    sources = args.sources or SOURCES
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources, args.seed)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
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
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources, args.seed)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
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
