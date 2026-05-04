"""
b5 — Onset/offset detection of each note in a sequence.

Question: given a sequence of N notes, can the ALM detect when every note
starts and ends?

Universal IVs: duration_ms (per note), source.
Experiment-specific IVs:
    n_notes:        {3, 5, 8}
    rhythm:         {regular, irregular}    (irregular = seeded random gaps)
    pitch_pattern:  {fixed_pitch, varied_pitches}

Fixed conditions: 30 s clip; non-overlapping; equal level. Per-cell RNG seed
makes the irregular-rhythm clip pattern reproducible.

Scoring strategy (academic-paper-friendly):
    Match predicted notes to ground-truth notes by Hungarian assignment on
    midpoint distance. Then aggregate:
      mean_iou        — average over matched pairs (0 for unmatched GT notes)
      mean_abs_on/off — mean |error| over matched pairs (seconds)
      precision       — matched / total predicted
      recall          — matched / total GT
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import midi_to_note, parse_mm_ss_cc
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

N_NOTES_OPTS:    list[int] = [3, 5, 8]
RHYTHMS:         list[str] = ["regular", "irregular"]
PITCH_PATTERNS:  list[str] = ["fixed_pitch", "varied_pitches"]
TOTAL_DUR_MS = config.DEFAULT_TOTAL_DUR_MS
DEFAULT_SEED = config.DEFAULT_SEED

IRR_GAP_MIN, IRR_GAP_MAX = 300, 2500

SOURCES: list[str] = config.ALL_SOURCES

PROMPT = (
    "This audio contains a sequence of musical notes separated by silence. "
    "List the onset and offset of EVERY note in order, as comma-separated "
    "MM:SS.cc timestamps (onset, offset, onset, offset, …). "
    "Example for 2 notes: '0:01.20, 0:02.50, 0:03.10, 0:04.00'. "
    "Reply with the timestamp list only."
)


def build_conditions(durations_ms: list[int], pitches: list[int], sources: list[str], seed: int) -> list[dict]:
    rng_seed = seed
    rows: list[dict] = []
    for src in sources:
        for n in N_NOTES_OPTS:
            for rhythm in RHYTHMS:
                for pattern in PITCH_PATTERNS:
                    for dur in durations_ms:
                        cell_seed = (rng_seed ^ hash((src, n, rhythm, pattern, dur))) & 0xFFFFFFFF
                        sub_rng   = random.Random(cell_seed)
                        if pattern == "fixed_pitch":
                            base = sub_rng.choice(pitches)
                            midis = [base] * n
                        else:
                            midis = sub_rng.sample(pitches, n)
                        if rhythm == "regular":
                            # Regular rhythm: equal gaps that exactly fill TOTAL_DUR_MS.
                            total_gap_budget = TOTAL_DUR_MS - dur * n
                            if total_gap_budget < 0:
                                continue
                            gap_size = total_gap_budget // (n + 1)
                            gaps = [gap_size] * (n + 1)
                        else:
                            gaps = [sub_rng.randint(IRR_GAP_MIN, IRR_GAP_MAX) for _ in range(n + 1)]
                            if sum(gaps) + dur * n > TOTAL_DUR_MS:
                                continue
                        onsets = []
                        cursor = gaps[0]
                        for _ in range(n):
                            onsets.append(cursor)
                            cursor += dur + gaps[len(onsets)]
                        rows.append({
                            "source":      src,
                            "n_notes":     n,
                            "rhythm":      rhythm,
                            "pitch_pattern": pattern,
                            "duration_ms": dur,
                            "midi_seq":    midis,
                            "onsets_ms":   onsets,
                            "seed":        cell_seed,
                        })
    return rows


def _wav_for(c: dict) -> Path:
    triples = [(m, c["onsets_ms"][i], c["duration_ms"]) for i, m in enumerate(c["midi_seq"])]
    path, _ = engine.clip_with_notes(triples, c["source"], TOTAL_DUR_MS, name_hint="b5")
    return path


def _greedy_match(gt_pairs: list[tuple[float, float]],
                  pred_pairs: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Greedy minimum-midpoint-distance matching (good enough for small N)."""
    used_p: set[int] = set()
    matches: list[tuple[int, int]] = []
    for i, (g_on, g_off) in enumerate(gt_pairs):
        g_mid = (g_on + g_off) / 2
        best_j, best_d = -1, float("inf")
        for j, (p_on, p_off) in enumerate(pred_pairs):
            if j in used_p:
                continue
            p_mid = (p_on + p_off) / 2
            d = abs(p_mid - g_mid)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used_p.add(best_j)
            matches.append((i, best_j))
    return matches


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav  = str(_wav_for(c))
        out  = query_alm(model_name, wav, PROMPT)
        raw  = (out["result"] or "").strip()
        ts   = parse_mm_ss_cc(raw)
        # Pair consecutive timestamps as (onset, offset)
        pred_pairs = [(ts[i], ts[i + 1]) for i in range(0, len(ts) - 1, 2)
                      if ts[i + 1] >= ts[i]]
        gt_pairs   = [(c["onsets_ms"][i] / 1000,
                       (c["onsets_ms"][i] + c["duration_ms"]) / 1000)
                      for i in range(c["n_notes"])]
        matches = _greedy_match(gt_pairs, pred_pairs)
        ious   = [_iou(gt_pairs[g], pred_pairs[p]) for g, p in matches]
        on_err = [abs(gt_pairs[g][0] - pred_pairs[p][0]) for g, p in matches]
        off_err = [abs(gt_pairs[g][1] - pred_pairs[p][1]) for g, p in matches]
        precision = len(matches) / len(pred_pairs) if pred_pairs else 0.0
        recall    = len(matches) / len(gt_pairs)
        mean_iou  = sum(ious) / len(matches) if matches else 0.0

        records.append({
            "source":         c["source"],
            "source_type":    "waveform" if c["source"] in config.WAVEFORMS else "instrument",
            "duration_ms":    c["duration_ms"],
            "n_notes":        c["n_notes"],
            "rhythm":         c["rhythm"],
            "pitch_pattern":  c["pitch_pattern"],
            "midi_seq":       str(c["midi_seq"]),
            "note_seq":       ", ".join(midi_to_note(m) for m in c["midi_seq"]),
            "onsets_ms":      str(c["onsets_ms"]),
            "n_pred":         len(pred_pairs),
            "n_matched":      len(matches),
            "wav":            wav,
            "raw_response":   raw,
            "prompt":         PROMPT,
            "mean_iou":       round(mean_iou, 4),
            "mean_abs_err_on":  round(sum(on_err)  / max(1, len(on_err)),  4) if on_err else None,
            "mean_abs_err_off": round(sum(off_err) / max(1, len(off_err)), 4) if off_err else None,
            "precision":      round(precision, 4),
            "recall":         round(recall, 4),
            "seed":           c["seed"],
            "model_params":   out["model_params"],
        })

    n = len(records)
    summary = {
        "total":     n,
        "mean_iou":  round(sum(r["mean_iou"]  for r in records) / max(1, n), 4),
        "precision": round(sum(r["precision"] for r in records) / max(1, n), 4),
        "recall":    round(sum(r["recall"]    for r in records) / max(1, n), 4),
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli   : {n}",
        f"  Mean IoU  : {summary['mean_iou']:.3f}",
        f"  Precision : {summary['precision']:.3f}",
        f"  Recall    : {summary['recall']:.3f}",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, durations_ms=config.DEFAULT_DURATIONS_MS,
        n_notes_opts=N_NOTES_OPTS, rhythms=RHYTHMS, pitch_patterns=PITCH_PATTERNS,
        total_dur_ms=TOTAL_DUR_MS,
        prompt=PROMPT,
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("mean_iou", "precision", "recall"),
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
        try: _wav_for(c)
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
