"""
d4 — Discrete melodic contour.

Question: given a sequence of separate notes, can the ALM correctly identify
the directional shape of the melody? The model is asked to output a
comma-separated list of ``up`` / ``down`` tokens, one per transition.

Universal IVs: duration_ms (per note), source.
Experiment-specific IVs:
    base_midi:      ∈ DEFAULT_PITCHES                anchor for the contour
    n_transitions:  {2, 3, 5, 7}                     number of up/down moves
    step_size_st:   {1, 2, 4, 7}                     semitones per move
    note_duration_ms: {250, 500, 1000}               (the "movement speed" axis)
    pattern_seed:   per-trial seed for the up/down sequence

Fixed conditions: notes are separated by silence = note_duration / 4. The
contour is a deterministic seeded sequence of ``up``/``down`` moves.

Scoring:
    transition_accuracy  — per-transition correctness, averaged
    sequence_correct     — full match across all transitions
"""

from __future__ import annotations

import argparse
import random
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

N_TRANSITIONS_OPTS:  list[int] = [2, 3, 5, 7]
STEP_SIZES_ST:       list[int] = [1, 2, 4, 7]
NOTE_DURATIONS_MS:   list[int] = [250, 500, 1000]
DEFAULT_TRIALS_PER_CELL = 2
DEFAULT_SEED = config.DEFAULT_SEED

SOURCES: list[str] = config.ALL_SOURCES

PROMPT = (
    "Listen to this sequence of separate musical notes. For each TRANSITION "
    "between consecutive notes, reply with 'up' or 'down', comma-separated, "
    "in order. Example for a 4-note ascending-then-descending pattern: "
    "'up, up, down'. Reply with ONLY the list."
)


def _build_pattern(n_transitions: int, seed: int) -> list[str]:
    """Deterministic ``up``/``down`` sequence avoiding all-up or all-down trivial patterns."""
    rng = random.Random(seed)
    while True:
        seq = [rng.choice(["up", "down"]) for _ in range(n_transitions)]
        if not (all(s == "up" for s in seq) or all(s == "down" for s in seq)) or n_transitions <= 1:
            return seq


def build_conditions(durations_ms: list[int], pitches: list[int]) -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for base in pitches:
            for n_t in N_TRANSITIONS_OPTS:
                for step in STEP_SIZES_ST:
                    for note_dur in NOTE_DURATIONS_MS:
                        for trial in range(DEFAULT_TRIALS_PER_CELL):
                            cell_seed = (DEFAULT_SEED ^ hash((src, base, n_t, step, note_dur, trial))) & 0xFFFFFFFF
                            pattern = _build_pattern(n_t, cell_seed)
                            # Build pitches: each transition moves up or down by step
                            midis = [base]
                            cur = base
                            for d in pattern:
                                cur = cur + step if d == "up" else cur - step
                                midis.append(cur)
                            if min(midis) < 12 or max(midis) > 96:
                                continue
                            rows.append({
                                "source":           src,
                                "duration_ms":      note_dur,        # universal IV slot
                                "base_midi":        base,
                                "n_transitions":    n_t,
                                "step_size_st":     step,
                                "note_duration_ms": note_dur,
                                "pattern":          pattern,
                                "midis":            midis,
                                "seed":             cell_seed,
                            })
    return rows


def _wav_for(c: dict) -> Path:
    gap = c["note_duration_ms"] // 4
    return engine.sequence(c["midis"], c["source"], c["note_duration_ms"], gap)


_TOKEN_RE = re.compile(r"\b(up|down)\b", re.IGNORECASE)


def _parse_pattern(text: str) -> list[str]:
    return [m.group(1).lower() for m in _TOKEN_RE.finditer(text or "")]


def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav  = str(_wav_for(c))
        out  = query_alm(model_name, wav, PROMPT)
        raw  = (out["result"] or "").strip()
        pred = _parse_pattern(raw)
        gt   = c["pattern"]
        # Per-transition accuracy: align by index, missing tokens count as wrong
        n_corr = sum(1 for i, g in enumerate(gt) if i < len(pred) and pred[i] == g)
        per_trans = n_corr / max(1, len(gt))
        seq_ok = (pred == gt)

        records.append({
            "duration_ms":          c["note_duration_ms"],
            "source":               c["source"],
            "source_type":          "waveform" if c["source"] in config.WAVEFORMS else "instrument",
            "base_midi":            c["base_midi"],
            "n_transitions":        c["n_transitions"],
            "step_size_st":         c["step_size_st"],
            "note_duration_ms":     c["note_duration_ms"],
            "midi_seq":             ",".join(str(m) for m in c["midis"]),
            "note_seq":             ",".join(midi_to_note(m) for m in c["midis"]),
            "pattern_gt":           ",".join(gt),
            "pattern_pred":         ",".join(pred),
            "transition_accuracy":  round(per_trans, 4),
            "sequence_correct":     int(seq_ok),
            "wav":                  wav,
            "raw_response":         raw,
            "prompt":               PROMPT,
            "seed":                 c["seed"],
            "model_params":         out["model_params"],
        })

    n = len(records)
    summary = {
        "total":               n,
        "transition_accuracy": round(sum(r["transition_accuracy"] for r in records) / max(1, n), 4),
        "sequence_correct":    round(sum(r["sequence_correct"]    for r in records) / max(1, n), 4),
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli              : {n}",
        f"  Per-transition acc   : {summary['transition_accuracy']:.1%}",
        f"  Full-sequence acc    : {summary['sequence_correct']:.1%}",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES,
        n_transitions_opts=N_TRANSITIONS_OPTS,
        step_sizes_st=STEP_SIZES_ST,
        note_durations_ms=NOTE_DURATIONS_MS,
        trials_per_cell=DEFAULT_TRIALS_PER_CELL,
        prompt=PROMPT,
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("transition_accuracy", "sequence_correct"),
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
