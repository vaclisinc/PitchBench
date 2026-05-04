"""
c2 — Chord pitch count.

Question: can the ALM correctly count the number of distinct pitches in a
*simultaneously*-sounding chord?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    n:                {1..6}                    (chord size)
    same_instrument:  {True, False}             (single-source vs mixed-timbre)
    chord_quality:    {maj, min, dim, aug, dom7, maj7, min7, random_set}
    root_midi:        anchor MIDI used to build the chord
    trial:            for the random_set quality, multiple seeded trials

Fixed conditions: equal level, root-position; trial-level RNG seeded for
reproducibility. Single integer-count prompt; chance grows with the
upper bound on N (we test up to 6).

Usage::
    pitchbench --id c2 --preview
    pitchbench --id c2 --models audio_flamingo_next_instruct
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

# ── IVs ───────────────────────────────────────────────────────────────────────

CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj":         (0, 4, 7),
    "min":         (0, 3, 7),
    "dim":         (0, 3, 6),
    "aug":         (0, 4, 8),
    "dom7":        (0, 4, 7, 10),
    "maj7":        (0, 4, 7, 11),
    "min7":        (0, 3, 7, 10),
    # "random_set" is handled specially per trial
}

QUALITIES_FIXED:   list[str] = list(CHORD_INTERVALS.keys())
QUALITIES:         list[str] = QUALITIES_FIXED + ["random_set"]

ROOT_MIDIS:        list[int] = config.DEFAULT_SELECTION
SAME_INSTRUMENT_OPTS: list[bool] = [True, False]

DEFAULT_N_TRIALS = 1                # per (n, quality, root, dur, source) cell
RANDOM_TRIALS    = 3                # for random_set quality only
DEFAULT_SEED = config.DEFAULT_SEED

# Counts to test. The fixed qualities define ``n`` directly via len(intervals);
# random_set sweeps n=1..6 explicitly.
RANDOM_NS:         list[int] = [1, 2, 3, 4, 5, 6]
RANDOM_PITCH_RANGE = (48, 84)        # pitch range for random_set draws

SOURCES: list[str] = config.ALL_SOURCES

PROMPT = (
    "Listen to this audio. How many distinct musical pitches are sounding "
    "at the same time? Reply with ONLY a single integer. Nothing else. Do not think."
)


# ── Conditions ────────────────────────────────────────────────────────────────

def _mixed_sources(n: int, src_idx_seed: int) -> list[str]:
    """Deterministic per-position source assignment for same_instrument=False."""
    rng = random.Random(src_idx_seed)
    pool = list(SOURCES)
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n)]


def build_conditions(durations_ms: list[int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    def _add(n: int, midis: list[int], quality: str, root: int, src_or_list, dur: int, trial: int):
        same = isinstance(src_or_list, str)
        rows.append({
            "duration_ms":    dur,
            "n":              n,
            "chord_quality":  quality,
            "root_midi":      root,
            "midis":          midis,
            "source":         src_or_list if same else "+".join(src_or_list),
            "_source_arg":    src_or_list,
            "same_instrument": same,
            "trial":          trial,
        })

    for src in SOURCES:
        for dur in durations_ms:
            # Fixed-quality chords (n is determined by quality)
            for quality in QUALITIES_FIXED:
                intervals = CHORD_INTERVALS[quality]
                n = len(intervals)
                for root in ROOT_MIDIS:
                    midis = [root + iv for iv in intervals]
                    if any(m > 96 for m in midis):       # skip out-of-range
                        continue
                    for same in SAME_INSTRUMENT_OPTS:
                        if same:
                            _add(n, midis, quality, root, src, dur, 0)
                        else:
                            srcs = _mixed_sources(n, seed + root + n * 1000)
                            _add(n, midis, quality, root, srcs, dur, 0)

            # Random-set chords for n=1..6
            for n in RANDOM_NS:
                for trial in range(RANDOM_TRIALS):
                    sub_rng = random.Random(seed + 7919 * (n * 100 + trial))
                    midis   = sorted(sub_rng.sample(range(*RANDOM_PITCH_RANGE), n))
                    for same in SAME_INSTRUMENT_OPTS:
                        if same:
                            _add(n, midis, "random_set", midis[0], src, dur, trial)
                        else:
                            srcs = _mixed_sources(n, seed + 991 * (n * 100 + trial))
                            _add(n, midis, "random_set", midis[0], srcs, dur, trial)
    return rows


def _wav_for(c: dict) -> Path:
    return engine.chord(c["midis"], c["_source_arg"], c["duration_ms"])


def _parse_count(text: str) -> int | None:
    m = re.search(r"\b(\d+)\b", (text or "").strip())
    return int(m.group(1)) if m else None


# ── Run one model ─────────────────────────────────────────────────────────────

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
        pred = _parse_count(raw)
        ok   = (pred == c["n"]) if pred is not None else False
        off  = abs(pred - c["n"]) if pred is not None else None

        records.append({
            "duration_ms":     c["duration_ms"],
            "source":          c["source"],
            "same_instrument": c["same_instrument"],
            "chord_quality":   c["chord_quality"],
            "root_midi":       c["root_midi"],
            "midi_set":        str(c["midis"]),
            "note_set":        ", ".join(midi_to_note(m) for m in c["midis"]),
            "n":               c["n"],
            "trial":           c["trial"],
            "wav":             wav,
            "raw_response":    raw,
            "count_pred":      pred,
            "count_correct":   int(ok),
            "off_by":          off,
            "prompt":          PROMPT,
            "model_params":    out["model_params"],
        })
        sym = "✓" if ok else (f"✗(pred={pred})" if pred is not None else "✗(?)")
        print(f"    n={c['n']} {c['chord_quality']:11s} root={midi_to_note(c['root_midi']):4s} "
              f"same={str(c['same_instrument']):5s} dur={c['duration_ms']:>5}ms  {sym}")

    n_total = len(records)
    n_ok    = sum(r["count_correct"] for r in records)
    summary = {
        "total":    n_total,
        "correct":  n_ok,
        "accuracy": round(n_ok / n_total, 4) if n_total else None,
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli  : {n_total}",
        f"  Accuracy : {n_ok}/{n_total} ({n_ok/max(1,n_total):.1%})",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, qualities=QUALITIES, roots=ROOT_MIDIS,
        random_ns=RANDOM_NS, random_trials=RANDOM_TRIALS,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        prompt=PROMPT,
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("count_correct",),
    )
    return summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, args.seed)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    for c in conds:
        try:    _wav_for(c)
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
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, args.seed)
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
