"""
d1 — Sequence pitch count.

Question: can the ALM correctly count the number of distinct pitches in a
sequential passage (notes played one after another)?

Universal IVs: duration_ms (per tone), source.
Experiment-specific IVs:
    n:                {1..10}                  (sequence length)
    rhythm:           {regular, irregular}     (irregular gaps drawn from a
                                                seeded uniform distribution)
    pitch_set_seed:   trial index — selects a different seeded random pitch set

Fixed conditions: equal level; non-overlapping notes; pitch range C3..C6;
single integer-count prompt.

Usage::
    pitchbench --id d1 --preview
    pitchbench --id d1 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import midi_to_note
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines


EXP_NAME = Path(__file__).stem

# ── IVs ───────────────────────────────────────────────────────────────────────

N_COUNTS:        list[int] = [1, 2, 3, 4, 5, 7, 10]
RHYTHMS:         list[str] = ["regular", "irregular"]
DEFAULT_GAP_MS               = config.DEFAULT_GAP_MS
DEFAULT_N_TRIALS             = config.DEFAULT_N_TRIALS
DEFAULT_SEED = config.DEFAULT_SEED

PITCH_MIN, PITCH_MAX = 48, 84

SOURCES: list[str] = config.ALL_SOURCES

PROMPT = (
    "Listen to this audio. How many distinct musical pitches are played in "
    "this sequence? Reply with ONLY a single integer. Nothing else. Output only the answer."
)


# ── Conditions ────────────────────────────────────────────────────────────────

def build_conditions(durations_ms: list[int], n_trials: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for src in SOURCES:
        for dur in durations_ms:
            for n in N_COUNTS:
                for trial in range(n_trials):
                    midis = sorted(rng.sample(range(PITCH_MIN, PITCH_MAX + 1), n))
                    # Distinct presented order: shuffled, but length == n (no repeats by construction)
                    presented = list(midis)
                    rng.shuffle(presented)
                    for rhythm in RHYTHMS:
                        if rhythm == "regular":
                            gaps_ms = [DEFAULT_GAP_MS] * (n - 1)
                        else:
                            gaps_ms = [
                                rng.randint(DEFAULT_GAP_MS // 2, DEFAULT_GAP_MS * 3)
                                for _ in range(n - 1)
                            ]
                        rows.append({
                            "source":      src,
                            "duration_ms": dur,
                            "n":           n,
                            "rhythm":      rhythm,
                            "gaps_ms":     gaps_ms,
                            "midi_set":    midis,
                            "presented":   presented,
                            "trial":       trial,
                        })
    return rows


def _wav_for(c: dict) -> Path:
    """Concatenate per-position tones with per-gap silences."""
    SR = engine.SR
    parts: list[np.ndarray] = []
    for i, m in enumerate(c["presented"]):
        # tone() caches its own wav; read it back, then tile gaps inline
        wav_path = engine.tone(m, c["source"], c["duration_ms"])
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        arr = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0)
        parts.append(arr)
        if i < len(c["presented"]) - 1:
            parts.append(np.zeros(int(SR * c["gaps_ms"][i] / 1000), dtype=np.float32))
    audio = np.concatenate(parts)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak

    notes_slug   = "-".join(engine._note_slug(m) for m in c["presented"])
    rhythm_slug  = "reg" if c["rhythm"] == "regular" else "irr"
    name = (
        f"{notes_slug}_{c['source']}_seqcount_n{c['n']}_dur{c['duration_ms']}ms_"
        f"{rhythm_slug}_t{c['trial']}.wav"
    )
    out = engine._audio_dir() / name
    if not out.exists():
        engine._write_wav(out, audio)
    return out


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
            "source":        c["source"],
            "duration_ms":   c["duration_ms"],
            "n":             c["n"],
            "rhythm":        c["rhythm"],
            "trial":         c["trial"],
            "midi_set":      str(c["midi_set"]),
            "note_set":      ", ".join(midi_to_note(m) for m in c["midi_set"]),
            "presented":     str(c["presented"]),
            "wav":           wav,
            "raw_response":  raw,
            "count_pred":    pred,
            "count_correct": int(ok),
            "off_by":        off,
            "prompt":        PROMPT,
            "model_params":  out["model_params"],
        })
        sym = "✓" if ok else (f"✗(pred={pred})" if pred is not None else "✗(?)")
        print(f"    n={c['n']:>2} {c['rhythm']:>9} {c['source']:>10} dur={c['duration_ms']:>5}ms t={c['trial']}  {sym}")

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
        sources=SOURCES, n_counts=N_COUNTS, rhythms=RHYTHMS,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        gap_ms=DEFAULT_GAP_MS,
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
    parser.add_argument("--preview",  action="store_true")
    parser.add_argument("--models",   nargs="+", metavar="MODEL")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, args.n_trials, args.seed)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
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
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, args.n_trials, args.seed)
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
