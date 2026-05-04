"""
d6 — Pitch ranking.

Question: given N tones in sequence, can the model rank them low→high? Tests
comparison across time beyond the pairwise case (d2).

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    n:              {3, 4, 5}                (number of tones to rank)
    rhythm:         {regular, irregular}     (irregular = seeded random gaps)
    delta_cents:    spacing between adjacent rank levels {25, 50, 100, 200, 400}
    base_name:      {A3, A4, A5}             (anchor frequency for the spacing)

Fixed conditions: pure-sine waveforms only (Hz tones unsupported on
instruments); equal level. The presented order of the N tones is permuted
deterministically per trial via a seeded RNG.

Scoring: ``answer_correct`` = exact match of the predicted permutation.
``kendall_tau`` and ``spearman_rho`` are also recorded for partial credit.

Usage::
    pitchbench --id d6 --preview
    pitchbench --id d6 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.results import (
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample


EXP_NAME = Path(__file__).stem

# ── IVs ───────────────────────────────────────────────────────────────────────

BASE_FREQS:  dict[str, float] = {"A3": 220.00, "A4": 440.00, "A5": 880.00}
DELTA_CENTS: list[int]        = [25, 50, 100, 200, 400]
N_TONES:     list[int]        = [3, 4, 5]
RHYTHMS:     list[str]        = ["regular", "irregular"]

DEFAULT_DURATION_MS = config.DEFAULT_DURATION_MS   # per tone
DEFAULT_GAP_MS      = 300    # base inter-tone silence (rhythm=regular)
DEFAULT_N_TRIALS    = 3
DEFAULT_SEED = config.DEFAULT_SEED

SOURCES: list[str] = list(config.WAVEFORMS)   # Hz tones unsupported on instruments


def _prompt(n: int) -> str:
    example = " ".join(str(i) for i in range(n, 0, -1))
    return (
        f"{n} pure tones play one after another, separated by brief silences. "
        f"Rank them from lowest pitch to highest pitch. "
        f'Reply with ONLY the order as positions (1..{n}), e.g. "{example}" '
        f"means the {n}th tone played is lowest and the 1st is highest."
    )


# ── Conditions ────────────────────────────────────────────────────────────────

def _cents_above(base_hz: float, cents: float) -> float:
    return base_hz * (2 ** (cents / 1200))


def build_conditions(
    durations_ms: list[int],
    n_trials: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for src in SOURCES:
        for base_name, base_hz in BASE_FREQS.items():
            for delta in DELTA_CENTS:
                for n in N_TONES:
                    sorted_freqs = [_cents_above(base_hz, delta * i) for i in range(n)]
                    for dur_ms in durations_ms:
                        for rhythm in RHYTHMS:
                            for trial in range(n_trials):
                                perm = list(range(n))
                                rng.shuffle(perm)
                                presented = [sorted_freqs[p] for p in perm]
                                # answer_gt: positions sorted by ascending pitch (1-indexed)
                                gt = " ".join(
                                    str(i + 1) for i in sorted(range(n), key=lambda i: presented[i])
                                )
                                if rhythm == "regular":
                                    gaps_ms = [DEFAULT_GAP_MS] * (n - 1)
                                else:
                                    gaps_ms = [
                                        rng.randint(DEFAULT_GAP_MS // 2, DEFAULT_GAP_MS * 3)
                                        for _ in range(n - 1)
                                    ]
                                rows.append({
                                    "source":       src,
                                    "duration_ms":  dur_ms,
                                    "base_name":    base_name,
                                    "delta_cents":  delta,
                                    "n_notes":      n,
                                    "rhythm":       rhythm,
                                    "presented_hz": [round(f, 4) for f in presented],
                                    "gaps_ms":      gaps_ms,
                                    "perm":         perm,
                                    "answer_gt":    gt,
                                    "trial":        trial,
                                })
    return rows


def _wav_for(c: dict) -> Path:
    """Manually concatenate Hz tones with per-position gaps (engine.sequence_hz uses constant gap)."""
    import numpy as np
    SR = engine.SR
    parts: list = []
    for i, f in enumerate(c["presented_hz"]):
        # render via tone_hz then read the WAV back; cheaper: synth inline.
        wav_path = engine.tone_hz(f, c["source"], c["duration_ms"])
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            n   = wf.getnframes()
            raw = wf.readframes(n)
        arr = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0)
        parts.append(arr)
        if i < len(c["presented_hz"]) - 1:
            parts.append(np.zeros(int(SR * c["gaps_ms"][i] / 1000), dtype=np.float32))
    audio = np.concatenate(parts)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak
    presented_slug = "-".join(f"{round(f):d}hz" for f in c["presented_hz"])
    rhythm_slug    = "reg" if c["rhythm"] == "regular" else "irr"
    name           = f"{presented_slug}_{c['source']}_rank_n{c['n_notes']}_dur{c['duration_ms']}ms_{rhythm_slug}_t{c['trial']}.wav"
    out = engine._audio_dir() / name
    if not out.exists():
        engine._write_wav(out, audio)
    return out


def _parse_perm(text: str, n: int) -> str | None:
    nums = [int(x) for x in re.findall(r"\b([1-9])\b", text or "")]
    seen: list[int] = []
    for v in nums:
        if 1 <= v <= n and v not in seen:
            seen.append(v)
        if len(seen) == n:
            return " ".join(str(x) for x in seen)
    return None


def _kendall_tau(pred: list[int], gt: list[int]) -> float:
    """Normalised Kendall tau in [-1, 1]. ``pred`` and ``gt`` are 1-indexed permutations."""
    if len(pred) != len(gt):
        return 0.0
    n = len(pred)
    if n < 2:
        return 1.0
    pairs = n * (n - 1) // 2
    concord = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pred.index(gt[i]) < pred.index(gt[j])):
                concord += 1
    return (2 * concord - pairs) / pairs


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav    = str(_wav_for(c))
        prompt = _prompt(c["n_notes"])
        out    = query_alm(model_name, wav, prompt)
        raw    = (out["result"] or "").strip()
        pred   = _parse_perm(raw, c["n_notes"])
        exact  = (pred == c["answer_gt"]) if pred else False
        tau    = _kendall_tau(
            [int(x) for x in pred.split()] if pred else [],
            [int(x) for x in c["answer_gt"].split()],
        ) if pred else 0.0

        records.append({
            "source":         c["source"],
            "duration_ms":    c["duration_ms"],
            "n_notes":        c["n_notes"],
            "rhythm":         c["rhythm"],
            "base_name":      c["base_name"],
            "delta_cents":    c["delta_cents"],
            "trial":          c["trial"],
            "presented_hz":   str(c["presented_hz"]),
            "wav":            wav,
            "raw_response":   raw,
            "answer_gt":      c["answer_gt"],
            "answer_pred":    pred,
            "answer_correct": int(exact),
            "kendall_tau":    round(tau, 4),
            "prompt":         prompt,
            "model_params":   out["model_params"],
        })
        sym = "✓" if exact else "✗"
        print(f"    {sym} n={c['n_notes']} {c['base_name']} Δ={c['delta_cents']}c "
              f"{c['rhythm']:>9}  [{c['answer_gt']}]  → {pred or '???'}  τ={tau:+.2f}")

    n     = len(records)
    n_ok  = sum(r["answer_correct"] for r in records)
    mean_tau = round(sum(r["kendall_tau"] for r in records) / max(1, n), 4)
    summary = {
        "total":       n,
        "correct":     n_ok,
        "accuracy":    round(n_ok / n, 4) if n else None,
        "kendall_tau": mean_tau,
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources       : {SOURCES}",
        f"  Stimuli       : {n}",
        f"  Exact match   : {n_ok}/{n}  ({n_ok/max(1,n):.1%})",
        f"  Mean τ        : {mean_tau:+.3f}",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, base_freqs=BASE_FREQS, delta_cents=DELTA_CENTS,
        n_tones=N_TONES, rhythms=RHYTHMS,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("answer_correct", "kendall_tau"),
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
    parser.add_argument("--sample-seed",  type=int, default=42,   metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, args.n_trials, args.seed)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    for c in conds:
        _wav_for(c)
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {SOURCES}")
    print(f"Trials     : {len(conds)}")
    print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, args.n_trials, args.seed)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
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


if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
