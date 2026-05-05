"""
d6 — Pitch ranking.

Given N tones in sequence, can the model rank them low→high? Tests
comparison across time beyond the pairwise case (d2).

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    n_notes:        {3, 4, 5}
    rhythm:         {regular, irregular}
    delta_cents:    {25, 50, 100, 200, 400}
    base_name:      {A3, A4, A5}

Headline metric: ``answer_correct`` — exact match of the predicted
permutation. ``kendall_tau`` is recorded as a per-stim diagnostic only
(continuous float, doesn't end in ``_correct``, so invisible to the
auto-marginals / accuracies CSV).
"""

from __future__ import annotations

import random
import re
import wave
from pathlib import Path

import numpy as np

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_d import CatDSpec, run_cat_d_experiment

EXP_NAME            = Path(__file__).stem
BASE_FREQS          = config.pitchbench_d6_BASE_FREQS
DELTA_CENTS         = config.pitchbench_d6_DELTA_CENTS
N_TONES             = config.pitchbench_d6_N_TONES
RHYTHMS             = config.pitchbench_d6_RHYTHMS
DEFAULT_DURATION_MS = config.pitchbench_d6_DURATION_MS
DEFAULT_GAP_MS      = config.pitchbench_d6_GAP_MS
DEFAULT_N_TRIALS    = config.pitchbench_d6_N_TRIALS
DEFAULT_SEED        = config.pitchbench_d6_SEED
SOURCES             = config.pitchbench_d6_SOURCES


def _prompt(n: int) -> str:
    example = " ".join(str(i) for i in range(n, 0, -1))
    return (
        f"{n} tones play one after another, separated by brief silences. "
        f"Rank them from lowest pitch to highest pitch. "
        f'Reply with ONLY the order as positions (1..{n}), space-separated, e.g. "{example}" '
        f"means the {n}th tone played is lowest and the 1st is highest."
    )


def _cents_above(base_hz: float, cents: float) -> float:
    return base_hz * (2 ** (cents / 1200))


def build_conditions() -> list[dict]:
    rng = random.Random(DEFAULT_SEED)
    rows: list[dict] = []
    for src in SOURCES:
        for base_name, base_hz in BASE_FREQS.items():
            for delta in DELTA_CENTS:
                for n in N_TONES:
                    sorted_freqs = [_cents_above(base_hz, delta * i) for i in range(n)]
                    for dur_ms in config.DEFAULT_DURATIONS_MS:
                        for rhythm in RHYTHMS:
                            for trial in range(DEFAULT_N_TRIALS):
                                perm = list(range(n))
                                rng.shuffle(perm)
                                presented = [sorted_freqs[p] for p in perm]
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


def wav_for(c: dict) -> Path:
    """Manually concatenate Hz tones with per-position gaps."""
    SR = engine.SR
    parts: list = []
    for i, f in enumerate(c["presented_hz"]):
        wav_path = engine.tone_hz(f, c["source"], c["duration_ms"])
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


def prompts_for(c: dict) -> dict[str, str]:
    return {"main": _prompt(c["n_notes"])}


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
    if len(pred) != len(gt):
        return 0.0
    n = len(pred)
    if n < 2:
        return 1.0
    pairs = n * (n - 1) // 2
    concord = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pred.index(gt[i]) < pred.index(gt[j]):
                concord += 1
    return (2 * concord - pairs) / pairs


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw   = responses["main"]
    pred  = _parse_perm(raw, c["n_notes"])
    exact = (pred == c["answer_gt"]) if pred else False
    tau   = _kendall_tau(
        [int(x) for x in pred.split()] if pred else [],
        [int(x) for x in c["answer_gt"].split()],
    ) if pred else 0.0
    return {
        "source":         c["source"],
        "duration_ms":    c["duration_ms"],
        "n_notes":        c["n_notes"],
        "rhythm":         c["rhythm"],
        "base_name":      c["base_name"],
        "delta_cents":    c["delta_cents"],
        "trial":          c["trial"],
        "presented_hz":   str(c["presented_hz"]),
        "answer_gt":      c["answer_gt"],
        "raw_response":   raw,
        "answer_pred":    pred,
        "answer_correct": int(exact),
        "kendall_tau":    round(tau, 4),
    }


SPEC = CatDSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="ranking",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("answer",),
    record_extras=(
        "duration_ms", "n_notes", "rhythm", "base_name", "delta_cents", "trial",
    ),
    label_fn=lambda j: (
        f"n={j['cond']['n_notes']} {j['cond']['base_name']} "
        f"Δ={j['cond']['delta_cents']}c {j['cond']['rhythm']:>9}"
    ),
)


def preview() -> None:
    run_cat_d_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_d_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
