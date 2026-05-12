"""
d2 — Pitch difference (binary higher/lower).

Question: given two sequential tones, which one is higher in pitch, as a
function of the cents difference between them and the absolute base frequency?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    base_name:       {A3, A4, A5}
    delta_cents:     {1, 2, 5, 10, 25, 50, 100, 200, 400, 700, 1200}
    separation_ms:   {200, 500, 1000, 2000}
    order:           {0=low first, 1=high first}

The model is asked a single binary question; chance = 50 %.
"""

from __future__ import annotations

import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.experiments.helpers.cat_d import CatDSpec, run_cat_d_experiment

EXP_NAME            = Path(__file__).stem
BASE_FREQS          = config.pitchbench_d2_BASE_FREQS
DELTA_CENTS         = config.pitchbench_d2_DELTA_CENTS
SEPARATION_MS       = config.pitchbench_d2_SEPARATION_MS
DEFAULT_DURATION_MS = config.pitchbench_d2_DURATION_MS
DEFAULT_N_TRIALS    = config.pitchbench_d2_N_TRIALS
DEFAULT_SEED        = config.pitchbench_d2_SEED
SOURCES             = config.pitchbench_d2_SOURCES

PROMPT = (
    "Two tones play one after another, separated by a brief silence. "
    "Which tone is higher in pitch — the first or the second? "
    'Reply with ONLY "first" or "second".'
)


def _cents_above(base_hz: float, cents: float) -> float:
    return base_hz * (2 ** (cents / 1200))


def build_conditions() -> list[dict]:
    rng = random.Random(DEFAULT_SEED)
    rows: list[dict] = []
    for src in SOURCES:
        for base_name, base_hz in BASE_FREQS.items():
            for delta in DELTA_CENTS:
                high_hz = _cents_above(base_hz, delta)
                for dur_ms in config.DEFAULT_DURATIONS_MS:
                    for sep_ms in SEPARATION_MS:
                        for trial in range(DEFAULT_N_TRIALS):
                            order = rng.randint(0, 1)
                            rows.append({
                                "source":        src,
                                "duration_ms":   dur_ms,
                                "separation_ms": sep_ms,
                                "base_name":     base_name,
                                "base_hz":       round(base_hz, 4),
                                "delta_cents":   delta,
                                "high_hz":       round(high_hz, 4),
                                "order":         order,
                                "answer_gt":     "first" if order == 1 else "second",
                                "trial":         trial,
                            })
    return rows


def wav_for(c: dict) -> Path:
    freqs = ([c["base_hz"], c["high_hz"]] if c["order"] == 0
             else [c["high_hz"], c["base_hz"]])
    return engine.sequence_hz(freqs, c["source"], c["duration_ms"], c["separation_ms"])


def prompts_for(_: dict) -> dict[str, str]:
    return {"main": PROMPT}


def _parse_binary(text: str) -> str | None:
    t = (text or "").strip().lower()
    if "first"  in t: return "first"
    if "second" in t: return "second"
    return None


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw  = responses["main"]
    pred = _parse_binary(raw)
    ok   = (pred == c["answer_gt"]) if pred else False
    return {
        "source":         c["source"],
        "duration_ms":    c["duration_ms"],
        "separation_ms":  c["separation_ms"],
        "base_name":      c["base_name"],
        "base_hz":        c["base_hz"],
        "delta_cents":    c["delta_cents"],
        "high_hz":        c["high_hz"],
        "order":          c["order"],
        "trial":          c["trial"],
        "answer_gt":      c["answer_gt"],
        "raw_response":   raw,
        "answer_pred":    pred,
        "answer_correct": int(ok),
    }


SPEC = CatDSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="binary",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("answer",),
    record_extras=(
        "duration_ms", "separation_ms", "base_name", "delta_cents", "order", "trial",
    ),
    label_fn=lambda j: (
        f"{j['cond']['base_name']} Δ={j['cond']['delta_cents']:>5}c "
        f"sep={j['cond']['separation_ms']:>4}ms "
        f"dur={j['cond']['duration_ms']:>4}ms"
    ),
)


def preview() -> None:
    run_cat_d_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_d_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
