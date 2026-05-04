"""
d2 — Pitch difference (binary higher/lower).

Question: given two sequential tones, which one is higher in pitch, as a
function of the cents difference between them and the absolute base frequency?

Universal IVs (per the v2 contract): duration_ms, source, base_midi.
Experiment-specific IVs:
    base_freq_hz:    {220, 440, 880}  (A3 / A4 / A5)
    delta_cents:     {1, 2, 5, 10, 25, 50, 100, 200, 400, 700, 1200}
    separation_ms:   {200, 500, 1000, 2000}   (silence between the two tones)
    order:           {first_higher, second_higher}

Fixed conditions: pure sine waveforms only (the relative-pitch judgment
should not depend on timbre); equal level; deterministic randomised order
seeded by `--seed`.

The model is asked a single binary question; chance = 50 %.

Usage::
    pitchbench --id d2 --preview
    pitchbench --id d2 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.results import (
    get_run_metadata, make_run_dir, save_comparison, save_results,
)

EXP_NAME = Path(__file__).stem

# ── IVs ───────────────────────────────────────────────────────────────────────

BASE_FREQS: dict[str, float] = {"A3": 220.00, "A4": 440.00, "A5": 880.00}

DELTA_CENTS: list[int] = [1, 2, 5, 10, 25, 50, 100, 200, 400, 700, 1200]

SEPARATION_MS: list[int] = [200, 500, 1000, 2000]

DEFAULT_DURATION_MS = 1000           # per tone — universal IV slot is "duration_ms"
DEFAULT_N_TRIALS    = 3
DEFAULT_SEED        = 42

SOURCES: list[str] = list(config.WAVEFORMS)   # waveforms only — Hz tones unsupported on instruments

PROMPT = (
    "Two pure tones play one after another, separated by a brief silence. "
    "Which tone is higher in pitch — the first or the second? "
    'Reply with ONLY "first" or "second".'
)


# ── Conditions ────────────────────────────────────────────────────────────────

def _cents_above(base_hz: float, cents: float) -> float:
    return base_hz * (2 ** (cents / 1200))


def build_conditions(
    durations_ms: list[int],
    separations_ms: list[int],
    n_trials: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for src in SOURCES:
        for base_name, base_hz in BASE_FREQS.items():
            for delta in DELTA_CENTS:
                high_hz = _cents_above(base_hz, delta)
                for dur_ms in durations_ms:
                    for sep_ms in separations_ms:
                        for trial in range(n_trials):
                            order = rng.randint(0, 1)   # 0 = low first, 1 = high first
                            rows.append({
                                "source":       src,
                                "duration_ms":  dur_ms,
                                "separation_ms": sep_ms,
                                "base_name":    base_name,
                                "base_hz":      round(base_hz, 4),
                                "delta_cents":  delta,
                                "high_hz":      round(high_hz, 4),
                                "order":        order,
                                "answer_gt":    "first" if order == 1 else "second",
                                "trial":        trial,
                            })
    return rows


def _wav_for(c: dict) -> Path:
    freqs = ([c["base_hz"], c["high_hz"]] if c["order"] == 0
             else [c["high_hz"], c["base_hz"]])
    return engine.sequence_hz(freqs, c["source"], c["duration_ms"], c["separation_ms"])


def _parse_binary(text: str) -> str | None:
    t = (text or "").strip().lower()
    if "first"  in t: return "first"
    if "second" in t: return "second"
    return None


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav  = str(_wav_for(c))
        out  = query_alm(model_name, wav, PROMPT)
        raw  = (out["result"] or "").strip()
        pred = _parse_binary(raw)
        ok   = (pred == c["answer_gt"]) if pred else False

        records.append({
            **c,
            "wav":            wav,
            "raw_response":   raw,
            "answer_pred":    pred,
            "answer_correct": int(ok),
            "prompt":         PROMPT,
            "model_params":   out["model_params"],
        })
        sym = "✓" if ok else "✗"
        print(f"    {sym} {c['base_name']} Δ={c['delta_cents']:>5}c sep={c['separation_ms']:>4}ms "
              f"dur={c['duration_ms']:>4}ms  [{c['answer_gt']}]  → {pred or '???'}")

    n        = len(records)
    n_corr   = sum(r["answer_correct"] for r in records)
    overall  = round(n_corr / n, 4) if n else None
    summary  = {"total": n, "correct": n_corr, "accuracy": overall, "chance": 0.5}

    summary_lines = [
        f"  Sources : {SOURCES}",
        f"  Stimuli : {n}",
        f"  Overall : {overall:.1%}  (chance = 50 %)" if overall is not None else "  Overall : N/A",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, base_freqs=BASE_FREQS,
        delta_cents=DELTA_CENTS, separations_ms=SEPARATION_MS,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        prompt=PROMPT,
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),                  # binary, no four-format pitch scoring
        extra_metrics=("answer_correct",),
    )
    return summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",   action="store_true")
    parser.add_argument("--models",    nargs="+", metavar="MODEL")
    parser.add_argument("--n-trials",  type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed",      type=int, default=DEFAULT_SEED)
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    conds = build_conditions(
        config.DEFAULT_DURATIONS_MS, SEPARATION_MS, args.n_trials, args.seed,
    )
    for c in conds:
        _wav_for(c)
    print(f"Experiment  : {EXP_NAME}")
    print(f"Sources     : {SOURCES}")
    print(f"Seed        : {args.seed}")
    print(f"Trials      : {len(conds)}  "
          f"({len(SOURCES)} sources × {len(BASE_FREQS)} bases × {len(DELTA_CENTS)} deltas × "
          f"{len(config.DEFAULT_DURATIONS_MS)} durs × {len(SEPARATION_MS)} seps × {args.n_trials} trials)")
    print(f"Audio dir   : {config.AUDIO_DIR}/{EXP_NAME}")
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    target_models = args.models or list(config.MODELS)
    conds = build_conditions(
        config.DEFAULT_DURATIONS_MS, SEPARATION_MS, args.n_trials, args.seed,
    )
    for c in conds:
        _wav_for(c)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for m in target_models:
        all_summaries[m] = run_one_model(m, conds, run_dir)
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
