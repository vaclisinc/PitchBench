"""
d3 — Sequential interval identification.

Question: given two notes played one after another, can the ALM correctly
report the interval between them — as a signed integer number of
semitones (positive = ascending, negative = descending)?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    base_midi:       ∈ DEFAULT_PITCHES
    interval_st:     {1..12}
    direction:       {ascending, descending}
    separation_ms:   {200, 500, 1000, 2000}

Headline metric: ``interval_correct`` (exact match on the signed integer).
``interval_within_1`` (off-by-one tolerance) is a per-stimulus diagnostic
auto-excluded from the headline accuracies CSV by the ``_within_1``
exclusion rule in ``_marginal_csv_rows``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.experiments.helpers.cat_d import CatDSpec, run_cat_d_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME       = Path(__file__).stem
INTERVALS_ST   = config.pitchbench_d6_INTERVALS_ST
DIRECTIONS     = config.pitchbench_d6_DIRECTIONS
SEPARATIONS_MS = config.pitchbench_d6_SEPARATIONS_MS
SOURCES        = config.pitchbench_d6_SOURCES
PITCHES        = config.pitchbench_d6_PITCHES
DURATIONS_MS   = config.pitchbench_d6_DURATIONS_MS

PROMPT = (
    "Two musical notes play in sequence, separated by a brief silence. "
    "How many semitones apart are they? Reply with a SIGNED integer where "
    "positive = ascending (the second note is higher) and negative = "
    "descending. Examples: '7' for an ascending fifth, '-3' for a descending "
    "minor third. Reply with ONLY the integer."
)

_INT_RE = re.compile(r"-?\d+")


def _parse_signed_st(text: str) -> int | None:
    m = _INT_RE.search(text or "")
    if not m:
        return None
    v = int(m.group(0))
    return v if -36 <= v <= 36 else None


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for base in PITCHES:
            for iv in INTERVALS_ST:
                for direction in DIRECTIONS:
                    other = base + iv if direction == "ascending" else base - iv
                    if other < 12 or other > 96:
                        continue
                    for dur in DURATIONS_MS:
                        for sep in SEPARATIONS_MS:
                            rows.append({
                                "source":        src,
                                "duration_ms":   dur,
                                "base_midi":     base,
                                "interval_st":   iv,
                                "direction":     direction,
                                "separation_ms": sep,
                                "midis":         [base, other],
                                "signed_st":     iv if direction == "ascending" else -iv,
                            })
    return rows


def wav_for(c: dict) -> Path:
    return engine.sequence(c["midis"], c["source"], c["duration_ms"], c["separation_ms"])


def prompts_for(_: dict) -> dict[str, str]:
    return {"main": PROMPT}


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw  = responses["main"]
    pred = _parse_signed_st(raw)
    ok        = (pred == c["signed_st"]) if pred is not None else False
    within_1  = (pred is not None and abs(pred - c["signed_st"]) <= 1)
    return {
        "duration_ms":       c["duration_ms"],
        "source":            c["source"],
        "base_midi":         c["base_midi"],
        "base_note":         midi_to_note(c["base_midi"]),
        "interval_st":       c["interval_st"],
        "direction":         c["direction"],
        "separation_ms":     c["separation_ms"],
        "signed_st":         c["signed_st"],
        "midis":             "+".join(str(m) for m in c["midis"]),
        "raw_response":      raw,
        "interval_pred":     pred,
        "interval_correct":  int(ok),
        "interval_within_1": int(within_1),
    }


SPEC = CatDSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="interval",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("interval",),
    record_extras=(
        "duration_ms", "base_midi", "interval_st", "direction",
        "separation_ms", "signed_st",
    ),
    label_fn=lambda j: (
        f"iv={j['cond']['signed_st']:+3d}st "
        f"base={midi_to_note(j['cond']['base_midi']):4s} "
        f"sep={j['cond']['separation_ms']:>4}ms {j['cond']['source']}"
    ),
)


def preview() -> None:
    run_cat_d_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_d_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
