"""
a3 — Pitch identification by duration.

Tests how the ALM's pitch identification accuracy varies with the duration
of the sustained tone.

Universal IVs: source, source_type, midi, duration_ms.

Usage::
    pitchbench --id a3 --preview
    pitchbench --id a3 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_a import (
    CatASpec, run_cat_a_experiment,
)

EXP_NAME     = Path(__file__).stem
SOURCES      = config.pitchbench_a3_SOURCES
PITCHES      = config.pitchbench_a3_PITCHES
DURATIONS_MS = config.pitchbench_a3_DURATIONS_MS


def build_conditions() -> list[dict]:
    return [
        {"source": s,
         "source_type": "waveform" if s in config.WAVEFORMS else "instrument",
         "midi": m,
         "duration_ms": d}
        for s in SOURCES for m in PITCHES for d in DURATIONS_MS
    ]


def wav_for(c: dict) -> Path:
    return engine.tone(c["midi"], c["source"], c["duration_ms"])


SPEC = CatASpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix="This audio contains a single musical note. ",
    record_extras=("duration_ms",),
)


def preview() -> None:
    run_cat_a_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_a_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
