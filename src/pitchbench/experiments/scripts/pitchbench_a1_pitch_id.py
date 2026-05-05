"""
a1 — Pitch identification.

Tests whether the ALM can identify a single sustained tone (one MIDI pitch
played by one source: waveform or GM instrument).

Universal IVs: source, source_type, midi.

Usage::
    pitchbench --id a1 --preview
    pitchbench --id a1 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_a import (
    CatASpec, run_cat_a_experiment,
)

EXP_NAME         = Path(__file__).stem
PITCHES          = config.pitchbench_a1_PITCHES
TONE_DURATION_MS = config.pitchbench_a1_TONE_DURATION_MS
ALL_SOURCES      = config.pitchbench_a1_SOURCES


def _available_sources() -> list[str]:
    """Drop GM instruments when FluidSynth isn't installed."""
    out: list[str] = []
    have_fluidsynth = find_spec("fluidsynth") is not None
    for src in ALL_SOURCES:
        if src in config.WAVEFORMS or have_fluidsynth:
            out.append(src)
        else:
            print(f"  [SKIP] Instrument {src!r}: FluidSynth not installed")
    return out


def build_conditions() -> list[dict]:
    sources = _available_sources()
    return [
        {"source": s,
         "source_type": "waveform" if s in config.WAVEFORMS else "instrument",
         "midi": m}
        for s in sources for m in PITCHES
    ]


def wav_for(c: dict) -> Path:
    return engine.tone(c["midi"], c["source"], TONE_DURATION_MS)


SPEC = CatASpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix="This audio contains a single musical note. ",
    record_extras=(),
)


def preview() -> None:
    run_cat_a_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_a_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
