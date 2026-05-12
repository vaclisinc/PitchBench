"""
e1 — Pitch recognition under loudness variation.

Tests whether the ALM's pitch identification holds when the stimulus
amplitude varies from near-silence (−30 dBFS) to full level (0 dBFS).

Universal IVs: midi, source.
Experiment-specific IV:
    loudness_db: ∈ config.pitchbench_a2_LOUDNESS_DB

4-format pitch-ID prompts (MIDI / SPN / Doremi / Hz). The auto-marginals
produce ``by_loudness.<db>.<format>`` rows in ``accuracies_<model>.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.experiments.helpers.cat_e import CatESpec, run_cat_e_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME    = Path(__file__).stem
PITCHES     = config.pitchbench_a2_PITCHES
LOUDNESS_DB = config.pitchbench_a2_LOUDNESS_DB
SOURCES     = config.pitchbench_a2_SOURCES
TONE_MS     = config.pitchbench_a2_TONE_MS

PROMPT_PREFIX = "Listen to this audio clip of a single musical note. "


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for midi in PITCHES:
            for db in LOUDNESS_DB:
                rows.append({
                    "source":      src,
                    "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":        midi,
                    "loudness_db": db,
                })
    return rows


def wav_for(c: dict) -> Path:
    return engine.tone_at_volume(c["midi"], c["source"], TONE_MS, c["loudness_db"])


SPEC = CatESpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix=PROMPT_PREFIX,
    record_extras=("loudness_db",),
    label_fn=lambda j: (
        f"{midi_to_note(j['cond']['midi']):4s}  "
        f"{j['cond']['loudness_db']:>4} dBFS  {j['cond']['source']}"
    ),
)


def preview() -> None:
    run_cat_e_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_e_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
