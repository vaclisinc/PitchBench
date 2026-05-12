"""
e4 — Pitch recognition under harmonic saturation.

Tests whether models can identify pitch when audio is processed with
tanh soft-clipping (harmonic saturation), which preserves the
fundamental frequency while enriching the harmonic spectrum.

Saturation levels:
  clean      — no processing
  sat_light  — tanh drive=2  (mild harmonic enrichment)
  sat_medium — tanh drive=5  (moderate, analogous to tape saturation)
  sat_heavy  — tanh drive=20 (strong, approaches hard clipping)

Universal IVs: midi, source.
Experiment-specific IVs:
    saturation_level:  ∈ config.pitchbench_e3_SATURATIONS
    saturation_type:   coarser category from params["type"]

4-format pitch-ID prompts. Auto-marginals produce
``by_saturation.<level>.<format>`` rows.
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.experiments.helpers.cat_e import CatESpec, run_cat_e_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME    = Path(__file__).stem
PITCHES     = config.pitchbench_e3_PITCHES
TONE_MS     = config.pitchbench_e3_TONE_MS
SATURATIONS = config.pitchbench_e3_SATURATIONS
SOURCES     = config.pitchbench_e3_SOURCES

PROMPT_PREFIX = "Listen to this audio clip of a single musical note. "


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for sat_idx, (sat_name, sat_params) in enumerate(SATURATIONS.items()):
            for midi in PITCHES:
                noise_seed = (sat_idx * 1000 + midi) % (2 ** 31)
                rows.append({
                    "source":            src,
                    "source_type":       "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":              midi,
                    "saturation_level":  sat_name,
                    "saturation_type":   sat_params.get("type", "clean"),
                    "saturation_params": str(sat_params),
                    "noise_seed":        noise_seed,
                })
    return rows


def wav_for(c: dict) -> Path:
    return engine.tone_with_effect(
        c["midi"], c["source"], TONE_MS,
        c["saturation_level"], SATURATIONS[c["saturation_level"]], c["noise_seed"],
    )


SPEC = CatESpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix=PROMPT_PREFIX,
    record_extras=(
        "saturation_level", "saturation_type", "saturation_params", "noise_seed",
    ),
    label_fn=lambda j: (
        f"{j['cond']['source']:10s}  {j['cond']['saturation_level']:12s}  "
        f"{midi_to_note(j['cond']['midi']):4s}"
    ),
)


def preview() -> None:
    run_cat_e_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_e_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
