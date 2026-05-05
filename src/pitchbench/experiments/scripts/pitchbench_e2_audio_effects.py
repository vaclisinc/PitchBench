"""
e2 — Pitch recognition under audio effects.

Tests whether models can identify pitch when audio is processed with
effects that genuinely threaten pitch cues. Effects use plugin-quality
DSP (pedalboard + scipy Butterworth) and every output is RMS-matched to
the dry signal so loudness does not leak into the experiment as effect
strength increases.

Universal IVs: midi, source.
Experiment-specific IVs:
    effect:        keyed into config.pitchbench_e2_EFFECTS
    effect_type:   coarser category (clean / filter / distort / reverb / chorus)
    effect_params: stringified params dict
    noise_seed:    deterministic per-condition seed

4-format pitch-ID prompts. Auto-marginals produce
``by_effect.<name>.<format>`` rows in the headline accuracies CSV.
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_e import CatESpec, run_cat_e_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME = Path(__file__).stem
PITCHES  = config.pitchbench_e2_PITCHES
TONE_MS  = config.pitchbench_e2_TONE_MS
EFFECTS  = config.pitchbench_e2_EFFECTS
SOURCES  = config.pitchbench_e2_SOURCES

PROMPT_PREFIX = "Listen to this audio clip of a single musical note. "


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for eff_idx, (eff_name, eff_params) in enumerate(EFFECTS.items()):
            for midi in PITCHES:
                noise_seed = (eff_idx * 1000 + midi) % (2 ** 31)
                rows.append({
                    "source":        src,
                    "source_type":   "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":          midi,
                    "effect":        eff_name,
                    "effect_type":   eff_params.get("type", "clean"),
                    "effect_params": str(eff_params),
                    "noise_seed":    noise_seed,
                })
    return rows


def wav_for(c: dict) -> Path:
    return engine.tone_with_effect(
        c["midi"], c["source"], TONE_MS,
        c["effect"], EFFECTS[c["effect"]], c["noise_seed"],
    )


SPEC = CatESpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix=PROMPT_PREFIX,
    record_extras=("effect", "effect_type", "effect_params", "noise_seed"),
    label_fn=lambda j: (
        f"{j['cond']['source']:10s}  {j['cond']['effect']:12s}  "
        f"{midi_to_note(j['cond']['midi']):4s}"
    ),
)


def preview() -> None:
    run_cat_e_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_e_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
