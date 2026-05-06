"""
e3 — Background-noise effects on pitch identification.

Question: do additive backgrounds (white noise + real-world recordings)
impair the ALM's absolute pitch hearing?

Universal IVs: duration_ms, midi, source.
Experiment-specific IVs:
    background:  ∈ {white_noise, church-bells, crowd-noise, rain, street-noise}
    snr_db:      ∈ {30, 20, 0, −6}

Backgrounds:
    white_noise           — Gaussian, deterministic per seed=0
    church-bells / crowd-noise / rain / street-noise
                          — loaded from data/downloaded/background/<name>.mp3,
                            truncated to the fragment length (no looping).

Each mix is normalised to peak 0.9 after combining tone + scaled background.
4-format pitch-ID prompts.
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_e import CatESpec, run_cat_e_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME     = Path(__file__).stem
BACKGROUNDS  = config.pitchbench_e2_BACKGROUNDS
SNR_DB       = config.pitchbench_e2_SNR_DB
SOURCES      = config.pitchbench_e2_SOURCES
PITCHES      = config.pitchbench_e2_PITCHES
DURATIONS_MS = config.pitchbench_e2_DURATIONS_MS

PROMPT_PREFIX = (
    "This audio contains a single sustained musical note mixed with a "
    "background sound. Identify the PITCH of the note, ignoring the "
    "background. "
)


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for midi in PITCHES:
            for dur in DURATIONS_MS:
                for bg in BACKGROUNDS:
                    for snr in SNR_DB:
                        rows.append({
                            "source":      src,
                            "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                            "midi":        midi,
                            "duration_ms": dur,
                            "background":  bg,
                            "snr_db":      snr,
                        })
    return rows


def wav_for(c: dict) -> Path:
    return engine.tone_with_background(
        c["midi"], c["source"], c["duration_ms"], c["background"], c["snr_db"],
    )


SPEC = CatESpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix=PROMPT_PREFIX,
    record_extras=("duration_ms", "background", "snr_db"),
    label_fn=lambda j: (
        f"{midi_to_note(j['cond']['midi']):4s} {j['cond']['source']:8s} "
        f"bg={j['cond']['background']:>13s} "
        f"snr={j['cond']['snr_db']:>+5.0f}dB"
    ),
)


def preview() -> None:
    run_cat_e_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_e_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
