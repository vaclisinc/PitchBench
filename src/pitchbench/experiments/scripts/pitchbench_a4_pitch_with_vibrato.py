"""
a4 — Pitch with vibrato.

Tests whether the ALM can identify the nominal centre pitch of a tone that
is frequency-modulated (vibrato).

Universal IVs: source, source_type, midi, duration_ms.
Experiment-specific IVs: vibrato_rate_hz, vibrato_depth_cents, is_control.

The control records (rate==0 OR depth==0 → flat-pitch) sit in the data for
ablation; the headline accuracy is reported on vibrato-active records only
(``is_control == 0``).

Usage::
    pitchbench --id a4 --preview
    pitchbench --id a4 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_a import (
    CatASpec, run_cat_a_experiment,
)

EXP_NAME             = Path(__file__).stem
SOURCES              = config.pitchbench_a4_SOURCES
PITCHES              = config.pitchbench_a4_PITCHES
DURATIONS_MS         = config.pitchbench_a4_DURATIONS_MS
VIBRATO_RATES_HZ     = config.pitchbench_a4_VIBRATO_RATES_HZ
VIBRATO_DEPTHS_CENTS = config.pitchbench_a4_VIBRATO_DEPTHS_CENTS

PROMPT_PREFIX = (
    "This audio contains a single sustained musical note that may have "
    "vibrato. Identify the nominal CENTRE pitch (ignore the vibrato "
    "modulation). "
)


def build_conditions() -> list[dict]:
    """Vibrato-active condition stims plus (optionally) a 1:1 flat-pitch baseline twin.

    Iterates over all (rate>0, depth>0) cells. For each, when
    ``config.INCLUDE_BASELINES`` is set we also emit a paired baseline
    record with rate=depth=0 sharing the same ``pair_id``. Result:
    ``condition_n == baseline_n``.
    """
    rows: list[dict] = []
    pair_id = 0
    for src in SOURCES:
        for midi in PITCHES:
            for dur in DURATIONS_MS:
                for rate in VIBRATO_RATES_HZ:
                    if rate == 0:
                        continue
                    for depth in VIBRATO_DEPTHS_CENTS:
                        if depth == 0:
                            continue
                        base = {
                            "source":      src,
                            "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                            "midi":        midi,
                            "duration_ms": dur,
                            "pair_id":     pair_id,
                        }
                        rows.append({
                            **base,
                            "vibrato_rate_hz":     rate,
                            "vibrato_depth_cents": depth,
                            "is_control":          0,
                        })
                        if config.INCLUDE_BASELINES:
                            rows.append({
                                **base,
                                "vibrato_rate_hz":     0,
                                "vibrato_depth_cents": 0,
                                "is_control":          1,
                            })
                        pair_id += 1
    return rows


def wav_for(c: dict) -> Path:
    return engine.tone_with_vibrato(
        c["midi"], c["source"], c["duration_ms"],
        c["vibrato_rate_hz"], c["vibrato_depth_cents"],
    )


SPEC = CatASpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix=PROMPT_PREFIX,
    record_extras=("duration_ms", "vibrato_rate_hz", "vibrato_depth_cents", "is_control"),
    primary_filter=lambda r: r.get("is_control") == 0,
    # Vibrato-active condition stim and its flat-pitch baseline twin share a
    # pair_id assigned in build_conditions(). Sampling on condition records
    # auto-pulls the matched baseline.
    pair_key=("pair_id",),
)


def preview() -> None:
    run_cat_a_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_a_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
