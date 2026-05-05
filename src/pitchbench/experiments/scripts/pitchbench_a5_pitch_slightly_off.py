"""
a5 — Pitch slightly off (nearest in-tune pitch).

When a tone is slightly detuned, does the ALM snap to the nearest in-tune
pitch? Each (pitch, duration) cell sweeps a symmetric set of detunes
bounded inside the basin of the target pitch (so the answer remains
unambiguous).

Universal IVs: source, source_type, midi, duration_ms.
Experiment-specific IV: detune_hz.

The ``detune_hz == 0`` records are the in-tune control; the headline
accuracy is reported on detuned records only (``detune_hz != 0``).

Usage::
    pitchbench --id a5 --preview
    pitchbench --id a5 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_a import (
    CatASpec, run_cat_a_experiment,
)
from pitchbench.experiments.helpers.music import midi_to_freq

EXP_NAME        = Path(__file__).stem
SOURCES         = config.pitchbench_a5_SOURCES
PITCHES         = config.pitchbench_a5_PITCHES
DURATIONS_MS    = config.pitchbench_a5_DURATIONS_MS
N_DETUNE_LEVELS = config.pitchbench_a5_N_DETUNE_LEVELS
DETUNE_FRACTION = config.pitchbench_a5_DETUNE_FRACTION

PROMPT_PREFIX = (
    "This audio contains a single sustained musical note that may be "
    "slightly out of tune. Identify the NEAREST in-tune pitch (the "
    "closest standard musical note). "
)


def _detune_grid_nonzero(midi: int) -> list[float]:
    """Symmetric NON-ZERO detune values bounded inside the basin of the pitch.

    The 0-detune control is no longer part of the IV grid — it is emitted
    as an explicit baseline twin per condition (see ``build_conditions``).
    """
    f0      = midi_to_freq(midi)
    half_lo = (f0 - midi_to_freq(midi - 1)) / 2
    half_hi = (midi_to_freq(midi + 1) - f0) / 2
    max_neg = -half_lo * DETUNE_FRACTION
    max_pos =  half_hi * DETUNE_FRACTION
    out: list[float] = []
    half = N_DETUNE_LEVELS // 2
    for i in range(-half, half + 1):
        if i == 0:
            continue
        if i < 0:
            out.append(round(max_neg * (i / -half), 4))
        else:
            out.append(round(max_pos * (i /  half), 4))
    return out


def build_conditions() -> list[dict]:
    """Detuned condition stims plus (optionally) a 1:1 in-tune baseline twin.

    For every (source, midi, duration_ms, detune_hz!=0) cell we emit one
    condition record. When ``config.INCLUDE_BASELINES`` is set we also emit a
    paired baseline (detune_hz=0, same other fields) sharing the same
    ``pair_id``. Result: ``condition_n == baseline_n``.
    """
    rows: list[dict] = []
    pair_id = 0
    for src in SOURCES:
        for midi in PITCHES:
            for dur in DURATIONS_MS:
                for detune_hz in _detune_grid_nonzero(midi):
                    base = {
                        "source":      src,
                        "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                        "midi":        midi,
                        "duration_ms": dur,
                        "pair_id":     pair_id,
                    }
                    rows.append({**base, "detune_hz": detune_hz})
                    if config.INCLUDE_BASELINES:
                        rows.append({**base, "detune_hz": 0.0})
                    pair_id += 1
    return rows


def wav_for(c: dict) -> Path:
    return engine.tone_hz(midi_to_freq(c["midi"]) + c["detune_hz"], c["source"], c["duration_ms"])


SPEC = CatASpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix=PROMPT_PREFIX,
    record_extras=("duration_ms", "detune_hz"),
    primary_filter=lambda r: r.get("detune_hz") != 0,
    # Detuned condition stim and its in-tune baseline twin share a pair_id
    # assigned in build_conditions(). Sampling on detuned records auto-pulls
    # the matched baseline.
    pair_key=("pair_id",),
)


def preview() -> None:
    run_cat_a_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_a_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
