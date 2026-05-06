"""
Analysis (Q1 ablation) configuration.

Imports BENCHMARK_* constants from benchmark_config.  Per-experiment
variables default to benchmark values; the ACTIVE_PRESET overrides exactly
the experiments in that preset (one variable each).

Loaded by config.py when MODE = "ANALYSIS".
Also used by run_analysis.py for preset-driven runs.
"""

from __future__ import annotations

import sys as _sys
from typing import Any

from pitchbench.benchmark_config import (
    BENCHMARK_NOTATION_FORMATS,
    BENCHMARK_PITCHES_FULL_RANGE, BENCHMARK_PITCHES_SELECTION,
    BENCHMARK_PITCHES_SELECTION_COMPACT, BENCHMARK_PITCHES_SELECTION_REDUCED,
    BENCHMARK_INTERVALS,
    BENCHMARK_DURATION_MS, BENCHMARK_DURATIONS_MS_SHORT_LONG,
    BENCHMARK_DURATIONS_MS_FULL_SWEEP, BENCHMARK_NOTE_DURATIONS_MS_D4,
    BENCHMARK_TOTAL_DUR_MS, BENCHMARK_TONE_POSITIONS_MS,
    BENCHMARK_GAP_MS, BENCHMARK_GAP_MS_D7, BENCHMARK_N_TRIALS, BENCHMARK_SEED,
    BENCHMARK_ALL_SOURCES, BENCHMARK_WAVEFORMS,
    BENCHMARK_GM_INSTRUMENTS, BENCHMARK_CONTINUOUS_GM_INSTRUMENTS,
    BENCHMARK_F1_MIXED_GROUPS, BENCHMARK_BASE_FREQS_A,
    BENCHMARK_C1_CHORD_INTERVALS, BENCHMARK_C4_CHORD_TYPES, BENCHMARK_C3_QUALITIES,
    BENCHMARK_D4_TRAJECTORIES,
    BENCHMARK_E1_EFFECTS, BENCHMARK_E3_SATURATIONS, BENCHMARK_E4_CONDITIONS,
    BENCHMARK_PITCHES_E5,
    SAMPLING,
)

# ─── Active preset ────────────────────────────────────────────────────────────
# Change ACTIVE_PRESET to switch which ablation analysis config.py loads.
ACTIVE_PRESET = "q1"

# ─── Q1 preset — MIDI-only, sine source, one variable per experiment ──────────
_SOURCES    = ["sine", "synth_lead", "cello", "flute", "piano", "voice"]
_PITCHES = [29, 38, 47, 56, 65, 74, 83]

_EFFECTS_NO_CLEAN: dict[str, dict[str, Any]] = {
    "highpass_above_f0": {"type": "highpass",    "cutoff_ratio": 1.5, "order": 6},
    "lowpass_at_f0":     {"type": "lowpass",     "cutoff_ratio": 1.2, "order": 6},
    "bitcrush_4bit":     {"type": "bitcrush",    "bit_depth": 4},
    "distortion_heavy":  {"type": "saturation",  "drive_db": 30.0},
    "reverb_long":       {"type": "reverb_room", "room_size": 0.9, "damping": 0.4,
                          "wet_level": 0.6, "dry_level": 0.4},
    "chorus_heavy":      {"type": "chorus",      "rate_hz": 1.2, "depth": 0.9, "mix": 0.6},
}
_SATS_NO_CLEAN: dict[str, dict[str, Any]] = {
    "sat_light":  {"type": "saturation", "drive_db":  6.0},
    "sat_medium": {"type": "saturation", "drive_db": 15.0},
    "sat_heavy":  {"type": "saturation", "drive_db": 30.0},
}
_TIME_NO_CLEAN: list[dict[str, Any]] = [
    {"name": "resample_0.5x", "mode": "resample", "factor": 0.5},
    {"name": "resample_2x",   "mode": "resample", "factor": 2.0},
    {"name": "stretch_0.5x",  "mode": "stretch",  "factor": 0.5},
    {"name": "stretch_2x",    "mode": "stretch",  "factor": 2.0},
]

PRESETS: dict[str, dict[str, Any]] = {
    "q1": {
        "notation_formats": ["midi"],
        "experiments": {
            "pitchbench_a2_single_pitch_by_loudness": {
                "pitchbench_a2_LOUDNESS_DB": [-30, -20, -12, 6, 12],
                "pitchbench_a2_TONE_MS":     5000,
                "pitchbench_a2_PITCHES":     _PITCHES,
                "pitchbench_a2_SOURCES":     _SOURCES,
            },
            "pitchbench_a3_single_pitch_by_duration": {
                "pitchbench_a3_DURATIONS_MS": [50, 250, 1000, 4000, 15000, 60000],
                "pitchbench_a3_PITCHES":      _PITCHES,
                "pitchbench_a3_SOURCES":      _SOURCES,
            },
            "pitchbench_y1_single_pitch_id_mcq": {
                "pitchbench_y1_OPTION_STEP_ST": [2, 4, 6],
                "pitchbench_y1_PITCHES":        _PITCHES,
                "pitchbench_y1_SOURCES":        _SOURCES,
                "pitchbench_y1_TONE_DURATION_MS": BENCHMARK_DURATION_MS,
            },
            "pitchbench_b1_single_pitch_within_silence": {
                "pitchbench_b1_TONE_POSITIONS_MS": [10_000, 30_000, 50_000],
                "pitchbench_b1_TONE_DURATION_MS":  5000,
                "pitchbench_b1_TOTAL_SILENCE_MS":  60_000,
                "pitchbench_b1_PITCHES":           _PITCHES,
                "pitchbench_b1_SOURCES":           _SOURCES,
            },
            "pitchbench_b2_pitch_at_timestamp": {
                "pitchbench_b2_N_NOTES_OPTS": [5, 10],
                "pitchbench_b2_DURATIONS_MS": [5000],
                "pitchbench_b2_TOTAL_DUR_MS": 60_000,
                "pitchbench_b2_MAIN_PITCHES":      _PITCHES,
                "pitchbench_b2_DISTRACTORS":      BENCHMARK_PITCHES_FULL_RANGE,
                "pitchbench_b2_SOURCES":      _SOURCES,
            },
            # Both d7a (concat-audio) and d7b (split-audio) share the same
            # ``pitchbench_d7_*`` parameter set — pointing both keys at one
            # block keeps the analysis-mode override consistent.
            "pitchbench_d7a_pitch_with_reference": {
                "pitchbench_d7_REFERENCE_PITCHES": [69, 53],
                "pitchbench_d7_INTERVALS":         BENCHMARK_INTERVALS,
                "pitchbench_d7_TONE_DURATION_MS":  5000,
                "pitchbench_d7_GAP_MS":            500,
                "pitchbench_d7_SOURCES":           _SOURCES,
            },
            "pitchbench_d7b_pitch_with_reference_split": {
                "pitchbench_d7_REFERENCE_PITCHES": [69, 53],
                "pitchbench_d7_INTERVALS":         BENCHMARK_INTERVALS,
                "pitchbench_d7_TONE_DURATION_MS":  5000,
                "pitchbench_d7_GAP_MS":            500,
                "pitchbench_d7_SOURCES":           _SOURCES,
            },
            "pitchbench_e1_audio_effects": {
                "pitchbench_e1_EFFECTS":  _EFFECTS_NO_CLEAN,
                "pitchbench_e1_TONE_MS":  5000,
                "pitchbench_e1_PITCHES":  _PITCHES,
                "pitchbench_e1_SOURCES":  _SOURCES,
            },
            "pitchbench_e2_background": {
                "pitchbench_e2_BACKGROUNDS":  ["white_noise", "church-bells",
                                               "crowd-noise", "rain", "street-noise", "oscillating-high-and-low-pitches"],
                "pitchbench_e2_SNR_DB":       [-6.0],
                "pitchbench_e2_DURATIONS_MS": [5000],
                "pitchbench_e2_PITCHES":      _PITCHES,
                "pitchbench_e2_SOURCES":      _SOURCES,
            },
            "pitchbench_e3_harmonic_saturation": {
                "pitchbench_e3_SATURATIONS": _SATS_NO_CLEAN,
                "pitchbench_e3_TONE_MS":     5000,
                "pitchbench_e3_PITCHES":     _PITCHES,
                "pitchbench_e3_SOURCES":     _SOURCES,
            },
            "pitchbench_e4_time_stretching": {
                "pitchbench_e4_CONDITIONS": _TIME_NO_CLEAN,
                "pitchbench_e4_TONE_MS":    3000,
                "pitchbench_e4_PITCHES":    BENCHMARK_PITCHES_E5,
                "pitchbench_e4_SOURCES":    _SOURCES,
            },
            "pitchbench_e6_slightly_off": {
                "pitchbench_e6_N_DETUNE_LEVELS": 6,
                "pitchbench_e6_DETUNE_FRACTION": 0.45,
                "pitchbench_e6_DURATIONS_MS":    [5000],
                "pitchbench_e6_PITCHES":         _PITCHES,
                "pitchbench_e6_SOURCES":         _SOURCES,
            },
        },
    },
}

# ─── Module-level pitchbench_* variables (benchmark defaults, then preset override)
# These are what config.py injects when MODE = "ANALYSIS".

# Start with benchmark defaults for all experiments not in the preset.
from pitchbench.benchmark_config import *  # noqa: F401,F403,E402

# Override with the active preset's experiment variables.
_this = _sys.modules[__name__]
_preset = PRESETS[ACTIVE_PRESET]
pitchbench_general_NOTATION_FORMATS = _preset["notation_formats"]
for _exp_overrides in _preset["experiments"].values():
    for _k, _v in _exp_overrides.items():
        setattr(_this, _k, _v)
del _this, _preset, _exp_overrides, _k, _v, _sys
