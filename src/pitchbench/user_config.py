"""
User (non-benchmark) configuration — freely editable.

Imports BENCHMARK_* constants from benchmark_config so the defaults mirror
paper values. Edit any line marked "← edit me" to explore a different
parameter range without touching the benchmark.

Loaded by config.py when MODE = "USER".
Also used by run_analysis.py --user-config for runtime patching.
"""

from __future__ import annotations
from typing import Any

from pitchbench.benchmark_config import (
    BENCHMARK_NOTATION_FORMATS,
    BENCHMARK_PITCHES_FULL_RANGE,    BENCHMARK_PITCHES_SELECTION,
    BENCHMARK_PITCHES_SELECTION_COMPACT,
    BENCHMARK_PITCHES_SELECTION_REDUCED,
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
)

# ── notation ────────────────────────────────────────────────────────────────
pitchbench_general_NOTATION_FORMATS = BENCHMARK_NOTATION_FORMATS  # ← edit me

# ── a1 ──────────────────────────────────────────────────────────────────────
pitchbench_a1_PITCHES          = BENCHMARK_PITCHES_FULL_RANGE   # ← edit me
pitchbench_a1_TONE_DURATION_MS = BENCHMARK_DURATION_MS          # ← edit me
pitchbench_a1_SOURCES          = BENCHMARK_ALL_SOURCES          # ← edit me

# ── y1 (single pitch with multiple-choice options) ─────────────────────────
pitchbench_y1_PITCHES          = BENCHMARK_PITCHES_FULL_RANGE   # ← edit me
pitchbench_y1_TONE_DURATION_MS = BENCHMARK_DURATION_MS          # ← edit me
pitchbench_y1_SOURCES          = BENCHMARK_ALL_SOURCES          # ← edit me
pitchbench_y1_N_OPTIONS        = 5                              # ← edit me
pitchbench_y1_OPTION_STEP_ST   = 2                              # ← edit me
pitchbench_y1_SEED             = BENCHMARK_SEED                 # ← edit me

# ── d7 (pitch with reference) ───────────────────────────────────────────────
pitchbench_d7_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION   # ← edit me
pitchbench_d7_INTERVALS         = [-12,-7,-5,-4,-3,-2,-1,0,
                                     1, 2, 3, 4, 5, 7,12]       # ← edit me
pitchbench_d7_TONE_DURATION_MS  = BENCHMARK_DURATION_MS         # ← edit me
pitchbench_d7_GAP_MS            = 500                           # ← edit me
pitchbench_d7_SOURCES           = BENCHMARK_ALL_SOURCES         # ← edit me

# ── a3 ──────────────────────────────────────────────────────────────────────
pitchbench_a3_PITCHES      = BENCHMARK_PITCHES_SELECTION        # ← edit me
pitchbench_a3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_FULL_SWEEP  # ← edit me
pitchbench_a3_SOURCES      = BENCHMARK_ALL_SOURCES              # ← edit me

# ── e5 (vibrato) ────────────────────────────────────────────────────────────
pitchbench_e5_VIBRATO_RATES_HZ     = [0, 3, 5, 7, 10]          # ← edit me
pitchbench_e5_VIBRATO_DEPTHS_CENTS = [0, 25, 50, 100, 200]      # ← edit me
pitchbench_e5_DURATIONS_MS         = [BENCHMARK_DURATION_MS]    # ← edit me
pitchbench_e5_PITCHES              = BENCHMARK_PITCHES_SELECTION # ← edit me
pitchbench_e5_SOURCES              = BENCHMARK_WAVEFORMS        # ← edit me

# ── e6 (slightly off) ───────────────────────────────────────────────────────
pitchbench_e6_N_DETUNE_LEVELS = 5                                # ← edit me
pitchbench_e6_DETUNE_FRACTION = 0.40                             # ← edit me
pitchbench_e6_PITCHES         = BENCHMARK_PITCHES_SELECTION      # ← edit me
pitchbench_e6_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
pitchbench_e6_SOURCES         = BENCHMARK_ALL_SOURCES            # ← edit me

# ── b1 ──────────────────────────────────────────────────────────────────────
pitchbench_b1_PITCHES           = BENCHMARK_PITCHES_SELECTION    # ← edit me
pitchbench_b1_TONE_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS    # ← edit me
pitchbench_b1_TONE_DURATION_MS  = BENCHMARK_DURATION_MS          # ← edit me
pitchbench_b1_TOTAL_SILENCE_MS  = BENCHMARK_TOTAL_DUR_MS         # ← edit me
pitchbench_b1_SOURCES           = BENCHMARK_ALL_SOURCES          # ← edit me

# ── b3 ──────────────────────────────────────────────────────────────────────
pitchbench_b3_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
pitchbench_b3_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS         # ← edit me
pitchbench_b3_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS              # ← edit me
pitchbench_b3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
pitchbench_b3_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

# ── b4 ──────────────────────────────────────────────────────────────────────
pitchbench_b4_N_DISTRACTORS   = [5, 7]                           # ← edit me
pitchbench_b4_TARGET_POS_OPTS = ["first", "middle", "last"]      # ← edit me
pitchbench_b4_TOTAL_DUR_MS    = BENCHMARK_TOTAL_DUR_MS           # ← edit me
pitchbench_b4_GAP_MIN_MS      = 500                              # ← edit me
pitchbench_b4_GAP_MAX_MS      = 2000                             # ← edit me
pitchbench_b4_PITCHES         = BENCHMARK_PITCHES_SELECTION      # ← edit me
pitchbench_b4_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
pitchbench_b4_SEED            = BENCHMARK_SEED                   # ← edit me
pitchbench_b4_SOURCES         = BENCHMARK_ALL_SOURCES            # ← edit me

# ── b2 ──────────────────────────────────────────────────────────────────────
pitchbench_b2_N_NOTES_OPTS = [5, 10]                             # ← edit me
pitchbench_b2_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS              # ← edit me
pitchbench_b2_GAP_MIN_MS   = 30                                  # ← edit me
pitchbench_b2_GAP_MAX_MS   = 1500                                # ← edit me
pitchbench_b2_MAIN_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
pitchbench_b2_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
pitchbench_b2_SEED         = BENCHMARK_SEED                      # ← edit me
pitchbench_b2_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me
pitchbench_b2_DISTRACTORS  = BENCHMARK_PITCHES_FULL_RANGE          # ← edit me

# ── b5 ──────────────────────────────────────────────────────────────────────
pitchbench_b5_N_NOTES_OPTS   = [3, 5, 8]                         # ← edit me
pitchbench_b5_RHYTHMS        = ["regular", "irregular"]          # ← edit me
pitchbench_b5_PITCH_PATTERNS = ["fixed_pitch", "varied_pitches"] # ← edit me
pitchbench_b5_TOTAL_DUR_MS   = BENCHMARK_TOTAL_DUR_MS            # ← edit me
pitchbench_b5_IRR_GAP_MIN_MS = 300                               # ← edit me
pitchbench_b5_IRR_GAP_MAX_MS = 2500                              # ← edit me
pitchbench_b5_PITCHES        = BENCHMARK_PITCHES_SELECTION       # ← edit me
pitchbench_b5_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
pitchbench_b5_SEED           = BENCHMARK_SEED                    # ← edit me
pitchbench_b5_SOURCES        = BENCHMARK_ALL_SOURCES             # ← edit me

# ── c2 ──────────────────────────────────────────────────────────────────────
pitchbench_c2_INTERVALS_ST         = list(range(1, 13))          # ← edit me
pitchbench_c2_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
pitchbench_c2_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
pitchbench_c2_PITCHES              = BENCHMARK_PITCHES_SELECTION # ← edit me
pitchbench_c2_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

# ── c1 ──────────────────────────────────────────────────────────────────────
pitchbench_c1_CHORD_INTERVALS      = BENCHMARK_C1_CHORD_INTERVALS # ← edit me
pitchbench_c1_QUALITIES            = list(BENCHMARK_C1_CHORD_INTERVALS.keys()) + ["random_set"] # ← edit me
pitchbench_c1_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION # ← edit me
pitchbench_c1_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
pitchbench_c1_RANDOM_NS            = [1, 2, 3, 4, 5, 6]          # ← edit me
pitchbench_c1_RANDOM_PITCH_RANGE   = (48, 84)                    # ← edit me
pitchbench_c1_N_TRIALS             = 1                           # ← edit me
pitchbench_c1_RANDOM_TRIALS        = 3                           # ← edit me
pitchbench_c1_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
pitchbench_c1_SEED                 = BENCHMARK_SEED              # ← edit me
pitchbench_c1_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

# ── c4 ──────────────────────────────────────────────────────────────────────
pitchbench_c4_CHORD_TYPES      = BENCHMARK_C4_CHORD_TYPES        # ← edit me
pitchbench_c4_BASE_ROOTS       = BENCHMARK_PITCHES_SELECTION     # ← edit me
pitchbench_c4_TONE_DURATION_MS = BENCHMARK_DURATION_MS           # ← edit me
pitchbench_c4_SOURCES          = BENCHMARK_ALL_SOURCES           # ← edit me

# ── c3 ──────────────────────────────────────────────────────────────────────
pitchbench_c3_QUALITIES            = BENCHMARK_C3_QUALITIES      # ← edit me
pitchbench_c3_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION # ← edit me
pitchbench_c3_TASKS                = ["quality_only"]            # ← edit me
pitchbench_c3_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
pitchbench_c3_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
pitchbench_c3_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

# ── d1 ──────────────────────────────────────────────────────────────────────
pitchbench_d1_N_COUNTS     = [1, 2, 3, 4, 5, 7, 10]              # ← edit me
pitchbench_d1_RHYTHMS      = ["regular", "irregular"]            # ← edit me
pitchbench_d1_PITCH_MIN    = 48                                  # ← edit me
pitchbench_d1_PITCH_MAX    = 84                                  # ← edit me
pitchbench_d1_GAP_MS       = BENCHMARK_GAP_MS                    # ← edit me
pitchbench_d1_N_TRIALS     = BENCHMARK_N_TRIALS                  # ← edit me
pitchbench_d1_SEED         = BENCHMARK_SEED                      # ← edit me
pitchbench_d1_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
pitchbench_d1_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

# ── d2 ──────────────────────────────────────────────────────────────────────
pitchbench_d2_BASE_FREQS    = BENCHMARK_BASE_FREQS_A             # ← edit me
pitchbench_d2_DELTA_CENTS   = [1, 2, 5, 10, 25, 50, 100, 200, 400, 700, 1200] # ← edit me
pitchbench_d2_SEPARATION_MS = [200, 500, 1000, 2000]             # ← edit me
pitchbench_d2_DURATION_MS   = BENCHMARK_DURATION_MS              # ← edit me
pitchbench_d2_N_TRIALS      = BENCHMARK_N_TRIALS                 # ← edit me
pitchbench_d2_SEED          = BENCHMARK_SEED                     # ← edit me
pitchbench_d2_SOURCES       = BENCHMARK_ALL_SOURCES              # ← edit me

# ── d6 ──────────────────────────────────────────────────────────────────────
pitchbench_d6_INTERVALS_ST   = list(range(1, 13))                # ← edit me
pitchbench_d6_DIRECTIONS     = ["ascending", "descending"]       # ← edit me
pitchbench_d6_SEPARATIONS_MS = [200, 500, 1000, 2000]            # ← edit me
pitchbench_d6_PITCHES        = BENCHMARK_PITCHES_SELECTION       # ← edit me
pitchbench_d6_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
pitchbench_d6_SOURCES        = BENCHMARK_ALL_SOURCES             # ← edit me

# ── d3 ──────────────────────────────────────────────────────────────────────
pitchbench_d3_N_TRANSITIONS_OPTS = [2, 3, 5, 7]                  # ← edit me
pitchbench_d3_STEP_SIZES_ST      = [1, 2, 4, 7]                  # ← edit me
pitchbench_d3_NOTE_DURATIONS_MS  = BENCHMARK_NOTE_DURATIONS_MS_D4 # ← edit me
pitchbench_d3_TRIALS_PER_CELL    = 2                             # ← edit me
pitchbench_d3_PITCHES            = BENCHMARK_PITCHES_SELECTION   # ← edit me
pitchbench_d3_SEED               = BENCHMARK_SEED                # ← edit me
pitchbench_d3_SOURCES            = BENCHMARK_ALL_SOURCES         # ← edit me

# ── d4 ──────────────────────────────────────────────────────────────────────
pitchbench_d4_START_PITCHES = BENCHMARK_PITCHES_SELECTION        # ← edit me
pitchbench_d4_INTERVALS_ST  = [1, 4, 7, 12]                      # ← edit me
pitchbench_d4_DURATION_MS   = BENCHMARK_DURATION_MS              # ← edit me
pitchbench_d4_SOURCES       = BENCHMARK_WAVEFORMS                # ← edit me
pitchbench_d4_TRAJECTORIES  = BENCHMARK_D4_TRAJECTORIES          # ← edit me

# ── d5 ──────────────────────────────────────────────────────────────────────
pitchbench_d5_BASE_FREQS  = BENCHMARK_BASE_FREQS_A               # ← edit me
pitchbench_d5_DELTA_CENTS = [25, 50, 100, 200, 400]              # ← edit me
pitchbench_d5_N_TONES     = [3, 4, 5, 7]                         # ← edit me
pitchbench_d5_RHYTHMS     = ["regular", "irregular"]             # ← edit me
pitchbench_d5_DURATION_MS = BENCHMARK_DURATION_MS                # ← edit me
pitchbench_d5_GAP_MS      = BENCHMARK_GAP_MS                     # ← edit me
pitchbench_d5_N_TRIALS    = BENCHMARK_N_TRIALS                   # ← edit me
pitchbench_d5_SEED        = BENCHMARK_SEED                       # ← edit me
pitchbench_d5_SOURCES     = BENCHMARK_WAVEFORMS                  # ← edit me

# ── d8 ──────────────────────────────────────────────────────────────────────
pitchbench_d8_PITCH_MIN    = 29                                  # ← edit me
pitchbench_d8_PITCH_MAX    = 89                                  # ← edit me
pitchbench_d8_N_NOTES_LIST = [3, 5, 10]                          # ← edit me
pitchbench_d8_TONE_MS      = BENCHMARK_DURATION_MS               # ← edit me
pitchbench_d8_GAP_MS       = BENCHMARK_GAP_MS_D7                 # ← edit me
pitchbench_d8_N_TRIALS     = 5                                   # ← edit me
pitchbench_d8_SEED         = BENCHMARK_SEED                      # ← edit me
pitchbench_d8_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

# ── a2 (loudness) ───────────────────────────────────────────────────────────
pitchbench_a2_LOUDNESS_DB = [-30, -20, -12, -6, -3, 0]           # ← edit me
pitchbench_a2_TONE_MS     = BENCHMARK_DURATION_MS                # ← edit me
pitchbench_a2_PITCHES     = BENCHMARK_PITCHES_SELECTION          # ← edit me
pitchbench_a2_SOURCES     = BENCHMARK_ALL_SOURCES                # ← edit me

# ── e1 (audio effects) ──────────────────────────────────────────────────────
pitchbench_e1_PITCHES = BENCHMARK_PITCHES_SELECTION              # ← edit me
pitchbench_e1_TONE_MS = BENCHMARK_DURATION_MS                    # ← edit me
pitchbench_e1_EFFECTS = BENCHMARK_E1_EFFECTS                     # ← edit me
pitchbench_e1_SOURCES = BENCHMARK_ALL_SOURCES                    # ← edit me

# ── e2 (background) ─────────────────────────────────────────────────────────
pitchbench_e2_BACKGROUNDS  = ["white_noise", "church-bells", "crowd-noise",
                               "rain", "street-noise", "oscillating-high-and-low-pitches"] # ← edit me
pitchbench_e2_SNR_DB       = [30.0, 20.0, 0.0, -6.0]             # ← edit me
pitchbench_e2_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
pitchbench_e2_DURATIONS_MS = [BENCHMARK_DURATION_MS]             # ← edit me
pitchbench_e2_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

# ── e3 (harmonic saturation) ────────────────────────────────────────────────
pitchbench_e3_PITCHES     = BENCHMARK_PITCHES_SELECTION          # ← edit me
pitchbench_e3_TONE_MS     = BENCHMARK_DURATION_MS                # ← edit me
pitchbench_e3_SATURATIONS = BENCHMARK_E3_SATURATIONS             # ← edit me
pitchbench_e3_SOURCES     = BENCHMARK_ALL_SOURCES                # ← edit me

# ── e4 (time stretching) ────────────────────────────────────────────────────
pitchbench_e4_PITCHES    = BENCHMARK_PITCHES_E5                  # ← edit me
pitchbench_e4_TONE_MS    = 3000                                  # ← edit me
pitchbench_e4_CONDITIONS = BENCHMARK_E4_CONDITIONS               # ← edit me
pitchbench_e4_SOURCES    = BENCHMARK_ALL_SOURCES                 # ← edit me

# ── g1/g2/g3 (analysis track, not in CLI) ───────────────────────────────────
pitchbench_g1_PITCHES = BENCHMARK_PITCHES_FULL_RANGE              # ← edit me
pitchbench_g1_SOURCES = BENCHMARK_ALL_SOURCES                     # ← edit me

pitchbench_g2_PITCHES          = [29, 43, 57, 71, 72]             # ← edit me
pitchbench_g2_SOURCES          = BENCHMARK_ALL_SOURCES            # ← edit me
pitchbench_g2_TONE_DURATION_MS = 2000                             # ← edit me
pitchbench_g2_TOP_K            = 200                              # ← edit me
pitchbench_g2_MAX_NEW_TOKENS   = 32                               # ← edit me

pitchbench_g3_PITCHES    = BENCHMARK_PITCHES_FULL_RANGE           # ← edit me
pitchbench_g3_SOURCES    = BENCHMARK_ALL_SOURCES                  # ← edit me
pitchbench_g3_K          = 1                                      # ← edit me
pitchbench_g3_SPLIT_SEED = BENCHMARK_SEED                         # ← edit me

# ── f1 ──────────────────────────────────────────────────────────────────────
pitchbench_f1_N_NOTES      = 10                                  # ← edit me
pitchbench_f1_N_PARTS_LIST = [2, 3, 4]                           # ← edit me
pitchbench_f1_N_TRIALS     = 2                                   # ← edit me
pitchbench_f1_DIST_N_MIN   = 1                                   # ← edit me
pitchbench_f1_DIST_N_MAX   = 20                                  # ← edit me
pitchbench_f1_DUR_JITTER   = 0.5                                 # ← edit me
pitchbench_f1_TEMPOS       = {"slow": 1000, "medium": 500}       # ← edit me
pitchbench_f1_PART_RANGES  = [(72, 83), (60, 71), (48, 59), (36, 47)] # ← edit me
pitchbench_f1_MIXED_GROUPS = BENCHMARK_F1_MIXED_GROUPS           # ← edit me
pitchbench_f1_SEED         = BENCHMARK_SEED                      # ← edit me
pitchbench_f1_SOURCES      = BENCHMARK_CONTINUOUS_GM_INSTRUMENTS # ← edit me

# ── f2 ──────────────────────────────────────────────────────────────────────
pitchbench_f2_N_VOICES         = 4                               # ← edit me
pitchbench_f2_MIN_SEG_NOTES    = 4                               # ← edit me
pitchbench_f2_MAX_SEG_SEC      = 30.0                            # ← edit me
pitchbench_f2_QPM              = 60.0                            # ← edit me
pitchbench_f2_VOICE_NAMES      = ["soprano", "alto", "tenor", "bass"] # ← edit me
pitchbench_f2_DEFAULT_CHORALES = ["bach/bwv66.6", "bach/bwv4.8",
                                  "bach/bwv7.7", "bach/bwv26.6",
                                  "bach/bwv57.8"]                # ← edit me
pitchbench_f2_MIXED_GROUPS     = {
    "classical_quartet": ["flute", "violin", "cello", "bass"],
    "mixed_timbres":     ["piano", "trumpet", "clarinet", "guitar"],
}                                                                # ← edit me
pitchbench_f2_SEED             = BENCHMARK_SEED                  # ← edit me
pitchbench_f2_SOURCES          = BENCHMARK_GM_INSTRUMENTS        # ← edit me

# ── z1 ──────────────────────────────────────────────────────────────────────
pitchbench_z1_N_PER_FAMILY = 10                                  # ← edit me
pitchbench_z1_SEED         = BENCHMARK_SEED                      # ← edit me

# ── Runtime patching (for run_analysis.py --user-config) ────────────────────
# Add any pitchbench_<id>_<VAR> key you want to override at runtime.
USER_OVERRIDES: dict[str, Any] = {
    # "pitchbench_a2_LOUDNESS_DB": [-20, -10, 0],
}

EXPERIMENTS: list[str] = [
    "pitchbench_a1_single_pitch_id",
    "pitchbench_a2_single_pitch_by_loudness",
    "pitchbench_a3_single_pitch_by_duration",
    "pitchbench_b1_single_pitch_within_silence",
    "pitchbench_b2_pitch_at_timestamp",
    "pitchbench_b3_timestamp_single_pitch",
    "pitchbench_b4_timestamp_specific_pitch",
    "pitchbench_b5_timestamp_multiple_pitches",
    "pitchbench_c1_chord_count_pitches",
    "pitchbench_c2_chord_dyad_interval",
    "pitchbench_c3_chord_quality",
    "pitchbench_c4_chord_pitches",
    "pitchbench_d1_sequence_count_pitches",
    "pitchbench_d2_dyad_lower_higher_difference",
    "pitchbench_d3_contour_discrete",
    "pitchbench_d4_contour_continuous",
    "pitchbench_d5_sequence_ranking_by_pitch",
    "pitchbench_d6_sequence_dyad_interval",
    "pitchbench_d7a_pitch_with_reference",
    "pitchbench_d7b_pitch_with_reference_split",
    "pitchbench_d8_sequence_pitches",
    "pitchbench_e1_audio_effects",
    "pitchbench_e2_background",
    "pitchbench_e3_harmonic_saturation",
    "pitchbench_e4_time_stretching",
    "pitchbench_e5_vibrato",
    "pitchbench_e6_slightly_off",
    "pitchbench_f1_melodic_line_atonal",
    "pitchbench_f2_melodic_line_tonal",
]
