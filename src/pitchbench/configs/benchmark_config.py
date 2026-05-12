"""
Benchmark (EVAL) configuration — paper-locked values.

Standalone: no imports from the rest of pitchbench.

Contains:
  - Source-list constants (mirrors config.py engine catalogues)
  - All BENCHMARK_* data-generation constants
  - SAMPLING dict (stratification specs for --sample-n)
  - All pitchbench_* experiment parameters at benchmark (paper) values

Loaded by config.py when MODE = "EVAL".
"""

from __future__ import annotations
from typing import Any

_WAVEFORMS = ["sine", "sawtooth", "square", "triangle"]
_GM_PROGRAMS_V1: dict[str, int] = {
    "piano": 0, "electric_keyboard": 4, "guitar": 24,
    "flute": 73, "trumpet": 56, "trombone": 57, "clarinet": 71, "oboe": 68,
    "violin": 40, "cello": 42, "organ": 19, "bass": 32,
    "synth_lead": 80, "synth_pad": 88, "voice": 52,
}
_ALL_INSTRUMENTS: list[str] = list(_GM_PROGRAMS_V1.keys())


# ═══ BENCHMARK CONSTANTS — DO NOT MODIFY for paper runs ═══════════════════
BENCHMARK_NOTATION_FORMATS          = ["midi", "spn", "hz", "doremi"]
BENCHMARK_PITCHES_FULL_RANGE        = list(range(29, 90))
BENCHMARK_PITCHES_SELECTION         = [30, 36, 43, 48, 54, 60, 64, 69, 79, 88]
BENCHMARK_PITCHES_SELECTION_REDUCED = [30, 43, 54, 64, 79]
BENCHMARK_PITCHES_SELECTION_COMPACT = [42, 53, 60, 67, 77]
BENCHMARK_INTERVALS                 = [-12, -7, -5, -4, -3, -1, 0, 1, 3, 4, 5, 7, 12]
BENCHMARK_DURATION_MS               = 5000
BENCHMARK_DURATIONS_MS_SHORT_LONG   = [1000, 5000]
BENCHMARK_DURATIONS_MS_FULL_SWEEP   = [50, 250, 500, 1000, 4000, 15000, 60000]
BENCHMARK_NOTE_DURATIONS_MS_D4      = [250, 500, 1000]
BENCHMARK_TOTAL_DUR_MS              = 60_000
BENCHMARK_TONE_POSITIONS_MS         = [2_000, 7_000, 14_000, 22_000,
                                       27_000, 41_000, 47_000, 53_000]
BENCHMARK_GAP_MS                    = 300
BENCHMARK_GAP_MS_D7                 = 250
BENCHMARK_N_TRIALS                  = 3
BENCHMARK_SEED                      = 100
BENCHMARK_TIMESTAMP_TOLERANCE_MS    = 250
BENCHMARK_ALL_SOURCES               = _WAVEFORMS + _ALL_INSTRUMENTS
BENCHMARK_WAVEFORMS                 = _WAVEFORMS
BENCHMARK_GM_INSTRUMENTS            = _ALL_INSTRUMENTS
BENCHMARK_CONTINUOUS_GM_INSTRUMENTS = [
    "flute", "trumpet", "trombone", "clarinet", "oboe",
    "violin", "cello", "organ", "synth_lead", "synth_pad", "voice",
]
BENCHMARK_F1_MIXED_GROUPS: dict[str, list[str]] = {
    "winds_and_strings": ["flute", "violin", "cello", "clarinet"],
    "brass_reeds_synth": ["trumpet", "trombone", "oboe", "synth_pad"],
}
BENCHMARK_BASE_FREQS_A = {"A3": 220.00, "A4": 440.00, "A5": 880.00}
BENCHMARK_C1_CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0,4,7), "min": (0,3,7), "dim": (0,3,6), "aug": (0,4,8),
    "dom7": (0,4,7,10), "maj7": (0,4,7,11), "min7": (0,3,7,10),
}
BENCHMARK_C4_CHORD_TYPES: dict[str, list[int]] = {
    "dyad_m2":[0,1],"dyad_M2":[0,2],"dyad_m3":[0,3],"dyad_M3":[0,4],
    "dyad_p4":[0,5],"dyad_tt":[0,6],"dyad_p5":[0,7],"dyad_m6":[0,8],
    "dyad_M6":[0,9],"dyad_m7":[0,10],"dyad_M7":[0,11],"dyad_oct":[0,12],
    "dyad_uni":[0,0],
    "triad_maj":[0,4,7],"triad_min":[0,3,7],"triad_dim":[0,3,6],"triad_aug":[0,4,8],
    "seventh_dom":[0,4,7,10],"seventh_maj":[0,4,7,11],"seventh_min":[0,3,7,10],
}
BENCHMARK_C3_QUALITIES: dict[str, tuple[tuple[int,...],str]] = {
    "major":      ((0,4,7),    "major"),       "minor":      ((0,3,7),    "minor"),
    "diminished": ((0,3,6),    "diminished"),  "augmented":  ((0,4,8),    "augmented"),
    "dom7":       ((0,4,7,10), "dominant seventh"), "maj7": ((0,4,7,11), "major seventh"),
    "min7":       ((0,3,7,10), "minor seventh"),    "m7b5": ((0,3,6,10), "half diminished"),
    "sus2":       ((0,2,7),    "sus2"),         "sus4":       ((0,5,7),    "sus4"),
}
BENCHMARK_D4_TRAJECTORIES: list[dict[str, Any]] = [
    {"name":"up",           "shape":"linear", "interval_sign":+1, "gt_seq":["up"]},
    {"name":"down",         "shape":"linear", "interval_sign":-1, "gt_seq":["down"]},
    {"name":"up_then_down", "shape":"arch",   "interval_sign":+1, "gt_seq":["up","down"]},
    {"name":"down_then_up", "shape":"valley", "interval_sign":-1, "gt_seq":["down","up"]},
]
BENCHMARK_E1_EFFECTS: dict[str, dict[str,Any]] = {
    "highpass_above_f0": {"type":"highpass",    "cutoff_ratio":1.5, "order":6},
    "lowpass_at_f0":     {"type":"lowpass",     "cutoff_ratio":1.2, "order":6},
    "bitcrush_4bit":     {"type":"bitcrush",    "bit_depth":4},
    "distortion_heavy":  {"type":"saturation",  "drive_db":30.0},
    "reverb_long":       {"type":"reverb_room", "room_size":0.9, "damping":0.4,
                          "wet_level":0.6, "dry_level":0.4},
    "chorus_heavy":      {"type":"chorus",      "rate_hz":1.2, "depth":0.9, "mix":0.6},
}
BENCHMARK_E3_SATURATIONS: dict[str, dict[str,Any]] = {
    "sat_light": {"type":"saturation","drive_db":6.0},
    "sat_medium": {"type":"saturation","drive_db":15.0},
    "sat_heavy":  {"type":"saturation","drive_db":30.0},
}
BENCHMARK_PITCHES_E5 = [36, 43, 48, 54, 58, 60, 64, 67, 69, 72, 77, 84]
BENCHMARK_E4_CONDITIONS: list[dict[str,Any]] = [
    {"name":"resample_0.5x", "mode":"resample", "factor":0.5},
    {"name":"resample_2x",   "mode":"resample", "factor":2.0},
    {"name":"stretch_0.5x",  "mode":"stretch",  "factor":0.5},
    {"name":"stretch_2x",    "mode":"stretch",  "factor":2.0},
]

# ═══ SAMPLING (stratification) ════════════════════════════════════════════════
SAMPLING: dict[str, dict[str, Any]] = {
    "pitchbench_a1_single_pitch_id":              {"per_stratum": 1,  "strata": ("midi", "source")},
    "pitchbench_y1_single_pitch_id_mcq":          {"per_stratum": 1,  "strata": ("midi", "source", "semitone_step")},
    "pitchbench_a2_single_pitch_by_loudness":     {"per_stratum": 4,  "strata": ("midi", "loudness_db")},
    "pitchbench_a3_single_pitch_by_duration":     {"per_stratum": 3,  "strata": ("midi", "duration_ms")},

    "pitchbench_b1_single_pitch_within_silence":  {"per_stratum": 2,  "strata": ("midi", "pos_ms")},
    "pitchbench_b2_pitch_at_timestamp":           {"per_stratum": 10, "strata": ("n_notes", "target_idx")},
    "pitchbench_b3_timestamp_single_pitch":       {"per_stratum": 1,  "strata": ("pos_ms", "duration_ms", "midi")},
    "pitchbench_b4_timestamp_specific_pitch":     {"per_stratum": 10, "strata": ("target_pos", "n_distractors", "duration_ms")},
    "pitchbench_b5_timestamp_multiple_pitches":   {"per_stratum": 10, "strata": ("rhythm", "n_notes", "duration_ms")},

    "pitchbench_c1_chord_count_pitches":          {"per_stratum": 1,  "strata": ("n", "chord_quality", "root_midi", "same_instrument")},
    "pitchbench_c2_chord_dyad_interval":          {"per_stratum": 1,  "strata": ("interval_st", "same_instrument", "root_midi")},
    "pitchbench_c3_chord_quality":                {"per_stratum": 1,  "strata": ("chord_quality_gt", "same_instrument", "root_midi")},
    "pitchbench_c4_chord_pitches":                {"per_stratum": 1,  "strata": ("chord_type", "n_notes", "root_midi")},

    "pitchbench_d1_sequence_count_pitches":       {"per_stratum": 5,  "strata": ("n", "rhythm", "duration_ms")},
    "pitchbench_d2_dyad_lower_higher_difference": {"per_stratum": 1,  "strata": ("delta_cents", "order", "duration_ms", "base_name")},
    "pitchbench_d3_contour_discrete":             {"per_stratum": 3,  "strata": ("n_transitions", "step_size_st", "note_duration_ms")}, #
    "pitchbench_d4_contour_continuous":           {"per_stratum": 1,  "strata": ("traj_name", "interval_st", "start_midi")},
    "pitchbench_d5_sequence_ranking_by_pitch":    {"per_stratum": 1,  "strata": ("n_notes", "delta_cents", "rhythm", "base_name")},
    "pitchbench_d6_sequence_dyad_interval":       {"per_stratum": 1,  "strata": ("signed_st", "base_midi")},
    "pitchbench_d7a_pitch_with_reference":        {"per_stratum": 2,  "strata": ("ref_midi", "interval")},
    "pitchbench_d7b_pitch_with_reference_split":  {"per_stratum": 2,  "strata": ("ref_midi", "interval")},
    "pitchbench_d8_sequence_pitches":             {"per_stratum": 3,  "strata": ("n_notes", "source")},
    "pitchbench_e1_audio_effects":                {"per_stratum": 2,  "strata": ("effect_type", "midi", "source_type")},
    "pitchbench_e2_background":                   {"per_stratum": 1,  "strata": ("background", "snr_db", "midi")},
    "pitchbench_e3_harmonic_saturation":          {"per_stratum": 2,  "strata": ("saturation_level", "midi", "source_type")},
    "pitchbench_e4_time_stretching":              {"per_stratum": 2,  "strata": ("condition", "midi", "source_type")},
    "pitchbench_e5_vibrato":                      {"per_stratum": 2,  "strata": ("vibrato_rate_hz", "vibrato_depth_cents", "midi")},
    "pitchbench_e6_slightly_off":                 {"per_stratum": 4,  "strata": ("midi", "detune_hz", "midi")},
    "pitchbench_f1_melodic_line_atonal":          {"per_stratum": 1,  "strata": ("n", "source_label", "tempo", "x")},
    "pitchbench_f2_melodic_line_tonal":           {"per_stratum": 8,  "strata": ("chorale_slug", "x")},
}

# ═══ PER-EXPERIMENT PARAMETERS ════════════════════════════════════════════════
pitchbench_general_NOTATION_FORMATS = BENCHMARK_NOTATION_FORMATS

pitchbench_a1_PITCHES          = BENCHMARK_PITCHES_FULL_RANGE
pitchbench_a1_TONE_DURATION_MS = BENCHMARK_DURATION_MS
pitchbench_a1_SOURCES          = BENCHMARK_ALL_SOURCES

pitchbench_y1_PITCHES         = BENCHMARK_PITCHES_FULL_RANGE
pitchbench_y1_TONE_DURATION_MS = BENCHMARK_DURATION_MS
pitchbench_y1_SOURCES         = BENCHMARK_ALL_SOURCES
pitchbench_y1_N_OPTIONS       = 5
pitchbench_y1_OPTION_STEP_ST  = 2
pitchbench_y1_SEED            = BENCHMARK_SEED

# Shared by d7a (concat-audio with anchor) and d7b (split-audio with anchor) —
# both experiments draw stimuli from the same condition grid so their accuracy
# numbers are directly comparable.
pitchbench_d7_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION_COMPACT
pitchbench_d7_INTERVALS         = BENCHMARK_INTERVALS
pitchbench_d7_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
pitchbench_d7_GAP_MS            = 500
pitchbench_d7_SOURCES           = BENCHMARK_ALL_SOURCES

# d7c (concat-audio) and d7d (split-audio) — same anchor design as d7a/d7b
# but target pitches come from an explicit list matching a1, not from intervals.
pitchbench_d7c_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION_COMPACT
pitchbench_d7c_PITCHES           = BENCHMARK_PITCHES_SELECTION
pitchbench_d7c_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
pitchbench_d7c_GAP_MS            = 500
pitchbench_d7c_SOURCES           = BENCHMARK_ALL_SOURCES

pitchbench_d7d_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION_COMPACT
pitchbench_d7d_PITCHES           = BENCHMARK_PITCHES_SELECTION
pitchbench_d7d_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
pitchbench_d7d_GAP_MS            = 500
pitchbench_d7d_SOURCES           = BENCHMARK_ALL_SOURCES

pitchbench_a3_PITCHES      = BENCHMARK_PITCHES_SELECTION
pitchbench_a3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_FULL_SWEEP
pitchbench_a3_SOURCES      = BENCHMARK_ALL_SOURCES

pitchbench_e5_VIBRATO_RATES_HZ     = [3, 5, 7, 10]
pitchbench_e5_VIBRATO_DEPTHS_CENTS = [25, 50, 100, 200]
pitchbench_e5_DURATIONS_MS         = [BENCHMARK_DURATION_MS]
pitchbench_e5_PITCHES              = BENCHMARK_PITCHES_SELECTION_REDUCED
pitchbench_e5_SOURCES              = BENCHMARK_WAVEFORMS

pitchbench_e6_N_DETUNE_LEVELS = 5
pitchbench_e6_DETUNE_FRACTION = 0.45
pitchbench_e6_PITCHES         = BENCHMARK_PITCHES_SELECTION
pitchbench_e6_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_e6_SOURCES         = BENCHMARK_ALL_SOURCES

pitchbench_b1_PITCHES           = BENCHMARK_PITCHES_SELECTION
pitchbench_b1_TONE_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS
pitchbench_b1_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
pitchbench_b1_TOTAL_SILENCE_MS  = BENCHMARK_TOTAL_DUR_MS
pitchbench_b1_SOURCES           = BENCHMARK_ALL_SOURCES

pitchbench_b3_PITCHES      = BENCHMARK_PITCHES_SELECTION
pitchbench_b3_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS
pitchbench_b3_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS
pitchbench_b3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_b3_SOURCES      = BENCHMARK_ALL_SOURCES

pitchbench_b4_N_DISTRACTORS   = [5, 7]
pitchbench_b4_TARGET_POS_OPTS = ["first", "middle", "last"]
pitchbench_b4_TOTAL_DUR_MS    = BENCHMARK_TOTAL_DUR_MS
pitchbench_b4_GAP_MIN_MS      = 500
pitchbench_b4_GAP_MAX_MS      = 2000
pitchbench_b4_PITCHES         = BENCHMARK_PITCHES_SELECTION
pitchbench_b4_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_b4_SEED            = BENCHMARK_SEED
pitchbench_b4_SOURCES         = BENCHMARK_ALL_SOURCES

pitchbench_b2_N_NOTES_OPTS = [5, 10]
pitchbench_b2_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS
pitchbench_b2_GAP_MIN_MS   = 30
pitchbench_b2_GAP_MAX_MS   = 1500
pitchbench_b2_MAIN_PITCHES = BENCHMARK_PITCHES_SELECTION
pitchbench_b2_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_b2_SEED         = BENCHMARK_SEED
pitchbench_b2_SOURCES      = BENCHMARK_ALL_SOURCES
pitchbench_b2_DISTRACTORS  = BENCHMARK_PITCHES_FULL_RANGE

pitchbench_b5_N_NOTES_OPTS   = [3, 5, 8]
pitchbench_b5_RHYTHMS        = ["regular", "irregular"]
pitchbench_b5_PITCH_PATTERNS = ["fixed_pitch", "varied_pitches"]
pitchbench_b5_TOTAL_DUR_MS   = BENCHMARK_TOTAL_DUR_MS
pitchbench_b5_IRR_GAP_MIN_MS = 300
pitchbench_b5_IRR_GAP_MAX_MS = 2500
pitchbench_b5_PITCHES        = BENCHMARK_PITCHES_SELECTION
pitchbench_b5_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_b5_SEED           = BENCHMARK_SEED
pitchbench_b5_SOURCES        = BENCHMARK_ALL_SOURCES

pitchbench_c2_INTERVALS_ST         = list(range(1, 13))
pitchbench_c2_SAME_INSTRUMENT_OPTS = [True, False]
pitchbench_c2_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_c2_PITCHES              = BENCHMARK_PITCHES_SELECTION
pitchbench_c2_SOURCES              = BENCHMARK_ALL_SOURCES

pitchbench_c1_CHORD_INTERVALS      = {q: BENCHMARK_C3_QUALITIES[q][0] for q in BENCHMARK_C3_QUALITIES}
pitchbench_c1_QUALITIES            = list(BENCHMARK_C3_QUALITIES.keys()) + ["random_set"]
pitchbench_c1_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION
pitchbench_c1_SAME_INSTRUMENT_OPTS = [True, False]
pitchbench_c1_RANDOM_NS            = [1, 2, 3, 4, 5, 6]
pitchbench_c1_RANDOM_PITCH_RANGE   = (BENCHMARK_PITCHES_FULL_RANGE[0], BENCHMARK_PITCHES_FULL_RANGE[-1])
pitchbench_c1_N_TRIALS             = 1
pitchbench_c1_RANDOM_TRIALS        = 3
pitchbench_c1_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_c1_SEED                 = BENCHMARK_SEED
pitchbench_c1_SOURCES              = BENCHMARK_ALL_SOURCES

pitchbench_c4_CHORD_TYPES      = BENCHMARK_C4_CHORD_TYPES
pitchbench_c4_BASE_ROOTS       = BENCHMARK_PITCHES_SELECTION
pitchbench_c4_TONE_DURATION_MS = BENCHMARK_DURATION_MS
pitchbench_c4_SOURCES          = BENCHMARK_ALL_SOURCES

pitchbench_c3_QUALITIES            = BENCHMARK_C3_QUALITIES
pitchbench_c3_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION
pitchbench_c3_TASKS                = ["quality_only"]
pitchbench_c3_SAME_INSTRUMENT_OPTS = [True, False]
pitchbench_c3_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_c3_SOURCES              = BENCHMARK_ALL_SOURCES

pitchbench_d1_N_COUNTS     = [1, 2, 3, 4, 5, 7, 10]
pitchbench_d1_RHYTHMS      = ["regular", "irregular"]
pitchbench_d1_PITCH_MIN    = 29
pitchbench_d1_PITCH_MAX    = 89
pitchbench_d1_GAP_MS       = BENCHMARK_GAP_MS
pitchbench_d1_N_TRIALS     = BENCHMARK_N_TRIALS
pitchbench_d1_SEED         = BENCHMARK_SEED
pitchbench_d1_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_d1_SOURCES      = BENCHMARK_ALL_SOURCES

pitchbench_d2_BASE_FREQS    = BENCHMARK_BASE_FREQS_A
pitchbench_d2_DELTA_CENTS   = [1, 2, 5, 10, 25, 50, 100, 200, 400, 700, 1200]
pitchbench_d2_SEPARATION_MS = [200, 500, 1000, 2000]
pitchbench_d2_DURATION_MS   = BENCHMARK_DURATION_MS
pitchbench_d2_N_TRIALS      = BENCHMARK_N_TRIALS
pitchbench_d2_SEED          = BENCHMARK_SEED
pitchbench_d2_SOURCES       = BENCHMARK_ALL_SOURCES

pitchbench_d6_INTERVALS_ST   = list(range(1, 13))
pitchbench_d6_DIRECTIONS     = ["ascending", "descending"]
pitchbench_d6_SEPARATIONS_MS = [200, 500, 1000, 2000]
pitchbench_d6_PITCHES        = BENCHMARK_PITCHES_SELECTION
pitchbench_d6_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG
pitchbench_d6_SOURCES        = BENCHMARK_ALL_SOURCES

pitchbench_d3_N_TRANSITIONS_OPTS = [2, 3, 5, 7]
pitchbench_d3_STEP_SIZES_ST      = [1, 2, 4, 7, 11]
pitchbench_d3_NOTE_DURATIONS_MS  = BENCHMARK_NOTE_DURATIONS_MS_D4
pitchbench_d3_TRIALS_PER_CELL    = 2
pitchbench_d3_PITCHES            = BENCHMARK_PITCHES_SELECTION
pitchbench_d3_SEED               = BENCHMARK_SEED
pitchbench_d3_SOURCES            = BENCHMARK_ALL_SOURCES

pitchbench_d4_START_PITCHES = BENCHMARK_PITCHES_SELECTION
pitchbench_d4_INTERVALS_ST  = [1, 4, 7, 12]
pitchbench_d4_DURATION_MS   = BENCHMARK_DURATION_MS
pitchbench_d4_SOURCES       = BENCHMARK_WAVEFORMS
pitchbench_d4_TRAJECTORIES  = BENCHMARK_D4_TRAJECTORIES

pitchbench_d5_BASE_FREQS  = BENCHMARK_BASE_FREQS_A
pitchbench_d5_DELTA_CENTS = [25, 50, 100, 200, 400]
pitchbench_d5_N_TONES     = [3, 4, 5, 7]
pitchbench_d5_RHYTHMS     = ["regular", "irregular"]
pitchbench_d5_DURATION_MS = BENCHMARK_DURATION_MS
pitchbench_d5_GAP_MS      = BENCHMARK_GAP_MS
pitchbench_d5_N_TRIALS    = BENCHMARK_N_TRIALS
pitchbench_d5_SEED        = BENCHMARK_SEED
pitchbench_d5_SOURCES     = BENCHMARK_WAVEFORMS

pitchbench_d8_PITCH_MIN    = 29
pitchbench_d8_PITCH_MAX    = 89
pitchbench_d8_N_NOTES_LIST = [3, 5, 10]
pitchbench_d8_TONE_MS      = BENCHMARK_DURATION_MS
pitchbench_d8_GAP_MS       = BENCHMARK_GAP_MS_D7
pitchbench_d8_N_TRIALS     = 5
pitchbench_d8_SEED         = BENCHMARK_SEED
pitchbench_d8_SOURCES      = BENCHMARK_ALL_SOURCES

pitchbench_a2_LOUDNESS_DB = [-30, -20, -12, -3, 6]
pitchbench_a2_TONE_MS     = BENCHMARK_DURATION_MS
pitchbench_a2_PITCHES     = BENCHMARK_PITCHES_SELECTION
pitchbench_a2_SOURCES     = BENCHMARK_ALL_SOURCES

pitchbench_e1_PITCHES = BENCHMARK_PITCHES_SELECTION
pitchbench_e1_TONE_MS = BENCHMARK_DURATION_MS
pitchbench_e1_EFFECTS = BENCHMARK_E1_EFFECTS
pitchbench_e1_SOURCES = BENCHMARK_ALL_SOURCES

pitchbench_e2_BACKGROUNDS  = ["white_noise", "church-bells", "crowd-noise",
                               "rain", "street-noise", "oscillating-high-and-low-pitches"]
pitchbench_e2_SNR_DB       = [10.0, 0.0, -6.0, -12.0]
pitchbench_e2_PITCHES      = BENCHMARK_PITCHES_SELECTION
pitchbench_e2_DURATIONS_MS = [BENCHMARK_DURATION_MS]
pitchbench_e2_SOURCES      = BENCHMARK_ALL_SOURCES

pitchbench_e3_PITCHES     = BENCHMARK_PITCHES_SELECTION
pitchbench_e3_TONE_MS     = BENCHMARK_DURATION_MS
pitchbench_e3_SATURATIONS = BENCHMARK_E3_SATURATIONS
pitchbench_e3_SOURCES     = BENCHMARK_ALL_SOURCES

pitchbench_e4_PITCHES    = BENCHMARK_PITCHES_E5
pitchbench_e4_TONE_MS    = 3000
pitchbench_e4_CONDITIONS = BENCHMARK_E4_CONDITIONS
pitchbench_e4_SOURCES    = BENCHMARK_ALL_SOURCES

pitchbench_f1_N_NOTES      = 10
pitchbench_f1_N_PARTS_LIST = [2, 3]
pitchbench_f1_N_TRIALS     = 2
pitchbench_f1_DIST_N_MIN   = 1
pitchbench_f1_DIST_N_MAX   = 20
pitchbench_f1_DUR_JITTER   = 0.5
pitchbench_f1_TEMPOS       = {"slow": 1000, "medium": 500}
pitchbench_f1_PART_RANGES  = [(72, 83), (60, 71), (48, 59), (36, 47)]
pitchbench_f1_MIXED_GROUPS = BENCHMARK_F1_MIXED_GROUPS
pitchbench_f1_SEED         = BENCHMARK_SEED
pitchbench_f1_SOURCES      = BENCHMARK_CONTINUOUS_GM_INSTRUMENTS

pitchbench_f2_N_VOICES         = 4
pitchbench_f2_MIN_SEG_NOTES    = 4
pitchbench_f2_MAX_SEG_SEC      = 30.0
pitchbench_f2_QPM              = 60.0
pitchbench_f2_VOICE_NAMES      = ["soprano", "alto", "tenor", "bass"]
pitchbench_f2_DEFAULT_CHORALES = ["bach/bwv66.6", "bach/bwv4.8",
                                  "bach/bwv7.7", "bach/bwv26.6", "bach/bwv57.8"]
pitchbench_f2_MIXED_GROUPS     = {
    "classical_quartet": ["flute", "violin", "cello", "bass"],
    "mixed_timbres":     ["piano", "trumpet", "clarinet", "guitar"],
}
pitchbench_f2_SEED             = BENCHMARK_SEED
pitchbench_f2_SOURCES          = BENCHMARK_GM_INSTRUMENTS

pitchbench_z1_N_PER_FAMILY = 10
pitchbench_z1_SEED         = BENCHMARK_SEED

# ── g1/g2/g3 (analysis track, not in CLI) ────────────────────────────────────
pitchbench_g1_PITCHES = BENCHMARK_PITCHES_FULL_RANGE
pitchbench_g1_SOURCES = BENCHMARK_ALL_SOURCES

pitchbench_g2_PITCHES          = [29, 43, 57, 71, 72]
pitchbench_g2_SOURCES          = BENCHMARK_ALL_SOURCES
pitchbench_g2_TONE_DURATION_MS = 2000
pitchbench_g2_TOP_K            = 200
pitchbench_g2_MAX_NEW_TOKENS   = 32

pitchbench_g3_PITCHES    = BENCHMARK_PITCHES_FULL_RANGE
pitchbench_g3_SOURCES    = BENCHMARK_ALL_SOURCES
pitchbench_g3_K          = 1
pitchbench_g3_SPLIT_SEED = BENCHMARK_SEED
