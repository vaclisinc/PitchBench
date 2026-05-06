"""
Central configuration for PitchBench.

All data-generation parameters live here, namespaced by experiment as
``pitchbench_<exp>_VARIABLE``. Experiment scripts shadow these names locally so
their generation code stays terse.

EVAL flag
---------
``EVAL = True``  → every per-experiment data-gen parameter is locked to its
                   ``BENCHMARK_*`` value. This is paper-mode: do not edit the
                   benchmark constants.
``EVAL = False`` → the ``else`` branch below runs; values default to the same
                   benchmark numbers but each line is marked ``# ← edit me`` so
                   you can tune individual experiments without touching the
                   benchmark section.

Prompts and parsing logic stay in their experiment scripts — they are not
data-gen knobs.
"""

from datetime import datetime
import os
from pathlib import Path

# ── Mode ──────────────────────────────────────────────────────────────────────
# Flip this to False to use the user-overridable values further down.
EVAL = True

# ── Runtime / project root ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get("PITCHBENCH_ROOT", ".")).resolve()

DATA_DIR     = _PROJECT_ROOT / "data"
AUDIO_DIR    = DATA_DIR / "audio"
RESULTS_DIR  = _PROJECT_ROOT / "results" / os.environ.get(
    "PITCHBENCH_RUN", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
STIMULI_DIR  = _PROJECT_ROOT / "stimuli"

# ── Audio infrastructure (engine-level; not data-gen) ─────────────────────────
SAMPLE_RATE  = 16_000
NOTE_ON_DUR  = 1.7
RELEASE_DUR  = 0.5
FADE_OUT     = 0.05
TARGET_PEAK  = 0.9

SF2_PATH = os.environ.get(
    "PITCHBENCH_SF2",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
)

# ── Sources (engine-level catalogues; per-experiment SOURCE lists reference these) ──
GM_PROGRAMS_V1: dict[str, int] = {
    "piano":             0,   # Acoustic Grand Piano
    "electric_keyboard": 4,   # Electric Piano 1
    "guitar":            24,  # Acoustic Guitar (nylon)
    "flute":             73,
    "trumpet":           56,
    "trombone":          57,
    "clarinet":          71,
    "oboe":              68,
    "violin":            40,
    "cello":             42,
    "organ":             19,  # Church Organ
    "bass":              32,
    "synth_lead":        80,
    "synth_pad":         88,
    "voice":             52,  # Choir Aahs
}

WAVEFORMS:   list[str] = ["sine", "sawtooth", "square", "triangle"]
ALL_SOURCES: list[str] = list(WAVEFORMS) + list(GM_PROGRAMS_V1.keys())

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS_DIR     = _PROJECT_ROOT / "_datasets"
NSYNTH_VALID_DIR = DATASETS_DIR / "NSynth" / "nsynth-valid"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "manual":                        "Manual (CLI input by human)",
    "music_flamingo":                "Music Flamingo",
    "audio_flamingo_next_instruct":  "Audio Flamingo Next – instruct",
    "audio_flamingo_next_think":     "Audio Flamingo Next – think",
    "audio_flamingo_next_captioner": "Audio Flamingo Next – captioner",
    "qwen3_omni":                    "Qwen3-Omni (30B-A3B-Instruct)",
    "kimi_audio":                    "Kimi-Audio 7B Instruct",
}

MODEL_URLS: dict[str, str] = {
    "music_flamingo":                os.environ.get("MF_URL",            "http://localhost:8000"),
    "audio_flamingo_next_instruct":  os.environ.get("AF_NEXT_INST_URL",  "http://localhost:8001"),
    "audio_flamingo_next_think":     os.environ.get("AF_NEXT_THINK_URL", "http://localhost:8002"),
    "audio_flamingo_next_captioner": os.environ.get("AF_NEXT_CAP_URL",   "http://localhost:8003"),
    "qwen3_omni":                    os.environ.get("QWEN3_OMNI_URL",    "http://lowland.cs.berkeley.edu:9999"),
    "kimi_audio":                    os.environ.get("KIMI_AUDIO_URL",    "http://localhost:8004"),
}

LOCAL_CONCURRENCY: int = int(os.environ.get("LOCAL_CONCURRENCY", "1"))

OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

DOREMI_PARSER_MODEL: str = "openrouter/google/gemini-2.5-flash-lite"
DEFAULT_MODEL:       str = "openrouter/google/gemini-3.1-flash-lite-preview"

MODEL_MUSIC_FLAMINGO = "nvidia/music-flamingo-hf"

AF_NEXT_CHECKPOINTS = {
    "audio_flamingo_next_instruct":  "nvidia/audio-flamingo-next-hf",
    "audio_flamingo_next_think":     "nvidia/audio-flamingo-next-think-hf",
    "audio_flamingo_next_captioner": "nvidia/audio-flamingo-next-captioner-hf",
}

# ── Dispatcher concurrency caps ───────────────────────────────────────────────
# Local concurrency is the single ``LOCAL_CONCURRENCY`` knob above. The caps
# here cover non-local cases: OpenRouter (network-bound, can run wide) and the
# interactive ``manual`` model (always 1).
CONCURRENCY: dict[str, int] = {
    "openrouter": int(os.environ.get("PITCHBENCH_CONCURRENCY_OPENROUTER", "20")),
    "manual":     1,
}


def concurrency_for(model_name: str) -> int:
    """Return the configured max-workers cap for ``model_name``."""
    if model_name == "manual":
        return CONCURRENCY["manual"]
    if model_name.startswith("openrouter/"):
        return CONCURRENCY["openrouter"]
    return max(1, LOCAL_CONCURRENCY)


# ── CLI sampling defaults (not data-gen) ──────────────────────────────────────
DEFAULT_SAMPLE_SEED = 42 

EXPERIMENT_DEFAULTS: dict[str, dict] = {
    # Strata design rule of thumb:
    # 1) Include only factors that should be evenly represented in every run.
    # 2) Exclude paired/ablation toggles and constant fields (they add noise).
    # 3) Prefer 1-2 primary axes to keep total sample size tractable.
    # Total sampled items = per_stratum x num_distinct_strata_cells.
    # Category A: Single-pitch identification / perturbations
    # a1: balance pitch *and* timbre family (waveform vs instrument);
    # full `source` would explode strata to ≈19 levels.
    "pitchbench_a1_single_pitch_id":              {"per_stratum": 1, "strata": ("midi", "source")},
    "pitchbench_a2_single_pitch_by_loudness":     {"per_stratum": 4, "strata": ("midi", "loudness_db")},
    "pitchbench_a3_single_pitch_by_duration":     {"per_stratum": 3, "strata": ("midi", "duration_ms",)},
    # a2: do NOT stratify by `condition` (anchored vs baseline) — both
    # always run as a matched pair via CatASpec.pair_key. `interval` IS
    # the experimental factor (does the reference help across intervals?).
    
    # a4: the experiment IS about vibrato — stratify by the manipulation,
    # not by pitch. is_control is an ablation, not a stratum.
    "pitchbench_e5_vibrato":                  {"per_stratum": 1, "strata": ("vibrato_rate_hz", "vibrato_depth_cents")},
    "pitchbench_e6_slightly_off":             {"per_stratum": 1, "strata": ("midi", "detune_hz")},

    # Category B: Onsets, offsets, and time-localised pitch
    # b1: hidden + baseline run as a matched pair via CatBSpec.pair_key.
    "pitchbench_b1_single_pitch_within_silence":  {"per_stratum": 2, "strata": ("midi", "pos_ms")},
    "pitchbench_b2_pitch_at_timestamp":          {"per_stratum": 10, "strata": ("n_notes", "target_idx" )},
    # b2 is a *timing* task — stratify on timing factors, not pitch identity.
    "pitchbench_b3_timestamp_single_pitch":    {"per_stratum": 1, "strata": ("pos_ms", "duration_ms", "midi")},
    "pitchbench_b4_timestamp_specific_pitch":  {"per_stratum": 10, "strata": ("target_pos", "n_distractors", "duration_ms")},
    "pitchbench_b5_timestamp_multiple_pitches": {"per_stratum": 10, "strata": ("rhythm", "n_notes", "duration_ms")},

    # Category C: Chords / dyads / simultaneous pitches
    # c1 / c4: same_instrument (single-timbre vs mixed-timbre) is a real DV.
    "pitchbench_c1_chord_count_pitches":     {"per_stratum": 1, "strata": ("n", "chord_quality", "root_midi")},
    "pitchbench_c2_chord_dyad_interval":     {"per_stratum": 1, "strata": ("interval_st", "same_instrument", "root_midi")},
    "pitchbench_c3_chord_quality":           {"per_stratum": 1, "strata": ("chord_quality_gt", "same_instrument", "root_midi")},
    "pitchbench_c4_chord_pitches":           {"per_stratum": 1, "strata": ("chord_type", "n_notes", "root_midi")},

    # Category D: Sequences, contour, and intervals
    "pitchbench_d1_sequence_count_pitches":          {"per_stratum": 5, "strata": ("n", "rhythm", "duration_ms")},
    "pitchbench_d2_dyad_lower_higher_difference":    {"per_stratum": 1, "strata": ("delta_cents", "order", "duration_ms", "base_name")},
    "pitchbench_d3_contour_discrete":                {"per_stratum": 1, "strata": ("n_transitions", "step_size_st", "note_duration_ms", "base_midi" )},
    "pitchbench_d4_contour_continuous":              {"per_stratum": 1, "strata": ("traj_name","interval_st", "start_midi")},
    # d6: spacing (delta_cents) is the core difficulty axis for ranking.
    "pitchbench_d5_sequence_ranking_by_pitch":          {"per_stratum": 1, "strata": ("n_notes", "delta_cents", "rhythm", "base_name")},
    "pitchbench_d6_sequence_dyad_interval":             {"per_stratum": 1, "strata": ("signed_st","base_midi", )},
    "pitchbench_d7_pitch_with_reference":             {"per_stratum": 2, "strata": ("ref_midi","interval")},
    "pitchbench_d8_sequence_pitches":                   {"per_stratum": 3, "strata": ("n_notes","source")},

    # Category E: Loudness and effects
    "pitchbench_e1_audio_effects":         {"per_stratum": 1, "strata": ("effect_type", "midi", "source_type")},
    "pitchbench_e2_background":            {"per_stratum": 1, "strata": ("background", "snr_db", "source_type")},
    "pitchbench_e3_harmonic_saturation":   {"per_stratum": 1, "strata": ("saturation_level", "midi", "source_type")},
    "pitchbench_e4_time_stretching":       {"per_stratum": 1, "strata": ("condition", "midi", "source_type")},

    # Category F: Polyphony / multi-instrument identification
    "pitchbench_f1_melodic_line_atonal":   {"per_stratum": 1, "strata": ("n", "source_label", "tempo", "x")},
    # f2: voice position (x ∈ {soprano, alto, tenor, bass}) is a core DV.
    "pitchbench_f2_melodic_line_tonal":    {"per_stratum": 1, "strata": ("chorale_slug", "x", "source_label", "tempo")},
}


# ═══════════════════════════════════════════════════════════════════════════
#                         BENCHMARK CONSTANTS
#  Canonical paper values. Used when EVAL=True. DO NOT MODIFY for paper runs.
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_NOTATION_FORMATS = ["midi", "spn", "hz", "doremi"]

BENCHMARK_PITCHES_FULL_RANGE = list(range(29, 90))   # F1–F6
BENCHMARK_PITCHES_SELECTION  = [30, 36, 43, 48, 54, 60, 64, 69, 79, 88]      # 10 hand-picked
BENCHMARK_PITCHES_SELECTION_REDUCED = [30, 43, 54, 64, 79] # smaller set for faster runs
BENCHMARK_PITCHES_SELECTION_COMPACT = [42, 53, 60, 67, 77]

BENCHMARK_INTERVALS = [-12, -7, -5, -4, -3, -1, 0, 1,  3,  4,  5,  7, 12] # 13
BENCHMARK_DURATION_MS              = 5000
BENCHMARK_DURATIONS_MS_SHORT_LONG  = [1000, 5000]
BENCHMARK_DURATIONS_MS_FULL_SWEEP  = [50, 250, 500, 1000, 4000, 15000, 60000]
BENCHMARK_NOTE_DURATIONS_MS_D4     = [250, 500, 1000]

BENCHMARK_TOTAL_DUR_MS      = 60_000
BENCHMARK_TONE_POSITIONS_MS = [2_000, 7_000, 14_000, 22_000,
                               27_000, 41_000, 47_000, 53_000]
# BENCHMARK_TONE_POSITIONS_MS = [2_000, 14_000, 22_000,
#                                27_000, 47_000, 53_000] new proposal
BENCHMARK_GAP_MS            = 300
BENCHMARK_GAP_MS_D7         = 250
BENCHMARK_N_TRIALS          = 3
BENCHMARK_SEED              = 100

# A predicted timestamp is considered correct iff it falls within
# ±BENCHMARK_TIMESTAMP_TOLERANCE_MS of the ground-truth timestamp (in either
# direction). Used by b2/b3/b5 to derive the binary ``correct`` field on each
# stimulus and the primary timing-accuracy metric.
BENCHMARK_TIMESTAMP_TOLERANCE_MS = 250

BENCHMARK_ALL_SOURCES    = ALL_SOURCES
BENCHMARK_WAVEFORMS      = WAVEFORMS
BENCHMARK_GM_INSTRUMENTS = list(GM_PROGRAMS_V1.keys())
BENCHMARK_CONTINUOUS_GM_INSTRUMENTS = [
    "flute",
    "trumpet",
    "trombone",
    "clarinet",
    "oboe",
    "violin",
    "cello",
    "organ",
    "synth_lead",
    "synth_pad",
    "voice",
]

BENCHMARK_F1_MIXED_GROUPS: dict[str, list[str]] = {
    "winds_and_strings": ["flute", "violin", "cello", "clarinet"],
    "brass_reeds_synth": ["trumpet", "trombone", "oboe", "synth_pad"],
}

BENCHMARK_BASE_FREQS_A   = {"A3": 220.00, "A4": 440.00, "A5": 880.00}

# When True, every cat-A condition stim is paired 1:1 with a matched
# baseline (no-vibrato / no-detune / no-reference) twin and the baseline
# numbers are reported alongside the headline accuracy. Set to False to
# skip baselines entirely (faster runs, no `accuracy_baseline` block).
INCLUDE_BASELINES: bool = True

# Large literal dicts/lists kept here so per-experiment blocks below stay scannable.

BENCHMARK_C1_CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj":  (0, 4, 7),
    "min":  (0, 3, 7),
    "dim":  (0, 3, 6),
    "aug":  (0, 4, 8),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
}

BENCHMARK_C4_CHORD_TYPES: dict[str, list[int]] = {
    # Dyads — all 13 intervals (unison through octave)
    "dyad_m2":  [0, 1],
    "dyad_M2":  [0, 2],
    "dyad_m3":  [0, 3],
    "dyad_M3":  [0, 4],
    "dyad_p4":  [0, 5],
    "dyad_tt":  [0, 6],
    "dyad_p5":  [0, 7],
    "dyad_m6":  [0, 8],
    "dyad_M6":  [0, 9],
    "dyad_m7":  [0, 10],
    "dyad_M7":  [0, 11],
    "dyad_oct": [0, 12],
    "dyad_uni": [0, 0],
    # Triads
    "triad_maj": [0, 4, 7],
    "triad_min": [0, 3, 7],
    "triad_dim": [0, 3, 6],
    "triad_aug": [0, 4, 8],
    # Seventh chords
    "seventh_dom": [0, 4, 7, 10],
    "seventh_maj": [0, 4, 7, 11],
    "seventh_min": [0, 3, 7, 10],
}

BENCHMARK_C3_QUALITIES: dict[str, tuple[tuple[int, ...], str]] = {
    "major":      ((0, 4, 7),     "major"),
    "minor":      ((0, 3, 7),     "minor"),
    "diminished": ((0, 3, 6),     "diminished"),
    "augmented":  ((0, 4, 8),     "augmented"),
    "dom7":       ((0, 4, 7, 10), "dominant seventh"),
    "maj7":       ((0, 4, 7, 11), "major seventh"),
    "min7":       ((0, 3, 7, 10), "minor seventh"),
    "m7b5":       ((0, 3, 6, 10), "half diminished"),
    "sus2":       ((0, 2, 7),     "sus2"),
    "sus4":       ((0, 5, 7),     "sus4"),
}


BENCHMARK_D4_TRAJECTORIES: list[dict] = [
    # gt_seq tokens are strictly alternating up/down — never two of the same in a row.
    # The "flat"/"same" trajectory was removed: this experiment counts direction
    # changes only (up vs down), matching d4's vocabulary.
    {"name": "up",           "shape": "linear", "interval_sign": +1, "gt_seq": ["up"]},
    {"name": "down",         "shape": "linear", "interval_sign": -1, "gt_seq": ["down"]},
    {"name": "up_then_down", "shape": "arch",   "interval_sign": +1, "gt_seq": ["up", "down"]},
    {"name": "down_then_up", "shape": "valley", "interval_sign": -1, "gt_seq": ["down", "up"]},
]

BENCHMARK_E1_EFFECTS: dict[str, dict] = {
    # Effects that genuinely threaten pitch perception. All RMS-matched to the
    # dry signal in engine._apply_effect, so loudness does not leak in.
    "clean":             {},
    # f0-relative steep Butterworth filters
    "highpass_above_f0": {"type": "highpass",    "cutoff_ratio": 1.5, "order": 6},   # removes fundamental → "missing fundamental" probe
    "lowpass_at_f0":     {"type": "lowpass",     "cutoff_ratio": 1.2, "order": 6},   # strips harmonics, leaves fundamental
    # Digital degradation
    "bitcrush_4bit":     {"type": "bitcrush",    "bit_depth": 4},
    # Plugin-quality non-linear / spatial / modulated
    "distortion_heavy":  {"type": "saturation",  "drive_db": 30.0},                  # fundamental drops, harmonics dominate
    "reverb_long":       {"type": "reverb_room", "room_size": 0.9, "damping": 0.4,
                          "wet_level": 0.6, "dry_level": 0.4},                       # algorithmic tail, smears attack
    "chorus_heavy":      {"type": "chorus",      "rate_hz": 1.2, "depth": 0.9, "mix": 0.6},  # detuned copies near f0
} if INCLUDE_BASELINES else {
    "highpass_above_f0": {"type": "highpass",    "cutoff_ratio": 1.5, "order": 6},   # removes fundamental → "missing fundamental" probe
    "lowpass_at_f0":     {"type": "lowpass",     "cutoff_ratio": 1.2, "order": 6},   # strips harmonics, leaves fundamental
    # Digital degradation
    "bitcrush_4bit":     {"type": "bitcrush",    "bit_depth": 4},
    # Plugin-quality non-linear / spatial / modulated
    "distortion_heavy":  {"type": "saturation",  "drive_db": 30.0},                  # fundamental drops, harmonics dominate
    "reverb_long":       {"type": "reverb_room", "room_size": 0.9, "damping": 0.4,
                          "wet_level": 0.6, "dry_level": 0.4},                       # algorithmic tail, smears attack
    "chorus_heavy":      {"type": "chorus",      "rate_hz": 1.2, "depth": 0.9, "mix": 0.6},  # detuned copies near f0
}

BENCHMARK_E3_SATURATIONS: dict[str, dict] = {
    "clean":      {},
    "sat_light":  {"type": "saturation", "drive_db":  6.0},
    "sat_medium": {"type": "saturation", "drive_db": 15.0},
    "sat_heavy":  {"type": "saturation", "drive_db": 30.0},
} if INCLUDE_BASELINES else {
    "sat_light":  {"type": "saturation", "drive_db":  6.0},
    "sat_medium": {"type": "saturation", "drive_db": 15.0},
    "sat_heavy":  {"type": "saturation", "drive_db": 30.0},
}

# Restricted to MIDI 36–84 so ±12 semitone shifts from resampling stay audible
BENCHMARK_PITCHES_E5 = [36, 43, 48, 54, 58, 60, 64, 67, 69, 72, 77, 84]

BENCHMARK_E4_CONDITIONS: list[dict] = [
    {"name": "clean",         "mode": "clean",    "factor": 1.0},
    {"name": "resample_0.5x", "mode": "resample", "factor": 0.5},  # 2× speed → pitch +12
    {"name": "resample_2x",   "mode": "resample", "factor": 2.0},  # ½× speed → pitch −12
    {"name": "stretch_0.5x",  "mode": "stretch",  "factor": 0.5},  # 2× speed, pitch unchanged
    {"name": "stretch_2x",    "mode": "stretch",  "factor": 2.0},  # ½× speed, pitch unchanged
] if INCLUDE_BASELINES else [
    {"name": "resample_0.5x", "mode": "resample", "factor": 0.5},  # 2× speed → pitch +12
    {"name": "resample_2x",   "mode": "resample", "factor": 2.0},  # ½× speed → pitch −12
    {"name": "stretch_0.5x",  "mode": "stretch",  "factor": 0.5},  # 2× speed, pitch unchanged
    {"name": "stretch_2x",    "mode": "stretch",  "factor": 2.0},  # ½× speed, pitch unchanged
]

# ═══════════════════════════════════════════════════════════════════════════
#                  PER-EXPERIMENT DATA-GENERATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

if EVAL:
    pitchbench_general_NOTATION_FORMATS = BENCHMARK_NOTATION_FORMATS
    # ── a1: pitch identification ──────────────────────────────────────────
    pitchbench_a1_PITCHES          = BENCHMARK_PITCHES_FULL_RANGE
    pitchbench_a1_TONE_DURATION_MS = BENCHMARK_DURATION_MS
    pitchbench_a1_SOURCES          = BENCHMARK_ALL_SOURCES

    # ── a2: pitch with reference ──────────────────────────────────────────
    pitchbench_d7_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION_COMPACT # smaller range since intervals can push the notes out of the audible MIDI range
    pitchbench_d7_INTERVALS         = BENCHMARK_INTERVALS
    pitchbench_d7_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
    pitchbench_d7_GAP_MS            = 500
    pitchbench_d7_SOURCES           = BENCHMARK_ALL_SOURCES

    # ── a3: pitch by duration ─────────────────────────────────────────────
    pitchbench_a3_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_a3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_FULL_SWEEP
    pitchbench_a3_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── a4: pitch with vibrato ────────────────────────────────────────────
    pitchbench_e5_VIBRATO_RATES_HZ     = [3, 5, 7, 10]
    pitchbench_e5_VIBRATO_DEPTHS_CENTS = [25, 50, 100, 200]
    pitchbench_e5_DURATIONS_MS         = [BENCHMARK_DURATION_MS]
    pitchbench_e5_PITCHES              = BENCHMARK_PITCHES_SELECTION_REDUCED
    pitchbench_e5_SOURCES              = BENCHMARK_WAVEFORMS

    # ── a5: pitch slightly off ────────────────────────────────────────────
    pitchbench_e6_N_DETUNE_LEVELS = 5
    pitchbench_e6_DETUNE_FRACTION = 0.45
    pitchbench_e6_PITCHES         = BENCHMARK_PITCHES_SELECTION
    pitchbench_e6_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_e6_SOURCES         = BENCHMARK_ALL_SOURCES

    # ── b1: pitch in silence ──────────────────────────────────────────────
    pitchbench_b1_PITCHES           = BENCHMARK_PITCHES_SELECTION
    pitchbench_b1_TONE_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS
    pitchbench_b1_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
    pitchbench_b1_TOTAL_SILENCE_MS  = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b1_SOURCES           = BENCHMARK_ALL_SOURCES

    # ── b2: onset/offset single ──────────────────────────────────────────
    pitchbench_b3_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_b3_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS
    pitchbench_b3_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_b3_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── b3: onset/offset specific note ──────────────────────────────────
    pitchbench_b4_N_DISTRACTORS   = [5, 7]
    pitchbench_b4_TARGET_POS_OPTS = ["first", "middle", "last"]
    pitchbench_b4_TOTAL_DUR_MS    = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b4_GAP_MIN_MS      = 500
    pitchbench_b4_GAP_MAX_MS      = 2000
    pitchbench_b4_PITCHES         = BENCHMARK_PITCHES_SELECTION
    pitchbench_b4_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_b4_SEED            = BENCHMARK_SEED
    pitchbench_b4_SOURCES         = BENCHMARK_ALL_SOURCES

    # ── b4: pitch at time ────────────────────────────────────────────────
    pitchbench_b2_N_NOTES_OPTS = [5, 10]
    pitchbench_b2_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b2_GAP_MIN_MS   = 30
    pitchbench_b2_GAP_MAX_MS   = 1500
    pitchbench_b2_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_b2_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_b2_SEED         = BENCHMARK_SEED
    pitchbench_b2_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── b5: onset/offset each note ───────────────────────────────────────
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

    # ── c1: dyad interval ────────────────────────────────────────────────
    pitchbench_c2_INTERVALS_ST         = list(range(1, 13))
    pitchbench_c2_SAME_INSTRUMENT_OPTS = [True, False]
    pitchbench_c2_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_c2_PITCHES              = BENCHMARK_PITCHES_SELECTION
    pitchbench_c2_SOURCES              = BENCHMARK_ALL_SOURCES

    # ── c2: chord pitch count ────────────────────────────────────────────
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

    # ── c3: chord pitch id ───────────────────────────────────────────────
    pitchbench_c4_CHORD_TYPES      = BENCHMARK_C4_CHORD_TYPES
    pitchbench_c4_BASE_ROOTS       = BENCHMARK_PITCHES_SELECTION
    pitchbench_c4_TONE_DURATION_MS = BENCHMARK_DURATION_MS
    pitchbench_c4_SOURCES          = BENCHMARK_ALL_SOURCES

    # ── c4: chord quality ────────────────────────────────────────────────
    pitchbench_c3_QUALITIES            = BENCHMARK_C3_QUALITIES
    pitchbench_c3_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION
    pitchbench_c3_TASKS                = ["quality_only"]
    pitchbench_c3_SAME_INSTRUMENT_OPTS = [True, False]
    pitchbench_c3_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_c3_SOURCES              = BENCHMARK_ALL_SOURCES

    # ── d1: seq pitch count ──────────────────────────────────────────────
    pitchbench_d1_N_COUNTS     = [1, 2, 3, 4, 5, 7, 10]
    pitchbench_d1_RHYTHMS      = ["regular", "irregular"]
    pitchbench_d1_PITCH_MIN    = 29
    pitchbench_d1_PITCH_MAX    = 89
    pitchbench_d1_GAP_MS       = BENCHMARK_GAP_MS
    pitchbench_d1_N_TRIALS     = BENCHMARK_N_TRIALS
    pitchbench_d1_SEED         = BENCHMARK_SEED
    pitchbench_d1_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_d1_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── d2: pitch difference ─────────────────────────────────────────────
    pitchbench_d2_BASE_FREQS    = BENCHMARK_BASE_FREQS_A
    pitchbench_d2_DELTA_CENTS   = [1, 2, 5, 10, 25, 50, 100, 200, 400, 700, 1200]
    pitchbench_d2_SEPARATION_MS = [200, 500, 1000, 2000]
    pitchbench_d2_DURATION_MS   = BENCHMARK_DURATION_MS
    pitchbench_d2_N_TRIALS      = BENCHMARK_N_TRIALS
    pitchbench_d2_SEED          = BENCHMARK_SEED
    pitchbench_d2_SOURCES       = BENCHMARK_ALL_SOURCES

    # ── d3: interval id seq ──────────────────────────────────────────────
    pitchbench_d6_INTERVALS_ST   = list(range(1, 13))
    pitchbench_d6_DIRECTIONS     = ["ascending", "descending"]
    pitchbench_d6_SEPARATIONS_MS = [200, 500, 1000, 2000]
    pitchbench_d6_PITCHES        = BENCHMARK_PITCHES_SELECTION
    pitchbench_d6_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_d6_SOURCES        = BENCHMARK_ALL_SOURCES

    # ── d4: contour discrete ─────────────────────────────────────────────
    pitchbench_d3_N_TRANSITIONS_OPTS = [2, 3, 5, 7]
    pitchbench_d3_STEP_SIZES_ST      = [1, 2, 4, 7, 11]
    pitchbench_d3_NOTE_DURATIONS_MS  = BENCHMARK_NOTE_DURATIONS_MS_D4
    pitchbench_d3_TRIALS_PER_CELL    = 2
    pitchbench_d3_PITCHES            = BENCHMARK_PITCHES_SELECTION
    pitchbench_d3_SEED               = BENCHMARK_SEED
    pitchbench_d3_SOURCES            = BENCHMARK_ALL_SOURCES

    # ── d5: contour continuous ───────────────────────────────────────────
    pitchbench_d4_START_PITCHES = BENCHMARK_PITCHES_SELECTION
    pitchbench_d4_INTERVALS_ST  = [1, 4, 7, 12]
    pitchbench_d4_DURATION_MS   = BENCHMARK_DURATION_MS
    pitchbench_d4_SOURCES       = BENCHMARK_WAVEFORMS
    pitchbench_d4_TRAJECTORIES  = BENCHMARK_D4_TRAJECTORIES

    # ── d6: pitch ranking ────────────────────────────────────────────────
    pitchbench_d5_BASE_FREQS  = BENCHMARK_BASE_FREQS_A
    pitchbench_d5_DELTA_CENTS = [25, 50, 100, 200, 400]
    pitchbench_d5_N_TONES     = [3, 4, 5, 7]
    pitchbench_d5_RHYTHMS     = ["regular", "irregular"]
    pitchbench_d5_DURATION_MS = BENCHMARK_DURATION_MS
    pitchbench_d5_GAP_MS      = BENCHMARK_GAP_MS
    pitchbench_d5_N_TRIALS    = BENCHMARK_N_TRIALS
    pitchbench_d5_SEED        = BENCHMARK_SEED
    pitchbench_d5_SOURCES     = BENCHMARK_WAVEFORMS

    # ── d7: seq pitch id ─────────────────────────────────────────────────
    pitchbench_d8_PITCH_MIN    = 29
    pitchbench_d8_PITCH_MAX    = 89
    pitchbench_d8_N_NOTES_LIST = [3, 5, 10]
    pitchbench_d8_TONE_MS      = BENCHMARK_DURATION_MS
    pitchbench_d8_GAP_MS       = BENCHMARK_GAP_MS_D7
    pitchbench_d8_N_TRIALS     = 5
    pitchbench_d8_SEED         = BENCHMARK_SEED
    pitchbench_d8_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── e1: loudness ─────────────────────────────────────────────────────
    pitchbench_a2_LOUDNESS_DB = [-30, -20, -12, -6, -3]
    pitchbench_a2_TONE_MS     = BENCHMARK_DURATION_MS
    pitchbench_a2_PITCHES     = BENCHMARK_PITCHES_SELECTION
    pitchbench_a2_SOURCES     = BENCHMARK_ALL_SOURCES

    # ── e1: audio effects ────────────────────────────────────────────────
    pitchbench_e1_PITCHES = BENCHMARK_PITCHES_SELECTION
    pitchbench_e1_TONE_MS = BENCHMARK_DURATION_MS
    pitchbench_e1_EFFECTS = BENCHMARK_E1_EFFECTS
    pitchbench_e1_SOURCES = BENCHMARK_ALL_SOURCES

    # ── e2: background effects ───────────────────────────────────────────
    pitchbench_e2_BACKGROUNDS  = ["white_noise", "church-bells",
                                  "crowd-noise", "rain", "street-noise"]
    pitchbench_e2_SNR_DB       = [30.0, 20.0, 0.0, -6.0]
    pitchbench_e2_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_e2_DURATIONS_MS = [BENCHMARK_DURATION_MS]
    pitchbench_e2_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── e3: harmonic saturation ──────────────────────────────────────────
    pitchbench_e3_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_e3_TONE_MS      = BENCHMARK_DURATION_MS
    pitchbench_e3_SATURATIONS  = BENCHMARK_E3_SATURATIONS
    pitchbench_e3_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── e4: time stretch vs resample ─────────────────────────────────────
    pitchbench_e4_PITCHES      = BENCHMARK_PITCHES_E5
    pitchbench_e4_TONE_MS      = 3000
    pitchbench_e4_CONDITIONS   = BENCHMARK_E4_CONDITIONS
    pitchbench_e4_SOURCES      = BENCHMARK_ALL_SOURCES

    # # ── g1: embedding geometry (analysis track, not in CLI) ──────────────
    # pitchbench_g1_PITCHES = BENCHMARK_PITCHES_FULL_RANGE
    # pitchbench_g1_SOURCES = BENCHMARK_ALL_SOURCES

    # # ── g2: token logits (analysis track, not in CLI) ────────────────────
    # pitchbench_g2_PITCHES          = [29, 43, 57, 71, 72]
    # pitchbench_g2_SOURCES          = BENCHMARK_ALL_SOURCES
    # pitchbench_g2_TONE_DURATION_MS = 2000
    # pitchbench_g2_TOP_K            = 200
    # pitchbench_g2_MAX_NEW_TOKENS   = 32

    # # ── g3: knn oracle (analysis track, not in CLI) ──────────────────────
    # pitchbench_g3_PITCHES    = BENCHMARK_PITCHES_FULL_RANGE
    # pitchbench_g3_SOURCES    = BENCHMARK_ALL_SOURCES
    # pitchbench_g3_K          = 1
    # pitchbench_g3_SPLIT_SEED = BENCHMARK_SEED

    # ── f1: melodic line id ──────────────────────────────────────────────
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

    # ── f2: chorale voice id ─────────────────────────────────────────────
    pitchbench_f2_N_VOICES         = 4
    pitchbench_f2_MIN_SEG_NOTES    = 4
    pitchbench_f2_MAX_SEG_SEC      = 30.0
    pitchbench_f2_QPM              = 60.0
    pitchbench_f2_VOICE_NAMES      = ["soprano", "alto", "tenor", "bass"]
    pitchbench_f2_DEFAULT_CHORALES = ["bach/bwv66.6", "bach/bwv4.8",
                                     "bach/bwv7.7", "bach/bwv26.6",
                                     "bach/bwv57.8"]
    pitchbench_f2_MIXED_GROUPS     = {
        "classical_quartet": ["flute", "violin", "cello", "bass"],
        "mixed_timbres":     ["piano", "trumpet", "clarinet", "guitar"],
    }
    pitchbench_f2_SEED             = BENCHMARK_SEED
    pitchbench_f2_SOURCES          = BENCHMARK_GM_INSTRUMENTS

    # ── z1: NSynth pitch id ──────────────────────────────────────────────
    pitchbench_z1_N_PER_FAMILY = 10
    pitchbench_z1_SEED         = BENCHMARK_SEED

else:
    # ═══════════════════════════════════════════════════════════════════════
    # User-overridable values for non-benchmark runs (EVAL=False).
    # Initialised to a copy of the benchmark values so the codebase still runs.
    # Edit any line marked "← edit me" to customise that experiment.
    # ═══════════════════════════════════════════════════════════════════════

    # ── a1: pitch identification ──────────────────────────────────────────
    pitchbench_a1_PITCHES          = BENCHMARK_PITCHES_FULL_RANGE   # ← edit me
    pitchbench_a1_TONE_DURATION_MS = BENCHMARK_DURATION_MS           # ← edit me
    pitchbench_a1_SOURCES          = BENCHMARK_ALL_SOURCES           # ← edit me

    # ── a2: pitch with reference ──────────────────────────────────────────
    pitchbench_d7_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION    # ← edit me
    pitchbench_d7_INTERVALS         = [-12, -7, -5, -4, -3, -2, -1, 0,
                                        1,  2,  3,  4,  5,  7, 12]   # ← edit me
    pitchbench_d7_TONE_DURATION_MS  = BENCHMARK_DURATION_MS          # ← edit me
    pitchbench_d7_GAP_MS            = 500                            # ← edit me
    pitchbench_d7_SOURCES           = BENCHMARK_ALL_SOURCES          # ← edit me

    # ── a3: pitch by duration ─────────────────────────────────────────────
    pitchbench_a3_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_a3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_FULL_SWEEP   # ← edit me
    pitchbench_a3_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── a4: pitch with vibrato ────────────────────────────────────────────
    pitchbench_e5_VIBRATO_RATES_HZ     = [0, 3, 5, 7, 10]            # ← edit me
    pitchbench_e5_VIBRATO_DEPTHS_CENTS = [0, 25, 50, 100, 200]       # ← edit me
    pitchbench_e5_DURATIONS_MS         = [BENCHMARK_DURATION_MS]     # ← edit me
    pitchbench_e5_PITCHES              = BENCHMARK_PITCHES_SELECTION # ← edit me
    pitchbench_e5_SOURCES              = BENCHMARK_WAVEFORMS         # ← edit me

    # ── a5: pitch slightly off ────────────────────────────────────────────
    pitchbench_e6_N_DETUNE_LEVELS = 5                                # ← edit me
    pitchbench_e6_DETUNE_FRACTION = 0.40                             # ← edit me
    pitchbench_e6_PITCHES         = BENCHMARK_PITCHES_SELECTION      # ← edit me
    pitchbench_e6_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_e6_SOURCES         = BENCHMARK_ALL_SOURCES            # ← edit me

    # ── b1: pitch in silence ──────────────────────────────────────────────
    pitchbench_b1_PITCHES           = BENCHMARK_PITCHES_SELECTION    # ← edit me
    pitchbench_b1_TONE_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS    # ← edit me
    pitchbench_b1_TONE_DURATION_MS  = BENCHMARK_DURATION_MS          # ← edit me
    pitchbench_b1_TOTAL_SILENCE_MS  = BENCHMARK_TOTAL_DUR_MS         # ← edit me
    pitchbench_b1_SOURCES           = BENCHMARK_ALL_SOURCES          # ← edit me

    # ── b2: onset/offset single ──────────────────────────────────────────
    pitchbench_b3_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_b3_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS         # ← edit me
    pitchbench_b3_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS              # ← edit me
    pitchbench_b3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
    pitchbench_b3_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── b3: onset/offset specific note ──────────────────────────────────
    pitchbench_b4_N_DISTRACTORS   = [5, 7]                           # ← edit me
    pitchbench_b4_TARGET_POS_OPTS = ["first", "middle", "last"]      # ← edit me
    pitchbench_b4_TOTAL_DUR_MS    = BENCHMARK_TOTAL_DUR_MS           # ← edit me
    pitchbench_b4_GAP_MIN_MS      = 500                              # ← edit me
    pitchbench_b4_GAP_MAX_MS      = 2000                             # ← edit me
    pitchbench_b4_PITCHES         = BENCHMARK_PITCHES_SELECTION      # ← edit me
    pitchbench_b4_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_b4_SEED            = BENCHMARK_SEED                   # ← edit me
    pitchbench_b4_SOURCES         = BENCHMARK_ALL_SOURCES            # ← edit me

    # ── b4: pitch at time ────────────────────────────────────────────────
    pitchbench_b2_N_NOTES_OPTS = [5, 10]                             # ← edit me
    pitchbench_b2_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS              # ← edit me
    pitchbench_b2_GAP_MIN_MS   = 30                                  # ← edit me
    pitchbench_b2_GAP_MAX_MS   = 1500                                # ← edit me
    pitchbench_b2_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_b2_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
    pitchbench_b2_SEED         = BENCHMARK_SEED                      # ← edit me
    pitchbench_b2_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── b5: onset/offset each note ───────────────────────────────────────
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

    # ── c1: dyad interval ────────────────────────────────────────────────
    pitchbench_c2_INTERVALS_ST         = list(range(1, 13))          # ← edit me
    pitchbench_c2_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
    pitchbench_c2_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_c2_PITCHES              = BENCHMARK_PITCHES_SELECTION # ← edit me
    pitchbench_c2_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

    # ── c2: chord pitch count ────────────────────────────────────────────
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

    # ── c3: chord pitch id ───────────────────────────────────────────────
    pitchbench_c4_CHORD_TYPES      = BENCHMARK_C4_CHORD_TYPES        # ← edit me
    pitchbench_c4_BASE_ROOTS       = BENCHMARK_PITCHES_SELECTION     # ← edit me
    pitchbench_c4_TONE_DURATION_MS = BENCHMARK_DURATION_MS           # ← edit me
    pitchbench_c4_SOURCES          = BENCHMARK_ALL_SOURCES           # ← edit me

    # ── c4: chord quality ────────────────────────────────────────────────
    pitchbench_c3_QUALITIES            = BENCHMARK_C3_QUALITIES      # ← edit me
    pitchbench_c3_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION # ← edit me
    pitchbench_c3_TASKS                = ["quality_only"]            # ← edit me
    pitchbench_c3_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
    pitchbench_c3_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_c3_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

    # ── d1: seq pitch count ──────────────────────────────────────────────
    pitchbench_d1_N_COUNTS     = [1, 2, 3, 4, 5, 7, 10]              # ← edit me
    pitchbench_d1_RHYTHMS      = ["regular", "irregular"]            # ← edit me
    pitchbench_d1_PITCH_MIN    = 48                                  # ← edit me
    pitchbench_d1_PITCH_MAX    = 84                                  # ← edit me
    pitchbench_d1_GAP_MS       = BENCHMARK_GAP_MS                    # ← edit me
    pitchbench_d1_N_TRIALS     = BENCHMARK_N_TRIALS                  # ← edit me
    pitchbench_d1_SEED         = BENCHMARK_SEED                      # ← edit me
    pitchbench_d1_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
    pitchbench_d1_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── d2: pitch difference ─────────────────────────────────────────────
    pitchbench_d2_BASE_FREQS    = BENCHMARK_BASE_FREQS_A             # ← edit me
    pitchbench_d2_DELTA_CENTS   = [1, 2, 5, 10, 25, 50, 100, 200, 400, 700, 1200] # ← edit me
    pitchbench_d2_SEPARATION_MS = [200, 500, 1000, 2000]             # ← edit me
    pitchbench_d2_DURATION_MS   = BENCHMARK_DURATION_MS              # ← edit me
    pitchbench_d2_N_TRIALS      = BENCHMARK_N_TRIALS                 # ← edit me
    pitchbench_d2_SEED          = BENCHMARK_SEED                     # ← edit me
    pitchbench_d2_SOURCES       = BENCHMARK_ALL_SOURCES              # ← edit me

    # ── d3: interval id seq ──────────────────────────────────────────────
    pitchbench_d6_INTERVALS_ST   = list(range(1, 13))                # ← edit me
    pitchbench_d6_DIRECTIONS     = ["ascending", "descending"]       # ← edit me
    pitchbench_d6_SEPARATIONS_MS = [200, 500, 1000, 2000]            # ← edit me
    pitchbench_d6_PITCHES        = BENCHMARK_PITCHES_SELECTION       # ← edit me
    pitchbench_d6_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_d6_SOURCES        = BENCHMARK_ALL_SOURCES             # ← edit me

    # ── d4: contour discrete ─────────────────────────────────────────────
    pitchbench_d3_N_TRANSITIONS_OPTS = [2, 3, 5, 7]                  # ← edit me
    pitchbench_d3_STEP_SIZES_ST      = [1, 2, 4, 7]                  # ← edit me
    pitchbench_d3_NOTE_DURATIONS_MS  = BENCHMARK_NOTE_DURATIONS_MS_D4 # ← edit me
    pitchbench_d3_TRIALS_PER_CELL    = 2                             # ← edit me
    pitchbench_d3_PITCHES            = BENCHMARK_PITCHES_SELECTION   # ← edit me
    pitchbench_d3_SEED               = BENCHMARK_SEED                # ← edit me
    pitchbench_d3_SOURCES            = BENCHMARK_ALL_SOURCES         # ← edit me

    # ── d5: contour continuous ───────────────────────────────────────────
    pitchbench_d4_START_PITCHES = BENCHMARK_PITCHES_SELECTION        # ← edit me
    pitchbench_d4_INTERVALS_ST  = [1, 4, 7, 12]                      # ← edit me
    pitchbench_d4_DURATION_MS   = BENCHMARK_DURATION_MS              # ← edit me
    pitchbench_d4_SOURCES       = BENCHMARK_WAVEFORMS                # ← edit me
    pitchbench_d4_TRAJECTORIES  = BENCHMARK_D4_TRAJECTORIES          # ← edit me

    # ── d6: pitch ranking ────────────────────────────────────────────────
    pitchbench_d5_BASE_FREQS  = BENCHMARK_BASE_FREQS_A               # ← edit me
    pitchbench_d5_DELTA_CENTS = [25, 50, 100, 200, 400]              # ← edit me
    pitchbench_d5_N_TONES     = [3, 4, 5, 7]                         # ← edit me
    pitchbench_d5_RHYTHMS     = ["regular", "irregular"]             # ← edit me
    pitchbench_d5_DURATION_MS = BENCHMARK_DURATION_MS                # ← edit me
    pitchbench_d5_GAP_MS      = BENCHMARK_GAP_MS                     # ← edit me
    pitchbench_d5_N_TRIALS    = BENCHMARK_N_TRIALS                   # ← edit me
    pitchbench_d5_SEED        = BENCHMARK_SEED                       # ← edit me
    pitchbench_d5_SOURCES     = BENCHMARK_WAVEFORMS                  # ← edit me

    # ── d7: seq pitch id ─────────────────────────────────────────────────
    pitchbench_d8_PITCH_MIN    = 29                                  # ← edit me
    pitchbench_d8_PITCH_MAX    = 89                                  # ← edit me
    pitchbench_d8_N_NOTES_LIST = [3, 5, 10]                          # ← edit me
    pitchbench_d8_TONE_MS      = BENCHMARK_DURATION_MS               # ← edit me
    pitchbench_d8_GAP_MS       = BENCHMARK_GAP_MS_D7                 # ← edit me
    pitchbench_d8_N_TRIALS     = 5                                   # ← edit me
    pitchbench_d8_SEED         = BENCHMARK_SEED                      # ← edit me
    pitchbench_d8_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── e1: loudness ─────────────────────────────────────────────────────
    pitchbench_a2_LOUDNESS_DB = [-30, -20, -12, -6, -3, 0]           # ← edit me
    pitchbench_a2_TONE_MS     = BENCHMARK_DURATION_MS                # ← edit me
    pitchbench_a2_PITCHES     = BENCHMARK_PITCHES_SELECTION          # ← edit me
    pitchbench_a2_SOURCES     = BENCHMARK_ALL_SOURCES                # ← edit me

    # ── e2: audio effects ────────────────────────────────────────────────
    pitchbench_e1_PITCHES = BENCHMARK_PITCHES_SELECTION              # ← edit me
    pitchbench_e1_TONE_MS = BENCHMARK_DURATION_MS                    # ← edit me
    pitchbench_e1_EFFECTS = BENCHMARK_E1_EFFECTS                     # ← edit me
    pitchbench_e1_SOURCES = BENCHMARK_ALL_SOURCES                    # ← edit me

    # ── e3: background effects ───────────────────────────────────────────
    pitchbench_e2_BACKGROUNDS  = ["white_noise", "church-bells",
                                  "crowd-noise", "rain", "street-noise"] # ← edit me
    pitchbench_e2_SNR_DB       = [30.0, 20.0, 0.0, -6.0]             # ← edit me
    pitchbench_e2_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_e2_DURATIONS_MS = [BENCHMARK_DURATION_MS]             # ← edit me
    pitchbench_e2_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── e4: harmonic saturation ──────────────────────────────────────────
    pitchbench_e3_PITCHES     = BENCHMARK_PITCHES_SELECTION          # ← edit me
    pitchbench_e3_TONE_MS     = BENCHMARK_DURATION_MS                # ← edit me
    pitchbench_e3_SATURATIONS = BENCHMARK_E3_SATURATIONS             # ← edit me
    pitchbench_e3_SOURCES     = BENCHMARK_ALL_SOURCES                # ← edit me

    # ── e5: time stretch vs resample ─────────────────────────────────────
    pitchbench_e4_PITCHES     = BENCHMARK_PITCHES_E5                 # ← edit me
    pitchbench_e4_TONE_MS     = 3000                                 # ← edit me
    pitchbench_e4_CONDITIONS  = BENCHMARK_E4_CONDITIONS              # ← edit me
    pitchbench_e4_SOURCES     = BENCHMARK_ALL_SOURCES                # ← edit me

    # ── g1: embedding geometry (analysis track, not in CLI) ──────────────
    pitchbench_g1_PITCHES = BENCHMARK_PITCHES_FULL_RANGE              # ← edit me
    pitchbench_g1_SOURCES = BENCHMARK_ALL_SOURCES                     # ← edit me

    # ── g2: token logits (analysis track, not in CLI) ────────────────────
    pitchbench_g2_PITCHES          = [29, 43, 57, 71, 72]             # ← edit me
    pitchbench_g2_SOURCES          = BENCHMARK_ALL_SOURCES            # ← edit me
    pitchbench_g2_TONE_DURATION_MS = 2000                             # ← edit me
    pitchbench_g2_TOP_K            = 200                              # ← edit me
    pitchbench_g2_MAX_NEW_TOKENS   = 32                               # ← edit me

    # ── g3: knn oracle (analysis track, not in CLI) ──────────────────────
    pitchbench_g3_PITCHES    = BENCHMARK_PITCHES_FULL_RANGE           # ← edit me
    pitchbench_g3_SOURCES    = BENCHMARK_ALL_SOURCES                  # ← edit me
    pitchbench_g3_K          = 1                                      # ← edit me
    pitchbench_g3_SPLIT_SEED = BENCHMARK_SEED                         # ← edit me

    # ── f1: melodic line id ──────────────────────────────────────────────
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

    # ── f2: chorale voice id ─────────────────────────────────────────────
    pitchbench_f2_N_VOICES         = 4                               # ← edit me
    pitchbench_f2_MIN_SEG_NOTES    = 4                               # ← edit me
    pitchbench_f2_MAX_SEG_SEC      = 30.0                            # ← edit me
    pitchbench_f2_QPM              = 60.0                            # ← edit me
    pitchbench_f2_VOICE_NAMES      = ["soprano", "alto", "tenor", "bass"]   # ← edit me
    pitchbench_f2_DEFAULT_CHORALES = ["bach/bwv66.6", "bach/bwv4.8",
                                     "bach/bwv7.7", "bach/bwv26.6",
                                     "bach/bwv57.8"]                # ← edit me
    pitchbench_f2_MIXED_GROUPS     = {
        "classical_quartet": ["flute", "violin", "cello", "bass"],
        "mixed_timbres":     ["piano", "trumpet", "clarinet", "guitar"],
    }                                                                # ← edit me
    pitchbench_f2_SEED             = BENCHMARK_SEED                  # ← edit me
    pitchbench_f2_SOURCES          = BENCHMARK_GM_INSTRUMENTS        # ← edit me

    # ── z1: NSynth pitch id ──────────────────────────────────────────────
    pitchbench_z1_N_PER_FAMILY = 10                                  # ← edit me
    pitchbench_z1_SEED         = BENCHMARK_SEED                      # ← edit me


# ── Legacy aliases (deprecated; kept so existing callers don't break) ─────────
# A handful of helpers and tests still refer to the old DEFAULT_* names; these
# point at the BENCHMARK_* values so behaviour is unchanged.
DEFAULT_PITCHES           = BENCHMARK_PITCHES_SELECTION
DEFAULT_DURATION_MS       = BENCHMARK_DURATION_MS
DEFAULT_DURATIONS_MS      = BENCHMARK_DURATIONS_MS_SHORT_LONG
DEFAULT_MIDI_MIN          = 29
DEFAULT_MIDI_MAX          = 89
DEFAULT_SELECTION         = BENCHMARK_PITCHES_SELECTION
DEFAULT_TOTAL_DUR_MS      = BENCHMARK_TOTAL_DUR_MS
DEFAULT_TONE_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS
DEFAULT_GAP_MS            = BENCHMARK_GAP_MS
DEFAULT_N_TRIALS          = BENCHMARK_N_TRIALS
DEFAULT_SEED              = BENCHMARK_SEED
