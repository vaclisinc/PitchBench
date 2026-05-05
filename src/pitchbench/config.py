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
RESULTS_DIR  = _PROJECT_ROOT / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
}

MODEL_URLS: dict[str, str] = {
    "music_flamingo":                os.environ.get("MF_URL",            "http://localhost:8000"),
    "audio_flamingo_next_instruct":  os.environ.get("AF_NEXT_INST_URL",  "http://localhost:8001"),
    "audio_flamingo_next_think":     os.environ.get("AF_NEXT_THINK_URL", "http://localhost:8002"),
    "audio_flamingo_next_captioner": os.environ.get("AF_NEXT_CAP_URL",   "http://localhost:8003"),
    "qwen3_omni":                    os.environ.get("QWEN3_OMNI_URL",    "http://lowland.cs.berkeley.edu:9999"),
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
    # Single-pitch ID (a)
    "pitchbench_a1_pitch_id":              {"per_stratum": None, "strata": ("midi",)},
    "pitchbench_a2_pitch_with_reference":  {"per_stratum": 10,   "strata": ("condition", "interval")},
    "pitchbench_a3_pitch_by_duration":     {"per_stratum":  5,   "strata": ("midi", "duration_ms")},
    "pitchbench_a4_pitch_with_vibrato":    {"per_stratum": 10,   "strata": ("midi", "is_control")},
    "pitchbench_a5_pitch_slightly_off":    {"per_stratum": 10,   "strata": ("midi", "detune_hz")},
    # Onsets / offsets (b)
    "pitchbench_b1_pitch_in_silence":      {"per_stratum": 10,   "strata": ("condition", "midi")},
    "pitchbench_b2_onset_offset_single":   {"per_stratum":  5,   "strata": ("midi", "pos_ms")},
    "pitchbench_b3_onset_offset_specific": {"per_stratum": 20,   "strata": ("target_pos", "n_distractors")},
    "pitchbench_b4_pitch_at_time":         {"per_stratum": 20,   "strata": ("n_notes", "target_idx")},
    "pitchbench_b5_onset_offset_each":     {"per_stratum": 25,   "strata": ("rhythm", "n_notes")},
    # Chords (c)
    "pitchbench_c1_dyad_interval":         {"per_stratum": 20,   "strata": ("interval_st",)},
    "pitchbench_c2_chord_pitch_count":     {"per_stratum": 15,   "strata": ("n", "chord_quality")},
    "pitchbench_c3_chord_pitch_id":        {"per_stratum": 20,   "strata": ("chord_type",)},
    "pitchbench_c4_chord_quality":         {"per_stratum": 25,   "strata": ("chord_quality",)},
    # Sequences (d)
    "pitchbench_d1_seq_pitch_count":       {"per_stratum": 20,   "strata": ("n", "rhythm")},
    "pitchbench_d2_pitch_difference":      {"per_stratum": 15,   "strata": ("delta_cents", "order")},
    "pitchbench_d3_interval_id_seq":       {"per_stratum": 12,   "strata": ("signed_st",)},
    "pitchbench_d4_contour_discrete":      {"per_stratum": 15,   "strata": ("n_transitions", "step_size_st")},
    "pitchbench_d5_contour_continuous":    {"per_stratum": None, "strata": ("traj_name",)},
    "pitchbench_d6_pitch_ranking":         {"per_stratum": 20,   "strata": ("rhythm", "n_notes")},
    "pitchbench_d7_seq_pitch_id":          {"per_stratum": 25,   "strata": ("n_notes",)},
    # Effects (e)
    "pitchbench_e1_loudness":              {"per_stratum":  5,   "strata": ("midi", "loudness_db")},
    "pitchbench_e2_audio_effects":         {"per_stratum": 10,   "strata": ("effect_type", "midi")},
    "pitchbench_e3_background_effects":    {"per_stratum": 10,   "strata": ("background", "snr_db")},
    # Polyphony (g)
    "pitchbench_g1_melodic_line_id":       {"per_stratum": 20,   "strata": ("n", "source_label")},
    "pitchbench_g2_chorale_voice_id":      {"per_stratum": None, "strata": ("chorale_slug",)},
}


# ═══════════════════════════════════════════════════════════════════════════
#                         BENCHMARK CONSTANTS
#  Canonical paper values. Used when EVAL=True. DO NOT MODIFY for paper runs.
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_PITCHES_FULL_RANGE = list(range(29, 90))   # F1–F6
BENCHMARK_PITCHES_SELECTION  = [30, 36, 43, 48, 54, 58, 60, 64, 67, 69,
                                73, 76, 79, 88]      # 14 hand-picked

BENCHMARK_DURATION_MS              = 5000
BENCHMARK_DURATIONS_MS_SHORT_LONG  = [1000, 5000]
BENCHMARK_DURATIONS_MS_FULL_SWEEP  = [50, 100, 250, 500, 1000, 2000,
                                      4000, 5000, 15000, 60000]
BENCHMARK_NOTE_DURATIONS_MS_D4     = [250, 500, 1000]

BENCHMARK_TOTAL_DUR_MS      = 60_000
BENCHMARK_TONE_POSITIONS_MS = [2_000, 7_000, 14_000, 22_000,
                               27_000, 41_000, 47_000, 53_000]
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

BENCHMARK_BASE_FREQS_A   = {"A3": 220.00, "A4": 440.00, "A5": 880.00}

# Large literal dicts/lists kept here so per-experiment blocks below stay scannable.

BENCHMARK_C2_CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj":  (0, 4, 7),
    "min":  (0, 3, 7),
    "dim":  (0, 3, 6),
    "aug":  (0, 4, 8),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
}

BENCHMARK_C3_CHORD_TYPES: dict[str, list[int]] = {
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

BENCHMARK_C4_QUALITIES: dict[str, tuple[tuple[int, ...], str]] = {
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

BENCHMARK_D5_TRAJECTORIES: list[dict] = [
    # gt_seq tokens are strictly alternating up/down — never two of the same in a row.
    # The "flat"/"same" trajectory was removed: this experiment counts direction
    # changes only (up vs down), matching d4's vocabulary.
    {"name": "up",           "shape": "linear", "interval_sign": +1, "gt_seq": ["up"]},
    {"name": "down",         "shape": "linear", "interval_sign": -1, "gt_seq": ["down"]},
    {"name": "up_then_down", "shape": "arch",   "interval_sign": +1, "gt_seq": ["up", "down"]},
    {"name": "down_then_up", "shape": "valley", "interval_sign": -1, "gt_seq": ["down", "up"]},
]

BENCHMARK_E2_EFFECTS: dict[str, dict] = {
    "clean":        {},
    "reverb_s":     {"type": "reverb", "delay_s": 0.05, "decay": 0.30},
    "reverb_l":     {"type": "reverb", "delay_s": 0.20, "decay": 0.70},
    "clip_50":      {"type": "clip",   "threshold": 0.50},
    "clip_25":      {"type": "clip",   "threshold": 0.25},
    "eq_lo_boost":  {"type": "eq_lo",  "cutoff_hz":  500, "gain_db":  12},
    "eq_hi_boost":  {"type": "eq_hi",  "cutoff_hz": 2000, "gain_db":  12},
    "eq_lo_cut":    {"type": "eq_lo",  "cutoff_hz":  500, "gain_db": -12},
    "eq_hi_cut":    {"type": "eq_hi",  "cutoff_hz": 2000, "gain_db": -12},
    "eq_telephone": {"type": "eq_hi",  "cutoff_hz": 1000, "gain_db": -24},
}


# ═══════════════════════════════════════════════════════════════════════════
#                  PER-EXPERIMENT DATA-GENERATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

if EVAL:
    # ── a1: pitch identification ──────────────────────────────────────────
    pitchbench_a1_PITCHES          = BENCHMARK_PITCHES_FULL_RANGE
    pitchbench_a1_TONE_DURATION_MS = BENCHMARK_DURATION_MS
    pitchbench_a1_SOURCES          = BENCHMARK_ALL_SOURCES

    # ── a2: pitch with reference ──────────────────────────────────────────
    pitchbench_a2_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION
    pitchbench_a2_INTERVALS         = [-12, -7, -5, -4, -3, -2, -1, 0,
                                        1,  2,  3,  4,  5,  7, 12]
    pitchbench_a2_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
    pitchbench_a2_GAP_MS            = 500
    pitchbench_a2_CONDITIONS        = ["anchored", "baseline"]
    pitchbench_a2_SOURCES           = BENCHMARK_ALL_SOURCES

    # ── a3: pitch by duration ─────────────────────────────────────────────
    pitchbench_a3_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_a3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_FULL_SWEEP
    pitchbench_a3_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── a4: pitch with vibrato ────────────────────────────────────────────
    pitchbench_a4_VIBRATO_RATES_HZ     = [0, 3, 5, 7, 10]
    pitchbench_a4_VIBRATO_DEPTHS_CENTS = [0, 25, 50, 100, 200]
    pitchbench_a4_DURATIONS_MS         = [BENCHMARK_DURATION_MS]
    pitchbench_a4_PITCHES              = BENCHMARK_PITCHES_SELECTION
    pitchbench_a4_SOURCES              = BENCHMARK_WAVEFORMS

    # ── a5: pitch slightly off ────────────────────────────────────────────
    pitchbench_a5_N_DETUNE_LEVELS = 5
    pitchbench_a5_DETUNE_FRACTION = 0.45
    pitchbench_a5_PITCHES         = BENCHMARK_PITCHES_SELECTION
    pitchbench_a5_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_a5_SOURCES         = BENCHMARK_ALL_SOURCES

    # ── b1: pitch in silence ──────────────────────────────────────────────
    pitchbench_b1_PITCHES           = BENCHMARK_PITCHES_SELECTION
    pitchbench_b1_TONE_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS
    pitchbench_b1_TONE_DURATION_MS  = BENCHMARK_DURATION_MS
    pitchbench_b1_TOTAL_SILENCE_MS  = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b1_CONDITIONS        = ["hidden", "baseline"]
    pitchbench_b1_SOURCES           = BENCHMARK_ALL_SOURCES

    # ── b2: onset/offset single ──────────────────────────────────────────
    pitchbench_b2_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_b2_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS
    pitchbench_b2_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b2_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_b2_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── b3: onset/offset specific note ──────────────────────────────────
    pitchbench_b3_N_DISTRACTORS   = [5, 7]
    pitchbench_b3_TARGET_POS_OPTS = ["first", "middle", "last"]
    pitchbench_b3_TOTAL_DUR_MS    = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b3_GAP_MIN_MS      = 500
    pitchbench_b3_GAP_MAX_MS      = 2000
    pitchbench_b3_PITCHES         = BENCHMARK_PITCHES_SELECTION
    pitchbench_b3_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_b3_SEED            = BENCHMARK_SEED
    pitchbench_b3_SOURCES         = BENCHMARK_ALL_SOURCES

    # ── b4: pitch at time ────────────────────────────────────────────────
    pitchbench_b4_N_NOTES_OPTS = [5, 10]
    pitchbench_b4_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS
    pitchbench_b4_GAP_MIN_MS   = 30
    pitchbench_b4_GAP_MAX_MS   = 1500
    pitchbench_b4_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_b4_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_b4_SEED         = BENCHMARK_SEED
    pitchbench_b4_SOURCES      = BENCHMARK_ALL_SOURCES

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
    pitchbench_c1_INTERVALS_ST         = list(range(1, 13))
    pitchbench_c1_SAME_INSTRUMENT_OPTS = [True, False]
    pitchbench_c1_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_c1_PITCHES              = BENCHMARK_PITCHES_SELECTION
    pitchbench_c1_SOURCES              = BENCHMARK_ALL_SOURCES

    # ── c2: chord pitch count ────────────────────────────────────────────
    pitchbench_c2_CHORD_INTERVALS      = BENCHMARK_C2_CHORD_INTERVALS
    pitchbench_c2_QUALITIES            = list(BENCHMARK_C2_CHORD_INTERVALS.keys()) + ["random_set"]
    pitchbench_c2_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION
    pitchbench_c2_SAME_INSTRUMENT_OPTS = [True, False]
    pitchbench_c2_RANDOM_NS            = [1, 2, 3, 4, 5, 6]
    pitchbench_c2_RANDOM_PITCH_RANGE   = (48, 84)
    pitchbench_c2_N_TRIALS             = 1
    pitchbench_c2_RANDOM_TRIALS        = 3
    pitchbench_c2_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_c2_SEED                 = BENCHMARK_SEED
    pitchbench_c2_SOURCES              = BENCHMARK_ALL_SOURCES

    # ── c3: chord pitch id ───────────────────────────────────────────────
    pitchbench_c3_CHORD_TYPES      = BENCHMARK_C3_CHORD_TYPES
    pitchbench_c3_BASE_ROOTS       = BENCHMARK_PITCHES_SELECTION
    pitchbench_c3_TONE_DURATION_MS = BENCHMARK_DURATION_MS
    pitchbench_c3_SOURCES          = BENCHMARK_ALL_SOURCES

    # ── c4: chord quality ────────────────────────────────────────────────
    pitchbench_c4_QUALITIES            = BENCHMARK_C4_QUALITIES
    pitchbench_c4_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION
    pitchbench_c4_TASKS                = ["quality_only"]
    pitchbench_c4_SAME_INSTRUMENT_OPTS = [True, False]
    pitchbench_c4_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_c4_SOURCES              = BENCHMARK_ALL_SOURCES

    # ── d1: seq pitch count ──────────────────────────────────────────────
    pitchbench_d1_N_COUNTS     = [1, 2, 3, 4, 5, 7, 10]
    pitchbench_d1_RHYTHMS      = ["regular", "irregular"]
    pitchbench_d1_PITCH_MIN    = 48
    pitchbench_d1_PITCH_MAX    = 84
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
    pitchbench_d3_INTERVALS_ST   = list(range(1, 13))
    pitchbench_d3_DIRECTIONS     = ["ascending", "descending"]
    pitchbench_d3_SEPARATIONS_MS = [200, 500, 1000, 2000]
    pitchbench_d3_PITCHES        = BENCHMARK_PITCHES_SELECTION
    pitchbench_d3_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG
    pitchbench_d3_SOURCES        = BENCHMARK_ALL_SOURCES

    # ── d4: contour discrete ─────────────────────────────────────────────
    pitchbench_d4_N_TRANSITIONS_OPTS = [2, 3, 5, 7]
    pitchbench_d4_STEP_SIZES_ST      = [1, 2, 4, 7]
    pitchbench_d4_NOTE_DURATIONS_MS  = BENCHMARK_NOTE_DURATIONS_MS_D4
    pitchbench_d4_TRIALS_PER_CELL    = 2
    pitchbench_d4_PITCHES            = BENCHMARK_PITCHES_SELECTION
    pitchbench_d4_SEED               = BENCHMARK_SEED
    pitchbench_d4_SOURCES            = BENCHMARK_ALL_SOURCES

    # ── d5: contour continuous ───────────────────────────────────────────
    pitchbench_d5_START_PITCHES = BENCHMARK_PITCHES_SELECTION
    pitchbench_d5_INTERVALS_ST  = [1, 4, 7, 12]
    pitchbench_d5_DURATION_MS   = BENCHMARK_DURATION_MS
    pitchbench_d5_SOURCES       = BENCHMARK_WAVEFORMS
    pitchbench_d5_TRAJECTORIES  = BENCHMARK_D5_TRAJECTORIES

    # ── d6: pitch ranking ────────────────────────────────────────────────
    pitchbench_d6_BASE_FREQS  = BENCHMARK_BASE_FREQS_A
    pitchbench_d6_DELTA_CENTS = [25, 50, 100, 200, 400]
    pitchbench_d6_N_TONES     = [3, 4, 5, 7]
    pitchbench_d6_RHYTHMS     = ["regular", "irregular"]
    pitchbench_d6_DURATION_MS = BENCHMARK_DURATION_MS
    pitchbench_d6_GAP_MS      = BENCHMARK_GAP_MS
    pitchbench_d6_N_TRIALS    = BENCHMARK_N_TRIALS
    pitchbench_d6_SEED        = BENCHMARK_SEED
    pitchbench_d6_SOURCES     = BENCHMARK_WAVEFORMS

    # ── d7: seq pitch id ─────────────────────────────────────────────────
    pitchbench_d7_PITCH_MIN    = 29
    pitchbench_d7_PITCH_MAX    = 89
    pitchbench_d7_N_NOTES_LIST = [3, 5, 10]
    pitchbench_d7_TONE_MS      = BENCHMARK_DURATION_MS
    pitchbench_d7_GAP_MS       = BENCHMARK_GAP_MS_D7
    pitchbench_d7_N_TRIALS     = 5
    pitchbench_d7_SEED         = BENCHMARK_SEED
    pitchbench_d7_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── e1: loudness ─────────────────────────────────────────────────────
    pitchbench_e1_LOUDNESS_DB = [-30, -20, -12, -6, -3, 0]
    pitchbench_e1_TONE_MS     = BENCHMARK_DURATION_MS
    pitchbench_e1_PITCHES     = BENCHMARK_PITCHES_SELECTION
    pitchbench_e1_SOURCES     = BENCHMARK_ALL_SOURCES

    # ── e2: audio effects ────────────────────────────────────────────────
    pitchbench_e2_PITCHES = BENCHMARK_PITCHES_SELECTION
    pitchbench_e2_TONE_MS = BENCHMARK_DURATION_MS
    pitchbench_e2_EFFECTS = BENCHMARK_E2_EFFECTS
    pitchbench_e2_SOURCES = BENCHMARK_ALL_SOURCES

    # ── e3: background effects ───────────────────────────────────────────
    pitchbench_e3_BACKGROUNDS  = ["white_noise", "church-bells",
                                  "crowd-noise", "rain", "street-noise"]
    pitchbench_e3_SNR_DB       = [30.0, 20.0, 0.0, -6.0]
    pitchbench_e3_PITCHES      = BENCHMARK_PITCHES_SELECTION
    pitchbench_e3_DURATIONS_MS = [BENCHMARK_DURATION_MS]
    pitchbench_e3_SOURCES      = BENCHMARK_ALL_SOURCES

    # ── f1: embedding geometry ───────────────────────────────────────────
    pitchbench_f1_PITCHES = BENCHMARK_PITCHES_FULL_RANGE
    pitchbench_f1_SOURCES = BENCHMARK_ALL_SOURCES

    # ── f2: token logits ─────────────────────────────────────────────────
    pitchbench_f2_PITCHES          = [29, 43, 57, 71, 72]
    pitchbench_f2_SOURCES          = BENCHMARK_ALL_SOURCES
    pitchbench_f2_TONE_DURATION_MS = 2000
    pitchbench_f2_TOP_K            = 200
    pitchbench_f2_MAX_NEW_TOKENS   = 32

    # ── f3: knn oracle ───────────────────────────────────────────────────
    pitchbench_f3_PITCHES    = BENCHMARK_PITCHES_FULL_RANGE
    pitchbench_f3_SOURCES    = BENCHMARK_ALL_SOURCES
    pitchbench_f3_K          = 1
    pitchbench_f3_SPLIT_SEED = BENCHMARK_SEED

    # ── g1: melodic line id ──────────────────────────────────────────────
    pitchbench_g1_N_NOTES      = 10
    pitchbench_g1_N_PARTS_LIST = [2, 3, 4]
    pitchbench_g1_N_TRIALS     = 2
    pitchbench_g1_DIST_N_MIN   = 1
    pitchbench_g1_DIST_N_MAX   = 20
    pitchbench_g1_DUR_JITTER   = 0.5
    pitchbench_g1_TEMPOS       = {"slow": 1000, "medium": 500}
    pitchbench_g1_PART_RANGES  = [(72, 83), (60, 71), (48, 59), (36, 47)]
    pitchbench_g1_SEED         = BENCHMARK_SEED
    pitchbench_g1_SOURCES      = BENCHMARK_GM_INSTRUMENTS

    # ── g2: chorale voice id ─────────────────────────────────────────────
    pitchbench_g2_N_VOICES         = 4
    pitchbench_g2_MIN_SEG_NOTES    = 4
    pitchbench_g2_MAX_SEG_SEC      = 30.0
    pitchbench_g2_QPM              = 60.0
    pitchbench_g2_VOICE_NAMES      = ["soprano", "alto", "tenor", "bass"]
    pitchbench_g2_DEFAULT_CHORALES = ["bach/bwv66.6", "bach/bwv4.8",
                                     "bach/bwv7.7", "bach/bwv26.6",
                                     "bach/bwv57.8"]
    pitchbench_g2_MIXED_GROUPS     = {
        "classical_quartet": ["flute", "violin", "cello", "bass"],
        "mixed_timbres":     ["piano", "trumpet", "clarinet", "guitar"],
    }
    pitchbench_g2_SEED             = BENCHMARK_SEED
    pitchbench_g2_SOURCES          = BENCHMARK_GM_INSTRUMENTS

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
    pitchbench_a2_REFERENCE_PITCHES = BENCHMARK_PITCHES_SELECTION    # ← edit me
    pitchbench_a2_INTERVALS         = [-12, -7, -5, -4, -3, -2, -1, 0,
                                        1,  2,  3,  4,  5,  7, 12]   # ← edit me
    pitchbench_a2_TONE_DURATION_MS  = BENCHMARK_DURATION_MS          # ← edit me
    pitchbench_a2_GAP_MS            = 500                            # ← edit me
    pitchbench_a2_CONDITIONS        = ["anchored", "baseline"]       # ← edit me
    pitchbench_a2_SOURCES           = BENCHMARK_ALL_SOURCES          # ← edit me

    # ── a3: pitch by duration ─────────────────────────────────────────────
    pitchbench_a3_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_a3_DURATIONS_MS = BENCHMARK_DURATIONS_MS_FULL_SWEEP   # ← edit me
    pitchbench_a3_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── a4: pitch with vibrato ────────────────────────────────────────────
    pitchbench_a4_VIBRATO_RATES_HZ     = [0, 3, 5, 7, 10]            # ← edit me
    pitchbench_a4_VIBRATO_DEPTHS_CENTS = [0, 25, 50, 100, 200]       # ← edit me
    pitchbench_a4_DURATIONS_MS         = [BENCHMARK_DURATION_MS]     # ← edit me
    pitchbench_a4_PITCHES              = BENCHMARK_PITCHES_SELECTION # ← edit me
    pitchbench_a4_SOURCES              = BENCHMARK_WAVEFORMS         # ← edit me

    # ── a5: pitch slightly off ────────────────────────────────────────────
    pitchbench_a5_N_DETUNE_LEVELS = 5                                # ← edit me
    pitchbench_a5_DETUNE_FRACTION = 0.40                             # ← edit me
    pitchbench_a5_PITCHES         = BENCHMARK_PITCHES_SELECTION      # ← edit me
    pitchbench_a5_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_a5_SOURCES         = BENCHMARK_ALL_SOURCES            # ← edit me

    # ── b1: pitch in silence ──────────────────────────────────────────────
    pitchbench_b1_PITCHES           = BENCHMARK_PITCHES_SELECTION    # ← edit me
    pitchbench_b1_TONE_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS    # ← edit me
    pitchbench_b1_TONE_DURATION_MS  = BENCHMARK_DURATION_MS          # ← edit me
    pitchbench_b1_TOTAL_SILENCE_MS  = BENCHMARK_TOTAL_DUR_MS         # ← edit me
    pitchbench_b1_CONDITIONS        = ["hidden", "baseline"]         # ← edit me
    pitchbench_b1_SOURCES           = BENCHMARK_ALL_SOURCES          # ← edit me

    # ── b2: onset/offset single ──────────────────────────────────────────
    pitchbench_b2_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_b2_POSITIONS_MS = BENCHMARK_TONE_POSITIONS_MS         # ← edit me
    pitchbench_b2_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS              # ← edit me
    pitchbench_b2_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
    pitchbench_b2_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── b3: onset/offset specific note ──────────────────────────────────
    pitchbench_b3_N_DISTRACTORS   = [5, 7]                           # ← edit me
    pitchbench_b3_TARGET_POS_OPTS = ["first", "middle", "last"]      # ← edit me
    pitchbench_b3_TOTAL_DUR_MS    = BENCHMARK_TOTAL_DUR_MS           # ← edit me
    pitchbench_b3_GAP_MIN_MS      = 500                              # ← edit me
    pitchbench_b3_GAP_MAX_MS      = 2000                             # ← edit me
    pitchbench_b3_PITCHES         = BENCHMARK_PITCHES_SELECTION      # ← edit me
    pitchbench_b3_DURATIONS_MS    = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_b3_SEED            = BENCHMARK_SEED                   # ← edit me
    pitchbench_b3_SOURCES         = BENCHMARK_ALL_SOURCES            # ← edit me

    # ── b4: pitch at time ────────────────────────────────────────────────
    pitchbench_b4_N_NOTES_OPTS = [5, 10]                             # ← edit me
    pitchbench_b4_TOTAL_DUR_MS = BENCHMARK_TOTAL_DUR_MS              # ← edit me
    pitchbench_b4_GAP_MIN_MS   = 30                                  # ← edit me
    pitchbench_b4_GAP_MAX_MS   = 1500                                # ← edit me
    pitchbench_b4_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_b4_DURATIONS_MS = BENCHMARK_DURATIONS_MS_SHORT_LONG   # ← edit me
    pitchbench_b4_SEED         = BENCHMARK_SEED                      # ← edit me
    pitchbench_b4_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

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
    pitchbench_c1_INTERVALS_ST         = list(range(1, 13))          # ← edit me
    pitchbench_c1_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
    pitchbench_c1_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_c1_PITCHES              = BENCHMARK_PITCHES_SELECTION # ← edit me
    pitchbench_c1_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

    # ── c2: chord pitch count ────────────────────────────────────────────
    pitchbench_c2_CHORD_INTERVALS      = BENCHMARK_C2_CHORD_INTERVALS # ← edit me
    pitchbench_c2_QUALITIES            = list(BENCHMARK_C2_CHORD_INTERVALS.keys()) + ["random_set"] # ← edit me
    pitchbench_c2_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION # ← edit me
    pitchbench_c2_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
    pitchbench_c2_RANDOM_NS            = [1, 2, 3, 4, 5, 6]          # ← edit me
    pitchbench_c2_RANDOM_PITCH_RANGE   = (48, 84)                    # ← edit me
    pitchbench_c2_N_TRIALS             = 1                           # ← edit me
    pitchbench_c2_RANDOM_TRIALS        = 3                           # ← edit me
    pitchbench_c2_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_c2_SEED                 = BENCHMARK_SEED              # ← edit me
    pitchbench_c2_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

    # ── c3: chord pitch id ───────────────────────────────────────────────
    pitchbench_c3_CHORD_TYPES      = BENCHMARK_C3_CHORD_TYPES        # ← edit me
    pitchbench_c3_BASE_ROOTS       = BENCHMARK_PITCHES_SELECTION     # ← edit me
    pitchbench_c3_TONE_DURATION_MS = BENCHMARK_DURATION_MS           # ← edit me
    pitchbench_c3_SOURCES          = BENCHMARK_ALL_SOURCES           # ← edit me

    # ── c4: chord quality ────────────────────────────────────────────────
    pitchbench_c4_QUALITIES            = BENCHMARK_C4_QUALITIES      # ← edit me
    pitchbench_c4_ROOT_MIDIS           = BENCHMARK_PITCHES_SELECTION # ← edit me
    pitchbench_c4_TASKS                = ["quality_only"]            # ← edit me
    pitchbench_c4_SAME_INSTRUMENT_OPTS = [True, False]               # ← edit me
    pitchbench_c4_DURATIONS_MS         = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_c4_SOURCES              = BENCHMARK_ALL_SOURCES       # ← edit me

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
    pitchbench_d3_INTERVALS_ST   = list(range(1, 13))                # ← edit me
    pitchbench_d3_DIRECTIONS     = ["ascending", "descending"]       # ← edit me
    pitchbench_d3_SEPARATIONS_MS = [200, 500, 1000, 2000]            # ← edit me
    pitchbench_d3_PITCHES        = BENCHMARK_PITCHES_SELECTION       # ← edit me
    pitchbench_d3_DURATIONS_MS   = BENCHMARK_DURATIONS_MS_SHORT_LONG # ← edit me
    pitchbench_d3_SOURCES        = BENCHMARK_ALL_SOURCES             # ← edit me

    # ── d4: contour discrete ─────────────────────────────────────────────
    pitchbench_d4_N_TRANSITIONS_OPTS = [2, 3, 5, 7]                  # ← edit me
    pitchbench_d4_STEP_SIZES_ST      = [1, 2, 4, 7]                  # ← edit me
    pitchbench_d4_NOTE_DURATIONS_MS  = BENCHMARK_NOTE_DURATIONS_MS_D4 # ← edit me
    pitchbench_d4_TRIALS_PER_CELL    = 2                             # ← edit me
    pitchbench_d4_PITCHES            = BENCHMARK_PITCHES_SELECTION   # ← edit me
    pitchbench_d4_SEED               = BENCHMARK_SEED                # ← edit me
    pitchbench_d4_SOURCES            = BENCHMARK_ALL_SOURCES         # ← edit me

    # ── d5: contour continuous ───────────────────────────────────────────
    pitchbench_d5_START_PITCHES = BENCHMARK_PITCHES_SELECTION        # ← edit me
    pitchbench_d5_INTERVALS_ST  = [1, 4, 7, 12]                      # ← edit me
    pitchbench_d5_DURATION_MS   = BENCHMARK_DURATION_MS              # ← edit me
    pitchbench_d5_SOURCES       = BENCHMARK_WAVEFORMS                # ← edit me
    pitchbench_d5_TRAJECTORIES  = BENCHMARK_D5_TRAJECTORIES          # ← edit me

    # ── d6: pitch ranking ────────────────────────────────────────────────
    pitchbench_d6_BASE_FREQS  = BENCHMARK_BASE_FREQS_A               # ← edit me
    pitchbench_d6_DELTA_CENTS = [25, 50, 100, 200, 400]              # ← edit me
    pitchbench_d6_N_TONES     = [3, 4, 5, 7]                         # ← edit me
    pitchbench_d6_RHYTHMS     = ["regular", "irregular"]             # ← edit me
    pitchbench_d6_DURATION_MS = BENCHMARK_DURATION_MS                # ← edit me
    pitchbench_d6_GAP_MS      = BENCHMARK_GAP_MS                     # ← edit me
    pitchbench_d6_N_TRIALS    = BENCHMARK_N_TRIALS                   # ← edit me
    pitchbench_d6_SEED        = BENCHMARK_SEED                       # ← edit me
    pitchbench_d6_SOURCES     = BENCHMARK_WAVEFORMS                  # ← edit me

    # ── d7: seq pitch id ─────────────────────────────────────────────────
    pitchbench_d7_PITCH_MIN    = 29                                  # ← edit me
    pitchbench_d7_PITCH_MAX    = 89                                  # ← edit me
    pitchbench_d7_N_NOTES_LIST = [3, 5, 10]                          # ← edit me
    pitchbench_d7_TONE_MS      = BENCHMARK_DURATION_MS               # ← edit me
    pitchbench_d7_GAP_MS       = BENCHMARK_GAP_MS_D7                 # ← edit me
    pitchbench_d7_N_TRIALS     = 5                                   # ← edit me
    pitchbench_d7_SEED         = BENCHMARK_SEED                      # ← edit me
    pitchbench_d7_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── e1: loudness ─────────────────────────────────────────────────────
    pitchbench_e1_LOUDNESS_DB = [-30, -20, -12, -6, -3, 0]           # ← edit me
    pitchbench_e1_TONE_MS     = BENCHMARK_DURATION_MS                # ← edit me
    pitchbench_e1_PITCHES     = BENCHMARK_PITCHES_SELECTION          # ← edit me
    pitchbench_e1_SOURCES     = BENCHMARK_ALL_SOURCES                # ← edit me

    # ── e2: audio effects ────────────────────────────────────────────────
    pitchbench_e2_PITCHES = BENCHMARK_PITCHES_SELECTION              # ← edit me
    pitchbench_e2_TONE_MS = BENCHMARK_DURATION_MS                    # ← edit me
    pitchbench_e2_EFFECTS = BENCHMARK_E2_EFFECTS                     # ← edit me
    pitchbench_e2_SOURCES = BENCHMARK_ALL_SOURCES                    # ← edit me

    # ── e3: background effects ───────────────────────────────────────────
    pitchbench_e3_BACKGROUNDS  = ["white_noise", "church-bells",
                                  "crowd-noise", "rain", "street-noise"] # ← edit me
    pitchbench_e3_SNR_DB       = [30.0, 20.0, 0.0, -6.0]             # ← edit me
    pitchbench_e3_PITCHES      = BENCHMARK_PITCHES_SELECTION         # ← edit me
    pitchbench_e3_DURATIONS_MS = [BENCHMARK_DURATION_MS]             # ← edit me
    pitchbench_e3_SOURCES      = BENCHMARK_ALL_SOURCES               # ← edit me

    # ── f1: embedding geometry ───────────────────────────────────────────
    pitchbench_f1_PITCHES = BENCHMARK_PITCHES_FULL_RANGE             # ← edit me
    pitchbench_f1_SOURCES = BENCHMARK_ALL_SOURCES                    # ← edit me

    # ── f2: token logits ─────────────────────────────────────────────────
    pitchbench_f2_PITCHES          = [29, 43, 57, 71, 72]            # ← edit me
    pitchbench_f2_SOURCES          = BENCHMARK_ALL_SOURCES           # ← edit me
    pitchbench_f2_TONE_DURATION_MS = 2000                            # ← edit me
    pitchbench_f2_TOP_K            = 200                             # ← edit me
    pitchbench_f2_MAX_NEW_TOKENS   = 32                              # ← edit me

    # ── f3: knn oracle ───────────────────────────────────────────────────
    pitchbench_f3_PITCHES    = BENCHMARK_PITCHES_FULL_RANGE          # ← edit me
    pitchbench_f3_SOURCES    = BENCHMARK_ALL_SOURCES                 # ← edit me
    pitchbench_f3_K          = 1                                     # ← edit me
    pitchbench_f3_SPLIT_SEED = BENCHMARK_SEED                        # ← edit me

    # ── g1: melodic line id ──────────────────────────────────────────────
    pitchbench_g1_N_NOTES      = 10                                  # ← edit me
    pitchbench_g1_N_PARTS_LIST = [2, 3, 4]                           # ← edit me
    pitchbench_g1_N_TRIALS     = 2                                   # ← edit me
    pitchbench_g1_DIST_N_MIN   = 1                                   # ← edit me
    pitchbench_g1_DIST_N_MAX   = 20                                  # ← edit me
    pitchbench_g1_DUR_JITTER   = 0.5                                 # ← edit me
    pitchbench_g1_TEMPOS       = {"slow": 1000, "medium": 500}       # ← edit me
    pitchbench_g1_PART_RANGES  = [(72, 83), (60, 71), (48, 59), (36, 47)] # ← edit me
    pitchbench_g1_SEED         = BENCHMARK_SEED                      # ← edit me
    pitchbench_g1_SOURCES      = BENCHMARK_GM_INSTRUMENTS            # ← edit me

    # ── g2: chorale voice id ─────────────────────────────────────────────
    pitchbench_g2_N_VOICES         = 4                               # ← edit me
    pitchbench_g2_MIN_SEG_NOTES    = 4                               # ← edit me
    pitchbench_g2_MAX_SEG_SEC      = 30.0                            # ← edit me
    pitchbench_g2_QPM              = 60.0                            # ← edit me
    pitchbench_g2_VOICE_NAMES      = ["soprano", "alto", "tenor", "bass"]   # ← edit me
    pitchbench_g2_DEFAULT_CHORALES = ["bach/bwv66.6", "bach/bwv4.8",
                                     "bach/bwv7.7", "bach/bwv26.6",
                                     "bach/bwv57.8"]                # ← edit me
    pitchbench_g2_MIXED_GROUPS     = {
        "classical_quartet": ["flute", "violin", "cello", "bass"],
        "mixed_timbres":     ["piano", "trumpet", "clarinet", "guitar"],
    }                                                                # ← edit me
    pitchbench_g2_SEED             = BENCHMARK_SEED                  # ← edit me
    pitchbench_g2_SOURCES          = BENCHMARK_GM_INSTRUMENTS        # ← edit me

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
