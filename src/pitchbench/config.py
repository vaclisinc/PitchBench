"""
Central configuration for PitchBench.
All hardcoded constants live here; nothing else should define paths or defaults.

Runtime directories (data/, results/, stimuli/, _datasets/) are resolved relative
to the working directory so that ``pip install pitchbench`` + ``pitchbench …``
works from any project root.  Override the base via the PITCHBENCH_ROOT env var.
"""

from datetime import datetime
import os
from pathlib import Path

DEV = False

# ── Runtime / project root ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get("PITCHBENCH_ROOT", ".")).resolve()

DATA_DIR     = _PROJECT_ROOT / "data"       # generated experiment stimuli
AUDIO_DIR    = DATA_DIR / "audio"           # central audio engine cache
RESULTS_DIR  = _PROJECT_ROOT / "results" / f"run_{datetime.now().strftime("%Y%m%d_%H%M%S")}"  # experiment outputs (never deleted)
STIMULI_DIR  = _PROJECT_ROOT / "stimuli"   # reference stimuli


# ── Audio defaults ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000   # Hz

# purpose: the two note durations (ms) each experiment tests
# exp: a4, a5, b2, b3, b4, b5, c1, c2, c4, d1, d2, d3, d4, d6, e3
DEFAULT_DURATIONS_MS: list[int] = [1000, 5000]

# purpose: note duration (ms) when an experiment uses only one fixed duration
# exp: a1, a2, b1, c3, d2, d5, d6, d7, e1, e2
DEFAULT_DURATION_MS = 5000

# purpose: lowest MIDI note number used in pitch sweeps (F1 normally, C4 in DEV mode)
# exp: a1, d7, f1, f3
DEFAULT_MIDI_MIN = 29 if not DEV else 60

# purpose: highest MIDI note number used in pitch sweeps (F6 normally, D4 in DEV mode)
# exp: a1, d7, f1, f3
DEFAULT_MIDI_MAX = 89 if not DEV else 62

# purpose: 14 hand-picked pitches spread across C1–F6 used as the standard test set (reduced to [C4, A4] in DEV mode)
# exp: a2, a3, c2, c3
DEFAULT_SELECTION = [30, 36, 43, 48, 54, 58, 60, 64, 67, 69, 73, 76, 79, 88] if not DEV else [60, 69]

# purpose: same list as DEFAULT_SELECTION, used by experiments that refer to it as "pitches"
# exp: a4, a5, b2, b3, b4, b5, c1, d3, d4, d5, e1, e3
DEFAULT_PITCHES: list[int] = DEFAULT_SELECTION

# purpose: silence gap between consecutive notes in a sequence (ms)
# exp: d1, d6
DEFAULT_GAP_MS   = 300

# purpose: how many times each condition is repeated
# exp: d1, d2, d6
DEFAULT_N_TRIALS = 3

# purpose: random seed that controls which notes/sequences get assigned to which trial
# exp: b3, b4, b5, c2, d1, d2, d4, d6, d7, f3, g1, g2, z1
DEFAULT_SEED = 100

# purpose: random seed for picking a smaller subset of conditions when --sample-n is used
# exp: a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c1, c2, c3, c4, d1, d2, d3, d4, d5, d6, d7, e1, e2, e3, g1, g2, z1
DEFAULT_SAMPLE_SEED = 42

# ── Stimulus generation (FluidSynth) ──────────────────────────────────────────
SF2_PATH = os.environ.get(
    "PITCHBENCH_SF2",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
)
NOTE_ON_DUR  = 1.7    # seconds of held note
RELEASE_DUR  = 0.5    # seconds of release tail captured after note-off
FADE_OUT     = 0.05   # seconds of linear fade at end
TARGET_PEAK  = 0.9    # normalize each file to this peak level

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS_DIR      = _PROJECT_ROOT / "_datasets"
NSYNTH_VALID_DIR  = DATASETS_DIR / "NSynth" / "nsynth-valid"

# ── Models ────────────────────────────────────────────────────────────────────
# Canonical slug → human-readable display name
MODELS: dict[str, str] = {
    "manual":                        "Manual (CLI input by human)",
    "music_flamingo":                "Music Flamingo",
    "audio_flamingo_next_instruct":  "Audio Flamingo Next – instruct",
    "audio_flamingo_next_think":     "Audio Flamingo Next – think",
    "audio_flamingo_next_captioner": "Audio Flamingo Next – captioner",
    "qwen3_omni":                    "Qwen3-Omni (30B-A3B-Instruct)",
}

# Each model's server URL (override via env vars to run on different hosts/ports).
# OpenRouter slugs (prefixed "openrouter/") are routed inside helpers/api.py and use
# the sentinel value "openrouter" as their endpoint.
MODEL_URLS: dict[str, str] = {
    "music_flamingo":                os.environ.get("MF_URL",           "http://localhost:8000"),
    "audio_flamingo_next_instruct":  os.environ.get("AF_NEXT_INST_URL", "http://localhost:8001"),
    "audio_flamingo_next_think":     os.environ.get("AF_NEXT_THINK_URL","http://localhost:8002"),
    "audio_flamingo_next_captioner": os.environ.get("AF_NEXT_CAP_URL",  "http://localhost:8003"),
    "qwen3_omni":                    os.environ.get("QWEN3_OMNI_URL",   "http://localhost:8004"),
    # OpenRouter slugs (prefixed "openrouter/") are routed inside helpers/api.py
    # and don't need to be registered here — pass any slug via --models.
}

OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# Cheap OpenRouter model used as a fallback parser when regex fails to
# extract a solfège pitch class from a model's free-text response.
# Set PITCHBENCH_DOREMI_LLM=0 to disable; the fallback also no-ops if no
# OPENROUTER_KEY is available.
DOREMI_PARSER_MODEL: str = "openrouter/google/gemini-2.5-flash-lite"

# Default model used when the user runs `pitchbench` without --models.
# The CLI prompts interactively with this as the default; pressing Enter accepts it.
# Any OpenRouter slug works — if the chosen model can't accept audio, OpenRouter's
# error response is surfaced verbatim rather than being pre-validated here.
DEFAULT_MODEL: str = "openrouter/google/gemini-3.1-flash-lite-preview"

# HuggingFace model IDs (used by model server scripts)
MODEL_MUSIC_FLAMINGO = "nvidia/music-flamingo-hf"

AF_NEXT_CHECKPOINTS = {
    "audio_flamingo_next_instruct":  "nvidia/audio-flamingo-next-hf",
    "audio_flamingo_next_think":     "nvidia/audio-flamingo-next-think-hf",
    "audio_flamingo_next_captioner": "nvidia/audio-flamingo-next-captioner-hf",
}

# Expanded instrument set for stimuli/v1 (used by exp_5 and exp_9–15)
GM_PROGRAMS_V1: dict[str, int] = {
    "piano":             0,   # Acoustic Grand Piano
    "electric_keyboard": 4,   # Electric Piano 1
    "guitar":            24,  # Acoustic Guitar (nylon)
    "flute":             73,  # Flute
    "trumpet":           56,  # Trumpet
    "trombone":          57,  # Trombone
    "clarinet":          71,  # Clarinet
    "oboe":              68,  # Oboe
    "violin":            40,  # Violin
    "cello":             42,  # Cello
    "organ":             19,  # Church Organ
    "bass":              32,  # Acoustic Bass
    "synth_lead":        80,  # Square Wave synth lead
    "synth_pad":         88,  # New Age synth pad
    "voice":             52,  # Choir Aahs
}

# Programmatic waveforms for v1 (generated without FluidSynth)
WAVEFORMS: list[str] = ["sine", "sawtooth", "square", "triangle"]

ALL_SOURCES: list[str] = list(WAVEFORMS) + list(GM_PROGRAMS_V1.keys())