"""
Central configuration for PitchBench.

Per-experiment data-generation parameters are namespaced as
``pitchbench_<exp>_VARIABLE`` and live in one of three config modules,
selected by the ``MODE`` flag:

  MODE = "EVAL"      → benchmark_config.py   (paper-locked values)
  MODE = "ANALYSIS"  → analysis_config.py    (Q1 ablation preset)
  MODE = "USER"      → user_config.py        (freely editable values)

Prompts and parsing logic stay in their experiment scripts — not here.
"""

from datetime import datetime
import os
import sys as _sys
from pathlib import Path

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE = "EVAL"           # "EVAL" | "ANALYSIS" | "USER"
EVAL = (MODE == "EVAL") # backward compat for code that checks `if EVAL:`

# ── Runtime / project root ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get("PITCHBENCH_ROOT", ".")).resolve()

print(f"Running in MODE={MODE}")

DATA_DIR     = _PROJECT_ROOT / "data" 
AUDIO_DIR    = DATA_DIR / "audio" if EVAL else DATA_DIR / "analysis_audio"
RESULTS_DIR  = _PROJECT_ROOT / "results" / os.environ.get(
    "PITCHBENCH_RUN", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
) if EVAL else _PROJECT_ROOT / "results" / "analysis"

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

# ── Sources (engine-level catalogues) ─────────────────────────────────────────
GM_PROGRAMS_V1: dict[str, int] = {
    "piano":             0,
    "electric_keyboard": 4,
    "guitar":            24,
    "flute":             73,
    "trumpet":           56,
    "trombone":          57,
    "clarinet":          71,
    "oboe":              68,
    "violin":            40,
    "cello":             42,
    "organ":             19,
    "bass":              32,
    "synth_lead":        80,
    "synth_pad":         88,
    "voice":             52,
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

# DashScope (Alibaba Model Studio) OpenAI-compatible endpoint. Region default is
# Singapore; switch via env var if you're in another region (US Virginia,
# Beijing, Hong Kong, Frankfurt — see DashScope docs).
DASHSCOPE_BASE_URL: str = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

DOREMI_PARSER_MODEL: str = "openrouter/google/gemini-2.5-flash-lite"
DEFAULT_MODEL:       str = "openrouter/google/gemini-3.1-flash-lite-preview"

MODEL_MUSIC_FLAMINGO = "nvidia/music-flamingo-hf"

AF_NEXT_CHECKPOINTS = {
    "audio_flamingo_next_instruct":  "nvidia/audio-flamingo-next-hf",
    "audio_flamingo_next_think":     "nvidia/audio-flamingo-next-think-hf",
    "audio_flamingo_next_captioner": "nvidia/audio-flamingo-next-captioner-hf",
}

CONCURRENCY: dict[str, int] = {
    "openrouter": int(os.environ.get("PITCHBENCH_CONCURRENCY_OPENROUTER", "20")),
    "dashscope":  int(os.environ.get("PITCHBENCH_CONCURRENCY_DASHSCOPE",  "10")),
    "manual":     1,
}


def concurrency_for(model_name: str) -> int:
    """Return the configured max-workers cap for ``model_name``."""
    if model_name == "manual":
        return CONCURRENCY["manual"]
    if model_name.startswith("openrouter/"):
        return CONCURRENCY["openrouter"]
    if model_name.startswith("dashscope/"):
        return CONCURRENCY["dashscope"]
    return max(1, LOCAL_CONCURRENCY)


DEFAULT_SAMPLE_SEED = 42

# ── Re-export BENCHMARK_* from benchmark_config (backward compat) ────────────
from pitchbench.benchmark_config import (  # noqa: E402
    INCLUDE_BASELINES,
    BENCHMARK_NOTATION_FORMATS,
    BENCHMARK_PITCHES_FULL_RANGE,
    BENCHMARK_PITCHES_SELECTION,
    BENCHMARK_PITCHES_SELECTION_REDUCED,
    BENCHMARK_PITCHES_SELECTION_COMPACT,
    BENCHMARK_INTERVALS,
    BENCHMARK_DURATION_MS,
    BENCHMARK_DURATIONS_MS_SHORT_LONG,
    BENCHMARK_DURATIONS_MS_FULL_SWEEP,
    BENCHMARK_NOTE_DURATIONS_MS_D4,
    BENCHMARK_TOTAL_DUR_MS,
    BENCHMARK_TONE_POSITIONS_MS,
    BENCHMARK_GAP_MS,
    BENCHMARK_GAP_MS_D7,
    BENCHMARK_N_TRIALS,
    BENCHMARK_SEED,
    BENCHMARK_TIMESTAMP_TOLERANCE_MS,
    BENCHMARK_ALL_SOURCES,
    BENCHMARK_WAVEFORMS,
    BENCHMARK_GM_INSTRUMENTS,
    BENCHMARK_CONTINUOUS_GM_INSTRUMENTS,
    BENCHMARK_F1_MIXED_GROUPS,
    BENCHMARK_BASE_FREQS_A,
    BENCHMARK_C1_CHORD_INTERVALS,
    BENCHMARK_C4_CHORD_TYPES,
    BENCHMARK_C3_QUALITIES,
    BENCHMARK_D4_TRAJECTORIES,
    BENCHMARK_E1_EFFECTS,
    BENCHMARK_E3_SATURATIONS,
    BENCHMARK_PITCHES_E5,
    BENCHMARK_E4_CONDITIONS,
    SAMPLING,
)
EXPERIMENT_DEFAULTS = SAMPLING

# ── Inject per-experiment pitchbench_* variables from the mode-selected file ──
if MODE == "EVAL":
    import pitchbench.benchmark_config as _params
elif MODE == "ANALYSIS":
    import pitchbench.analysis_config as _params
else:  # USER
    import pitchbench.user_config as _params

_this = _sys.modules[__name__]
for _n in dir(_params):
    if _n.startswith("pitchbench_"):
        setattr(_this, _n, getattr(_params, _n))
del _sys, _this, _params, _n

# ── Legacy aliases ────────────────────────────────────────────────────────────
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
