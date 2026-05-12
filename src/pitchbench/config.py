"""
Central configuration for PitchBench.

Per-experiment data-generation parameters are namespaced as
``pitchbench_<exp>_VARIABLE`` and live in ``benchmark_config.py``
(paper-locked values loaded by default) or ``analysis_config.py``
(loaded at runtime by ``pitchbench analyze``).

Prompts and parsing logic stay in their experiment scripts — not here.
"""

import os
import sys as _sys
from pathlib import Path

# ── Runtime / project root ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get("PITCHBENCH_ROOT", ".")).resolve()

DATA_DIR      = _PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
ANALYSIS_DIR  = DATA_DIR / "analysis"
RESULTS_DIR   = _PROJECT_ROOT / "results"

# ── Audio infrastructure (engine-level; not data-gen) ─────────────────────────
SAMPLE_RATE  = 16_000
NOTE_ON_DUR  = 1.7
RELEASE_DUR  = 0.5
FADE_OUT     = 0.05
TARGET_PEAK  = 0.9

def _resolve_sf2() -> str:
    if val := os.environ.get("PITCHBENCH_SF2"):
        print(f"Local soundfont: env override? {val}")
        return val
    soundfonts_dir = DATA_DIR / "soundfonts"
    for ext in ("*.sf2", "*.sf3"):
        found = sorted(soundfonts_dir.glob(ext))
        if found:
            print("Local soundfont: found in data/soundfonts")
            return str(found[0])
    print("Local soundfont: using default /usr/share/sounds/sf2/FluidR3_GM.sf2")
    return "/usr/share/sounds/sf2/FluidR3_GM.sf2"

SF2_PATH: str = _resolve_sf2()

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

# ── API endpoints ─────────────────────────────────────────────────────────────
LOCAL_CONCURRENCY: int = int(os.environ.get("LOCAL_CONCURRENCY", "2"))

OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

DASHSCOPE_BASE_URL: str = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

DOREMI_PARSER_MODEL: str = "openrouter/google/gemini-2.5-flash-lite"
DEFAULT_LOCAL_URL:   str = os.environ.get("PITCHBENCH_LOCAL_URL", "http://localhost:8001")

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
from pitchbench.configs.benchmark_config import (  # noqa: E402
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

# ── Always start with benchmark_config pitchbench_* variables ────────────────
# ``pitchbench analyze`` overrides these at runtime via _inject_analysis_config().
import pitchbench.configs.benchmark_config as _params  # noqa: E402
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
