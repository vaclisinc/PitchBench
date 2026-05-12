"""
y1 — Single-pitch identification with five answer options.

Same stimulus grid as a1 (single sustained tone), but each prompt provides five
candidate answers where exactly one is correct. The correct option position is
randomized deterministically per condition.

Universal IVs: source, source_type, midi.
Additional IV: semitone_step.

Usage::
    pitchbench --id y1 --preview
    pitchbench --id y1 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import random
import string
from importlib.util import find_spec
from pathlib import Path

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.experiments.helpers.cat_a import CatASpec, run_cat_a_experiment
from pitchbench.experiments.helpers.music import midi_to_freq, midi_to_note, midi_to_solfege
from pitchbench.experiments.helpers.timing_layout import stable_cell_seed

EXP_NAME = Path(__file__).stem
PITCHES = config.pitchbench_y1_PITCHES
TONE_DURATION_MS = config.pitchbench_y1_TONE_DURATION_MS
ALL_SOURCES = config.pitchbench_y1_SOURCES
N_OPTIONS = config.pitchbench_y1_N_OPTIONS
OPTION_STEP_ST_RAW = config.pitchbench_y1_OPTION_STEP_ST
SEED = config.pitchbench_y1_SEED


def _option_steps() -> list[int]:
    """Normalize config value to a non-empty list of positive semitone steps."""
    raw = OPTION_STEP_ST_RAW
    if isinstance(raw, int):
        steps = [raw]
    elif isinstance(raw, (list, tuple)):
        steps = [int(x) for x in raw]
    else:
        raise ValueError("pitchbench_y1_OPTION_STEP_ST must be an int or list[int]")
    if not steps:
        raise ValueError("pitchbench_y1_OPTION_STEP_ST must not be empty")
    if any(s <= 0 for s in steps):
        raise ValueError("pitchbench_y1_OPTION_STEP_ST values must be positive")
    return steps


def _available_sources() -> list[str]:
    """Drop GM instruments when FluidSynth isn't installed."""
    out: list[str] = []
    have_fluidsynth = find_spec("fluidsynth") is not None
    for src in ALL_SOURCES:
        if src in config.WAVEFORMS or have_fluidsynth:
            out.append(src)
        else:
            print(f"  [SKIP] Instrument {src!r}: FluidSynth not installed")
    return out


def _valid_correct_indices(midi: int, n_options: int, step_st: int) -> list[int]:
    """Return option indices that keep every candidate within MIDI [0, 127]."""
    valid: list[int] = []
    for idx in range(n_options):
        low = midi - idx * step_st
        high = midi + (n_options - 1 - idx) * step_st
        if 0 <= low and high <= 127:
            valid.append(idx)
    return valid


def _build_options(midi: int, source: str, step_st: int) -> tuple[list[int], int]:
    """Build equally spaced options with randomized correct answer index."""
    valid_indices = _valid_correct_indices(midi, N_OPTIONS, step_st)
    if not valid_indices:
        raise ValueError(
            f"No valid {N_OPTIONS}-choice option set for midi={midi}, "
            f"step={step_st} semitones"
        )

    seed = stable_cell_seed(SEED, source, midi, step_st, N_OPTIONS)
    rng = random.Random(seed)
    correct_idx = rng.choice(valid_indices)
    options = [midi + (i - correct_idx) * step_st for i in range(N_OPTIONS)]
    return options, correct_idx


def build_conditions() -> list[dict]:
    if N_OPTIONS != 5:
        raise ValueError("y1 requires exactly 5 options")
    steps = _option_steps()

    out: list[dict] = []
    for s in _available_sources():
        source_type = "waveform" if s in config.WAVEFORMS else "instrument"
        for step_st in steps:
            for m in PITCHES:
                options_midi, correct_option_idx = _build_options(m, s, step_st)
                out.append(
                    {
                        "source": s,
                        "source_type": source_type,
                        "midi": m,
                        "semitone_step": step_st,
                        "correct_option_idx": correct_option_idx,
                        "options_midi": options_midi,
                    }
                )
    return out


def wav_for(c: dict) -> Path:
    return engine.tone(c["midi"], c["source"], TONE_DURATION_MS)


def _fmt_options(options: list[str]) -> str:
    labels = list(string.ascii_uppercase[: len(options)])
    lines = [f"{lab}) {opt}" for lab, opt in zip(labels, options, strict=True)]
    return "\n".join(lines)


def prompts_for(c: dict) -> dict[str, str]:
    options_midi = [int(x) for x in c["options_midi"]]
    options_spn = [midi_to_note(m) for m in options_midi]
    options_doremi = [midi_to_solfege(m) for m in options_midi]
    options_hz = [f"{midi_to_freq(m):.2f} Hz" for m in options_midi]

    common = (
        "This audio contains a single musical note. "
        "choose from the following options.\n"
    )

    return {
        "midi": (
            common
            + _fmt_options([str(x) for x in options_midi])
            + "\nReply with ONLY the chosen MIDI note number (not the option letter). "
            "Output only the answer."
        ),
        "spn": (
            common
            + _fmt_options(options_spn)
            + "\nReply with ONLY the chosen note name and octave (not the option letter). "
            "Output only the answer."
        ),
        "doremi": (
            common
            + _fmt_options(options_doremi)
            + "\nReply with ONLY the chosen solfege syllable and accidental if needed "
            "(not the option letter). Output only the answer."
        ),
        "hz": (
            common
            + _fmt_options(options_hz)
            + "\nReply with ONLY the chosen frequency in Hz (not the option letter). "
            "Output only the answer."
        ),
    }


SPEC = CatASpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompts_fn=prompts_for,
    record_extras=("semitone_step", "correct_option_idx", "options_midi"),
)


def preview() -> None:
    run_cat_a_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_a_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
