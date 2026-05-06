"""
e5 — Time stretching vs resampling: pitch robustness under tempo change.

Tests whether ALMs can correctly identify pitch when audio duration is
altered via two fundamentally different operations:

  resample  — simulates playback speed change; duration AND pitch both change
              (analogous to slowing/speeding a record or tape)
  stretch   — phase-vocoder time stretching; duration changes, pitch is preserved

Conditions:
  clean         — no modification (3 s)
  resample_0.5x — 2× speed (shorter, ~1.5 s), pitch RISES 12 st  (GT = midi + 12)
  resample_2x   — ½× speed (longer, ~6 s),    pitch DROPS 12 st  (GT = midi − 12)
  stretch_0.5x  — 2× speed (shorter, ~1.5 s), pitch UNCHANGED    (GT = midi)
  stretch_2x    — ½× speed (longer,  ~6 s),   pitch UNCHANGED    (GT = midi)

Pitches: restricted to MIDI 36–84 so ±12-semitone shifts remain in audible range.
4-format pitch-ID prompts. The prompt warns the model the audio may have been
sped up or slowed down.

The headline ``midi_gt`` per record is the **perceived** pitch (after the
manipulation), not the ``original_midi`` — that's what
``standard_pitch_record`` scores against.
"""

from __future__ import annotations

import math
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_e import CatESpec, run_cat_e_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME   = Path(__file__).stem
PITCHES    = config.pitchbench_e4_PITCHES
TONE_MS    = config.pitchbench_e4_TONE_MS
CONDITIONS = config.pitchbench_e4_CONDITIONS
SOURCES    = config.pitchbench_e4_SOURCES

PROMPT_PREFIX = (
    "Listen to this audio clip of a single musical note. The recording "
    "may have been sped up or slowed down. Identify the PITCH of the note "
    "as it sounds in the audio. "
)


def _gt_midi(original_midi: int, mode: str, factor: float) -> int:
    """Ground-truth MIDI of the perceived pitch.

    factor is a duration ratio (output / input):
        factor=2.0 → twice as long → half speed → pitch drops one octave (−12 st)
        factor=0.5 → half as long → double speed → pitch rises one octave (+12 st)
    For stretch / clean, pitch is unchanged.
    """
    if mode == "resample" and factor != 1.0:
        return original_midi + round(-12.0 * math.log2(factor))
    return original_midi


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for cond in CONDITIONS:
            for midi in PITCHES:
                gt = _gt_midi(midi, cond["mode"], cond["factor"])
                rows.append({
                    "source":        src,
                    "source_type":   "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":          gt,           # perceived pitch — used as midi_gt
                    "condition":     cond["name"],
                    "mode":          cond["mode"],
                    "speed_factor":  cond["factor"],
                    "original_midi": midi,
                    "original_note": midi_to_note(midi),
                })
    return rows


def wav_for(c: dict) -> Path:
    return engine.tone_time_modified(
        c["original_midi"], c["source"], TONE_MS,
        c["mode"], c["speed_factor"],
    )


SPEC = CatESpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompt_prefix=PROMPT_PREFIX,
    record_extras=("condition", "mode", "speed_factor", "original_midi", "original_note"),
    label_fn=lambda j: (
        f"{j['cond']['source']:10s}  {j['cond']['condition']:14s}  "
        f"orig={midi_to_note(j['cond']['original_midi']):4s} → "
        f"gt={midi_to_note(j['cond']['midi']):4s}"
    ),
)


def preview() -> None:
    run_cat_e_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_e_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
