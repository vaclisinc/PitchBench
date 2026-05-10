"""
d7c — Pitch with reference, concat-audio format, a1-style pitch list.

Companion to d7a: same anchor design (reference tone + target tone in one
concatenated WAV, prompt identifies the reference) but the target pitches
come from an explicit list (``pitchbench_d7c_PITCHES``) rather than being
derived from ``ref_midi + interval``.  This lets the experiment share exactly
the same target pitch set as a1, so results are directly comparable.

In analysis mode (MODE=ANALYSIS, q1 preset) the pitch list is set to the 7
ablation pitches [29, 38, 47, 56, 65, 74, 83], giving 2 ref × 7 pitches × 6
sources = 84 anchored conditions — the same 42 notes per reference that a1
uses in the ablation study.

The interval is computed as ``tgt_midi − ref_midi`` and stored in each record
for analysis; it is not an independent variable here.

Companion experiment: ``d7d`` delivers the same conditions with split-audio
input (only multi-audio capable models can run d7d).

Usage::
    pitchbench --id d7c --preview
    pitchbench --id d7c --models openrouter/google/gemini-2.5-flash
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_a import (
    CatASpec, run_cat_a_experiment,
)
from pitchbench.experiments.helpers.music import (
    midi_to_freq, midi_to_note, midi_to_solfege,
)

EXP_NAME          = Path(__file__).stem
SOURCES           = config.pitchbench_d7c_SOURCES
REFERENCE_PITCHES = config.pitchbench_d7c_REFERENCE_PITCHES
PITCHES           = config.pitchbench_d7c_PITCHES
TONE_DURATION_MS  = config.pitchbench_d7c_TONE_DURATION_MS
GAP_MS            = config.pitchbench_d7c_GAP_MS


def build_conditions() -> list[dict]:
    """For each (ref, target_pitch, source), emit one anchored + one baseline row."""
    rows: list[dict] = []
    for ref_midi in REFERENCE_PITCHES:
        for tgt_midi in PITCHES:
            interval = tgt_midi - ref_midi
            for src in SOURCES:
                base = {
                    "source":      src,
                    "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":        tgt_midi,
                    "ref_midi":    ref_midi,
                    "interval":    interval,
                }
                rows.append({**base, "condition": "anchored"})
                if config.INCLUDE_BASELINES:
                    rows.append({**base, "condition": "baseline"})
    return rows


def wav_for(c: dict) -> Path:
    if c["condition"] == "anchored":
        return engine.sequence([c["ref_midi"], c["midi"]], c["source"], TONE_DURATION_MS, GAP_MS)
    return engine.tone(c["midi"], c["source"], TONE_DURATION_MS)


def prompts_for(c: dict) -> dict[str, str]:
    if c["condition"] == "baseline":
        return {
            "midi":   "This audio contains a single musical note. "
                      "What is the MIDI note number (an integer from 0 to 127)? "
                      "Reply with ONLY the integer.",
            "spn":    "This audio contains a single musical note. "
                      "What is the note name and octave, e.g. C4, F#3? "
                      "Reply with ONLY the note name in Scientific Pitch Notation.",
            "doremi": "This audio contains a single musical note. "
                      "What is the solfège syllable and accidental (if needed) "
                      "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B)? "
                      "Reply with ONLY the syllable and accidental (if needed).",
            "hz":     "This audio contains a single musical note. "
                      "What is the main pitch frequency, expressed in Hertz? "
                      "Reply with ONLY a number (the frequency in Hz).",
        }
    # anchored
    ref_midi = c["ref_midi"]
    ref_note = midi_to_note(ref_midi)
    ref_solf = midi_to_solfege(ref_midi)
    ref_hz   = f"{midi_to_freq(ref_midi):.2f}"
    return {
        "midi":   (f"You will hear two tones separated by a silence. "
                   f"The FIRST tone is MIDI note {ref_midi}. "
                   f"What is the MIDI note number of the SECOND tone? "
                   f"Reply with ONLY the integer."),
        "spn":    (f"You will hear two tones separated by a silence. "
                   f"The FIRST tone is {ref_note}. "
                   f"What is the note name and octave of the SECOND tone, "
                   f"e.g. C4, F#3? Reply with ONLY the note name in Scientific Pitch Notation."),
        "doremi": (f"You will hear two tones separated by a silence. "
                   f"The FIRST tone is '{ref_solf}' "
                   f"(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B). "
                   f"What is the solfège syllable and accidental (if needed) "
                   f"of the SECOND tone? Reply with ONLY the syllable and accidental."),
        "hz":     (f"You will hear two tones separated by a silence. "
                   f"The FIRST tone is {ref_hz} Hz. "
                   f"What is the pitch frequency of the SECOND tone in Hertz? "
                   f"Reply with ONLY a number (the frequency in Hz)."),
    }


SPEC = CatASpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    prompts_fn=prompts_for,
    record_extras=("condition", "ref_midi", "interval"),
    primary_filter=lambda r: r.get("condition") == "anchored",
    pair_key=("source", "ref_midi", "midi"),
    label_fn=lambda j: (
        f"{j['cond']['condition']:8s} r={midi_to_note(j['cond']['ref_midi']):>3} "
        f"t={midi_to_note(j['cond']['midi']):>3} {j['cond']['source']}"
    ),
)


def preview() -> None:
    run_cat_a_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_a_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
