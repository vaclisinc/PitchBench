"""
a2 — Pitch with reference (anchored vs baseline).

A reference tone is played first, then the target tone; the prompt tells
the model what the reference is. Tests whether the model can use that
anchor to identify the target more accurately — i.e. whether it can combine
relative pitch perception with an absolute reference.

For every ``(ref_midi, interval, source)`` cell the script always emits BOTH
an anchored record (audio = sequence([ref, target])) AND a baseline record
(audio = target tone alone) with the SAME target pitch — the baseline is
literally a copy of the anchored cell minus the reference. The headline
accuracy is reported on the anchored records only; baseline records become
``by_condition.baseline.<format>`` ablation rows in the accuracies CSV.

Universal IVs: source, source_type, midi (target).
Experiment-specific IVs: condition, ref_midi, interval.

Usage::
    pitchbench --id a2 --preview
    pitchbench --id a2 --models audio_flamingo_next_instruct
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
SOURCES           = config.pitchbench_a2_SOURCES
REFERENCE_PITCHES = config.pitchbench_a2_REFERENCE_PITCHES
INTERVALS         = config.pitchbench_a2_INTERVALS
TONE_DURATION_MS  = config.pitchbench_a2_TONE_DURATION_MS
GAP_MS            = config.pitchbench_a2_GAP_MS


def build_conditions() -> list[dict]:
    """For each (ref, interval, source), emit one anchored + one baseline row."""
    rows: list[dict] = []
    for ref_midi in REFERENCE_PITCHES:
        for interval in INTERVALS:
            tgt_midi = ref_midi + interval
            if tgt_midi < 0 or tgt_midi > 127:
                continue
            for src in SOURCES:
                base = {
                    "source":      src,
                    "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":        tgt_midi,         # the target pitch (graded by std record)
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
    ref_midi   = c["ref_midi"]
    ref_note   = midi_to_note(ref_midi)
    ref_solf   = midi_to_solfege(ref_midi)
    ref_hz     = f"{midi_to_freq(ref_midi):.2f}"
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
    # Anchored and baseline at the same (source, ref_midi, interval) cell are
    # the matched pair: sampling picks the anchored stim and the baseline
    # twin always comes along.
    pair_key=("source", "ref_midi", "interval"),
    label_fn=lambda j: (
        f"{j['cond']['condition']:8s} r={midi_to_note(j['cond']['ref_midi']):>3} "
        f"i={j['cond']['interval']:+3d} {j['cond']['source']}"
    ),
)


def preview() -> None:
    run_cat_a_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_a_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
