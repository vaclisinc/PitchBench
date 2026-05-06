"""
b1 — Hidden pitch in silence.

A single musical note is embedded at a specific position inside a long silent
clip (default: 60 s). The prompt tells the model exactly when the note occurs;
the task is to identify its pitch.

Universal IVs: source, source_type, midi.
Experiment-specific IVs: pos_ms, condition.

Each ``(source, midi, pos_ms)`` cell emits a *hidden* condition record
(tone embedded in silence) and — when ``config.INCLUDE_BASELINES`` is set —
a paired *baseline* record (same tone played in isolation) with the same
``pair_id``. Headline accuracy is reported on hidden records only;
baseline records become ``by_condition.baseline.<format>`` ablation rows
in the accuracies CSV.

Usage::
    pitchbench --id b1 --preview
    pitchbench --id b1 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_b import CatBSpec, run_cat_b_experiment

EXP_NAME          = Path(__file__).stem
PITCHES           = config.pitchbench_b1_PITCHES
TONE_POSITIONS_MS = config.pitchbench_b1_TONE_POSITIONS_MS
TONE_DURATION_MS  = config.pitchbench_b1_TONE_DURATION_MS
TOTAL_SILENCE_MS  = config.pitchbench_b1_TOTAL_SILENCE_MS
SOURCES           = config.pitchbench_b1_SOURCES


def build_conditions() -> list[dict]:
    """For each (source, midi, pos_ms): one hidden + (optionally) one baseline twin."""
    rows: list[dict] = []
    pair_id = 0
    for src in SOURCES:
        for midi in PITCHES:
            for pos_ms in TONE_POSITIONS_MS:
                base = {
                    "source":      src,
                    "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":        midi,
                    "pos_ms":      pos_ms,
                    "pair_id":     pair_id,
                }
                rows.append({**base, "condition": "hidden"})
                if config.INCLUDE_BASELINES:
                    rows.append({**base, "condition": "baseline"})
                pair_id += 1
    return rows


def wav_for(c: dict) -> Path:
    if c["condition"] == "hidden":
        return engine.tone_in_silence(
            c["midi"], c["source"],
            tone_start_ms=c["pos_ms"],
            tone_dur_ms=TONE_DURATION_MS,
            total_dur_ms=TOTAL_SILENCE_MS,
        )
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
    total_s = TOTAL_SILENCE_MS / 1000
    context = (f"You will hear a {total_s:.0f}-second audio clip. "
               f"A single musical note sounds in the clip; the rest is silence. ")
    return {
        "midi":   context + "What is the MIDI note number (0–127) of that note? "
                            "Reply with ONLY the integer.",
        "spn":    context + "What is the note name and octave of that note, "
                            "e.g. C4, F#3? Reply with ONLY the note name in "
                            "Scientific Pitch Notation.",
        "doremi": context + "What is the solfège syllable and accidental (if needed) "
                            "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B) of that "
                            "note? Reply with ONLY the syllable and accidental.",
        "hz":     context + "What is the pitch frequency of that note in Hertz? "
                            "Reply with ONLY a number (the frequency in Hz).",
    }


SPEC = CatBSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="pitch",
    prompts_fn=prompts_for,
    record_extras=("pos_ms", "condition"),
    primary_filter=lambda r: r.get("condition") == "hidden",
    pair_key=("pair_id",),
)


def preview() -> None:
    run_cat_b_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_b_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
