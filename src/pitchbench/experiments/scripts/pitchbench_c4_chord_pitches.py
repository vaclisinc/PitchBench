"""
c4 — Simultaneous-pitch identification (chord pitches).

Question: can the ALM identify the individual pitches of a chord? Each
chord is queried in three formats (MIDI / SPN / Doremi); each format is
scored as exact-match between the predicted set and the ground-truth set
(order-agnostic). One record per chord with `midi_correct`,
`spn_correct`, `doremi_correct` columns — same shape as cat-A.

Universal IVs: source.
Experiment-specific IVs:
    chord_type:  ∈ BENCHMARK_C4_CHORD_TYPES (dyads, triads, sevenths)
    n_notes:     chord size
    root_midi:   root note
"""

from __future__ import annotations

import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_c import CatCSpec, run_cat_c_experiment
from pitchbench.experiments.helpers.music import (
    SOLFEGE_TO_PC, extract_all_notes, midi_to_note, midi_to_solfege,
    note_to_midi,
)

EXP_NAME         = Path(__file__).stem
TONE_DURATION_MS = config.pitchbench_c4_TONE_DURATION_MS
SOURCES          = config.pitchbench_c4_SOURCES
CHORD_TYPES      = config.pitchbench_c4_CHORD_TYPES
BASE_ROOTS       = config.pitchbench_c4_BASE_ROOTS

PROMPT_MIDI = (
    "This audio contains multiple musical pitches played simultaneously. "
    "List ALL MIDI note numbers you hear, from lowest to highest. "
    "Reply with ONLY the integers separated by spaces. Nothing else. "
    "Output only the answer."
)

PROMPT_SPN = (
    "This audio contains multiple musical pitches played simultaneously. "
    "List ALL note names you hear, from lowest to highest, expressed in "
    "Scientific Pitch Notation. "
    "Reply with ONLY the note names expressed in Scientific Pitch Notation, "
    "separated by spaces, e.g. C4 E4 G#4. Nothing else. "
    "Output only the answer."
)

PROMPT_DOREMI = (
    "This audio contains multiple musical pitches played simultaneously. "
    "List ALL solfège syllable and accidentals (if needed) you hear "
    "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B). "
    "Reply with ONLY the syllables with accidental (if needed) separated "
    "by spaces, e.g. do mi sol#. Nothing else. Output only the answer."
)

PROMPTS = {"midi": PROMPT_MIDI, "spn": PROMPT_SPN, "doremi": PROMPT_DOREMI}


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for chord_name, intervals in CHORD_TYPES.items():
        for root in BASE_ROOTS:
            midi_notes = [root + iv for iv in intervals]
            if any(m < 0 or m > 127 for m in midi_notes):
                continue
            for source in SOURCES:
                rows.append({
                    "chord_type": chord_name,
                    "n_notes":    len(midi_notes),
                    "root_midi":  root,
                    "root_note":  midi_to_note(root),
                    "midi_notes": midi_notes,
                    "note_names": [midi_to_note(m) for m in midi_notes],
                    "doremi_seq": [midi_to_solfege(m) for m in midi_notes],
                    "source":     source,
                })
    return rows


def wav_for(c: dict) -> Path:
    return engine.chord(c["midi_notes"], c["source"], TONE_DURATION_MS)


def prompts_for(_: dict) -> dict[str, str]:
    return PROMPTS


def _parse_midi_list(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"\b(\d{1,3})\b", text) if 0 <= int(m) <= 127]


def _parse_spn_strings(text: str) -> list[str]:
    """Return the SPN tokens the model emitted, in order, as strings.

    Internal scoring goes via ``note_to_midi`` (see :func:`record_for`); the
    string list is preserved so the audit / records CSV shows what the model
    actually said (e.g. ``["C4", "E4", "G4"]`` rather than ``[60, 64, 67]``).
    """
    return list(extract_all_notes(text or ""))


_DOREMI_SYL_RE = re.compile(
    r"\b(do|re|mi|fa|sol|la|si|ti)\s*(?:(#|♯|sharp)|(b|♭|flat))?\b",
    re.IGNORECASE,
)


def _parse_doremi_syllables(text: str) -> list[str]:
    """Return the solfège tokens the model emitted, normalised to lowercase
    base + ``#`` / ``b`` (e.g. ``do``, ``sol#``, ``mib``).
    """
    out: list[str] = []
    for m in _DOREMI_SYL_RE.finditer((text or "").lower()):
        base    = m.group(1)
        sharp   = m.group(2)
        flat    = m.group(3)
        if sharp:
            out.append(f"{base}#")
        elif flat:
            out.append(f"{base}b")
        else:
            out.append(base)
    return out


def _syllable_to_pc(syl: str) -> int | None:
    """Map a normalised syllable token to a pitch class via SOLFEGE_TO_PC."""
    return SOLFEGE_TO_PC.get(syl)


def _set_exact(gt: list, pred: list) -> int:
    return int(set(gt) == set(pred))


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw_m, raw_s, raw_d = responses["midi"], responses["spn"], responses["doremi"]
    midi_pred       = _parse_midi_list(raw_m)
    spn_strings     = _parse_spn_strings(raw_s)
    doremi_syllables = _parse_doremi_syllables(raw_d)

    # Convert to comparable scalars for set-exact-match scoring.
    spn_midis = [m for n in spn_strings if (m := note_to_midi(n)) is not None]
    doremi_pcs = list(dict.fromkeys(
        pc for syl in doremi_syllables if (pc := _syllable_to_pc(syl)) is not None
    ))

    gt_midis = list(c["midi_notes"])
    gt_pcs   = [m % 12 for m in gt_midis]
    return {
        "chord_type":     c["chord_type"],
        "n_notes":        c["n_notes"],
        "root_midi":      c["root_midi"],
        "root_note":      c["root_note"],
        "midi_set":       str(gt_midis),
        "note_set":       ", ".join(c["note_names"]),
        "doremi_set":     ", ".join(c["doremi_seq"]),
        "source":         c["source"],
        "raw_midi":       raw_m,
        "raw_spn":        raw_s,
        "raw_doremi":     raw_d,
        # Display the model's tokens as the model emitted them.
        "midi_pred":      str(midi_pred),
        "spn_pred":       str(spn_strings),
        "doremi_pred":    str(doremi_syllables),
        # Scoring uses the converted scalars.
        "midi_correct":   _set_exact(gt_midis, midi_pred),
        "spn_correct":    _set_exact(gt_midis, spn_midis),
        "doremi_correct": _set_exact(gt_pcs,   doremi_pcs),
    }


SPEC = CatCSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="set",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("midi", "spn", "doremi"),
    record_extras=("chord_type", "n_notes", "root_midi"),
    label_fn=lambda j: (
        f"{j['cond']['chord_type']:12s} "
        f"r={j['cond']['root_note']:3s} "
        f"{j['cond']['source']:12s}"
    ),
)


def preview() -> None:
    run_cat_c_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_c_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
