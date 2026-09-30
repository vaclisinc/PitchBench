"""D8 — Ordered pitch identification in a sequence of separated notes.

Headline scores are macro Ordered Note F1 (LCS) per response format. ANY
is the per-stimulus maximum of MIDI, SPN and Hz (solfege lacks octave).
Full parsed sequences retain extra notes. Strict full-sequence correctness
and positional matches remain diagnostics; Hz retains its <=1 Hz tolerance.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.experiments.helpers.ordered_notes import ordered_note_f1
from pitchbench.experiments.helpers.cat_d import CatDSpec, run_cat_d_experiment
from pitchbench.experiments.helpers.music import (
    PC_TO_SOLFEGE,
    extract_all_notes,
    extract_all_solfege,
    midi_to_freq,
    midi_to_note,
    semitone_distance,
)

EXP_NAME         = Path(__file__).stem
PITCH_MIN        = config.pitchbench_d8_PITCH_MIN
PITCH_MAX        = config.pitchbench_d8_PITCH_MAX
N_NOTES_LIST     = config.pitchbench_d8_N_NOTES_LIST
DEFAULT_N_TRIALS = config.pitchbench_d8_N_TRIALS
DEFAULT_SEED     = config.pitchbench_d8_SEED
TONE_MS          = config.pitchbench_d8_TONE_MS
GAP_MS           = config.pitchbench_d8_GAP_MS
SOURCES          = config.pitchbench_d8_SOURCES


def _prompt_midi(n: int) -> str:
    return (
        f"You will hear {n} musical notes played one after another, "
        "each separated by a brief silence. "
        f"Identify all {n} MIDI note numbers in order from first to last. "
        "Reply with ONLY the integers separated by spaces "
        "(e.g. 60 64 68). Nothing else. Output only the answer."
    )


def _prompt_spn(n: int) -> str:
    return (
        f"You will hear {n} musical notes played one after another, "
        "each separated by a brief silence. "
        f"Identify all {n} notes in order from first to last. "
        "Reply with ONLY the note names expressed in Scientific Pitch Notation, separated by spaces "
        "(e.g. C4 E4 G#4). Nothing else. Output only the answer."
    )


def _prompt_doremi(n: int) -> str:
    return (
        f"You will hear {n} musical notes played one after another, "
        "each separated by a brief silence. "
        f"Identify all {n} solfège syllable and accidentals (if needed) in order from first to last "
        "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B; include sharps e.g. do# re#). "
        "Reply with ONLY the syllable and accidentals (if needed) separated by spaces "
        "(e.g. do mi sol#). Nothing else. Output only the answer."
    )


def _prompt_hz(n: int) -> str:
    return (
        f"You will hear {n} musical notes played one after another, "
        "each separated by a brief silence. "
        f"Identify the main pitch frequency, the one that characterizes the note, in Hertz of all {n} notes in order from first to last. "
        "Reply with ONLY the frequencies in Hz separated by spaces "
        "(e.g. 261.6 329.6 392.0). Nothing else. Output only the answer."
    )


def build_conditions() -> list[dict]:
    rng = random.Random(DEFAULT_SEED)
    rows: list[dict] = []
    for n in N_NOTES_LIST:
        for trial in range(DEFAULT_N_TRIALS):
            midi_seq   = rng.sample(range(PITCH_MIN, PITCH_MAX + 1), n)
            note_seq   = [midi_to_note(m) for m in midi_seq]
            doremi_seq = [PC_TO_SOLFEGE.get(m % 12, "?") for m in midi_seq]
            hz_seq     = [round(midi_to_freq(m), 4) for m in midi_seq]
            for src in SOURCES:
                rows.append({
                    "source":          src,
                    "n_notes":         n,
                    "trial":           trial,
                    "midi_sequence":   midi_seq,
                    "note_sequence":   note_seq,
                    "doremi_sequence": doremi_seq,
                    "hz_sequence":     hz_seq,
                })
    return rows


def wav_for(c: dict) -> Path:
    return engine.sequence(c["midi_sequence"], c["source"], TONE_MS, GAP_MS)


def prompts_for(c: dict) -> dict[str, str]:
    n = c["n_notes"]
    return {
        "midi":   _prompt_midi(n),
        "spn":    _prompt_spn(n),
        "doremi": _prompt_doremi(n),
        "hz":     _prompt_hz(n),
    }


# ── Response parsing ────────────────────────────────────────────────────────

def _pad(seq: list, n: int) -> list:
    out = list(seq[:n])
    while len(out) < n:
        out.append(None)
    return out


def _parse_midi_seq(text: str) -> list[int]:
    nums: list[int] = []
    for m in re.finditer(r"\b(\d{1,3})\b", text or ""):
        v = int(m.group(1))
        if 0 <= v <= 127:
            nums.append(v)
    return nums


def _parse_spn_seq(text: str) -> list[str]:
    return extract_all_notes(text or "")


def _parse_doremi_seq(text: str) -> list[int]:
    return extract_all_solfege(text or "")


def _parse_hz_seq(text: str) -> list[float]:
    nums: list[float] = []
    for m in re.finditer(r"\d+(?:\.\d+)?", text or ""):
        v = float(m.group(0))
        if 16.0 <= v <= 20000.0:
            nums.append(v)
    return nums


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    n          = c["n_notes"]
    raw_midi   = responses["midi"]
    raw_spn    = responses["spn"]
    raw_doremi = responses["doremi"]
    raw_hz     = responses["hz"]

    # MIDI scoring — exact integer match.
    pred_midi = _parse_midi_seq(raw_midi)
    midi_per_pos: list[bool | None] = [
        None if p is None else (gt == p)
        for gt, p in zip(c["midi_sequence"], _pad(pred_midi, n))
    ]

    # SPN scoring — semitone distance == 0.
    pred_spn = _parse_spn_seq(raw_spn)
    spn_per_pos: list[bool | None] = []
    for gt, pred in zip(c["note_sequence"], _pad(pred_spn, n)):
        if pred is None:
            spn_per_pos.append(None)
        else:
            dist = semitone_distance(gt, pred)
            spn_per_pos.append(dist == 0 if dist is not None else False)

    # Doremi scoring — pitch class match (mod-12 distance == 0).
    pred_doremi = _parse_doremi_seq(raw_doremi)
    doremi_per_pos: list[bool | None] = []
    gt_pcs = [m % 12 for m in c["midi_sequence"]]
    for gt_pc, pred_pc in zip(gt_pcs, _pad(pred_doremi, n)):
        if pred_pc is None:
            doremi_per_pos.append(None)
        else:
            diff = abs(gt_pc - pred_pc)
            doremi_per_pos.append(min(diff, 12 - diff) == 0)

    # Hz scoring — ≤1 Hz tolerance (matches standard_pitch_record).
    pred_hz = _parse_hz_seq(raw_hz)
    hz_per_pos: list[bool | None] = [
        None if p is None else (abs(p - gt) <= 1.0)
        for gt, p in zip(c["hz_sequence"], _pad(pred_hz, n))
    ]

    def _n_match(per_pos: list[bool | None]) -> int:
        return sum(1 for v in per_pos if v is True)

    n_midi_match   = _n_match(midi_per_pos)
    n_spn_match    = _n_match(spn_per_pos)
    n_doremi_match = _n_match(doremi_per_pos)
    n_hz_match     = _n_match(hz_per_pos)

    note_scores = {
        "midi": ordered_note_f1(c["midi_sequence"], pred_midi),
        "spn": ordered_note_f1(
            c["note_sequence"], pred_spn,
            lambda a, b: semitone_distance(a, b) == 0,
        ),
        "doremi": ordered_note_f1(gt_pcs, pred_doremi),
        "hz": ordered_note_f1(
            c["hz_sequence"], pred_hz, lambda a, b: abs(a - b) <= 1.0,
        ),
    }

    return {
        **{
            f"{fmt}_note_{metric}": value
            for fmt, scores in note_scores.items()
            for metric, value in scores.items()
        },
        "any_note_f1": max(note_scores[fmt]["f1"] for fmt in ("midi", "spn", "hz")),
        "source":                  c["source"],
        "n_notes":                 n,
        "trial":                   c["trial"],
        "midi_sequence_gt":        str(c["midi_sequence"]),
        "spn_sequence_gt":         str(c["note_sequence"]),
        "doremi_sequence_gt":      str(c["doremi_sequence"]),
        "hz_sequence_gt":          str(c["hz_sequence"]),
        # MIDI
        "midi_pred":               str(pred_midi),
        "midi_per_pos":            str(midi_per_pos),
        "midi_n_pos_match":        n_midi_match,
        "midi_sequence_correct":   int(n_midi_match == n and len(pred_midi) == n),
        # SPN
        "spn_pred":                str(pred_spn),
        "spn_per_pos":             str(spn_per_pos),
        "spn_n_pos_match":         n_spn_match,
        "spn_sequence_correct":    int(n_spn_match == n and len(pred_spn) == n),
        # Doremi
        "doremi_pred":             str(pred_doremi),
        "doremi_per_pos":          str(doremi_per_pos),
        "doremi_n_pos_match":      n_doremi_match,
        "doremi_sequence_correct": int(n_doremi_match == n and len(pred_doremi) == n),
        # Hz
        "hz_pred":                 str(pred_hz),
        "hz_per_pos":              str(hz_per_pos),
        "hz_n_pos_match":          n_hz_match,
        "hz_sequence_correct":     int(n_hz_match == n and len(pred_hz) == n),
        # Any format correct (doremi excluded — pitch-class only, reduced task)
        "any_sequence_correct":    int(
            (n_midi_match == n and len(pred_midi) == n)
            or (n_spn_match == n and len(pred_spn) == n)
            or (n_hz_match == n and len(pred_hz) == n)
        ),
        # Raw
        "raw_midi":                (raw_midi or "").strip(),
        "raw_spn":                 (raw_spn or "").strip(),
        "raw_doremi":              (raw_doremi or "").strip(),
        "raw_hz":                  (raw_hz or "").strip(),
    }


SPEC = CatDSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="sequence_id",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("midi", "spn", "doremi", "hz", "any"),
    metric_suffix="_note_f1",
    metadata_fn=lambda: {"score_name": "ordered_note_f1_lcs", "hz_tolerance_hz": 1.0},
    record_extras=("n_notes", "trial"),
    label_fn=lambda j: (
        f"n={j['cond']['n_notes']} t={j['cond']['trial']} {j['cond']['source']}"
    ),
)


def preview() -> None:
    run_cat_d_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_d_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
