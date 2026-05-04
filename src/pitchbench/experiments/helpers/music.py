"""Shared music / pitch utilities used across experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

NOTE_NAMES    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
                 "Ab": "G#", "Bb": "A#", "Cb": "B"}

# Solfege syllable and accidental (if needed) -> pitch class index (0=C ... 11=B), fixed-do system.
SOLFEGE_BASE_TO_PC: dict[str, int] = {
    "do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11, "ti": 11,
}

SOLFEGE_TO_PC: dict[str, int] = {
    "do": 0, "do#": 1, "reb": 1,
    "re": 2, "re#": 3, "mib": 3,
    "mi": 4,
    "fa": 5, "fa#": 6, "solb": 6,
    "sol": 7, "sol#": 8, "lab": 8,
    "la": 9, "la#": 10, "sib": 10,
    "si": 11, "ti": 11,
}

PC_TO_SOLFEGE: dict[int, str] = {
    0: "do",  1: "do#", 2: "re",  3: "re#", 4: "mi", 5: "fa",
    6: "fa#", 7: "sol", 8: "sol#", 9: "la", 10: "la#", 11: "si",
}

_ACCIDENTAL_TO_OFFSET = {"#": 1, "♯": 1, "sharp": 1, "b": -1, "♭": -1, "flat": -1}
_SOLFEGE_RE = re.compile(
    r"\b(do|re|mi|fa|sol|la|si|ti)(?:\s*(?:([#♯b♭])|[- ]?(sharp|flat)))?(?![A-Za-z])",
    re.IGNORECASE,
)


# ── MIDI / note name conversion ───────────────────────────────────────────────

def midi_to_note(midi: int) -> str:
    """Convert MIDI number to note name with octave, e.g. 69 → 'A4'."""
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def note_to_midi(note: str) -> int | None:
    """Convert note name (e.g. 'C#4', 'Bb3') to MIDI number, or None if invalid."""
    body = FLAT_TO_SHARP.get(note[:-1], note[:-1])
    if body not in NOTE_NAMES:
        return None
    try:
        return NOTE_NAMES.index(body) + (int(note[-1]) + 1) * 12
    except (ValueError, IndexError):
        return None


def midi_to_freq(midi: int) -> float:
    """Equal-temperament frequency in Hz for a MIDI integer, A4 = 440 Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12)


def semitone_distance(a: str, b: str) -> int | None:
    """Absolute semitone distance between two note names, or None if either is invalid."""
    m1, m2 = note_to_midi(a), note_to_midi(b)
    return abs(m1 - m2) if m1 is not None and m2 is not None else None


# ── Free-text response parsers ────────────────────────────────────────────────

def extract_note(text: str) -> str | None:
    """Parse the first note name (e.g. 'C#4') from a free-text model response.

    Flat marker is preserved as lowercase 'b' (so the dict key ``Bb`` matches);
    only the letter is uppercased.
    """
    m = re.search(r"\b([A-Ga-g][#b♯♭]?\d)\b", text)
    if not m:
        return None
    raw     = m.group(1)
    letter  = raw[0].upper()
    acc     = raw[1:-1].replace("♯", "#").replace("♭", "b")
    octave  = raw[-1]
    name    = letter + acc
    name    = FLAT_TO_SHARP.get(name, name)
    return name + octave


def extract_all_notes(text: str) -> list[str]:
    """Parse all note names in order from a free-text model response."""
    out: list[str] = []
    for m in re.finditer(r"\b([A-Ga-g][#b♯♭]?\d)\b", text):
        raw  = m.group(1)
        name = raw[:-1].upper().replace("♯", "#").replace("♭", "b")
        name = FLAT_TO_SHARP.get(name, name)
        if name in NOTE_NAMES:
            out.append(name + raw[-1])
    return out


def extract_freq(text: str) -> float | None:
    """Parse the first frequency in Hz from a model response.

    Accepts forms like ``440 Hz``, ``440Hz``, ``440 hertz``. If no Hz suffix
    is present, falls back to the first plausible decimal number > 16 Hz so
    that a model that replies just ``440`` still scores.
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:Hz|hz|HZ|hertz)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{2,5}(?:\.\d+)?)", text)
    if m:
        v = float(m.group(1))
        return v if 16.0 <= v <= 20000.0 else None
    return None


def extract_midi(text: str) -> int | None:
    """Parse a bare MIDI integer (0–127) from a model response."""
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        val = int(m.group(1))
        return val if 0 <= val <= 127 else None
    return None


def extract_solfege(text: str) -> int | None:
    """Parse a solfege syllable and accidental (if needed) from text and return pitch class 0-11, or None."""
    match = _SOLFEGE_RE.search(text.strip().lower())
    if not match:
        return None
    base       = match.group(1).lower()
    accidental = match.group(2) or match.group(3)
    pc = SOLFEGE_BASE_TO_PC[base]
    if accidental:
        pc = (pc + _ACCIDENTAL_TO_OFFSET[accidental.lower()]) % 12
    return pc


def extract_all_solfege(text: str) -> list[int]:
    """Parse all solfege syllable and accidental (if needed)s in order and return pitch classes."""
    pcs: list[int] = []
    for match in _SOLFEGE_RE.finditer(text.strip().lower()):
        base       = match.group(1).lower()
        accidental = match.group(2) or match.group(3)
        pc = SOLFEGE_BASE_TO_PC[base]
        if accidental:
            pc = (pc + _ACCIDENTAL_TO_OFFSET[accidental.lower()]) % 12
        pcs.append(pc)
    return pcs


def solfege_pc_to_midi(reference_midi: int, pred_pc: int) -> int:
    """Anchor a predicted pitch class to the nearest MIDI note around a reference."""
    octave_base = reference_midi - (reference_midi % 12)
    base = octave_base + pred_pc
    candidates = [base - 12, base, base + 12]
    return min(candidates, key=lambda midi: abs(midi - reference_midi))


def extract_solfege_midi(text: str, reference_midi: int) -> int | None:
    """Parse a solfege answer and map it to the nearest MIDI around a reference note."""
    pred_pc = extract_solfege(text)
    if pred_pc is None:
        return None
    return solfege_pc_to_midi(reference_midi, pred_pc)


def solfege_midi_distance(reference_midi: int, pred_midi: int) -> int:
    """Absolute semitone distance between the reference MIDI and parsed solfege MIDI."""
    return abs(reference_midi - pred_midi)


def midi_to_solfege(midi: int) -> str:
    """Return the fixed-do solfege syllable and accidental (if needed) for a MIDI note number."""
    return PC_TO_SOLFEGE.get(midi % 12, "?")


def solfege_pc_distance(gt_midi: int, pred_pc: int) -> int:
    """Min semitone distance from ground-truth MIDI to the nearest octave of pred_pc."""
    gt_pc = gt_midi % 12
    diff  = abs(gt_pc - pred_pc)
    return min(diff, 12 - diff)


def note_pc(spn_str: str) -> tuple[str | None, int | None]:
    """Split an SPN note string ('C#4') into (letter+accidental, octave_int).

    Returns ``(None, None)`` if the input cannot be parsed.
    """
    if not spn_str or len(spn_str) < 2:
        return None, None
    head, tail = spn_str[:-1], spn_str[-1]
    head = FLAT_TO_SHARP.get(head, head)
    if head not in NOTE_NAMES:
        return None, None
    try:
        return head, int(tail)
    except ValueError:
        return None, None


# ── Time-string parsing (MM:SS.cc) ────────────────────────────────────────────

_TIME_RE = re.compile(r"(\d{1,2}):(\d{1,2}(?:\.\d+)?)")


def parse_mm_ss_cc(text: str) -> list[float]:
    """Parse comma-separated ``MM:SS.cc`` timestamps into floats (seconds).

    Examples::

        parse_mm_ss_cc("0:01.20, 0:02.50")   → [1.20, 2.50]
        parse_mm_ss_cc("01:23.45")           → [83.45]
        parse_mm_ss_cc("garbage")            → []
    """
    out: list[float] = []
    for m in _TIME_RE.finditer(text):
        minutes = int(m.group(1))
        seconds = float(m.group(2))
        out.append(minutes * 60 + seconds)
    return out


def timing_metrics(
    gt_on: float, gt_off: float,
    pred_on: float | None, pred_off: float | None,
) -> dict[str, float | int | None]:
    """Onset/offset accuracy: abs error, IoU, and ±100/250/500 ms thresholds.

    Returns a dict with::
        abs_error_on, abs_error_off            float seconds
        iou                                    float in [0, 1]
        within_100ms_on / off                  0 / 1
        within_250ms_on / off                  0 / 1
        within_500ms_on / off                  0 / 1
        valid                                  bool — both endpoints parsed

    All threshold metrics are reported per-endpoint so callers can aggregate
    onset and offset separately.
    """
    if pred_on is None or pred_off is None:
        return {"valid": False, "iou": 0.0,
                "abs_error_on": None,  "abs_error_off": None,
                "within_100ms_on": 0,  "within_100ms_off": 0,
                "within_250ms_on": 0,  "within_250ms_off": 0,
                "within_500ms_on": 0,  "within_500ms_off": 0}

    if pred_off < pred_on:                       # tolerate reversed predictions
        pred_on, pred_off = pred_off, pred_on

    abs_on, abs_off = abs(pred_on - gt_on), abs(pred_off - gt_off)
    inter           = max(0.0, min(pred_off, gt_off) - max(pred_on, gt_on))
    union           = max(pred_off, gt_off) - min(pred_on, gt_on)
    iou             = inter / union if union > 0 else 0.0

    return {
        "valid":            True,
        "iou":              round(iou, 4),
        "abs_error_on":     round(abs_on, 4),
        "abs_error_off":    round(abs_off, 4),
        "within_100ms_on":  int(abs_on  <= 0.100),
        "within_100ms_off": int(abs_off <= 0.100),
        "within_250ms_on":  int(abs_on  <= 0.250),
        "within_250ms_off": int(abs_off <= 0.250),
        "within_500ms_on":  int(abs_on  <= 0.500),
        "within_500ms_off": int(abs_off <= 0.500),
    }


# ── Interval naming ──────────────────────────────────────────────────────────

# Semitone count → list of acceptable text aliases.
# Long-form aliases are matched case-insensitively; short letter+digit aliases
# (``m3`` vs ``M3``, ``m7`` vs ``M7``) live in a separate case-sensitive pool
# so the parser does not confuse them.
INTERVAL_NAMES: dict[int, list[str]] = {
    0:  ["unison", "perfect unison"],
    1:  ["minor second",   "minor 2nd", "half step", "halfstep", "semitone"],
    2:  ["major second",   "major 2nd", "whole step", "wholestep", "tone"],
    3:  ["minor third",    "minor 3rd"],
    4:  ["major third",    "major 3rd"],
    5:  ["perfect fourth", "perfect 4th"],
    6:  ["tritone", "augmented fourth", "augmented 4th",
         "diminished fifth", "diminished 5th"],
    7:  ["perfect fifth",  "perfect 5th"],
    8:  ["minor sixth",    "minor 6th", "augmented fifth", "augmented 5th"],
    9:  ["major sixth",    "major 6th", "diminished seventh", "diminished 7th"],
    10: ["minor seventh",  "minor 7th"],
    11: ["major seventh",  "major 7th"],
    12: ["octave", "perfect octave"],
}

# Case-INsensitive long-form aliases (long enough to be unambiguous).
_INTERVAL_LOOKUP_CI: list[tuple[str, int]] = sorted(
    [(alias.lower(), st) for st, names in INTERVAL_NAMES.items() for alias in names],
    key=lambda x: -len(x[0]),
)

# Case-SENSITIVE short aliases — minor/major distinction depends on case.
_INTERVAL_LOOKUP_CS: dict[str, int] = {
    "P1": 0,  "p1": 0,
    "m2": 1,  "M2": 2,
    "m3": 3,  "M3": 4,
    "P4": 5,  "p4": 5,
    "TT": 6,  "tt": 6,
    "A4": 6,  "d5": 6,
    "P5": 7,  "p5": 7,
    "m6": 8,  "M6": 9,
    "m7": 10, "M7": 11,
    "P8": 12, "p8": 12,
}


def extract_interval(text: str) -> int | None:
    """Parse an interval name from text and return its semitone count.

    Long-form aliases (``"perfect fifth"``) are matched case-insensitively and
    longest-first to avoid the ``"minor seventh"`` / ``"minor"`` collision.
    Short letter+digit aliases (``"M7"`` vs ``"m7"``) are matched
    case-sensitively against the original text. Falls back to a bare integer
    in [0, 24].
    """
    low = text.lower()
    for alias, st in _INTERVAL_LOOKUP_CI:
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return st
    for alias, st in _INTERVAL_LOOKUP_CS.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return st
    m = re.search(r"\b(\d{1,2})\b", low)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 24:
            return v
    return None


# ── Chord quality naming ─────────────────────────────────────────────────────

# Canonical quality → list of accepted aliases (lowercase, longest first inside
# the value list — _CHORD_QUALITY_LOOKUP re-sorts globally).
CHORD_QUALITIES: dict[str, list[str]] = {
    "major":      ["major", "maj", "M"],
    "minor":      ["minor", "min", "m"],
    "diminished": ["diminished", "dim", "°"],
    "augmented":  ["augmented",  "aug", "+"],
    "dom7":       ["dominant seventh", "dominant 7th", "dom7", "7th", "dominant"],
    "maj7":       ["major seventh",    "major 7th",    "maj7", "M7"],
    "min7":       ["minor seventh",    "minor 7th",    "min7", "m7"],
    "m7b5":       ["half-diminished",  "half diminished", "m7b5", "ø", "min7b5"],
    "sus2":       ["sus2", "suspended 2"],
    "sus4":       ["sus4", "suspended 4", "suspended"],
}

_CHORD_QUALITY_LOOKUP: list[tuple[str, str]] = sorted(
    [(alias.lower(), q) for q, names in CHORD_QUALITIES.items() for alias in names],
    key=lambda x: -len(x[0]),
)


def extract_chord_quality(text: str) -> str | None:
    """Return the canonical quality name (e.g. ``"major"``) found in text.

    Matching is longest-alias-first and case-insensitive; trailing letters
    are forbidden so ``"majoring"`` does not match ``"major"``. Compact chord
    notation such as ``"Cmaj7"`` is supported (no leading word boundary).
    """
    low = text.lower()
    for alias, q in _CHORD_QUALITY_LOOKUP:
        if re.search(rf"{re.escape(alias)}(?![a-z])", low):
            return q
    return None


# ── Standard prompts ─────────────────────────────────────────────────────────

PROMPT_MIDI = (
    "What is the MIDI note number (an integer from 0 to 127)? "
    "Reply with ONLY the integer. Nothing else. Don't think."
)

PROMPT_SPN = (
    "What is the note name and octave? "
    "Reply with ONLY the note name, for example: C4, F#3, Bb5. Nothing else. Don't think."
)
PROMPT_ABC = PROMPT_SPN   # legacy alias — many older scripts import PROMPT_ABC

PROMPT_DOREMI = (
    "What is the solfege syllable and accidental (if needed) of this pitch? "
    "Use fixed-do (do=C, re=D, mi=E, fa=F, sol=G, la=A, si=B). "
    "Include sharps or flats when needed (e.g. do#, reb, fa#, sib). "
    "Do NOT include the octave. Reply with ONLY the syllable and accidental (if needed). Nothing else. Don't think."
)
PROMPT_SOLFEGE = PROMPT_DOREMI   # alias

PROMPT_HZ = (
    "What is the pitch frequency in Hertz? "
    "Reply with ONLY a number (the frequency in Hz). Nothing else. Don't think."
)


# ── Standard pitch record (4-format scoring) ─────────────────────────────────

def standard_pitch_record(
    *,
    wav: str | Path,
    source: str,
    source_type: str,
    midi_gt: int,
    raw_midi:   str,
    raw_spn:    str | None = None,
    raw_doremi: str,
    raw_hz:     str | None = None,
    prompt_midi:   str,
    prompt_spn:    str | None = None,
    prompt_doremi: str,
    prompt_hz:     str | None = None,
    # legacy aliases — kept for migration callers; ignored if raw_spn is set
    raw_abc:    str | None = None,
    prompt_abc: str | None = None,
    **extra_meta: Any,
) -> dict[str, Any]:
    """Build a standardised CSV row for single-pitch experiments (4 formats).

    Scoring columns (per the PitchBench v2 plan):

    | Column                 | Definition                                       |
    | midi_correct           | predicted MIDI int == GT                         |
    | midi_pc_correct        | predicted MIDI mod 12 == GT mod 12               |
    | midi_octave_correct    | predicted MIDI // 12 == GT // 12                 |
    | midi_within_1          | |pred − GT| ≤ 1 semitone                         |
    | spn_correct            | predicted SPN matches GT (full note + octave)    |
    | spn_pc_correct         | predicted note letter+accidental matches GT      |
    | spn_octave_correct     | predicted octave digit matches GT                |
    | doremi_correct         | predicted solfège pitch class matches GT mod 12  |
    | hz_correct             | |pred Hz − GT Hz| ≤ 1.0 Hz                       |
    | hz_abs_error           | |pred Hz − GT Hz|                                |

    The ``raw_abc`` / ``prompt_abc`` kwargs are accepted as drop-in aliases for
    ``raw_spn`` / ``prompt_spn`` so existing scripts that haven't migrated to
    the SPN naming yet still work.
    """
    # accept legacy alias names
    if raw_spn    is None: raw_spn    = raw_abc    or ""
    if prompt_spn is None: prompt_spn = prompt_abc or ""
    if raw_hz     is None: raw_hz     = ""
    if prompt_hz  is None: prompt_hz  = ""

    wav           = str(wav)
    note_gt       = midi_to_note(midi_gt)
    gt_letter, gt_octave = note_pc(note_gt)
    doremi_gt_str = PC_TO_SOLFEGE.get(midi_gt % 12, "?")
    hz_gt         = midi_to_freq(midi_gt)

    # ── MIDI ──────────────────────────────────────────────────────────────────
    midi_pred = extract_midi(raw_midi)
    midi_err  = abs(midi_pred - midi_gt) if midi_pred is not None else None

    # ── SPN ───────────────────────────────────────────────────────────────────
    spn_pred                 = extract_note(raw_spn) if raw_spn else None
    spn_letter, spn_octave   = note_pc(spn_pred) if spn_pred else (None, None)

    # ── Doremi ────────────────────────────────────────────────────────────────
    doremi_pred_pc  = extract_solfege(raw_doremi)
    doremi_pred_str = PC_TO_SOLFEGE.get(doremi_pred_pc) if doremi_pred_pc is not None else None
    doremi_dist     = solfege_pc_distance(midi_gt, doremi_pred_pc) if doremi_pred_pc is not None else None

    # ── Hz ────────────────────────────────────────────────────────────────────
    hz_pred  = extract_freq(raw_hz) if raw_hz else None
    hz_abs   = abs(hz_pred - hz_gt) if hz_pred is not None else None

    return {
        **extra_meta,
        "source":          source,
        "source_type":     source_type,
        # ── MIDI format ───────────────────────────────────────────────────────
        "midi_gt":             midi_gt,
        "midi_pred":           midi_pred,
        "midi_correct":        int(midi_err == 0) if midi_err is not None else 0,
        "midi_within_1":       int(midi_err is not None and midi_err <= 1),
        "midi_pc_correct":     int(midi_pred is not None and midi_pred % 12 == midi_gt % 12),
        "midi_octave_correct": int(midi_pred is not None and midi_pred // 12 == midi_gt // 12),
        # ── SPN format ────────────────────────────────────────────────────────
        "spn_gt":              note_gt,
        "spn_pred":            spn_pred,
        "spn_correct":         int(spn_pred == note_gt),
        "spn_pc_correct":      int(spn_letter is not None and spn_letter == gt_letter),
        "spn_octave_correct":  int(spn_octave is not None and spn_octave == gt_octave),
        # ── Doremi format ─────────────────────────────────────────────────────
        "doremi_gt":           doremi_gt_str,
        "doremi_pred":         doremi_pred_str,
        "doremi_correct":      int(doremi_dist == 0) if doremi_dist is not None else 0,
        # ── Hz format ─────────────────────────────────────────────────────────
        "hz_gt":               round(hz_gt, 4),
        "hz_pred":             hz_pred,
        "hz_correct":          int(hz_abs is not None and hz_abs <= 1.0),
        "hz_abs_error":        round(hz_abs, 4) if hz_abs is not None else None,
        # ── Legacy aliases (kept so existing plot helpers/aggregations work) ──
        "abc_gt":              note_gt,
        "abc_pred":            spn_pred,
        "abc_correct":         int(spn_pred == note_gt),
        # ── Raw responses ─────────────────────────────────────────────────────
        "raw_midi":            (raw_midi or "").strip(),
        "raw_spn":             (raw_spn or "").strip(),
        "raw_doremi":          (raw_doremi or "").strip(),
        "raw_hz":              (raw_hz or "").strip(),
        # ── Audio path ────────────────────────────────────────────────────────
        "wav":                 wav,
        # ── Prompts (long strings, pushed to end of CSV) ──────────────────────
        "prompt_midi":         prompt_midi,
        "prompt_spn":          prompt_spn,
        "prompt_doremi":       prompt_doremi,
        "prompt_hz":           prompt_hz,
    }


def wide_to_long_records(
    records: list[dict[str, Any]],
    formats: tuple[str, ...] = ("midi", "spn", "doremi", "hz"),
) -> list[dict[str, Any]]:
    """Expand wide pitch records (one row per audio) to long format (one row per format).

    Used to feed the existing plot helpers that expect prompt_variant / exact_match.
    The default formats list now includes ``hz``; pass an explicit subset (e.g.
    ``("midi", "spn", "doremi")``) when an experiment didn't query Hz.
    """
    long: list[dict[str, Any]] = []
    for r in records:
        for fmt in formats:
            if f"{fmt}_correct" not in r:
                continue
            long.append({
                **r,
                "prompt_variant": fmt,
                "exact_match":    r[f"{fmt}_correct"],
                "within_1":       r.get(f"{fmt}_within_1", r[f"{fmt}_correct"]),
                "instrument":     r.get("source", r.get("instrument", "")),
            })
    return long
