"""Shared music / pitch utilities used across experiments."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

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


POINT_PITCH_FORMATS: tuple[str, ...] = ("midi", "spn", "doremi", "hz", "any")


def any_format_correct(values: Iterable[Any]) -> int:
    """Return 1 iff any provided format score counts as correct."""
    for value in values:
        if isinstance(value, (int, float, bool)) and bool(value):
            return 1
    return 0


def format_accuracy_dict(
    records: list[dict[str, Any]],
    metric_map: dict[str, str],
    *,
    include_all: bool = True,
) -> dict[str, float]:
    """Average the binary correctness columns listed in ``metric_map``.

    When ``include_all`` is true, also emit ``any`` = fraction of records for
    which any listed format was correct.
    """
    out: dict[str, float] = {}
    present_cols: list[str] = []
    for fmt, col in metric_map.items():
        vals = [r[col] for r in records if isinstance(r.get(col), (int, float, bool))]
        out[fmt] = round(sum(vals) / len(vals), 4) if vals else 0.0
        if vals:
            present_cols.append(col)
    if include_all and present_cols:
        # doremi is a reduced/pitch-class task — excluded from 'any'
        any_cols = [col for col in present_cols if not col.startswith("doremi")]
        any_vals = [
            any_format_correct(r.get(col) for col in any_cols)
            for r in records
        ]
        out["any"] = round(sum(any_vals) / len(any_vals), 4) if any_vals else 0.0
    return out


def extract_midi(text: str) -> int | None:
    """Parse a bare MIDI integer (0–127) from a model response."""
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        val = int(m.group(1))
        return val if 0 <= val <= 127 else None
    return None


def extract_solfege(text: str) -> int | None:
    """Parse a solfege syllable and accidental (if needed) from text and return pitch class 0-11, or None.

    Falls back to a cheap OpenRouter LLM (``config.DOREMI_PARSER_MODEL``) when
    the regex does not match — only fires if ``OPENROUTER_KEY`` is set and
    ``PITCHBENCH_DOREMI_LLM`` is not ``"0"``. Returns ``None`` if the LLM
    fallback is unavailable or fails.
    """
    match = _SOLFEGE_RE.search(text.strip().lower())
    if match:
        base       = match.group(1).lower()
        accidental = match.group(2) or match.group(3)
        pc = SOLFEGE_BASE_TO_PC[base]
        if accidental:
            pc = (pc + _ACCIDENTAL_TO_OFFSET[accidental.lower()]) % 12
        return pc
    if _doremi_llm_enabled():
        return _doremi_llm_parse_one(text)
    return None


def extract_all_solfege(text: str) -> list[int]:
    """Parse all solfege syllable and accidentals (if needed) in order and return pitch classes.

    Falls back to a cheap OpenRouter LLM (``config.DOREMI_PARSER_MODEL``) when
    the regex matches no syllables; same gating as :func:`extract_solfege`.
    """
    pcs: list[int] = []
    for match in _SOLFEGE_RE.finditer(text.strip().lower()):
        base       = match.group(1).lower()
        accidental = match.group(2) or match.group(3)
        pc = SOLFEGE_BASE_TO_PC[base]
        if accidental:
            pc = (pc + _ACCIDENTAL_TO_OFFSET[accidental.lower()]) % 12
        pcs.append(pc)
    if not pcs and _doremi_llm_enabled():
        return _doremi_llm_parse_many(text)
    return pcs


# ── LLM fallback for solfège parsing ──────────────────────────────────────────

def _doremi_llm_enabled() -> bool:
    if os.environ.get("PITCHBENCH_DOREMI_LLM", "1") == "0":
        return False
    return bool(
        os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    )


@lru_cache(maxsize=2048)
def _doremi_llm_call(text: str, expect_many: bool) -> str | None:
    """Single OpenRouter call to parse a solfège answer. Returns content string or None."""
    print("asking llm for solfege parse of:", repr(text))
    try:
        import requests
    except ImportError:
        return None
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    import pitchbench.config as config
    model_id = config.DOREMI_PARSER_MODEL
    if model_id.startswith("openrouter/"):
        model_id = model_id[len("openrouter/"):]

    if expect_many:
        instruction = (
            'You parse solfège answers. Given the user message (a model response '
            'naming one or more solfège syllables in fixed-do), reply with a '
            'compact JSON object: {"pcs": [<int>, ...]} where each integer is a '
            'pitch class 0..11 with do=0, do#/reb=1, re=2, re#/mib=3, mi=4, fa=5, '
            'fa#/solb=6, sol=7, sol#/lab=8, la=9, la#/sib=10, si/ti=11. Order them '
            'as they appear. If no solfège is present, return {"pcs": []}. '
            'Reply with ONLY the JSON object, nothing else.'
        )
    else:
        instruction = (
            'You parse solfège answers. Given the user message (a model response '
            'naming a single solfège syllable in fixed-do), reply with a compact '
            'JSON object: {"pc": <int|null>} where the integer is a pitch class '
            '0..11 with do=0, do#/reb=1, re=2, re#/mib=3, mi=4, fa=5, fa#/solb=6, '
            'sol=7, sol#/lab=8, la=9, la#/sib=10, si/ti=11. If no solfège is '
            'present, return {"pc": null}. The solfège notation has to be in strict format (syllable with accidental, not more, not less). Reply with ONLY the JSON object, '
            'nothing else.'
        )

    body = {
        "model":       model_id,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user",   "content": text},
        ],
        "max_tokens":      64,
        "temperature":     0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/vaclis-CNMAT/PitchBench",
        "X-Title":       "PitchBench",
    }
    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            json=body, headers=headers, timeout=30,
        )
        if not resp.ok:
            return None
        data    = resp.json()
        choice  = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content
    except Exception:
        return None


def _doremi_llm_parse_one(text: str) -> int | None:
    raw = _doremi_llm_call(text, expect_many=False)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    v = obj.get("pc") if isinstance(obj, dict) else None
    if isinstance(v, int) and 0 <= v <= 11:
        return v
    return None


def _doremi_llm_parse_many(text: str) -> list[int]:
    raw = _doremi_llm_call(text, expect_many=True)
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    seq = obj.get("pcs") if isinstance(obj, dict) else None
    if not isinstance(seq, list):
        return []
    return [int(x) for x in seq if isinstance(x, int) and 0 <= x <= 11]


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


# ── Time-string parsing ───────────────────────────────────────────────────────

# MM:SS.cc or MM:SS (colon-separated)
_RE_COLON   = re.compile(r"(\d{1,2}):(\d{1,2}(?:\.\d+)?)")
# MM.SS.cc (dot-separated, e.g. "0.05.20" → 5.20 s, "01.23.45" → 83.45 s)
# Requires exactly 3 dot-separated parts to avoid clashing with bare floats.
_RE_DOT     = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d+)")
# Bare seconds with an explicit 's' suffix (e.g. "5.2s", "83s")
_RE_S_SUFF  = re.compile(r"(\d+(?:\.\d+)?)s\b", re.IGNORECASE)


def parse_mm_ss_cc(text: str) -> list[float]:
    """Extract timestamps from text, accepting several formats (seconds-based result).

    Tried in priority order — first pattern that yields at least one hit wins:

    1. ``MM:SS.cc`` / ``MM:SS``   e.g. ``0:05.20``, ``1:23``
    2. ``MM.SS.cc``               e.g. ``0.05.20`` → 5.20 s, ``01.23.45`` → 83.45 s
    3. ``XXXs`` / ``XX.Xs``       e.g. ``5.2s``, ``83s``
    4. Bare number(s)             last resort — only if the cleaned text parses
                                  entirely as space/comma-separated floats

    Examples::

        parse_mm_ss_cc("0:01.20, 0:02.50")  → [1.20, 2.50]
        parse_mm_ss_cc("01:23.45")          → [83.45]
        parse_mm_ss_cc("0.05.20, 0.08.50")  → [5.20, 8.50]
        parse_mm_ss_cc("5.2s, 8.5s")        → [5.2, 8.5]
        parse_mm_ss_cc("5.2, 8.5")          → [5.2, 8.5]
        parse_mm_ss_cc("garbage")           → []
    """
    # 1) MM:SS.cc or MM:SS
    out = [int(m.group(1)) * 60 + float(m.group(2))
           for m in _RE_COLON.finditer(text)]
    if out:
        return out

    # 2) MM.SS.cc (dot-separated triple)
    out = [int(m.group(1)) * 60 + int(m.group(2)) + float("0." + m.group(3))
           for m in _RE_DOT.finditer(text)]
    if out:
        return out

    # 3) Bare seconds with 's' suffix
    out = [float(m.group(1)) for m in _RE_S_SUFF.finditer(text)]
    if out:
        return out

    # 4) Bare number fallback — only if the entire content is numbers + separators
    stripped = re.sub(r"[^\d.,\s]", "", text).strip()
    if stripped:
        try:
            floats = [float(p) for p in re.split(r"[\s,]+", stripped) if p]
            if floats:
                return floats
        except ValueError:
            pass

    return []


def timing_metrics(
    gt_on: float, gt_off: float,
    pred_on: float | None, pred_off: float | None,
) -> dict[str, float | int | None]:
    """Onset/offset accuracy: abs error, IoU, the primary ``correct`` flag,
    plus fixed ±100 / 250 / 500 ms diagnostic thresholds.

    The primary ``correct`` field is binary: 1 iff BOTH endpoints are within
    ±``config.BENCHMARK_TIMESTAMP_TOLERANCE_MS`` ms of GT (default 250 ms).
    The ``within_{100,250,500}ms_both`` fields stay fixed at their named
    thresholds so cross-tolerance comparisons keep working even if the global
    tolerance is changed.

    Returns a dict with::
        abs_error_on, abs_error_off            float seconds
        iou                                    float in [0, 1]
        correct                                0 / 1 — at the configured tolerance
        within_100ms_both                      0 / 1 — fixed diagnostic
        within_250ms_both                      0 / 1 — fixed diagnostic
        within_500ms_both                      0 / 1 — fixed diagnostic
        valid                                  bool — both endpoints parsed
    """
    # Lazy import to avoid a circular dependency on config at module load.
    import pitchbench.config as _config
    tol_s = _config.BENCHMARK_TIMESTAMP_TOLERANCE_MS / 1000.0

    if pred_on is None or pred_off is None:
        return {
            "valid": False, "iou": 0.0,
            "abs_error_on": None, "abs_error_off": None,
            "correct": 0,
            "within_100ms_both": 0,
            "within_250ms_both": 0,
            "within_500ms_both": 0,
        }

    if pred_off < pred_on:                       # tolerate reversed predictions
        pred_on, pred_off = pred_off, pred_on

    abs_on, abs_off = abs(pred_on - gt_on), abs(pred_off - gt_off)
    inter           = max(0.0, min(pred_off, gt_off) - max(pred_on, gt_on))
    union           = max(pred_off, gt_off) - min(pred_on, gt_on)
    iou             = inter / union if union > 0 else 0.0

    return {
        "valid":             True,
        "iou":               round(iou, 4),
        "abs_error_on":      round(abs_on, 4),
        "abs_error_off":     round(abs_off, 4),
        "correct":           int(abs_on <= tol_s and abs_off <= tol_s),
        "within_100ms_both": int(abs_on <= 0.100 and abs_off <= 0.100),
        "within_250ms_both": int(abs_on <= 0.250 and abs_off <= 0.250),
        "within_500ms_both": int(abs_on <= 0.500 and abs_off <= 0.500),
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
    "Reply with ONLY the integer. Nothing else. Output only the answer."
)

PROMPT_SPN = (
    "What is the note name and octave? "
    "Reply with ONLY the note name in Scientific Pitch Notation (e.g. C4, F#3, Bb5). Nothing else. Output only the answer."
)
PROMPT_ABC = PROMPT_SPN   # legacy alias — many older scripts import PROMPT_ABC

PROMPT_DOREMI = (
    "What is the solfege syllable and accidental (if needed) of this pitch? "
    "Use fixed-do (do=C, re=D, mi=E, fa=F, sol=G, la=A, si=B). "
    "Include sharps or flats when needed (e.g. do#, reb, fa#, sib). "
    "Do NOT include the octave. Reply with ONLY the syllable and accidental (if needed). Nothing else. Output only the answer."
)
PROMPT_SOLFEGE = PROMPT_DOREMI   # alias

PROMPT_HZ = (
    "What is the main pitch frequency, expressed in Hertz? "
    "Reply with ONLY a number (the frequency in Hz). Nothing else. Output only the answer."
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
    | hz_correct             | 0.99 ≤ pred / gt ≤ 1.01  (binary, ±1% ratio)     |
    | hz_abs_error           | |pred Hz − GT Hz|                                |
    | hz_ratio               | pred / gt                                         |

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
    hz_pred   = extract_freq(raw_hz) if raw_hz else None
    hz_abs    = abs(hz_pred - hz_gt) if hz_pred is not None else None
    hz_ratio  = (hz_pred / hz_gt) if (hz_pred is not None and hz_gt > 0) else None
    hz_ok     = int(hz_ratio is not None and 0.99 <= hz_ratio <= 1.01)
    midi_ok   = int(midi_err == 0) if midi_err is not None else 0
    spn_ok    = int(spn_pred == note_gt)
    doremi_ok = int(doremi_dist == 0) if doremi_dist is not None else 0
    any_ok    = any_format_correct((midi_ok, spn_ok, hz_ok))

    return {
        **extra_meta,
        "source":          source,
        "source_type":     source_type,
        # ── MIDI format ───────────────────────────────────────────────────────
        "midi_gt":             midi_gt,
        "midi_pred":           midi_pred,
        "midi_correct":        midi_ok,
        "midi_within_1":       int(midi_err is not None and midi_err <= 1),
        "midi_pc_correct":     int(midi_pred is not None and midi_pred % 12 == midi_gt % 12),
        "midi_octave_correct": int(midi_pred is not None and midi_pred // 12 == midi_gt // 12),
        # ── SPN format ────────────────────────────────────────────────────────
        "spn_gt":              note_gt,
        "spn_pred":            spn_pred,
        "spn_correct":         spn_ok,
        "spn_pc_correct":      int(spn_letter is not None and spn_letter == gt_letter),
        "spn_octave_correct":  int(spn_octave is not None and spn_octave == gt_octave),
        # ── Doremi format ─────────────────────────────────────────────────────
        "doremi_gt":           doremi_gt_str,
        "doremi_pred":         doremi_pred_str,
        "doremi_correct":      doremi_ok,
        # ── Hz format ─────────────────────────────────────────────────────────
        # hz_correct is binary: pred / gt within [0.99, 1.01] (±1% tolerance).
        # The legacy ≤1Hz absolute-error rule was scale-dependent (lenient at
        # high pitches, strict at low) — the ratio is scale-invariant and
        # roughly equivalent to ±17 cents.
        "hz_gt":               round(hz_gt, 4),
        "hz_pred":             hz_pred,
        "hz_correct":          hz_ok,
        "any_correct":         any_ok,
        "hz_abs_error":        round(hz_abs, 4) if hz_abs is not None else None,
        "hz_ratio":            round(hz_ratio, 6) if hz_ratio is not None else None,
        # ── Legacy aliases (kept so existing plot helpers/aggregations work) ──
        "abc_gt":              note_gt,
        "abc_pred":            spn_pred,
        "abc_correct":         spn_ok,
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
