"""
Per-stimulus audit logging.

Every experiment emits one short, self-describing line per stimulus so reviewers
can verify gt / pred / correctness without opening result CSVs. The standard
shape is::

    <label>  gt=<gt>  pred=<pred>  [score=<score>]  ✅|❌

For four-format pitch experiments (a1, a2, a3, a4, a5, b1, b4, e1, e2, e3, z1)
``audit_pitch_record`` collapses the four formats onto a single line so output
stays compact even with thousands of stimuli.
"""

from __future__ import annotations

from typing import Any


def audit_line(
    label: str,
    gt: Any,
    pred: Any,
    *,
    score: float | int | None = None,
    correct: bool | None = None,
) -> str:
    """Build a uniform per-stimulus audit string.

    Always shows ``gt`` and ``pred`` (in whatever shape they share); ``score``
    and ``correct`` are optional.
    """
    parts: list[str] = []
    if label:
        parts.append(label.rstrip())
    parts.append(f"gt={gt!s}")
    parts.append(f"pred={pred!s}")
    if score is not None:
        if isinstance(score, float):
            parts.append(f"score={score:.3f}")
        else:
            parts.append(f"score={score}")
    if correct is not None:
        parts.append("✅" if correct else "❌")
    return "  ".join(parts)


def audit_print(
    label: str,
    gt: Any,
    pred: Any,
    *,
    score: float | int | None = None,
    correct: bool | None = None,
    indent: str = "    ",
) -> None:
    """Print a uniform audit line to stdout."""
    print(indent + audit_line(label, gt, pred, score=score, correct=correct))


_PITCH_FORMATS: tuple[tuple[str, str, str, str], ...] = (
    ("midi",   "midi_gt",   "midi_pred",   "midi_correct"),
    ("spn",    "spn_gt",    "spn_pred",    "spn_correct"),
    ("doremi", "doremi_gt", "doremi_pred", "doremi_correct"),
    ("hz",     "hz_gt",     "hz_pred",     "hz_correct"),
)


def pitch_record_audit_str(record: dict[str, Any], label: str = "") -> str:
    """Return one compact audit line summarising all 4 formats from a pitch record.

    Always includes gt and pred for every format the record carries; an unset
    pred shows as ``None`` rather than being hidden.
    """
    parts: list[str] = [label.rstrip()] if label else []
    for fmt, gk, pk, ck in _PITCH_FORMATS:
        if gk not in record:
            continue
        gt   = record.get(gk)
        pred = record.get(pk)
        ok   = bool(record.get(ck))
        sym  = "✅" if ok else "❌"
        parts.append(f"{fmt}: gt={gt!s} pred={pred!s} {sym}")
    return "  | ".join(parts)


def audit_pitch_record(record: dict[str, Any], label: str = "", indent: str = "    ") -> None:
    """Print one compact audit line per pitch record (all 4 formats)."""
    print(indent + pitch_record_audit_str(record, label))


_SEQ_PITCH_FORMATS: tuple[tuple[str, str, str, str], ...] = (
    ("midi",   "midi_sequence_gt",   "midi_pred",   "midi_sequence_correct"),
    ("abc",    "abc_sequence_gt",    "abc_pred",    "abc_sequence_correct"),
    ("doremi", "doremi_sequence_gt", "doremi_pred", "doremi_sequence_correct"),
    ("hz",     "hz_sequence_gt",     "hz_pred",     "hz_sequence_correct"),
)


def seq_pitch_audit_str(record: dict[str, Any], label: str = "") -> str:
    """One audit line per record for d7-style sequence pitch IDs (4 formats)."""
    parts: list[str] = [label.rstrip()] if label else []
    for fmt, gk, pk, ck in _SEQ_PITCH_FORMATS:
        if gk not in record:
            continue
        gt   = record.get(gk)
        pred = record.get(pk)
        ok   = bool(record.get(ck))
        sym  = "✅" if ok else "❌"
        parts.append(f"{fmt}: gt={gt!s} pred={pred!s} {sym}")
    return "  | ".join(parts)


_MELODY_FORMATS: tuple[tuple[str, str, str, str], ...] = (
    ("midi",   "target_midi_gt",   "midi_pred",   "midi_seq_correct"),
    ("spn",    "target_spn_gt",    "spn_pred",    "spn_seq_correct"),
    ("doremi", "target_doremi_gt", "doremi_pred", "doremi_seq_correct"),
)


def melody_audit_str(record: dict[str, Any], label: str = "") -> str:
    """One audit line for g1/g2-style melody/chorale records (3 formats)."""
    parts: list[str] = [label.rstrip()] if label else []
    for fmt, gk, pk, ck in _MELODY_FORMATS:
        if gk not in record:
            continue
        gt   = record.get(gk)
        pred = record.get(pk)
        ok   = bool(record.get(ck))
        sym  = "✅" if ok else "❌"
        parts.append(f"{fmt}: gt={gt!s} pred={pred!s} {sym}")
    return "  | ".join(parts)
