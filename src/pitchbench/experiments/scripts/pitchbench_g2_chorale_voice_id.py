"""
Experiment g2 — Voice identification in Bach chorales (music21 corpus)

Same task as g1 (identify the x-th voice in a polyphonic mix) but using real
Bach chorales sourced from the music21 corpus instead of synthetic random
material.

For each (chorale, x) pair, the audio is the longest contiguous segment of
the chorale during which the target voice maintains its register rank — i.e.
exactly (x-1) voices are above it at every moment in the segment.  Within
this span, the target voice never crosses with any other voice; the audio
runs for as long as that condition holds.  Other voices may cross each other
freely, so the segment length depends on which voice is selected.

Independent variables:
  chorale_id       : Bach chorale corpus IDs (default: 6 chorales)
  x                : 1..4  (1=soprano · 2=alto · 3=tenor · 4=bass)
  instrument_config: "similar" – all 4 voices on the same source
                                 (iterated over all GM v1 instruments)
                     "mixed"   – each voice on a different source
                                 (iterated over MIXED_GROUPS)

Audio length is variable; rendered at the chorale's metronome mark
(default 90 qpm if none is specified) and capped at MAX_SEG_SEC.

Prompt formats: MIDI · SPN · Solfège.

Requires the ``music21`` package (``uv add music21`` or ``pip install music21``).

Usage:
    pitchbench --id g2 --preview
    pitchbench --id g2
    pitchbench --id g2 --chorales bach/bwv66.6 bach/bwv4.8
"""

import argparse
import re
from pathlib import Path
from typing import Any

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.audit import melody_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    PC_TO_SOLFEGE,
    extract_all_notes,
    extract_all_solfege,
    midi_to_note,
    semitone_distance,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata,
    make_run_dir,
    save_comparison,
    save_results,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

# Data-generation parameters (sourced from config.pitchbench_g2_*)
N_VOICES         = config.pitchbench_g2_N_VOICES
DEFAULT_SEED     = config.pitchbench_g2_SEED
MIN_SEG_NOTES    = config.pitchbench_g2_MIN_SEG_NOTES
MAX_SEG_SEC      = config.pitchbench_g2_MAX_SEG_SEC
DEFAULT_QPM      = config.pitchbench_g2_QPM
VOICE_NAMES      = config.pitchbench_g2_VOICE_NAMES
DEFAULT_CHORALES = config.pitchbench_g2_DEFAULT_CHORALES
SOURCES          = config.pitchbench_g2_SOURCES
MIXED_GROUPS     = config.pitchbench_g2_MIXED_GROUPS


# ── music21 helpers ───────────────────────────────────────────────────────────

def _import_music21():
    try:
        import music21
        return music21
    except ImportError:
        raise SystemExit(
            "music21 is required for experiment g2.\n"
            "Install with:  uv add music21   (or  pip install music21)"
        )


def _voice_to_events(stream_obj) -> list[tuple[float, int, float]]:
    """Convert a music21 Stream/Voice/Part to ``(offset_qL, midi, dur_qL)``
    triples.  Chorale parts are nominally monophonic; chords (rare) collapse
    to their top pitch."""
    m21 = _import_music21()
    events: list[tuple[float, int, float]] = []
    for n in stream_obj.flat.notes:
        if isinstance(n, m21.note.Note):
            events.append((float(n.offset), int(n.pitch.midi), float(n.quarterLength)))
        elif isinstance(n, m21.chord.Chord):
            top = max(p.midi for p in n.pitches)
            events.append((float(n.offset), int(top), float(n.quarterLength)))
    events.sort(key=lambda e: e[0])
    return events


def _extract_voices(piece) -> list[list[tuple[float, int, float]]]:
    """Return one event list per voice.  Handles both 4-staff and 2-staff
    (treble + bass with sub-voices) chorale layouts."""
    voices: list[list[tuple[float, int, float]]] = []
    for part in piece.parts:
        if part.hasVoices():
            for v in part.voices:
                voices.append(_voice_to_events(v))
        else:
            voices.append(_voice_to_events(part))
    return voices


def _sort_by_register(
    voices: list[list[tuple[float, int, float]]],
) -> list[list[tuple[float, int, float]]]:
    """Sort voices by mean MIDI pitch (descending) so index 0 = soprano."""
    def mean_pitch(events: list[tuple[float, int, float]]) -> float:
        if not events:
            return -1.0
        return sum(p for _, p, _ in events) / len(events)
    return sorted(voices, key=mean_pitch, reverse=True)


def _get_qpm(piece) -> float:
    # try:
    #     for tempo in piece.flat.getElementsByClass("MetronomeMark"):
    #         if tempo.number is not None:
    #             return float(tempo.number)
    # except Exception:
    #     pass
    return DEFAULT_QPM


def _voice_pitch_at(events: list[tuple[float, int, float]], t: float) -> int | None:
    """MIDI pitch sounding at quarter-length time ``t``, or None during a rest."""
    for onset, pitch, dur in events:
        if onset <= t < onset + dur:
            return pitch
        if onset > t:
            return None
    return None


def _time_grid(voices: list[list[tuple[float, int, float]]]) -> list[float]:
    """Sorted union of all onset/offset times across all voices."""
    times: set[float] = set()
    for events in voices:
        for onset, _, dur in events:
            times.add(float(onset))
            times.add(float(onset + dur))
    return sorted(times)


def _longest_no_cross_segment(
    voices: list[list[tuple[float, int, float]]], x: int,
) -> tuple[float, float] | None:
    """Find the longest contiguous span where voice ``x-1`` holds rank ``x``
    among the four voices (1 = highest pitch, 4 = lowest)."""
    grid = _time_grid(voices)
    if len(grid) < 2:
        return None

    best_len  = 0.0
    best: tuple[float, float] | None = None
    cur_start: float | None = None

    for i in range(len(grid) - 1):
        t = grid[i]
        pitches = [_voice_pitch_at(events, t) for events in voices]
        target  = pitches[x - 1]
        ok = (
            all(p is not None for p in pitches)
            and target is not None
            and pitches.count(target) == 1
            and (sum(1 for p in pitches if p is not None and p > target) + 1) == x
        )
        if ok:
            if cur_start is None:
                cur_start = grid[i]
            seg_end = grid[i + 1]
            seg_len = seg_end - cur_start
            if seg_len > best_len:
                best_len = seg_len
                best = (cur_start, seg_end)
        else:
            cur_start = None
    return best


def _segment_voice_notes(
    events: list[tuple[float, int, float]],
    start_qL: float,
    end_qL: float,
    qpm: float,
) -> list[tuple[int, int, int]]:
    """Clip a voice's events to ``[start_qL, end_qL)`` and convert to
    ``(midi, onset_ms, dur_ms)`` triples relative to the segment start."""
    sec_per_qL = 60.0 / qpm
    out: list[tuple[int, int, int]] = []
    for onset, pitch, dur in events:
        s = max(onset, start_qL)
        e = min(onset + dur, end_qL)
        if e <= s:
            continue
        out.append((
            int(pitch),
            int(round((s - start_qL) * sec_per_qL * 1000)),
            int(round((e - s)        * sec_per_qL * 1000)),
        ))
    out.sort(key=lambda n: n[1])
    return out


# ── Condition construction ────────────────────────────────────────────────────

def _instrument_choices() -> list[tuple[str, str, list[str]]]:
    out: list[tuple[str, str, list[str]]] = []
    for src in SOURCES:
        out.append(("similar", src, [src] * N_VOICES))
    for label, group in MIXED_GROUPS.items():
        if len(group) >= N_VOICES:
            out.append(("mixed", label, group[:N_VOICES]))
    return out


def _slug(chorale_id: str) -> str:
    return chorale_id.replace("/", "_").replace(".", "_")


def build_conditions(chorales: list[str], seed: int) -> list[dict]:
    m21 = _import_music21()
    rows: list[dict] = []

    for chorale_id in chorales:
        try:
            piece = m21.corpus.parse(chorale_id)
        except Exception as exc:
            print(f"  [SKIP] {chorale_id}: parse failed ({exc})")
            continue

        voices = _sort_by_register(_extract_voices(piece))
        if len(voices) != N_VOICES:
            print(f"  [SKIP] {chorale_id}: expected {N_VOICES} voices, got {len(voices)}")
            continue
        if any(not v for v in voices):
            print(f"  [SKIP] {chorale_id}: at least one voice has no notes")
            continue

        qpm        = _get_qpm(piece)
        sec_per_qL = 60.0 / qpm

        for x in range(1, N_VOICES + 1):
            seg = _longest_no_cross_segment(voices, x)
            if seg is None:
                print(f"  [SKIP] {chorale_id} x={x}: no non-crossing segment found")
                continue
            start_qL, end_qL = seg

            # Cap the audio length without breaking the no-cross guarantee
            # (we always cut from the end, never the start).
            seg_dur_sec = (end_qL - start_qL) * sec_per_qL
            if seg_dur_sec > MAX_SEG_SEC:
                end_qL = start_qL + MAX_SEG_SEC / sec_per_qL

            voice_notes = [
                _segment_voice_notes(events, start_qL, end_qL, qpm) for events in voices
            ]

            n_target = len(voice_notes[x - 1])
            if n_target < MIN_SEG_NOTES:
                print(f"  [SKIP] {chorale_id} x={x}: target voice has {n_target} notes")
                continue

            target_pitches = [m for m, _, _ in voice_notes[x - 1]]
            total_ms       = int(round((end_qL - start_qL) * sec_per_qL * 1000))

            for inst_cfg, source_label, sources in _instrument_choices():
                rows.append({
                    "chorale_id":     chorale_id,
                    "chorale_slug":   _slug(chorale_id),
                    "x":              x,
                    "voice_name":     VOICE_NAMES[x - 1],
                    "qpm":            qpm,
                    "start_qL":       start_qL,
                    "end_qL":         end_qL,
                    "total_ms":       total_ms,
                    "inst_cfg":       inst_cfg,
                    "source_label":   source_label,
                    "sources":        sources,
                    "all_notes":      voice_notes,
                    "n_target":       n_target,
                    "all_n_notes":    [len(v) for v in voice_notes],
                    "target_pitches": target_pitches,
                })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        lines = list(zip(c["all_notes"], c["sources"]))
        hint  = f"{c['chorale_slug']}_x{c['x']}_{c['inst_cfg']}_{c['source_label']}"
        try:
            engine.polyphonic_mix(lines, c["total_ms"], name_hint=hint)
        except ValueError as exc:
            print(f"  [SKIP] {exc}")


# ── Prompts ───────────────────────────────────────────────────────────────────

def _instrument_list_str(sources: list[str]) -> str:
    parts = []
    for i, src in enumerate(sources):
        parts.append(f"{src} ({VOICE_NAMES[i]})")
    return ", ".join(parts)


def _preamble(x: int, n_target: int, inst_cfg: str, sources: list[str]) -> str:
    voice = VOICE_NAMES[x - 1]
    if inst_cfg == "similar":
        timbre = "all four voices played on the same instrument"
    else:
        timbre = (
            f"each voice played by a different instrument"
        )
    return (
        f"You will hear an excerpt from a Bach four-part chorale (soprano, alto, "
        f"tenor, bass), {timbre}. The voices do not cross with respect to the "
        f"{voice}: it stays in its register the whole time. "
        f"Focus only on the {voice} voice. It plays exactly {n_target} notes."
    )


def make_prompt_midi(x: int, n_target: int, inst_cfg: str, sources: list[str]) -> str:
    return (
        f"{_preamble(x, n_target, inst_cfg, sources)} "
        f"List all {n_target} MIDI note numbers in order from first to last. "
        "Reply with ONLY the integers separated by spaces. Nothing else. Output only the answer."
    )


def make_prompt_spn(x: int, n_target: int, inst_cfg: str, sources: list[str]) -> str:
    return (
        f"{_preamble(x, n_target, inst_cfg, sources)} "
        f"List all {n_target} note names in order from first to last. "
        "Reply with ONLY the note names expressed in Scientific Pitch Notation, separated by spaces (e.g. C5 D#5 E5). Nothing else. Output only the answer."
    )


def make_prompt_doremi(x: int, n_target: int, inst_cfg: str, sources: list[str]) -> str:
    return (
        f"{_preamble(x, n_target, inst_cfg, sources)} "
        f"List all {n_target} solfège syllable and accidentals (if needed) in order from first to last "
        "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B; include sharps e.g. do# re#). "
        "Reply with ONLY the syllable and accidentals (if needed) separated by spaces. Nothing else. Output only the answer."
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_midi_seq(text: str, n: int) -> list[int | None]:
    nums = [int(m) for m in re.findall(r"\b(\d{1,3})\b", text) if 0 <= int(m) <= 127]
    out: list[int | None] = list(nums[:n])
    while len(out) < n:
        out.append(None)
    return out


def _parse_spn_seq(text: str, n: int) -> list[str | None]:
    tokens = extract_all_notes(text)
    out: list[str | None] = list(tokens[:n])
    while len(out) < n:
        out.append(None)
    return out


def _parse_doremi_seq(text: str, n: int) -> list[int | None]:
    pcs = extract_all_solfege(text)
    out: list[int | None] = list(pcs[:n])
    while len(out) < n:
        out.append(None)
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_midi(gt: list[int], pred: list[int | None]) -> tuple[list[bool | None], int]:
    n = len(gt)
    per_pos = [(gt[i] == pred[i] if pred[i] is not None else None) for i in range(n)]
    return per_pos, int(all(v is True for v in per_pos))


def _score_spn(gt_midi: list[int], pred: list[str | None]) -> tuple[list[bool | None], int]:
    gt_spn = [midi_to_note(m) for m in gt_midi]
    per_pos: list[bool | None] = []
    for gt, pr in zip(gt_spn, pred):
        if pr is None:
            per_pos.append(None)
        else:
            d = semitone_distance(gt, pr)
            per_pos.append(d == 0 if d is not None else False)
    return per_pos, int(all(v is True for v in per_pos))


def _score_doremi(gt_midi: list[int], pred_pcs: list[int | None]) -> tuple[list[bool | None], int]:
    gt_pcs = [m % 12 for m in gt_midi]
    per_pos: list[bool | None] = []
    for gt, pr in zip(gt_pcs, pred_pcs):
        if pr is None:
            per_pos.append(None)
        else:
            d = abs(gt - pr)
            per_pos.append(min(d, 12 - d) == 0)
    return per_pos, int(all(v is True for v in per_pos))


# ── Summary helpers ───────────────────────────────────────────────────────────

def _acc(records: list[dict], sc_key: str) -> float:
    if not records:
        return 0.0
    return round(sum(r[sc_key] for r in records) / len(records), 4)


def _compute_summary(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {}

    FMTS = ("midi", "spn", "doremi")

    def breakdown(sub: list[dict]) -> dict[str, float]:
        return {f: _acc(sub, f"{f}_seq_correct") for f in FMTS}

    chorales = sorted({r["chorale_id"] for r in records})
    labels   = sorted({r["source_label"] for r in records})

    overall = breakdown(records)

    return {
        "total":           len(records),
        "accuracy":         overall,
        "by_voice": {
            VOICE_NAMES[x - 1]: breakdown([r for r in records if r["x"] == x])
            for x in range(1, N_VOICES + 1)
            if any(r["x"] == x for r in records)
        },
        "by_instrumentation": {
            cfg: breakdown([r for r in records if r["inst_cfg"] == cfg])
            for cfg in ("similar", "mixed")
            if any(r["inst_cfg"] == cfg for r in records)
        },
        "by_source": {
            lab: breakdown([r for r in records if r["source_label"] == lab])
            for lab in labels
        },
        "by_chorale": {
            c: breakdown([r for r in records if r["chorale_id"] == c]) for c in chorales
        },
    }


def _format_summary(records: list[dict], summary: dict) -> list[str]:
    HDR = f"  {'':24s}  {'MIDI':>7}  {'SPN':>7}  {'Doremi':>7}"
    SEP = f"  {'─' * 52}"

    def row(label: str, d: dict[str, float]) -> str:
        return (
            f"  {label:24s}  {d['midi']:>7.1%}  {d['spn']:>7.1%}  {d['doremi']:>7.1%}"
        )

    lines: list[str] = [
        f"  Total records : {summary.get('total', 0)}",
        "  (Accuracy = fraction of stimuli where the FULL sequence is correct.)",
        "",
    ]
    for section, key in [
        ("Overall",              "overall"),
        ("By voice",             "by_voice"),
        ("By instrument config", "by_instrumentation"),
        ("By source",             "by_source"),
        ("By chorale",           "by_chorale"),
    ]:
        data = summary.get(key)
        if not data:
            continue
        lines += [f"  {section}:", HDR, SEP]
        if isinstance(data, dict) and "midi" in data:
            lines.append(row("all", data))
        else:
            for k, d in data.items():
                lines.append(row(str(k), d))
        lines.append("")
    return lines


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    chorales: list[str],
    seed: int,
    run_dir: Path,
    sample_info: dict | None = None,
) -> dict[str, Any]:
    info = get_model_info(model_name)
    print(f"\n  Model: {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially.
    jobs: list[dict[str, Any]] = []
    for c in conds:
        lines = list(zip(c["all_notes"], c["sources"]))
        hint  = f"{c['chorale_slug']}_x{c['x']}_{c['inst_cfg']}_{c['source_label']}"
        try:
            wav = str(engine.polyphonic_mix(lines, c["total_ms"], name_hint=hint))
        except ValueError as exc:
            print(f"  [SKIP] {exc}")
            continue

        x        = c["x"]
        n_target = c["n_target"]
        prompt_midi   = make_prompt_midi(x, n_target, c["inst_cfg"], c["sources"])
        prompt_spn    = make_prompt_spn(x, n_target, c["inst_cfg"], c["sources"])
        prompt_doremi = make_prompt_doremi(x, n_target, c["inst_cfg"], c["sources"])
        jobs.append({
            "wav": wav, "cond": c,
            "prompt_midi": prompt_midi, "prompt_spn": prompt_spn,
            "prompt_doremi": prompt_doremi,
        })

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict[str, Any]) -> dict[str, Any]:
        c        = job["cond"]
        x        = c["x"]
        n_target = c["n_target"]
        prompt_midi   = job["prompt_midi"]
        prompt_spn    = job["prompt_spn"]
        prompt_doremi = job["prompt_doremi"]

        raw_midi   = query_alm(model_name, job["wav"], prompt_midi)["result"]   or ""
        raw_spn    = query_alm(model_name, job["wav"], prompt_spn)["result"]    or ""
        raw_doremi = query_alm(model_name, job["wav"], prompt_doremi)["result"] or ""

        tgt        = c["target_pitches"]
        tgt_spn    = [midi_to_note(m) for m in tgt]
        tgt_doremi = [PC_TO_SOLFEGE[m % 12] for m in tgt]

        pred_midi   = _parse_midi_seq(raw_midi, n_target)
        pred_spn    = _parse_spn_seq(raw_spn, n_target)
        pred_doremi = _parse_doremi_seq(raw_doremi, n_target)

        midi_pos, midi_sc = _score_midi(tgt, pred_midi)
        spn_pos,  spn_sc  = _score_spn(tgt, pred_spn)
        dor_pos,  dor_sc  = _score_doremi(tgt, pred_doremi)

        return {
            # ── condition ─────────────────────────────────────────────────────
            "chorale_id":   c["chorale_id"],
            "x":            x,
            "voice_name":   c["voice_name"],
            "inst_cfg":     c["inst_cfg"],
            "source_label": c["source_label"],
            "sources":      str(c["sources"]),
            "target_source": c["sources"][x - 1],
            "qpm":          c["qpm"],
            "start_qL":     c["start_qL"],
            "end_qL":       c["end_qL"],
            "total_dur_ms": c["total_ms"],
            "n_target":     n_target,
            "all_n_notes":  str(c["all_n_notes"]),
            # ── ground truth ──────────────────────────────────────────────────
            "target_midi_gt":   str(tgt),
            "target_spn_gt":    str(tgt_spn),
            "target_doremi_gt": str(tgt_doremi),
            "wav":              job["wav"],
            # ── MIDI ─────────────────────────────────────────────────────────
            "midi_pred":        str(pred_midi),
            "midi_per_pos":     str(midi_pos),
            "midi_seq_correct": midi_sc,
            # ── SPN ──────────────────────────────────────────────────────────
            "spn_pred":         str(pred_spn),
            "spn_per_pos":      str(spn_pos),
            "spn_seq_correct":  spn_sc,
            # ── Doremi ───────────────────────────────────────────────────────
            "doremi_pred":      str(pred_doremi),
            "doremi_per_pos":   str(dor_pos),
            "doremi_seq_correct": dor_sc,
            # ── raw responses ─────────────────────────────────────────────────
            "raw_midi":         raw_midi.strip(),
            "raw_spn":          raw_spn.strip(),
            "raw_doremi":       raw_doremi.strip(),
            # ── prompts ───────────────────────────────────────────────────────
            "prompt_midi":      prompt_midi,
            "prompt_spn":       prompt_spn,
            "prompt_doremi":    prompt_doremi,
        }

    raw_results = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{j['cond']['chorale_id']:16s} x={j['cond']['x']}({j['cond']['voice_name']:7s}) {j['cond']['inst_cfg']:7s} {j['cond']['source_label']:18s} [{j['cond']['n_target']} notes]",
        result_label_fn=lambda j, r: melody_audit_str(
            r, label=f"{j['cond']['chorale_id']:16s} x={j['cond']['x']}({j['cond']['voice_name']:7s}) {j['cond']['inst_cfg']:7s}"
        ),
    )
    records: list[dict[str, Any]] = [r for r in raw_results if r is not None]

    summary       = _compute_summary(records)
    summary_lines = sampling_summary_lines(sample_info or {}) + _format_summary(records, summary)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name,
        model_info=info,
        chorales=chorales,
        n_voices=N_VOICES,
        min_seg_notes=MIN_SEG_NOTES,
        max_seg_sec=MAX_SEG_SEC,
        default_qpm=DEFAULT_QPM,
        sources=SOURCES,
        mixed_groups=MIXED_GROUPS,
        seed=seed,
        prompt_midi_example=make_prompt_midi(1, 12, "similar", ["piano"] * N_VOICES),
        prompt_spn_example=make_prompt_spn(1, 12, "similar", ["piano"] * N_VOICES),
        prompt_doremi_example=make_prompt_doremi(1, 12, "similar", ["piano"] * N_VOICES),
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    return summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",  action="store_true")
    parser.add_argument("--chorales", nargs="+", default=DEFAULT_CHORALES,
                        metavar="ID",
                        help=f"music21 corpus chorale IDs (default: {DEFAULT_CHORALES})")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--models",   nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by chorale)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args      = _parse_args()
    all_conds = build_conditions(args.chorales, args.seed)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)

    n_audio   = len(conds)
    total_dur = sum(c["total_ms"] for c in conds) / 1000
    n_queries = n_audio * 3

    print(f"Experiment    : {EXP_NAME}")
    print(f"Chorales      : {args.chorales}")
    print(f"Sources (sim) : {SOURCES}")
    print(f"Mixed groups  : {list(MIXED_GROUPS)}")
    print(f"Conditions    : {n_audio}")
    print(f"Audio files   : {n_audio}")
    print(f"Queries/model : {n_queries}  (MIDI + SPN + Doremi)")
    print(f"Total audio   : {total_dur:.1f} s  ({total_dur / 60:.1f} min)")
    print(f"Audio dir     : {config.AUDIO_DIR}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print()
    print(
        f"  {'chorale':16s}  {'x':>2}  {'voice':8s}  {'cfg':7s}  "
        f"{'src_label':18s}  {'#notes':>6}  {'sec':>5}"
    )
    for c in conds[:24]:
        print(
            f"  {c['chorale_id']:16s}  {c['x']:>2}  {c['voice_name']:8s}  "
            f"{c['inst_cfg']:7s}  {c['source_label']:18s}  "
            f"{c['n_target']:>6}  {c['total_ms'] / 1000:>5.1f}"
        )
    if len(conds) > 24:
        print(f"  … ({len(conds) - 24} more)")
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args         = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds    = build_conditions(args.chorales, args.seed)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Conditions : {len(conds)}  × 3 formats = {len(conds) * 3} queries/model")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict[str, Any]] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(
            model_name, conds, args.chorales, args.seed, run_dir, s_meta,
        )
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
