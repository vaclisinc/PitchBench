"""
f2 — Voice identification in Bach chorales (music21 corpus).

Same task as f1 (identify the x-th voice in a polyphonic mix) but using
real Bach chorales sourced from the music21 corpus instead of synthetic
random material.

For each (chorale, x) pair, the audio is the longest contiguous segment
of the chorale during which the target voice maintains its register rank
— i.e. exactly (x-1) voices are above it at every moment in the segment.
The target voice never crosses with any other voice in this span; other
voices may cross each other freely.

Independent variables:
  chorale_id : Bach chorale corpus IDs
  x          : 1..4 (1=soprano · 2=alto · 3=tenor · 4=bass)
  inst_cfg   : "similar" – all 4 voices on the same source
               "mixed"   – each voice on a different source

Headline metrics: ``midi_seq``, ``spn_seq``, ``doremi_seq`` — full-sequence
exact-match per format. Note count is variable (per chorale segment).

Requires the ``music21`` package.
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_f import (
    CatFSpec, run_cat_f_experiment, score_polyphonic_record,
)

EXP_NAME         = Path(__file__).stem
N_VOICES         = config.pitchbench_f2_N_VOICES
DEFAULT_SEED     = config.pitchbench_f2_SEED
MIN_SEG_NOTES    = config.pitchbench_f2_MIN_SEG_NOTES
MAX_SEG_SEC      = config.pitchbench_f2_MAX_SEG_SEC
DEFAULT_QPM      = config.pitchbench_f2_QPM
VOICE_NAMES      = config.pitchbench_f2_VOICE_NAMES
DEFAULT_CHORALES = config.pitchbench_f2_DEFAULT_CHORALES
SOURCES          = config.pitchbench_f2_SOURCES
MIXED_GROUPS     = config.pitchbench_f2_MIXED_GROUPS


# ── music21 helpers ──────────────────────────────────────────────────────────

def _import_music21():
    try:
        import music21
        return music21
    except ImportError:
        raise SystemExit(
            "music21 is required for experiment f2.\n"
            "Install with:  uv add music21   (or  pip install music21)"
        )


def _voice_to_events(stream_obj) -> list[tuple[float, int, float]]:
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
    def mean_pitch(events: list[tuple[float, int, float]]) -> float:
        if not events:
            return -1.0
        return sum(p for _, p, _ in events) / len(events)
    return sorted(voices, key=mean_pitch, reverse=True)


def _voice_pitch_at(events: list[tuple[float, int, float]], t: float) -> int | None:
    for onset, pitch, dur in events:
        if onset <= t < onset + dur:
            return pitch
        if onset > t:
            return None
    return None


def _time_grid(voices: list[list[tuple[float, int, float]]]) -> list[float]:
    times: set[float] = set()
    for events in voices:
        for onset, _, dur in events:
            times.add(float(onset))
            times.add(float(onset + dur))
    return sorted(times)


def _longest_no_cross_segment(
    voices: list[list[tuple[float, int, float]]], x: int,
) -> tuple[float, float] | None:
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


def _slug(chorale_id: str) -> str:
    return chorale_id.replace("/", "_").replace(".", "_")


def _instrument_choices() -> list[tuple[str, str, list[str]]]:
    out: list[tuple[str, str, list[str]]] = []
    for src in SOURCES:
        out.append(("similar", src, [src] * N_VOICES))
    for label, group in MIXED_GROUPS.items():
        if len(group) >= N_VOICES:
            out.append(("mixed", label, group[:N_VOICES]))
    return out


# ── Conditions ───────────────────────────────────────────────────────────────

def build_conditions() -> list[dict]:
    m21 = _import_music21()
    rows: list[dict] = []

    for chorale_id in DEFAULT_CHORALES:
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

        qpm        = DEFAULT_QPM
        sec_per_qL = 60.0 / qpm

        for x in range(1, N_VOICES + 1):
            seg = _longest_no_cross_segment(voices, x)
            if seg is None:
                print(f"  [SKIP] {chorale_id} x={x}: no non-crossing segment found")
                continue
            start_qL, end_qL = seg

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


def wav_for(c: dict) -> Path:
    lines = list(zip(c["all_notes"], c["sources"]))
    hint  = f"{c['chorale_slug']}_x{c['x']}_{c['inst_cfg']}_{c['source_label']}"
    return engine.polyphonic_mix(lines, c["total_ms"], name_hint=hint)


# ── Prompts ──────────────────────────────────────────────────────────────────

def _preamble(x: int, n_target: int, inst_cfg: str) -> str:
    voice  = VOICE_NAMES[x - 1]
    timbre = (
        "all four voices played on the same instrument" if inst_cfg == "similar"
        else "each voice played by a different instrument"
    )
    return (
        f"You will hear an excerpt from a Bach four-part chorale (soprano, alto, "
        f"tenor, bass), {timbre}. The voices do not cross with respect to the "
        f"{voice}: it stays in its register the whole time. "
        f"Focus only on the {voice} voice. It plays exactly {n_target} notes."
    )


def prompts_for(c: dict) -> dict[str, str]:
    pre = _preamble(c["x"], c["n_target"], c["inst_cfg"])
    n   = c["n_target"]
    midi = (
        f"{pre} List all {n} MIDI note numbers in order from first to last. "
        "Reply with ONLY the integers separated by spaces. Nothing else. "
        "Output only the answer."
    )
    spn = (
        f"{pre} List all {n} note names in order from first to last. "
        "Reply with ONLY the note names expressed in Scientific Pitch Notation, "
        "separated by spaces (e.g. C5 D#5 E5). Nothing else. Output only the answer."
    )
    doremi = (
        f"{pre} List all {n} solfège syllable and accidentals (if needed) "
        "in order from first to last "
        "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B; include sharps e.g. do# re#). "
        "Reply with ONLY the syllable and accidentals (if needed) separated by spaces. "
        "Nothing else. Output only the answer."
    )
    hz = (
        f"{pre} List the main pitch frequency in Hertz of all {n} notes "
        "in order from first to last. "
        "Reply with ONLY the frequencies in Hz separated by spaces "
        "(e.g. 261.6 329.6 392.0). Nothing else. Output only the answer."
    )
    return {"midi": midi, "spn": spn, "doremi": doremi, "hz": hz}


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    rec = score_polyphonic_record(c, wav, responses)
    rec.update({
        "chorale_id":    c["chorale_id"],
        "chorale_slug":  c["chorale_slug"],
        "x":             c["x"],
        "voice_name":    c["voice_name"],
        "qpm":           c["qpm"],
        "start_qL":      round(c["start_qL"], 4),
        "end_qL":        round(c["end_qL"], 4),
        "total_dur_ms":  c["total_ms"],
        "inst_cfg":      c["inst_cfg"],
        "source_label":  c["source_label"],
        "sources":       str(c["sources"]),
        "target_source": c["sources"][c["x"] - 1],
        "all_n_notes":   str(c["all_n_notes"]),
    })
    return rec


SPEC = CatFSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="chorale_voice",
    prompts_fn=prompts_for,
    record_fn=record_for,
    record_extras=("chorale_id", "x", "voice_name", "inst_cfg", "source_label"),
    label_fn=lambda j: (
        f"{j['cond']['chorale_slug']:20s} "
        f"x={j['cond']['x']} {j['cond']['voice_name']:8s} "
        f"{j['cond']['inst_cfg']:7s}"
    ),
)


def preview() -> None:
    run_cat_f_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_f_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
