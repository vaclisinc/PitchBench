"""
f1 — Melodic-line identification in n-part polyphony.

Tests whether a model can isolate and transcribe a single designated
melodic line from a synthetic polyphonic texture in which n = 2–4 lines
play simultaneously and continuously, with no silence — every voice
always sustains a note.

Independent variables:
  n             : number of simultaneous parts (2, 3, 4)
  x             : which line to identify by register rank
                  (1 = top/highest … n = bottom/lowest)
  tempo         : avg note duration of the target line —
                  slow=1000ms · medium=500ms · fast=250ms
  inst_cfg      : "similar" – all parts on the same source.
                  "mixed"   – each part on a different source.
  source_label  : single source ("similar") or group label ("mixed").

Polyphony rules:
  • every voice plays a note at every moment in the clip — no rests
  • adjacent notes within a voice flow into one another (no silent gaps)
  • each voice has its own rhythm and varying per-note durations
  • the target voice always has exactly N_NOTES notes
  • each distractor voice has between DIST_N_MIN and DIST_N_MAX notes

Headline metrics: ``midi_seq``, ``spn_seq``, ``doremi_seq`` — full-sequence
exact-match per format. Per-position correctness is preserved as a
``<format>_per_pos`` list-string in the records CSV.
"""

from __future__ import annotations

import random
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_f import (
    CatFSpec, run_cat_f_experiment, score_polyphonic_record,
)

EXP_NAME     = Path(__file__).stem
N_NOTES      = config.pitchbench_f1_N_NOTES
N_PARTS_LIST = config.pitchbench_f1_N_PARTS_LIST
N_TRIALS     = config.pitchbench_f1_N_TRIALS
DEFAULT_SEED = config.pitchbench_f1_SEED
DIST_N_MIN   = config.pitchbench_f1_DIST_N_MIN
DIST_N_MAX   = config.pitchbench_f1_DIST_N_MAX
DUR_JITTER   = config.pitchbench_f1_DUR_JITTER
TEMPOS       = config.pitchbench_f1_TEMPOS
PART_RANGES  = config.pitchbench_f1_PART_RANGES
MIXED_GROUPS = config.pitchbench_f1_MIXED_GROUPS
SOURCES      = config.pitchbench_f1_SOURCES


# ── Stimulus helpers ─────────────────────────────────────────────────────────

def _vary_durations(n: int, total_ms: int, rng: random.Random) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [total_ms]
    weights = [1.0 + rng.uniform(-DUR_JITTER, DUR_JITTER) for _ in range(n)]
    s = sum(weights)
    durs = [max(1, int(round(w * total_ms / s))) for w in weights]
    durs[-1] += total_ms - sum(durs)
    if durs[-1] < 1:
        durs[-1] = 1
        durs[0]  = total_ms - sum(durs[1:])
    return durs


def _sample_pitches_no_adj_repeat(
    rng: random.Random, lo: int, hi: int, n: int,
) -> list[int]:
    if n <= 0:
        return []
    pop = list(range(lo, hi + 1))
    if len(pop) == 1:
        return pop * n
    out = [rng.choice(pop)]
    for _ in range(n - 1):
        choices = [p for p in pop if p != out[-1]]
        out.append(rng.choice(choices))
    return out


def _flow_notes(
    pitches: list[int], total_dur_ms: int, rng: random.Random,
) -> list[tuple[int, int, int]]:
    n = len(pitches)
    if n == 0:
        return []
    durs  = _vary_durations(n, total_dur_ms, rng)
    notes: list[tuple[int, int, int]] = []
    onset = 0
    for midi, dur in zip(pitches, durs):
        notes.append((midi, onset, dur))
        onset += dur
    return notes


def _instrument_choices(n: int) -> list[tuple[str, str, list[str]]]:
    out: list[tuple[str, str, list[str]]] = []
    for src in SOURCES:
        out.append(("similar", src, [src] * n))
    for label, group in MIXED_GROUPS.items():
        if len(group) >= n:
            out.append(("mixed", label, group[:n]))
    return out


def _cond_hint(c: dict) -> str:
    return (
        f"n{c['n']}_x{c['x']}_{c['tempo']}_{c['inst_cfg']}_"
        f"{c['source_label']}_t{c['trial']}"
    )


# ── Conditions ───────────────────────────────────────────────────────────────

def build_conditions() -> list[dict]:
    rng  = random.Random(DEFAULT_SEED)
    rows: list[dict] = []
    for n in N_PARTS_LIST:
        for x in range(1, n + 1):
            for tempo, tone_ms in TEMPOS.items():
                total_ms = N_NOTES * tone_ms
                for trial in range(N_TRIALS):
                    parts_pitches: list[list[int]] = []
                    for part_idx in range(n):
                        lo, hi = PART_RANGES[part_idx]
                        if part_idx == x - 1:
                            pitches = rng.sample(range(lo, hi + 1), N_NOTES)
                        else:
                            n_di    = rng.randint(DIST_N_MIN, DIST_N_MAX)
                            pitches = _sample_pitches_no_adj_repeat(rng, lo, hi, n_di)
                        parts_pitches.append(pitches)

                    parts_notes = [
                        _flow_notes(p, total_ms, rng) for p in parts_pitches
                    ]
                    all_n_notes = [len(p) for p in parts_pitches]

                    for inst_cfg, source_label, sources in _instrument_choices(n):
                        rows.append({
                            "n":              n,
                            "x":              x,
                            "tempo":          tempo,
                            "tone_ms":        tone_ms,
                            "total_ms":       total_ms,
                            "inst_cfg":       inst_cfg,
                            "source_label":   source_label,
                            "sources":        sources,
                            "all_notes":      parts_notes,
                            "all_n_notes":    all_n_notes,
                            "target_pitches": parts_pitches[x - 1],
                            "trial":          trial,
                        })
    return rows


def wav_for(c: dict) -> Path:
    lines = list(zip(c["all_notes"], c["sources"]))
    return engine.polyphonic_mix(lines, c["total_ms"], name_hint=_cond_hint(c))


# ── Prompts ──────────────────────────────────────────────────────────────────

_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def _register_label(x: int, n: int) -> str:
    if x == 1:
        return "highest"
    if x == n:
        return "lowest"
    return f"{_ORDINALS.get(x, str(x) + 'th')}-from-top"


def _instrument_list_str(sources: list[str], n: int) -> str:
    parts = []
    for i, src in enumerate(sources):
        if i == 0:
            parts.append(f"{src} (highest)")
        elif i == n - 1:
            parts.append(f"{src} (lowest)")
        else:
            parts.append(src)
    return ", ".join(parts)


def _preamble(n: int, x: int, inst_cfg: str, sources: list[str]) -> str:
    reg = _register_label(x, n)
    flow = (
        "Every line plays continuously: each line always sustains a note, "
        "and consecutive notes flow directly into one another with no silence. "
        "The lines have different rhythms and numbers of notes, and individual "
        "note durations also vary within each line."
    )
    if inst_cfg == "similar":
        return (
            f"You will hear {n} simultaneous melodic lines, all played with the same "
            f"instrument, each in a distinct pitch register from highest to lowest. "
            f"{flow} "
            f"Focus only on the {reg} melodic line. "
            f"That line has exactly {N_NOTES} notes."
        )
    target_src = sources[x - 1]
    inst_list  = _instrument_list_str(sources, n)
    return (
        f"You will hear {n} simultaneous melodic lines, each played by a different "
        f"instrument: {inst_list}. {flow} "
        f"Focus only on the {target_src} ({reg} line). "
        f"That line has exactly {N_NOTES} notes."
    )


def prompts_for(c: dict) -> dict[str, str]:
    n, x = c["n"], c["x"]
    pre = _preamble(n, x, c["inst_cfg"], c["sources"])
    midi = (
        f"{pre} List all {N_NOTES} MIDI note numbers in order from first to last. "
        "Reply with ONLY the integers separated by spaces. Nothing else. Output only the answer."
    )
    spn = (
        f"{pre} List all {N_NOTES} note names in order from first to last. "
        "Reply with ONLY the note names expressed in Scientific Pitch Notation, "
        "separated by spaces (e.g. C5 D#5 E5). Nothing else. Output only the answer."
    )
    doremi = (
        f"{pre} List all {N_NOTES} solfège syllable and accidentals (if needed) "
        "in order from first to last "
        "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B; include sharps e.g. do# re#). "
        "Reply with ONLY the syllable and accidentals (if needed) separated by spaces. "
        "Nothing else. Output only the answer."
    )
    hz = (
        f"{pre} List the main pitch frequency in Hertz of all {N_NOTES} notes "
        "in order from first to last. "
        "Reply with ONLY the frequencies in Hz separated by spaces "
        "(e.g. 261.6 329.6 392.0). Nothing else. Output only the answer."
    )
    return {"midi": midi, "spn": spn, "doremi": doremi, "hz": hz}


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    rec = score_polyphonic_record(c, wav, responses)
    rec.update({
        "n":             c["n"],
        "x":             c["x"],
        "tempo":         c["tempo"],
        "tone_ms":       c["tone_ms"],
        "total_dur_ms":  c["total_ms"],
        "inst_cfg":      c["inst_cfg"],
        "source_label":  c["source_label"],
        "sources":       str(c["sources"]),
        "target_source": c["sources"][c["x"] - 1],
        "trial":         c["trial"],
        "all_n_notes":   str(c["all_n_notes"]),
    })
    return rec


SPEC = CatFSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="melodic_line",
    prompts_fn=prompts_for,
    record_fn=record_for,
    record_extras=("n", "x", "tempo", "inst_cfg", "source_label", "trial"),
    label_fn=lambda j: (
        f"n={j['cond']['n']} x={j['cond']['x']} "
        f"{j['cond']['tempo']:6s} {j['cond']['inst_cfg']:7s} "
        f"t={j['cond']['trial']}"
    ),
)


def preview() -> None:
    run_cat_f_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_f_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
