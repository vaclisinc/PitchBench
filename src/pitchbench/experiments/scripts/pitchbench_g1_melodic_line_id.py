"""
Experiment g1 — Melodic-line identification in n-part polyphony

Tests whether a model can isolate and transcribe a single designated melodic
line from a polyphonic texture in which n = 2–4 lines play simultaneously
and continuously, with no silence — every voice always sustains a note.

Independent variables:
  n                : number of simultaneous parts (2, 3, 4)
  x                : which line to identify by register rank
                     (1 = top/highest … n = bottom/lowest)
  tempo            : average note duration of the target line —
                     slow = 1000 ms · medium = 500 ms · fast = 250 ms
  instrument_config: "similar" – all parts use the same source.
                                 The source is varied across all 19 sources
                                 (4 waveforms + 15 GM v1 instruments).
                     "mixed"   – each part uses a different source. The
                                 source group is varied across MIXED_GROUPS.

Polyphony rules:
  • every voice plays a note at every moment in the clip — no rests
  • adjacent notes within a voice flow into one another (no silent gaps)
  • each voice has its own rhythm and varying per-note durations
  • the target voice always has exactly 10 notes
  • each distractor voice has between 1 and 20 notes (uniform random)

Register ranges (top → bottom, one octave each):
  part 1  72–83  (C5–B5)
  part 2  60–71  (C4–B4)
  part 3  48–59  (C3–B3)
  part 4  36–47  (C2–B2)

Prompt formats: MIDI integers · SPN note names · Solfège syllables

Scoring: per-position accuracy + full-sequence exact match, broken down by
         (n, x, tempo, instrument_config, source_label, format).

Usage:
    pitchbench pitchbench_g1_melodic_line_id
    pitchbench pitchbench_g1_melodic_line_id --preview
    pitchbench --id g1
"""

import argparse
import random
import re
from pathlib import Path
from typing import Any

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import (
    PC_TO_SOLFEGE,
    extract_all_notes,
    extract_all_solfege,
    midi_to_note,
    semitone_distance,
)
from pitchbench.experiments.helpers.results import (
    get_run_metadata,
    make_run_dir,
    save_comparison,
    save_results,
)
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample

EXP_NAME = Path(__file__).stem

# ── Fixed parameters ──────────────────────────────────────────────────────────

N_NOTES       = 10
N_PARTS_LIST  = [2, 3, 4]
N_TRIALS      = 2
DEFAULT_SEED  = 42
DIST_N_MIN    = 1
DIST_N_MAX    = 20
DUR_JITTER    = 0.5      # ± fraction around mean duration for per-note jitter

TEMPOS: dict[str, int] = {
    "slow":   1000,
    "medium":  500,
    "fast":    250,
}

# One-octave register ranges, top → bottom (index 0 = highest part)
PART_RANGES: list[tuple[int, int]] = [
    (72, 83),  # part 1 : C5–B5
    (60, 71),  # part 2 : C4–B4
    (48, 59),  # part 3 : C3–B3
    (36, 47),  # part 4 : C2–B2
]

# Full source pool (waveforms + GM v1 instruments).
SOURCES: list[str] = list(config.GM_PROGRAMS_V1.keys())

# "Mixed" instrument groups — each group provides up to 4 distinct sources
# assigned top→bottom across the n parts.
MIXED_GROUPS: dict[str, list[str]] = {
    "classical_quartet": ["flute", "violin", "cello", "bass"],
    "mixed_timbres":     ["piano", "trumpet", "clarinet", "guitar"],
}


# ── Stimulus helpers ──────────────────────────────────────────────────────────

def _total_dur_ms(tone_ms: int) -> int:
    """Total clip duration with no silent gaps: N_NOTES × tone_ms."""
    return N_NOTES * tone_ms


def _vary_durations(n: int, total_ms: int, rng: random.Random) -> list[int]:
    """``n`` positive integer durations summing to ``total_ms``, jittered around
    the mean ``total_ms / n``.  Used to shape rhythm within each voice."""
    if n <= 0:
        return []
    if n == 1:
        return [total_ms]
    weights = [1.0 + rng.uniform(-DUR_JITTER, DUR_JITTER) for _ in range(n)]
    s = sum(weights)
    durs = [max(1, int(round(w * total_ms / s))) for w in weights]
    durs[-1] += total_ms - sum(durs)
    if durs[-1] < 1:                 # extreme jitter edge case
        durs[-1] = 1
        durs[0]  = total_ms - sum(durs[1:])
    return durs


def _sample_pitches_no_adj_repeat(
    rng: random.Random, lo: int, hi: int, n: int,
) -> list[int]:
    """``n`` pitches from ``range(lo, hi+1)`` with no two adjacent equal.

    Without this, adjacent notes of the same pitch in a flowing voice would
    merge perceptually into a single sustained note.
    """
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
    """Build a continuously-sounding voice: each pitch gets a varied duration,
    notes are placed back-to-back so the voice has no silence."""
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


def _cond_hint(c: dict) -> str:
    return (
        f"n{c['n']}_x{c['x']}_{c['tempo']}_{c['inst_cfg']}_"
        f"{c['source_label']}_t{c['trial']}"
    )


# ── Condition construction ────────────────────────────────────────────────────

def _instrument_choices(n: int) -> list[tuple[str, str, list[str]]]:
    """Return all (inst_cfg, source_label, sources) options for a given n.

    ``similar`` yields one option per source in SOURCES; ``mixed`` yields one
    option per group in MIXED_GROUPS (sliced to ``n`` voices).
    """
    out: list[tuple[str, str, list[str]]] = []
    for src in SOURCES:
        out.append(("similar", src, [src] * n))
    for label, group in MIXED_GROUPS.items():
        if len(group) >= n:
            out.append(("mixed", label, group[:n]))
    return out


def build_conditions(n_trials: int, seed: int) -> list[dict]:
    rng  = random.Random(seed)
    rows: list[dict] = []

    for n in N_PARTS_LIST:
        for x in range(1, n + 1):
            for tempo, tone_ms in TEMPOS.items():
                total_ms = _total_dur_ms(tone_ms)
                for trial in range(n_trials):
                    # Pre-sample the pitch material once per (n, x, tempo, trial)
                    # so all instrument options share the same musical content.
                    parts_pitches: list[list[int]] = []
                    for part_idx in range(n):
                        lo, hi = PART_RANGES[part_idx]
                        if part_idx == x - 1:
                            pitches = rng.sample(range(lo, hi + 1), N_NOTES)
                        else:
                            n_di    = rng.randint(DIST_N_MIN, DIST_N_MAX)
                            pitches = _sample_pitches_no_adj_repeat(rng, lo, hi, n_di)
                        parts_pitches.append(pitches)

                    parts_notes: list[list[tuple[int, int, int]]] = [
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
                            "all_pitches":    parts_pitches,
                            "all_n_notes":    all_n_notes,
                            "target_pitches": parts_pitches[x - 1],
                            "trial":          trial,
                        })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        lines = list(zip(c["all_notes"], c["sources"]))
        try:
            engine.polyphonic_mix(lines, c["total_ms"], name_hint=_cond_hint(c))
        except ValueError as exc:
            print(f"  [SKIP] {exc}")


# ── Prompts ───────────────────────────────────────────────────────────────────

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


def make_prompt_midi(n: int, x: int, inst_cfg: str, sources: list[str]) -> str:
    return (
        f"{_preamble(n, x, inst_cfg, sources)} "
        f"List all {N_NOTES} MIDI note numbers in order from first to last. "
        "Reply with ONLY the integers separated by spaces. Nothing else."
    )


def make_prompt_spn(n: int, x: int, inst_cfg: str, sources: list[str]) -> str:
    return (
        f"{_preamble(n, x, inst_cfg, sources)} "
        f"List all {N_NOTES} note names in order from first to last. "
        "Reply with ONLY the note names separated by spaces (e.g. C5 D5 E5). Nothing else."
    )


def make_prompt_doremi(n: int, x: int, inst_cfg: str, sources: list[str]) -> str:
    return (
        f"{_preamble(n, x, inst_cfg, sources)} "
        f"List all {N_NOTES} solfège syllables in order from first to last "
        "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B; include sharps e.g. do# re#). "
        "Reply with ONLY the syllables separated by spaces. Nothing else."
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_midi_seq(text: str) -> list[int | None]:
    nums = [int(m) for m in re.findall(r"\b(\d{1,3})\b", text) if 0 <= int(m) <= 127]
    result: list[int | None] = list(nums[:N_NOTES])
    while len(result) < N_NOTES:
        result.append(None)
    return result


def _parse_spn_seq(text: str) -> list[str | None]:
    tokens = extract_all_notes(text)
    result: list[str | None] = list(tokens[:N_NOTES])
    while len(result) < N_NOTES:
        result.append(None)
    return result


def _parse_doremi_seq(text: str) -> list[int | None]:
    pcs = extract_all_solfege(text)
    result: list[int | None] = list(pcs[:N_NOTES])
    while len(result) < N_NOTES:
        result.append(None)
    return result


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_midi(
    gt: list[int], pred: list[int | None]
) -> tuple[list[bool | None], int, int]:
    per_pos = [
        (gt[i] == pred[i] if pred[i] is not None else None)
        for i in range(N_NOTES)
    ]
    n_correct = sum(1 for v in per_pos if v is True)
    return per_pos, n_correct, int(n_correct == N_NOTES)


def _score_spn(
    gt_midi: list[int], pred: list[str | None]
) -> tuple[list[bool | None], int, int]:
    gt_spn = [midi_to_note(m) for m in gt_midi]
    per_pos: list[bool | None] = []
    for gt, pr in zip(gt_spn, pred):
        if pr is None:
            per_pos.append(None)
        else:
            dist = semitone_distance(gt, pr)
            per_pos.append(dist == 0 if dist is not None else False)
    n_correct = sum(1 for v in per_pos if v is True)
    return per_pos, n_correct, int(n_correct == N_NOTES)


def _score_doremi(
    gt_midi: list[int], pred_pcs: list[int | None]
) -> tuple[list[bool | None], int, int]:
    gt_pcs = [m % 12 for m in gt_midi]
    per_pos: list[bool | None] = []
    for gt_pc, pred_pc in zip(gt_pcs, pred_pcs):
        if pred_pc is None:
            per_pos.append(None)
        else:
            diff = abs(gt_pc - pred_pc)
            per_pos.append(min(diff, 12 - diff) == 0)
    n_correct = sum(1 for v in per_pos if v is True)
    return per_pos, n_correct, int(n_correct == N_NOTES)


# ── Summary helpers ───────────────────────────────────────────────────────────

def _acc_pair(records: list[dict], nc_key: str, sc_key: str) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    note_acc = sum(r[nc_key] for r in records) / (len(records) * N_NOTES)
    seq_acc  = sum(r[sc_key] for r in records) / len(records)
    return round(note_acc, 4), round(seq_acc, 4)


def _compute_summary(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {}

    FMTS = ("midi", "spn", "doremi")

    def breakdown(sub: list[dict]) -> dict:
        return {f: _acc_pair(sub, f"{f}_n_correct", f"{f}_seq_correct") for f in FMTS}

    by_nx: dict[str, dict] = {}
    for n in N_PARTS_LIST:
        for x in range(1, n + 1):
            sub = [r for r in records if r["n"] == n and r["x"] == x]
            if sub:
                by_nx[f"n{n}_x{x}"] = breakdown(sub)

    labels_seen = sorted({r["source_label"] for r in records})

    return {
        "total":       len(records),
        "overall":     breakdown(records),
        "by_inst_cfg": {
            cfg: breakdown([r for r in records if r["inst_cfg"] == cfg])
            for cfg in ("similar", "mixed")
            if any(r["inst_cfg"] == cfg for r in records)
        },
        "by_tempo":    {
            t: breakdown([r for r in records if r["tempo"] == t])
            for t in TEMPOS
            if any(r["tempo"] == t for r in records)
        },
        "by_n":        {
            n: breakdown([r for r in records if r["n"] == n])
            for n in N_PARTS_LIST
            if any(r["n"] == n for r in records)
        },
        "by_source_label": {
            label: breakdown([r for r in records if r["source_label"] == label])
            for label in labels_seen
        },
        "by_nx":       by_nx,
    }


def _format_summary(records: list[dict], summary: dict) -> list[str]:
    HDR = (
        f"  {'':20s}  {'MIDI note':>9}  {'MIDI seq':>8}"
        f"  {'SPN note':>8}  {'SPN seq':>7}"
        f"  {'Do note':>7}  {'Do seq':>6}"
    )
    SEP = f"  {'─' * 72}"

    def row(label: str, d: dict) -> str:
        mn, ms = d["midi"]
        sn, ss = d["spn"]
        dn, ds = d["doremi"]
        return (
            f"  {label:20s}  {mn:>9.1%}  {ms:>8.1%}"
            f"  {sn:>8.1%}  {ss:>7.1%}"
            f"  {dn:>7.1%}  {ds:>6.1%}"
        )

    lines: list[str] = [f"  Total records : {summary.get('total', 0)}", ""]

    for section, key in [
        ("Overall", "overall"),
        ("By instrument config", "by_inst_cfg"),
        ("By tempo", "by_tempo"),
        ("By number of parts (n)", "by_n"),
        ("By source label", "by_source_label"),
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

    # (n, x) table — note accuracy only to keep it compact
    by_nx = summary.get("by_nx", {})
    if by_nx:
        lines += ["  By (n, x) — note accuracy:", f"  {'─' * 44}"]
        lines.append(f"  {'cond':12s}  {'MIDI':>7}  {'SPN':>7}  {'Doremi':>7}")
        for key, d in by_nx.items():
            mn, _ = d["midi"]
            sn, _ = d["spn"]
            dn, _ = d["doremi"]
            lines.append(f"  {key:12s}  {mn:>7.1%}  {sn:>7.1%}  {dn:>7.1%}")

    return lines


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    n_trials: int,
    seed: int,
    run_dir: Path,
    sample_info: dict | None = None,
) -> dict[str, Any]:
    info = get_model_info(model_name)
    print(f"\n  Model: {config.MODELS.get(model_name, model_name)}")

    records: list[dict[str, Any]] = []
    for c in conds:
        lines = list(zip(c["all_notes"], c["sources"]))
        try:
            wav = str(engine.polyphonic_mix(lines, c["total_ms"], name_hint=_cond_hint(c)))
        except ValueError as exc:
            print(f"  [SKIP] {exc}")
            continue

        n, x = c["n"], c["x"]
        prompt_midi   = make_prompt_midi(n, x, c["inst_cfg"], c["sources"])
        prompt_spn    = make_prompt_spn(n, x, c["inst_cfg"], c["sources"])
        prompt_doremi = make_prompt_doremi(n, x, c["inst_cfg"], c["sources"])

        print(f"    n={n} x={x} {c['tempo']:6s} {c['inst_cfg']:7s} t={c['trial']}")

        raw_midi   = query_alm(model_name, wav, prompt_midi)["result"]   or ""
        raw_spn    = query_alm(model_name, wav, prompt_spn)["result"]    or ""
        raw_doremi = query_alm(model_name, wav, prompt_doremi)["result"] or ""

        tgt = c["target_pitches"]
        tgt_spn    = [midi_to_note(m) for m in tgt]
        tgt_doremi = [PC_TO_SOLFEGE[m % 12] for m in tgt]

        pred_midi   = _parse_midi_seq(raw_midi)
        pred_spn    = _parse_spn_seq(raw_spn)
        pred_doremi = _parse_doremi_seq(raw_doremi)

        midi_pos, midi_nc, midi_sc = _score_midi(tgt, pred_midi)
        spn_pos,  spn_nc,  spn_sc  = _score_spn(tgt, pred_spn)
        dor_pos,  dor_nc,  dor_sc  = _score_doremi(tgt, pred_doremi)

        records.append({
            # ── condition ─────────────────────────────────────────────────────
            "n":              n,
            "x":              x,
            "tempo":          c["tempo"],
            "tone_ms":        c["tone_ms"],
            "total_dur_ms":   c["total_ms"],
            "inst_cfg":       c["inst_cfg"],
            "source_label":   c["source_label"],
            "sources":        str(c["sources"]),
            "target_source":  c["sources"][x - 1],
            "trial":          c["trial"],
            "all_n_notes":    str(c["all_n_notes"]),
            # ── ground truth ──────────────────────────────────────────────────
            "target_midi_gt":   str(tgt),
            "target_spn_gt":    str(tgt_spn),
            "target_doremi_gt": str(tgt_doremi),
            "wav":              wav,
            # ── MIDI ─────────────────────────────────────────────────────────
            "midi_pred":        str(pred_midi),
            "midi_per_pos":     str(midi_pos),
            "midi_n_correct":   midi_nc,
            "midi_seq_correct": midi_sc,
            # ── SPN ──────────────────────────────────────────────────────────
            "spn_pred":         str(pred_spn),
            "spn_per_pos":      str(spn_pos),
            "spn_n_correct":    spn_nc,
            "spn_seq_correct":  spn_sc,
            # ── Doremi ───────────────────────────────────────────────────────
            "doremi_pred":      str(pred_doremi),
            "doremi_per_pos":   str(dor_pos),
            "doremi_n_correct": dor_nc,
            "doremi_seq_correct": dor_sc,
            # ── raw responses ─────────────────────────────────────────────────
            "raw_midi":         raw_midi.strip(),
            "raw_spn":          raw_spn.strip(),
            "raw_doremi":       raw_doremi.strip(),
            # ── prompts ───────────────────────────────────────────────────────
            "prompt_midi":      prompt_midi,
            "prompt_spn":       prompt_spn,
            "prompt_doremi":    prompt_doremi,
        })

    summary      = _compute_summary(records)
    summary_lines = sampling_summary_lines(sample_info or {}) + _format_summary(records, summary)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name,
        model_info=info,
        n_parts_list=N_PARTS_LIST,
        n_notes_target=N_NOTES,
        dist_n_range=(DIST_N_MIN, DIST_N_MAX),
        dur_jitter=DUR_JITTER,
        tempos=TEMPOS,
        part_ranges=PART_RANGES,
        sources=SOURCES,
        mixed_groups=MIXED_GROUPS,
        n_trials=n_trials,
        seed=seed,
        prompt_midi_example=make_prompt_midi(2, 1, "similar", ["sine", "sine"]),
        prompt_spn_example=make_prompt_spn(2, 1, "similar", ["sine", "sine"]),
        prompt_doremi_example=make_prompt_doremi(2, 1, "similar", ["sine", "sine"]),
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    return summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",  action="store_true")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help=f"Trials per condition (default: {N_TRIALS})")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--models",   nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by (n_voices, source_label))")
    parser.add_argument("--sample-seed",  type=int, default=42,   metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args      = _parse_args()
    all_conds = build_conditions(args.n_trials, args.seed)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(
            all_conds, args.sample_n,
            lambda c: (c["n"], c["source_label"]),
            seed=args.sample_seed,
        )
    s_meta    = sampling_meta(len(all_conds), "(n_voices, source_label)", args.sample_n, args.sample_seed)
    generate_stimuli(conds)

    n_audio    = len(conds)
    total_dur  = sum(c["total_ms"] for c in conds) / 1000
    n_queries  = n_audio * 3

    print(f"Experiment     : {EXP_NAME}")
    print(f"Seed           : {args.seed}")
    print(f"Sources (sim)  : {SOURCES}")
    print(f"Mixed groups   : {list(MIXED_GROUPS)}")
    print(f"Conditions     : {n_audio}  (n × x × tempo × trial × source)")
    print(f"Audio files    : {n_audio}")
    print(f"Queries/model  : {n_queries}  (MIDI + SPN + Doremi)")
    print(f"Total audio    : {total_dur:.1f} s  ({total_dur / 60:.1f} min)")
    print(f"Audio dir      : {config.AUDIO_DIR}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print()
    print(f"  {'n':>2}  {'x':>2}  {'tempo':6s}  {'cfg':7s}  {'src_label':17s}  "
          f"{'t':>1}  {'dist_notes'}")
    for c in conds[:24]:
        dist_ns = [c["all_n_notes"][i] for i in range(c["n"]) if i != c["x"] - 1]
        print(f"  {c['n']:>2}  {c['x']:>2}  {c['tempo']:6s}  {c['inst_cfg']:7s}  "
              f"{c['source_label']:17s}  {c['trial']:>1}  {dist_ns}")
    if len(conds) > 24:
        print(f"  … ({len(conds) - 24} more)")
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args         = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds    = build_conditions(args.n_trials, args.seed)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(
            all_conds, args.sample_n,
            lambda c: (c["n"], c["source_label"]),
            seed=args.sample_seed,
        )
    s_meta = sampling_meta(len(all_conds), "(n_voices, source_label)", args.sample_n, args.sample_seed)
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
            model_name, conds, args.n_trials, args.seed, run_dir, s_meta,
        )
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
