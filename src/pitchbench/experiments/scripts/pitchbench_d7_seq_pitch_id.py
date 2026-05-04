"""
Experiment 08 — Multi-pitch sequence identification
Tests whether models can identify all pitches in a sequence of N notes
played one after another, and whether accuracy decays with position or
sequence length.

Stimuli: all waveforms + instruments, N notes randomly drawn without
         replacement from MIDI 48–84 (C3–C6), played with 0.25 s gaps.
Sequence lengths: N ∈ {3, 5, 10}, 5 random trials each.
Three prompts per stimulus (one wide row per audio):
  MIDI:   list all MIDI integers in order
  ABC:    list all note names in order
  Doremi: list all solfège syllable and accidental (if needed)s in order

Scoring: per-note accuracy at each position + sequence-level (all correct).

Usage:
    python experiments/run.py exp_8_multi_pitch
    python experiments/run.py exp_8_multi_pitch --preview
    python experiments/run.py exp_8_multi_pitch --n-notes 3 5 10 --n-trials 5
    python experiments/run.py exp_8_multi_pitch --models audio_flamingo_next_instruct
"""

import argparse
import ast
import random
import re
from pathlib import Path
from typing import Any

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_three_formats
from pitchbench.experiments.helpers.music import (
    PC_TO_SOLFEGE,
    extract_all_notes,
    extract_all_solfege,
    midi_to_note,
    semitone_distance,
)
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results

EXP_NAME = Path(__file__).stem

# ── Test parameters ───────────────────────────────────────────────────────────

PITCH_MIN = config.DEFAULT_MIDI_MIN
PITCH_MAX = config.DEFAULT_MIDI_MAX

N_NOTES_LIST: list[int] = [3, 5, 10]
DEFAULT_N_TRIALS = 5
DEFAULT_SEED     = config.DEFAULT_SEED

TONE_DURATION = 0.75   # seconds per note
GAP_DURATION  = 0.25   # silence between notes

TONE_MS = int(TONE_DURATION * 1000)
GAP_MS  = int(GAP_DURATION  * 1000)

SOURCES: list[str] = config.ALL_SOURCES


def make_prompt_abc(n: int) -> str:
    return (
        f"You will hear {n} musical notes played one after another, "
        "each separated by a brief silence. "
        f"Identify all {n} notes in order from first to last. "
        "Reply with ONLY the note names separated by spaces "
        "(e.g. C4 E4 G4). Nothing else. Do not think."
    )


def make_prompt_midi(n: int) -> str:
    return (
        f"You will hear {n} musical notes played one after another, "
        "each separated by a brief silence. "
        f"Identify all {n} MIDI note numbers in order from first to last. "
        "Reply with ONLY the integers separated by spaces "
        "(e.g. 60 64 67). Nothing else. Do not think."
    )


def make_prompt_doremi(n: int) -> str:
    return (
        f"You will hear {n} musical notes played one after another, "
        "each separated by a brief silence. "
        f"Identify all {n} solfège syllable and accidental (if needed)s in order from first to last "
        "(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B; include sharps e.g. do# re#). "
        "Reply with ONLY the syllable and accidental (if needed)s separated by spaces "
        "(e.g. do mi sol). Nothing else. Do not think."
    )


# ── Stimulus generation ───────────────────────────────────────────────────────

def build_sequences(
    n_notes_list: list[int],
    n_trials: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for n in n_notes_list:
        for trial in range(n_trials):
            midi_seq = rng.sample(range(PITCH_MIN, PITCH_MAX + 1), n)
            note_seq = [midi_to_note(m) for m in midi_seq]
            doremi_seq = [PC_TO_SOLFEGE.get(m % 12, "?") for m in midi_seq]
            rows.append({
                "n_notes":        n,
                "trial":          trial,
                "midi_sequence":  midi_seq,
                "note_sequence":  note_seq,
                "doremi_sequence": doremi_seq,
            })
    return rows


def generate_stimuli(seqs: list[dict]) -> None:
    for s in seqs:
        for src in SOURCES:
            engine.sequence(s["midi_sequence"], src, TONE_MS, GAP_MS)


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_abc_sequence(text: str, n: int) -> list[str | None]:
    tokens = extract_all_notes(text)
    result: list[str | None] = tokens[:n]
    while len(result) < n:
        result.append(None)
    return result


def parse_midi_sequence(text: str, n: int) -> list[int | None]:
    nums: list[int] = []
    for m in re.finditer(r"\b(\d{1,3})\b", text):
        v = int(m.group(1))
        if 0 <= v <= 127:
            nums.append(v)
    result: list[int | None] = list(nums[:n])
    while len(result) < n:
        result.append(None)
    return result


def parse_doremi_sequence(text: str, n: int) -> list[int | None]:
    pcs = extract_all_solfege(text)
    result: list[int | None] = list(pcs[:n])
    while len(result) < n:
        result.append(None)
    return result


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    seqs: list[dict],
    n_trials: int,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict[str, Any]] = []
    for seq in seqs:
        n = seq["n_notes"]
        prompt_midi   = make_prompt_midi(n)
        prompt_abc    = make_prompt_abc(n)
        prompt_doremi = make_prompt_doremi(n)

        for src in SOURCES:
            wav = str(engine.sequence(seq["midi_sequence"], src, TONE_MS, GAP_MS))

            print(f"    n={n} t={seq['trial']} {src}")
            raw_midi, raw_abc, raw_doremi = query_three_formats(
                model_name, wav, prompt_midi, prompt_abc, prompt_doremi
            )

            # ── ABC scoring ───────────────────────────────────────────────────
            pred_abc     = parse_abc_sequence(raw_abc, n)
            abc_per_pos: list[bool | None] = []
            for gt, pred in zip(seq["note_sequence"], pred_abc):
                if pred is None:
                    abc_per_pos.append(None)
                else:
                    dist = semitone_distance(gt, pred)
                    abc_per_pos.append(dist == 0 if dist is not None else False)
            n_abc_correct = sum(1 for v in abc_per_pos if v is True)

            # ── MIDI scoring ──────────────────────────────────────────────────
            pred_midi_seq = parse_midi_sequence(raw_midi, n)
            midi_per_pos: list[bool | None] = []
            for gt, pred in zip(seq["midi_sequence"], pred_midi_seq):
                if pred is None:
                    midi_per_pos.append(None)
                else:
                    midi_per_pos.append(gt == pred)
            n_midi_correct = sum(1 for v in midi_per_pos if v is True)

            # ── Doremi scoring ────────────────────────────────────────────────
            pred_doremi_seq = parse_doremi_sequence(raw_doremi, n)
            doremi_per_pos: list[bool | None] = []
            gt_pcs = [m % 12 for m in seq["midi_sequence"]]
            for gt_pc, pred_pc in zip(gt_pcs, pred_doremi_seq):
                if pred_pc is None:
                    doremi_per_pos.append(None)
                else:
                    diff = abs(gt_pc - pred_pc)
                    doremi_per_pos.append(min(diff, 12 - diff) == 0)
            n_doremi_correct = sum(1 for v in doremi_per_pos if v is True)

            records.append({
                "source":               src,
                "source_type":          "waveform" if src in config.WAVEFORMS else "instrument",
                "n_notes":              n,
                "trial":                seq["trial"],
                "midi_sequence_gt":     str(seq["midi_sequence"]),
                "abc_sequence_gt":      str(seq["note_sequence"]),
                "doremi_sequence_gt":   str(seq["doremi_sequence"]),
                "wav":                  wav,
                # MIDI format
                "midi_pred":            str(pred_midi_seq),
                "midi_per_pos":         str(midi_per_pos),
                "midi_n_correct":       n_midi_correct,
                "midi_sequence_correct":int(n_midi_correct == n),
                # ABC format
                "abc_pred":             str(pred_abc),
                "abc_per_pos":          str(abc_per_pos),
                "abc_n_correct":        n_abc_correct,
                "abc_sequence_correct": int(n_abc_correct == n),
                # Doremi format
                "doremi_pred":          str(pred_doremi_seq),
                "doremi_per_pos":       str(doremi_per_pos),
                "doremi_n_correct":     n_doremi_correct,
                "doremi_sequence_correct": int(n_doremi_correct == n),
                # Raw responses
                "raw_midi":             raw_midi.strip(),
                "raw_abc":              raw_abc.strip(),
                "raw_doremi":           raw_doremi.strip(),
                # Prompts
                "prompt_midi":          prompt_midi,
                "prompt_abc":           prompt_abc,
                "prompt_doremi":        prompt_doremi,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    per_n: dict[int, dict[str, Any]] = {}
    for n in N_NOTES_LIST:
        sub = [r for r in records if r["n_notes"] == n]
        if not sub:
            continue
        total_notes = n * len(sub)
        per_n[n] = {
            "trials":             len(sub),
            "midi_note_acc":      round(sum(r["midi_n_correct"]  for r in sub) / total_notes, 4),
            "midi_seq_acc":       round(sum(r["midi_sequence_correct"] for r in sub) / len(sub),  4),
            "abc_note_acc":       round(sum(r["abc_n_correct"]   for r in sub) / total_notes, 4),
            "abc_seq_acc":        round(sum(r["abc_sequence_correct"]  for r in sub) / len(sub),  4),
            "doremi_note_acc":    round(sum(r["doremi_n_correct"] for r in sub) / total_notes, 4),
            "doremi_seq_acc":     round(sum(r["doremi_sequence_correct"] for r in sub) / len(sub), 4),
        }

    per_position: dict[str, dict[str, float]] = {}
    for n in N_NOTES_LIST:
        sub = [r for r in records if r["n_notes"] == n]
        for pos in range(n):
            key = f"n{n}_pos{pos+1}"
            midi_vals, abc_vals, doremi_vals = [], [], []
            for r in sub:
                mp = ast.literal_eval(r["midi_per_pos"])
                ap = ast.literal_eval(r["abc_per_pos"])
                dp = ast.literal_eval(r["doremi_per_pos"])
                if pos < len(mp) and mp[pos] is not None:
                    midi_vals.append(int(mp[pos]))
                if pos < len(ap) and ap[pos] is not None:
                    abc_vals.append(int(ap[pos]))
                if pos < len(dp) and dp[pos] is not None:
                    doremi_vals.append(int(dp[pos]))
            per_position[key] = {
                "midi":   round(sum(midi_vals)   / len(midi_vals),   4) if midi_vals   else 0.0,
                "abc":    round(sum(abc_vals)    / len(abc_vals),    4) if abc_vals    else 0.0,
                "doremi": round(sum(doremi_vals) / len(doremi_vals), 4) if doremi_vals else 0.0,
            }

    summary: dict[str, Any] = {
        "total_sequences": len(records),
        "per_n":           per_n,
        "per_position":    per_position,
    }

    summary_lines: list[str] = [
        f"  Sources   : {SOURCES}",
        f"  Sequences : {len(records)}  ({N_NOTES_LIST} notes × {n_trials} trials × {len(SOURCES)} sources)",
        "",
        f"  {'N':>3}  {'MIDI note':>9}  {'MIDI seq':>8}  {'ABC note':>9}  {'ABC seq':>8}  {'Do note':>8}  {'Do seq':>7}",
        f"  {'─' * 65}",
    ]
    for n, d in per_n.items():
        summary_lines.append(
            f"  {n:>3}  {d['midi_note_acc']:>9.1%}  {d['midi_seq_acc']:>8.1%}"
            f"  {d['abc_note_acc']:>9.1%}  {d['abc_seq_acc']:>8.1%}"
            f"  {d['doremi_note_acc']:>8.1%}  {d['doremi_seq_acc']:>7.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, n_notes_list=N_NOTES_LIST, n_trials=n_trials,
        seed=seed, pitch_min=PITCH_MIN, pitch_max=PITCH_MAX,
        tone_duration=TONE_DURATION, gap_duration=GAP_DURATION,
        prompt_midi=make_prompt_midi(3), prompt_abc=make_prompt_abc(3),
        prompt_doremi=make_prompt_doremi(3),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    _save_plot(records, run_dir, model_name)
    return summary


def _save_plot(records: list[dict[str, Any]], run_dir: Path, model_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_panels = len(N_NOTES_LIST)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    fig.suptitle(
        f"Multi-pitch sequence accuracy by position — {config.MODELS.get(model_name, model_name)}",
        fontsize=12,
    )

    for ax, n in zip(axes, N_NOTES_LIST):
        sub = [r for r in records if r["n_notes"] == n]
        if not sub:
            continue
        abc_pos, midi_pos, doremi_pos = [], [], []
        for pos in range(n):
            abc_vals, midi_vals, doremi_vals = [], [], []
            for r in sub:
                ap = ast.literal_eval(r["abc_per_pos"])
                mp = ast.literal_eval(r["midi_per_pos"])
                dp = ast.literal_eval(r["doremi_per_pos"])
                if pos < len(ap) and ap[pos] is not None:
                    abc_vals.append(int(ap[pos]))
                if pos < len(mp) and mp[pos] is not None:
                    midi_vals.append(int(mp[pos]))
                if pos < len(dp) and dp[pos] is not None:
                    doremi_vals.append(int(dp[pos]))
            abc_pos.append(sum(abc_vals)    / len(abc_vals)    * 100 if abc_vals    else float("nan"))
            midi_pos.append(sum(midi_vals)  / len(midi_vals)   * 100 if midi_vals   else float("nan"))
            doremi_pos.append(sum(doremi_vals) / len(doremi_vals) * 100 if doremi_vals else float("nan"))

        positions = list(range(1, n + 1))
        ax.plot(positions, abc_pos,    "o-", linewidth=2, label="ABC")
        ax.plot(positions, midi_pos,   "s-", linewidth=2, label="MIDI")
        ax.plot(positions, doremi_pos, "^-", linewidth=2, label="Doremi")
        ax.set_xlabel("Note position")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{n}-note sequence")
        ax.set_xticks(positions)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = run_dir / f"multi_pitch_{model_name}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved  → {p}")


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",  action="store_true")
    parser.add_argument("--n-notes",  nargs="+", type=int, default=N_NOTES_LIST,
                        metavar="N", help=f"Sequence lengths (default: {N_NOTES_LIST})")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                        help=f"Trials per length (default: {DEFAULT_N_TRIALS})")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--models",   nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    seqs = build_sequences(args.n_notes, args.n_trials, args.seed)
    generate_stimuli(seqs)
    total_dur = sum(
        s["n_notes"] * TONE_DURATION + (s["n_notes"] - 1) * GAP_DURATION
        for s in seqs
    )
    print(f"Experiment   : {EXP_NAME}")
    print(f"Sources      : {SOURCES}")
    print(f"Seed         : {args.seed}")
    print(f"Lengths      : {args.n_notes}  × {args.n_trials} trials = {len(seqs)} sequences")
    print(f"Audio files  : {len(seqs) * len(SOURCES)}")
    print(f"Pitch range  : MIDI {PITCH_MIN}–{PITCH_MAX} (C3–C6)")
    print(f"Total audio  : {total_dur:.1f} s  ({total_dur/60:.1f} min)")
    print(f"Queries/model: {len(seqs) * len(SOURCES) * 3}  (MIDI + ABC + doremi)")
    print(f"Audio dir    : {config.AUDIO_DIR}")
    for s in seqs:
        print(f"  n={s['n_notes']} t={s['trial']}  {s['note_sequence']}")
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    seqs = build_sequences(args.n_notes, args.n_trials, args.seed)
    generate_stimuli(seqs)
    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Sequences  : {len(seqs) * len(SOURCES)}  × 3 formats = {len(seqs) * len(SOURCES) * 3} queries/model")
    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict[str, Any]] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, seqs, args.n_trials, args.seed, run_dir)
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
