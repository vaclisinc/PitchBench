"""
Experiment 09 — Simultaneous pitches (chords)
Tests whether models can identify individual pitches when they sound at the
same time. Covers dyads (all 13 intervals), triads (maj/min/dim/aug), and
seventh chords (dom7/maj7/min7) across all 12 roots.

Three prompt variants per stimulus:
  MIDI:    list all MIDI note numbers
  ABC:     list all note names
  Solfège: list all solfège syllable and accidentals (if needed)

Scoring: set-level exact match (order-agnostic) + per-note recall.

Usage:
    python experiments/run.py exp_9_simultaneous_pitches
    python experiments/run.py exp_9_simultaneous_pitches --preview
    python experiments/run.py exp_9_simultaneous_pitches --models audio_flamingo_next_instruct
    python experiments/run.py exp_9_simultaneous_pitches --sources sine piano
"""

import argparse
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.audit import audit_line
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    SOLFEGE_TO_PC,
    extract_all_notes,
    midi_to_note,
    midi_to_solfege,
    note_to_midi,
)
from pitchbench.experiments.helpers.plots import save_accuracy_plots
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

# Data-generation parameters (sourced from config.pitchbench_c3_*)
TONE_DURATION_MS = config.pitchbench_c3_TONE_DURATION_MS
SOURCES          = config.pitchbench_c3_SOURCES
CHORD_TYPES      = config.pitchbench_c3_CHORD_TYPES
BASE_ROOTS       = config.pitchbench_c3_BASE_ROOTS

PROMPT_MIDI = (
    "This audio contains multiple musical pitches played simultaneously. "
    "List ALL MIDI note numbers you hear, from lowest to highest. "
    "Reply with ONLY the integers separated by spaces. Nothing else. Output only the answer."
)

PROMPT_ABC = (
    "This audio contains multiple musical pitches played simultaneously. "
    "List ALL note names you hear, from lowest to highest, expressed in Scientific Pitch Notation. "
    "Reply with ONLY the note names expressed in Scientific Pitch Notation, separated by spaces, e.g. C4 E4 G#4. Nothing else. Output only the answer."
)

PROMPT_DOREMI = (
    "This audio contains multiple musical pitches played simultaneously. "
    "List ALL solfège syllable and accidentals (if needed) you hear (fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B). "
    "Reply with ONLY the syllables with accidental (if needed) separated by spaces, e.g. do mi sol#. Nothing else. Output only the answer."
)

PROMPTS: dict[str, str] = {
    "midi":   PROMPT_MIDI,
    "abc":    PROMPT_ABC,
    "doremi": PROMPT_DOREMI,
}


# ── Conditions ────────────────────────────────────────────────────────────────

def build_conditions(sources: list[str]) -> list[dict]:
    rows: list[dict] = []
    for chord_name, intervals in CHORD_TYPES.items():
        for root in BASE_ROOTS:
            midi_notes = [root + iv for iv in intervals]
            if any(m < 0 or m > 127 for m in midi_notes):
                continue
            note_seq   = [midi_to_note(m) for m in midi_notes]
            doremi_seq = [midi_to_solfege(m) for m in midi_notes]
            for source in sources:
                rows.append({
                    "chord_type": chord_name,
                    "n_notes":    len(midi_notes),
                    "root_midi":  root,
                    "root_note":  midi_to_note(root),
                    "midi_notes": midi_notes,
                    "note_names": note_seq,
                    "doremi_seq": doremi_seq,
                    "source":     source,
                })
    return rows


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        try:
            engine.chord(c["midi_notes"], c["source"], TONE_DURATION_MS)
        except ValueError:
            pass


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_midi_list(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"\b(\d{1,3})\b", text) if 0 <= int(m) <= 127]


def parse_abc_list(text: str) -> list[str]:
    return extract_all_notes(text)


def parse_solfege_list(text: str) -> list[int]:
    results = []
    for syl, pc in SOLFEGE_TO_PC.items():
        if re.search(rf"\b{re.escape(syl)}\b", text.lower()):
            results.append(pc)
    return list(dict.fromkeys(results))  # preserve order, deduplicate


def score_set(gt: list, pred: list) -> dict:
    gt_set   = set(gt)
    pred_set = set(pred)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    recall    = tp / len(gt_set) if gt_set else 1.0
    precision = tp / len(pred_set) if pred_set else 0.0
    exact     = (gt_set == pred_set)
    return {
        "exact_match": int(exact),
        "recall":      round(recall, 4),
        "precision":   round(precision, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    sources: list[str],
    run_dir: Path,
    sample_info: dict | None = None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially. One job per (cond, variant).
    jobs: list[dict] = []
    for c in conds:
        try:
            wav = str(engine.chord(c["midi_notes"], c["source"], TONE_DURATION_MS))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        for variant, prompt in PROMPTS.items():
            jobs.append({"wav": wav, "cond": c, "variant": variant, "prompt": prompt})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        variant = job["variant"]
        prompt  = job["prompt"]
        raw = query_alm(model_name, job["wav"], prompt)["result"]
        if variant == "midi":
            pred = parse_midi_list(raw)
            scores = score_set(c["midi_notes"], pred)
        elif variant == "abc":
            pred_notes = parse_abc_list(raw)
            pred_midi  = [note_to_midi(n) for n in pred_notes if note_to_midi(n) is not None]
            scores = score_set(c["midi_notes"], pred_midi)
        else:  # doremi — compare pitch classes
            pred_pcs = parse_solfege_list(raw)
            gt_pcs   = [m % 12 for m in c["midi_notes"]]
            scores = score_set(gt_pcs, pred_pcs)
        return {
            "chord_type":     c["chord_type"],
            "n_notes":        c["n_notes"],
            "root_midi":      c["root_midi"],
            "root_note":      c["root_note"],
            "midi_notes":     str(c["midi_notes"]),
            "note_names":     str(c["note_names"]),
            "source":         c["source"],
            "wav":            job["wav"],
            "prompt_variant": variant,
            "prompt":         prompt,
            "raw_response":   raw.strip(),
            # standard plot keys
            "instrument":     c["source"],
            "midi":           c["root_midi"],
            **scores,
        }

    raw_results = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"[{j['variant']:6s}] {j['cond']['chord_type']:12s} r={j['cond']['root_note']:3s} {j['cond']['source']:12s}",
        result_label_fn=lambda j, r: audit_line(
            f"[{j['variant']:6s}] {j['cond']['chord_type']:12s} r={j['cond']['root_note']:3s} {j['cond']['source']:12s}",
            gt=r['midi_notes'],
            pred=r.get('raw_response'),
            score=r.get('exact_match'),
            correct=bool(r.get('exact_match')),
        ),
    )
    records: list[dict] = [r for r in raw_results if r is not None]

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_chord: dict[str, dict] = {}
    for chord_name in CHORD_TYPES:
        sub = [r for r in records if r["chord_type"] == chord_name]
        if not sub:
            continue
        recall_vals = [r["recall"] for r in sub]
        per_chord[chord_name] = {
            "n":              len(sub),
            "exact_accuracy": round(sum(r["exact_match"] for r in sub) / len(sub), 4),
            "mean_recall":    round(sum(recall_vals) / len(recall_vals), 4),
        }

    per_variant: dict[str, dict] = {}
    for variant in PROMPTS:
        sub = [r for r in records if r["prompt_variant"] == variant]
        if not sub:
            continue
        recall_vals = [r["recall"] for r in sub]
        per_variant[variant] = {
            "n":              len(sub),
            "exact_accuracy": round(sum(r["exact_match"] for r in sub) / len(sub), 4),
            "mean_recall":    round(sum(recall_vals) / len(recall_vals), 4),
        }

    summary = {
        "total":       n,
        "per_chord":   per_chord,
        "per_variant": per_variant,
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources : {sources}",
        f"  Stimuli : {n // len(PROMPTS)}  chords × {len(PROMPTS)} variants = {n} queries",
        "",
        f"  {'Variant':10s}  {'n':>5}  {'Exact%':>8}  {'Recall%':>8}",
        f"  {'─' * 38}",
    ]
    for v, d in per_variant.items():
        summary_lines.append(
            f"  {v:10s}  {d['n']:>5}  {d['exact_accuracy']:>8.1%}  {d['mean_recall']:>8.1%}"
        )
    summary_lines += ["", "  Per chord type (exact accuracy):"]
    for chord_name, d in per_chord.items():
        summary_lines.append(f"    {chord_name:14s}: n={d['n']:>4}  {d['exact_accuracy']:.1%}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=sources, chord_types=list(CHORD_TYPES.keys()),
        base_roots=BASE_ROOTS, tone_duration_ms=TONE_DURATION_MS,
        prompts=PROMPTS,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    save_accuracy_plots(
        records, run_dir, model_name,
        instrument_key="source",
        pitch_key="n_notes",
        prompt_key="prompt_variant",
        accuracy_key="exact_match",
    )
    return summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    all_sources = config.WAVEFORMS + list(config.GM_PROGRAMS_V1.keys())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",  action="store_true")
    parser.add_argument("--models",   nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sources",  nargs="+", metavar="SRC", default=None,
                        help=f"Sources to use (default: all). Available: {all_sources}")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    sources = args.sources or SOURCES
    all_conds = build_conditions(sources)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)
    print(f"Experiment  : {EXP_NAME}")
    print(f"Sources     : {sources}")
    print(f"Chord types : {len(CHORD_TYPES)}")
    print(f"Roots       : {len(BASE_ROOTS)}")
    print(f"Stimuli     : {len(conds)} audio files × {len(PROMPTS)} variants = {len(conds)*len(PROMPTS)} queries/model")
    print(f"Audio dir   : {config.AUDIO_DIR}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = args.sources or SOURCES
    all_conds = build_conditions(sources)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    generate_stimuli(conds)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)} × {len(PROMPTS)} variants = {len(conds)*len(PROMPTS)} queries/model")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, conds, sources, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
