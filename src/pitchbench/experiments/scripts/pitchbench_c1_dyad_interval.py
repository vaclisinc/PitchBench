"""
c1 — Dyad interval identification.

Question: given two simultaneous tones, can the ALM correctly name the
musical interval between them?

Universal IVs: duration_ms, source.
Experiment-specific IVs:
    same_instrument:  {True, False}              (single timbre vs mixed)
    root_midi:        ∈ DEFAULT_PITCHES
    interval_st:      {1..12}                    (m2..octave)

Fixed conditions: equal level, identical onset/offset, root-position dyad.
Mixed-source dyads use deterministic per-position source assignment for
``same_instrument=False``.

Scoring: ``extract_interval`` parses semitone-count from the response;
``interval_correct`` = predicted == ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    INTERVAL_NAMES, extract_interval, midi_to_note,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies,
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

INTERVALS_ST: list[int] = list(range(1, 13))                 # m2 .. P8
SOURCES:      list[str] = config.ALL_SOURCES
SAME_INSTRUMENT_OPTS: list[bool] = [True, False]

PROMPT = (
    "This audio contains two simultaneous musical notes. "
    "How many semitones apart are they? "
    "Reply with ONLY the integer number of semitones."
)


# ── Conditions ────────────────────────────────────────────────────────────────

def _mixed_pair(seed_int: int) -> tuple[str, str]:
    """Deterministic 2-source pair for same_instrument=False."""
    pool = list(SOURCES)
    rng_idx = seed_int % len(pool)
    a = pool[rng_idx]
    b = pool[(rng_idx + 1 + (seed_int >> 8) % (len(pool) - 1)) % len(pool)]
    return a, b


def build_conditions(durations_ms: list[int], pitches: list[int]) -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for root in pitches:
            for iv in INTERVALS_ST:
                pair = root + iv
                if pair > 96:
                    continue
                for dur in durations_ms:
                    for same in SAME_INSTRUMENT_OPTS:
                        if same:
                            src_arg = src
                            src_label = src
                        else:
                            src_arg = list(_mixed_pair(hash((src, root, iv, dur))))
                            src_label = "+".join(src_arg)
                        rows.append({
                            "duration_ms":     dur,
                            "source":          src_label,
                            "_source_arg":     src_arg,
                            "same_instrument": same,
                            "root_midi":       root,
                            "interval_st":     iv,
                            "midis":           [root, pair],
                        })
    return rows


def _wav_for(c: dict) -> Path:
    return engine.chord(c["midis"], c["_source_arg"], c["duration_ms"])


def _interval_label(iv: int) -> str:
    return INTERVAL_NAMES[iv][0] if iv in INTERVAL_NAMES else str(iv)


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        try:
            wav = str(_wav_for(c))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        jobs.append({"wav": wav, "cond": c})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict) -> dict:
        c    = job["cond"]
        out  = query_alm(model_name, job["wav"], PROMPT)
        raw  = (out["result"] or "").strip()
        pred = extract_interval(raw)
        ok   = (pred == c["interval_st"]) if pred is not None else False
        return {
            "duration_ms":      c["duration_ms"],
            "source":           c["source"],
            "same_instrument":  c["same_instrument"],
            "root_midi":        c["root_midi"],
            "root_note":        midi_to_note(c["root_midi"]),
            "interval_st":      c["interval_st"],
            "interval_name":    _interval_label(c["interval_st"]),
            "midi_pair":        f"{c['midis'][0]}+{c['midis'][1]}",
            "wav":              job["wav"],
            "raw_response":     raw,
            "interval_pred":    pred,
            "interval_correct": int(ok),
            "prompt":           PROMPT,
            "model_params":     out["model_params"],
        }

    raw_results = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"iv={j['cond']['interval_st']:>2}st root={midi_to_note(j['cond']['root_midi']):4s} same={str(j['cond']['same_instrument']):5s} dur={j['cond']['duration_ms']:>5}ms",
        result_label_fn=lambda j, r: audit_line(
            f"iv={j['cond']['interval_st']:>2}st root={midi_to_note(j['cond']['root_midi']):4s} same={str(j['cond']['same_instrument']):5s}",
            gt=r['interval_st'],
            pred=r['interval_pred'],
            correct=bool(r['interval_correct']),
        ),
    )
    records: list[dict] = [r for r in raw_results if r is not None]

    n  = len(records)
    n_ok = sum(r["interval_correct"] for r in records)
    summary = {
        "total":    n,
        "correct":  n_ok,
        "accuracy": round(n_ok / n, 4) if n else None,
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli  : {n}",
        f"  Accuracy : {n_ok}/{n} ({n_ok/max(1,n):.1%})",
    ]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, intervals_st=INTERVALS_ST,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        prompt=PROMPT,
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=("interval_correct",),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {SOURCES}")
    print(f"Stimuli    : {len(conds)}")
    print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args  = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError: pass

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for m in target_models:
        all_summaries[m] = run_one_model(m, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
