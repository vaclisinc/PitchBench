"""
Experiment d5 — Pitch trajectory (open-response)
A continuously varying (gliding) pitch is synthesised and the model must
describe its trajectory as a comma-separated sequence of "higher", "lower",
or "same".

Trajectories and their ground-truth sequences:
  flat        → "same"
  up          → "higher"
  down        → "lower"
  up_then_down → "higher, lower"
  down_then_up → "lower, higher"

Parameters:
  start pitches  — 7 representative notes (F#1–G5)
  intervals      — small (4 st), medium (7 st), large (12 st)
                   for arch/valley the end = start (range = 2× interval)
  duration       — 3 s
  sources        — all waveforms (instruments excluded: FluidSynth glide
                   requires pitch-bend which not all presets support)

Scoring: exact match on normalized token sequence.

Usage:
    pitchbench d5
    pitchbench d5 --preview
    pitchbench --id d5 --models audio_flamingo_next_instruct
"""

import argparse
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import midi_to_note
from pitchbench.experiments.helpers.plots import save_accuracy_plots
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

START_PITCHES: list[int] = config.DEFAULT_PITCHES 

INTERVALS_ST: list[int] = [1, 4, 7, 12]   # semitones of change

DURATION_MS = config.DEFAULT_DURATION_MS

SOURCES: list[str] = list(config.WAVEFORMS)  # glide works best on waveforms

TRAJECTORIES: list[dict] = [
    {"name": "flat",         "shape": "linear",  "interval_sign":  0, "gt_seq": ["same"]},
    {"name": "up",           "shape": "linear",  "interval_sign": +1, "gt_seq": ["higher"]},
    {"name": "down",         "shape": "linear",  "interval_sign": -1, "gt_seq": ["lower"]},
    {"name": "up_then_down", "shape": "arch",    "interval_sign": +1, "gt_seq": ["higher", "lower"]},
    {"name": "down_then_up", "shape": "valley",  "interval_sign": -1, "gt_seq": ["lower", "higher"]},
]

PROMPT = (
    "Listen to this audio. Describe how the pitch changes over time as a "
    "comma-separated list using only the words 'higher', 'lower', and 'same'.\n"
    "Examples:\n"
    "  'higher'         — pitch rises throughout\n"
    "  'lower'          — pitch falls throughout\n"
    "  'same'           — pitch stays constant\n"
    "  'higher, lower'  — pitch rises then falls\n"
    "  'lower, higher'  — pitch falls then rises\n"
    "Reply with ONLY the comma-separated list. Nothing else."
)

# ── Response normalisation ────────────────────────────────────────────────────

_SYNONYMS: dict[str, set[str]] = {
    "higher": {
        "higher", "up", "rise", "rises", "rising", "ascend", "ascending",
        "increase", "increases", "increasing", "goes up", "went up",
        "pitch goes up", "pitch rises", "pitch increases",
    },
    "lower": {
        "lower", "down", "fall", "falls", "falling", "descend", "descending",
        "decrease", "decreases", "decreasing", "goes down", "went down",
        "pitch goes down", "pitch falls", "pitch decreases",
    },
    "same": {
        "same", "flat", "constant", "steady", "unchanged", "stays",
        "stable", "equal", "no change", "stays the same", "remains the same",
        "remains constant", "stays constant", "pitch stays", "pitch remains",
    },
}

_WORD_TO_TOKEN: dict[str, str] = {
    s: canonical
    for canonical, syns in _SYNONYMS.items()
    for s in syns
}


def _normalize_token(tok: str) -> str | None:
    return _WORD_TO_TOKEN.get(tok.strip().lower())


def _parse_sequence(raw: str) -> list[str] | None:
    """Return normalised token list, or None if any token is unrecognised."""
    parts = [p.strip() for p in raw.strip().split(",") if p.strip()]
    if not parts:
        return None
    result: list[str] = []
    for p in parts:
        norm = _normalize_token(p)
        if norm is None:
            return None
        result.append(norm)
    return result


# ── Condition builder ─────────────────────────────────────────────────────────

def build_conditions(sources: list[str]) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        for start_midi in START_PITCHES:
            traj = next(t for t in TRAJECTORIES if t["name"] == "flat")
            rows.append({
                "source":      src,
                "start_midi":  start_midi,
                "end_midi":    start_midi,
                "interval_st": 0,
                "traj_name":   "flat",
                "traj_shape":  traj["shape"],
                "gt_seq":      traj["gt_seq"],
            })
            for interval in INTERVALS_ST:
                for traj in TRAJECTORIES:
                    if traj["name"] == "flat":
                        continue
                    end_midi = start_midi + traj["interval_sign"] * interval
                    end_midi = max(12, min(115, end_midi))
                    rows.append({
                        "source":      src,
                        "start_midi":  start_midi,
                        "end_midi":    end_midi,
                        "interval_st": interval,
                        "traj_name":   traj["name"],
                        "traj_shape":  traj["shape"],
                        "gt_seq":      traj["gt_seq"],
                    })
    return rows


def _get_wav(c: dict) -> Path:
    if c["traj_name"] == "flat":
        return engine.tone(c["start_midi"], c["source"], DURATION_MS)
    return engine.glide(
        c["start_midi"], c["end_midi"],
        c["source"], DURATION_MS, c["traj_shape"],
    )


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    run_dir: Path,
    sample_info: dict | None = None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []

    for c in conds:
        wav  = _get_wav(c)
        result = query_alm(model_name, wav, PROMPT)
        raw  = result["result"] or ""

        pred_seq = _parse_sequence(raw)
        gt_str   = ", ".join(c["gt_seq"])
        pred_str = ", ".join(pred_seq) if pred_seq is not None else None
        exact    = int(pred_seq == c["gt_seq"]) if pred_seq is not None else 0

        records.append({
            "source":               c["source"],
            "source_type":          "waveform",
            "start_midi":           c["start_midi"],
            "end_midi":             c["end_midi"],
            "start_note":           midi_to_note(c["start_midi"]),
            "traj_name":            c["traj_name"],
            "interval_st":          c["interval_st"],
            "wav":                  str(wav),
            "prompt":               PROMPT,
            "raw_response":         raw.strip(),
            "trajectory_gt":        gt_str,
            "trajectory_pred":      pred_str,
            "trajectory_correct":   exact,
            # standard plot keys
            "instrument":           c["source"],
            "midi":                 c["start_midi"],
            "prompt_variant":       c["traj_name"],
        })
        sym = "✓" if exact else f"✗(pred={pred_str or '?'})"
        print(f"    {c['traj_name']:14s} start={midi_to_note(c['start_midi']):4s} "
              f"Δ={c['interval_st']:+2d}st {c['source']:10s}  {sym}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    overall = round(sum(r["trajectory_correct"] for r in records) / n, 4) if n else 0.0

    per_traj: dict[str, dict] = {}
    for traj in TRAJECTORIES:
        sub = [r for r in records if r["traj_name"] == traj["name"]]
        if sub:
            per_traj[traj["name"]] = {
                "accuracy": round(sum(r["trajectory_correct"] for r in sub) / len(sub), 4),
                "n": len(sub),
            }

    per_interval: dict[int, float] = {}
    for iv in INTERVALS_ST:
        sub = [r for r in records if r["interval_st"] == iv]
        if sub:
            per_interval[iv] = round(sum(r["trajectory_correct"] for r in sub) / len(sub), 4)

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources  : {SOURCES}",
        f"  Stimuli  : {n}",
        f"  Overall  : {overall:.1%}",
        f"  Chance   : {1/len(TRAJECTORIES):.1%}  ({len(TRAJECTORIES)} classes)",
        "",
        f"  {'Trajectory':16s}  {'Accuracy':>9}",
        f"  {'─' * 28}",
    ]
    for name, d in per_traj.items():
        summary_lines.append(f"  {name:16s}  {d['accuracy']:>9.1%}")
    summary_lines += ["", "  By interval (non-flat):"]
    for iv, acc in per_interval.items():
        summary_lines.append(f"    Δ={iv:+2d}st : {acc:.1%}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    summary = {
        "total": n, "overall": overall,
        "per_trajectory": per_traj,
        "per_interval": {str(k): v for k, v in per_interval.items()},
    }
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, start_pitches=START_PITCHES,
        intervals_st=INTERVALS_ST, duration_ms=DURATION_MS,
        trajectories=[t["name"] for t in TRAJECTORIES],
        prompt=PROMPT,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    save_accuracy_plots(
        records, run_dir, model_name,
        instrument_key="source", pitch_key="midi",
        prompt_key="prompt_variant", accuracy_key="trajectory_correct",
    )
    return {
        "overall": overall,
        **{f"acc_{t['name']}": per_traj.get(t["name"], {}).get("accuracy") for t in TRAJECTORIES},
    }


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    all_conds = build_conditions(SOURCES)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    print(f"Experiment   : {EXP_NAME}")
    print(f"Sources      : {SOURCES}")
    print(f"Trajectories : {[t['name'] for t in TRAJECTORIES]}")
    print(f"Start pitches: {[midi_to_note(m) for m in START_PITCHES]}")
    print(f"Intervals    : {INTERVALS_ST} semitones")
    print(f"Duration     : {DURATION_MS} ms")
    print(f"Stimuli      : {len(conds)}  (generating …)")
    for c in conds:
        _get_wav(c)
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds = build_conditions(SOURCES)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
