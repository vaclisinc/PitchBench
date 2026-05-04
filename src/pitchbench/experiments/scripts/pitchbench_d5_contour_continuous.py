"""
Experiment 11 — Pitch trajectory
A continuously varying (gliding) pitch is synthesised and the model must
describe its trajectory: does the pitch go up, down, up then down, etc.?

The experiment probes two capabilities:
  (a) directional sensitivity — can the model detect the direction of change?
  (b) shape discrimination    — can it distinguish linear, arch, and valley?

Trajectories:
  flat        — constant pitch (no change)              label: "flat"
  up          — linearly rising                         label: "up"
  down        — linearly falling                        label: "down"
  arch        — rises then falls (peak in the middle)   label: "up_then_down"
  valley      — falls then rises (trough in the middle) label: "down_then_up"

Parameters:
  start pitches  — 5 representative notes (C3, G3, C4, G4, C5)
  intervals      — small (4 st), medium (7 st), large (12 st)
                   for arch/valley the end = start (range = 2× interval)
  duration       — 3 s
  sources        — all waveforms (instruments excluded: FluidSynth glide
                   requires pitch-bend which not all presets support)

Prompt (multiple-choice, forced single letter):
  "Listen to this audio. The pitch of the sound changes over time.
   Which best describes the pitch trajectory?
   (A) goes up  (B) goes down  (C) goes up then down
   (D) goes down then up  (E) stays the same
   Reply with ONLY the letter A, B, C, D, or E."

Scoring: exact match on trajectory label.

Usage:
    python experiments/run.py exp_11_pitch_trajectory
    python experiments/run.py exp_11_pitch_trajectory --preview
    python experiments/run.py exp_11_pitch_trajectory --models audio_flamingo_next_instruct
"""

import argparse
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.music import midi_to_note
from pitchbench.experiments.helpers.plots import save_accuracy_plots
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results

EXP_NAME = Path(__file__).stem

START_PITCHES: list[int] = [30, 48, 55, 60, 67, 72, 82]   # F#1, C3, G3, C4, G4, C5, G5
#       

INTERVALS_ST: list[int] = [4, 7, 12]   # semitones of change

DURATION_MS = 3_000

SOURCES: list[str] = list(config.WAVEFORMS)  # glide works best on waveforms

# Trajectory definitions: (shape, answer_letter, answer_label)
#   shape    → passed to engine.glide()
#   label    → ground truth for scoring
#   For "flat" we use engine.tone(); end_midi = start_midi, shape irrelevant
TRAJECTORIES: list[dict] = [
    {"name": "flat",         "letter": "E", "shape": "linear",  "interval_sign":  0},
    {"name": "up",           "letter": "A", "shape": "linear",  "interval_sign": +1},
    {"name": "down",         "letter": "B", "shape": "linear",  "interval_sign": -1},
    {"name": "up_then_down", "letter": "C", "shape": "arch",    "interval_sign": +1},
    {"name": "down_then_up", "letter": "D", "shape": "valley",  "interval_sign": -1},
]

PROMPT = (
    "Listen to this audio. The pitch of the sound changes (or stays the same) "
    "over time. Which best describes the pitch trajectory?\n"
    "(A) goes up\n"
    "(B) goes down\n"
    "(C) goes up then down\n"
    "(D) goes down then up\n"
    "(E) stays the same\n"
    "Reply with ONLY the letter A, B, C, D, or E. Nothing else."
)

_LETTER_RE = re.compile(r"\b([A-Ea-e])\b")


def _parse_letter(raw: str) -> str | None:
    m = _LETTER_RE.search(raw.strip())
    return m.group(1).upper() if m else None


def build_conditions(sources: list[str]) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        for start_midi in START_PITCHES:
            # flat — no interval
            traj = next(t for t in TRAJECTORIES if t["name"] == "flat")
            rows.append({
                "source":       src,
                "start_midi":   start_midi,
                "end_midi":     start_midi,
                "interval_st":  0,
                "traj_name":    "flat",
                "traj_letter":  traj["letter"],
                "traj_shape":   traj["shape"],
            })
            for interval in INTERVALS_ST:
                for traj in TRAJECTORIES:
                    if traj["name"] == "flat":
                        continue
                    end_midi = start_midi + traj["interval_sign"] * interval
                    end_midi = max(12, min(115, end_midi))  # keep in audible range
                    rows.append({
                        "source":      src,
                        "start_midi":  start_midi,
                        "end_midi":    end_midi,
                        "interval_st": interval,
                        "traj_name":   traj["name"],
                        "traj_letter": traj["letter"],
                        "traj_shape":  traj["shape"],
                    })
    return rows


def _get_wav(c: dict) -> Path:
    if c["traj_name"] == "flat":
        return engine.tone(c["start_midi"], c["source"], DURATION_MS)
    return engine.glide(
        c["start_midi"], c["end_midi"],
        c["source"], DURATION_MS, c["traj_shape"],
    )


def run_one_model(
    model_name: str,
    sources: list[str],
    run_dir: Path,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    conds   = build_conditions(sources)
    records: list[dict] = []

    for c in conds:
        wav = _get_wav(c)
        raw = query_alm(model_name, str(wav)["result"], PROMPT)
        pred_letter = _parse_letter(raw)
        exact = int(pred_letter == c["traj_letter"]) if pred_letter else 0

        records.append({
            "source":            c["source"],
            "source_type":       "waveform",
            "start_midi":        c["start_midi"],
            "end_midi":          c["end_midi"],
            "start_note":        midi_to_note(c["start_midi"]),
            "traj_name":         c["traj_name"],
            "interval_st":       c["interval_st"],
            "wav":               str(wav),
            "prompt":            PROMPT,
            "raw_response":      raw.strip(),
            "trajectory_gt":     c["traj_letter"],
            "trajectory_pred":   pred_letter,
            "trajectory_correct": exact,
            "within_1":          exact,   # for plot compat
            # standard plot keys
            "instrument":        c["source"],
            "midi":              c["start_midi"],
            "prompt_variant":    c["traj_name"],
        })
        sym = "✓" if exact else f"✗(pred={pred_letter or '?'})"
        print(f"    {c['traj_name']:14s} start={midi_to_note(c['start_midi']):4s} "
              f"Δ={c['interval_st']:+2d}st {c['source']:10s}  {sym}  {Path(wav).name}")

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

    summary_lines = [
        f"  Sources  : {sources}",
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
        sources=sources, start_pitches=START_PITCHES,
        intervals_st=INTERVALS_ST, duration_ms=DURATION_MS,
        trajectories=[t["name"] for t in TRAJECTORIES],
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    save_accuracy_plots(
        records, run_dir, model_name,
        instrument_key="source", pitch_key="midi",
        prompt_key="prompt_variant", accuracy_key="trajectory_correct",
    )
    return {"overall": overall, **{f"acc_{t['name']}": per_traj.get(t["name"], {}).get("accuracy") for t in TRAJECTORIES}}


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    conds = build_conditions(SOURCES)
    print(f"Experiment   : {EXP_NAME}")
    print(f"Sources      : {SOURCES}")
    print(f"Trajectories : {[t['name'] for t in TRAJECTORIES]}")
    print(f"Start pitches: {[midi_to_note(m) for m in START_PITCHES]}")
    print(f"Intervals    : {INTERVALS_ST} semitones")
    print(f"Duration     : {DURATION_MS} ms")
    print(f"Stimuli      : {len(conds)}  (generating …)")
    for c in conds:
        _get_wav(c)
    print(f"Queries/model: {len(conds)}")
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, SOURCES, run_dir)
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
