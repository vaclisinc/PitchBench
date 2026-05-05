"""
d5 — Pitch trajectory (continuous glide).

A continuously varying (gliding) pitch is synthesised and the model must
describe its trajectory as a comma-separated sequence of ``up`` / ``down``
tokens — the same vocabulary as d4. The sequence MUST strictly alternate
(``up, down, up, down`` is valid; ``up, up, down`` is not).

Trajectories and ground-truth sequences:
  up           → "up"
  down         → "down"
  up_then_down → "up, down"
  down_then_up → "down, up"

Headline metric: ``trajectory_correct`` — binary exact match on the
alternating up/down sequence (with synonym normalisation).
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_d import CatDSpec, run_cat_d_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME      = Path(__file__).stem
START_PITCHES = config.pitchbench_d5_START_PITCHES
INTERVALS_ST  = config.pitchbench_d5_INTERVALS_ST
DURATION_MS   = config.pitchbench_d5_DURATION_MS
SOURCES       = config.pitchbench_d5_SOURCES
TRAJECTORIES  = config.pitchbench_d5_TRAJECTORIES

PROMPT = (
    "Listen to this audio. Describe how the pitch changes over time as a "
    "comma-separated list using ONLY the words 'up' and 'down', alternating. "
    "Never write the same direction twice in a row — count each movement as a single 'up' or 'down'. A change of "
    "direction asks for a new token.\n"
    "Examples:\n"
    "  'up'           — pitch rises throughout\n"
    "  'down'         — pitch falls throughout\n"
    "  'up, down'     — pitch rises then falls\n"
    "  'down, up'     — pitch falls then rises\n"
    "Reply with ONLY the comma-separated list. Nothing else."
)

_SYNONYMS: dict[str, set[str]] = {
    "up": {
        "up", "higher", "rise", "rises", "rising", "ascend", "ascending",
        "increase", "increases", "increasing", "goes up", "went up",
        "pitch goes up", "pitch rises", "pitch increases",
    },
    "down": {
        "down", "lower", "fall", "falls", "falling", "descend", "descending",
        "decrease", "decreases", "decreasing", "goes down", "went down",
        "pitch goes down", "pitch falls", "pitch decreases",
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
    parts = [p.strip() for p in (raw or "").strip().split(",") if p.strip()]
    if not parts:
        return None
    result: list[str] = []
    for p in parts:
        norm = _normalize_token(p)
        if norm is None:
            return None
        result.append(norm)
    return result


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for start_midi in START_PITCHES:
            for interval in INTERVALS_ST:
                for traj in TRAJECTORIES:
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


def wav_for(c: dict) -> Path:
    return engine.glide(
        c["start_midi"], c["end_midi"],
        c["source"], DURATION_MS, c["traj_shape"],
    )


def prompts_for(_: dict) -> dict[str, str]:
    return {"main": PROMPT}


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw      = responses["main"]
    pred_seq = _parse_sequence(raw)
    gt_str   = ", ".join(c["gt_seq"])
    pred_str = ", ".join(pred_seq) if pred_seq is not None else None
    exact    = int(pred_seq == c["gt_seq"]) if pred_seq is not None else 0
    return {
        "source":             c["source"],
        "source_type":        "waveform",
        "start_midi":         c["start_midi"],
        "end_midi":           c["end_midi"],
        "start_note":         midi_to_note(c["start_midi"]),
        "traj_name":          c["traj_name"],
        "interval_st":        c["interval_st"],
        "raw_response":       (raw or "").strip(),
        "trajectory_gt":      gt_str,
        "trajectory_pred":    pred_str,
        "trajectory_correct": exact,
    }


SPEC = CatDSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="trajectory",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("trajectory",),
    record_extras=("start_midi", "end_midi", "interval_st", "traj_name"),
    label_fn=lambda j: (
        f"{j['cond']['traj_name']:14s} "
        f"start={midi_to_note(j['cond']['start_midi']):4s} "
        f"Δ={j['cond']['interval_st']:+2d}st {j['cond']['source']:10s}"
    ),
)


def preview() -> None:
    run_cat_d_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_d_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
