"""
d4 — Discrete melodic contour.

Question: given a sequence of separate notes, can the ALM correctly identify
the directional shape of the melody? The model is asked to output a
comma-separated list of ``up`` / ``down`` tokens, one per transition.

Universal IVs: duration_ms (per note), source.
Experiment-specific IVs:
    base_midi:        ∈ DEFAULT_PITCHES
    n_transitions:    {2, 3, 5, 7}
    step_size_st:     {1, 2, 4, 7}
    note_duration_ms: {250, 500, 1000}

Headline metric: ``sequence_correct`` — binary exact-match across all
transitions.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_d import CatDSpec, run_cat_d_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME                = Path(__file__).stem
N_TRANSITIONS_OPTS      = config.pitchbench_d4_N_TRANSITIONS_OPTS
STEP_SIZES_ST           = config.pitchbench_d4_STEP_SIZES_ST
NOTE_DURATIONS_MS       = config.pitchbench_d4_NOTE_DURATIONS_MS
DEFAULT_TRIALS_PER_CELL = config.pitchbench_d4_TRIALS_PER_CELL
DEFAULT_SEED            = config.pitchbench_d4_SEED
SOURCES                 = config.pitchbench_d4_SOURCES
PITCHES                 = config.pitchbench_d4_PITCHES

PROMPT = (
    "Listen to this sequence of separate musical notes. For each TRANSITION "
    "between consecutive notes, reply with 'up' or 'down', comma-separated, "
    "in order. Example for a 4-note ascending-then-descending pattern: "
    "'up, up, down'. Reply with ONLY the list."
)

_TOKEN_RE = re.compile(r"\b(up|down)\b", re.IGNORECASE)


def _build_pattern(n_transitions: int, seed: int) -> list[str]:
    """Deterministic ``up``/``down`` sequence avoiding all-up or all-down patterns."""
    rng = random.Random(seed)
    while True:
        seq = [rng.choice(["up", "down"]) for _ in range(n_transitions)]
        if not (all(s == "up" for s in seq) or all(s == "down" for s in seq)) or n_transitions <= 1:
            return seq


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for base in PITCHES:
            for n_t in N_TRANSITIONS_OPTS:
                for step in STEP_SIZES_ST:
                    for note_dur in NOTE_DURATIONS_MS:
                        for trial in range(DEFAULT_TRIALS_PER_CELL):
                            cell_seed = (DEFAULT_SEED ^ hash((src, base, n_t, step, note_dur, trial))) & 0xFFFFFFFF
                            pattern = _build_pattern(n_t, cell_seed)
                            midis = [base]
                            cur = base
                            for d in pattern:
                                cur = cur + step if d == "up" else cur - step
                                midis.append(cur)
                            if min(midis) < 12 or max(midis) > 96:
                                continue
                            rows.append({
                                "source":           src,
                                "duration_ms":      note_dur,
                                "base_midi":        base,
                                "n_transitions":    n_t,
                                "step_size_st":     step,
                                "note_duration_ms": note_dur,
                                "pattern":          pattern,
                                "midis":            midis,
                                "seed":             cell_seed,
                            })
    return rows


def wav_for(c: dict) -> Path:
    gap = c["note_duration_ms"] // 4
    return engine.sequence(c["midis"], c["source"], c["note_duration_ms"], gap)


def prompts_for(_: dict) -> dict[str, str]:
    return {"main": PROMPT}


def _parse_pattern(text: str) -> list[str]:
    return [m.group(1).lower() for m in _TOKEN_RE.finditer(text or "")]


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw  = responses["main"]
    pred = _parse_pattern(raw)
    gt   = c["pattern"]
    seq_ok = (pred == gt)
    return {
        "duration_ms":          c["note_duration_ms"],
        "source":               c["source"],
        "base_midi":            c["base_midi"],
        "n_transitions":        c["n_transitions"],
        "step_size_st":         c["step_size_st"],
        "note_duration_ms":     c["note_duration_ms"],
        "midi_seq":             ",".join(str(m) for m in c["midis"]),
        "note_seq":             ",".join(midi_to_note(m) for m in c["midis"]),
        "pattern_gt":           ",".join(gt),
        "pattern_pred":         ",".join(pred),
        "sequence_correct":     int(seq_ok),
        "raw_response":         raw,
        "seed":                 c["seed"],
    }


SPEC = CatDSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="contour",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("sequence",),
    record_extras=(
        "duration_ms", "base_midi", "n_transitions", "step_size_st",
        "note_duration_ms",
    ),
    label_fn=lambda j: (
        f"nT={j['cond']['n_transitions']} step={j['cond']['step_size_st']}st "
        f"dur={j['cond']['note_duration_ms']:>4}ms {j['cond']['source']}"
    ),
)


def preview() -> None:
    run_cat_d_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_d_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
