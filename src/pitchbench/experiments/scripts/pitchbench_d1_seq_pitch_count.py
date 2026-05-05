"""
d1 — Sequence pitch count.

Question: can the ALM correctly count the number of distinct pitches in a
sequential passage (notes played one after another)?

Universal IVs: duration_ms (per tone), source.
Experiment-specific IVs:
    n:                {1..10}                  (sequence length)
    rhythm:           {regular, irregular}     (irregular gaps drawn from a
                                                seeded uniform distribution)

Usage::
    pitchbench --id d1 --preview
    pitchbench --id d1 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import random
import re
import wave
from pathlib import Path

import numpy as np

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_d import CatDSpec, run_cat_d_experiment
from pitchbench.experiments.helpers.music import midi_to_note

EXP_NAME         = Path(__file__).stem
N_COUNTS         = config.pitchbench_d1_N_COUNTS
RHYTHMS          = config.pitchbench_d1_RHYTHMS
DEFAULT_GAP_MS   = config.pitchbench_d1_GAP_MS
DEFAULT_N_TRIALS = config.pitchbench_d1_N_TRIALS
DEFAULT_SEED     = config.pitchbench_d1_SEED
PITCH_MIN        = config.pitchbench_d1_PITCH_MIN
PITCH_MAX        = config.pitchbench_d1_PITCH_MAX
SOURCES          = config.pitchbench_d1_SOURCES
DURATIONS_MS     = config.pitchbench_d1_DURATIONS_MS

PROMPT = (
    "Listen to this audio. How many distinct musical pitches are played in "
    "this sequence? No note is played twice-so don't worry about duplicates. "
    "Reply with ONLY a single integer. Nothing else. Output only the answer."
)


def build_conditions() -> list[dict]:
    rng = random.Random(DEFAULT_SEED)
    rows: list[dict] = []
    for src in SOURCES:
        for dur in DURATIONS_MS:
            for n in N_COUNTS:
                for trial in range(DEFAULT_N_TRIALS):
                    midis = sorted(rng.sample(range(PITCH_MIN, PITCH_MAX + 1), n))
                    presented = list(midis)
                    rng.shuffle(presented)
                    for rhythm in RHYTHMS:
                        if rhythm == "regular":
                            gaps_ms = [DEFAULT_GAP_MS] * (n - 1)
                        else:
                            gaps_ms = [
                                rng.randint(DEFAULT_GAP_MS // 2, DEFAULT_GAP_MS * 3)
                                for _ in range(n - 1)
                            ]
                        rows.append({
                            "source":      src,
                            "duration_ms": dur,
                            "n":           n,
                            "rhythm":      rhythm,
                            "gaps_ms":     gaps_ms,
                            "midi_set":    midis,
                            "presented":   presented,
                            "trial":       trial,
                        })
    return rows


def wav_for(c: dict) -> Path:
    """Concatenate per-position tones with per-gap silences."""
    SR = engine.SR
    parts: list[np.ndarray] = []
    for i, m in enumerate(c["presented"]):
        wav_path = engine.tone(m, c["source"], c["duration_ms"])
        with wave.open(str(wav_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        arr = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0)
        parts.append(arr)
        if i < len(c["presented"]) - 1:
            parts.append(np.zeros(int(SR * c["gaps_ms"][i] / 1000), dtype=np.float32))
    audio = np.concatenate(parts)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak

    notes_slug   = "-".join(engine._note_slug(m) for m in c["presented"])
    rhythm_slug  = "reg" if c["rhythm"] == "regular" else "irr"
    name = (
        f"{notes_slug}_{c['source']}_seqcount_n{c['n']}_dur{c['duration_ms']}ms_"
        f"{rhythm_slug}_t{c['trial']}.wav"
    )
    out = engine._audio_dir() / name
    if not out.exists():
        engine._write_wav(out, audio)
    return out


def prompts_for(_: dict) -> dict[str, str]:
    return {"main": PROMPT}


def _parse_count(text: str) -> int | None:
    m = re.search(r"\b(\d+)\b", (text or "").strip())
    return int(m.group(1)) if m else None


def record_for(c: dict, wav: str, responses: dict[str, str]) -> dict:
    raw  = responses["main"]
    pred = _parse_count(raw)
    ok   = (pred == c["n"]) if pred is not None else False
    off  = abs(pred - c["n"]) if pred is not None else None
    return {
        "source":        c["source"],
        "duration_ms":   c["duration_ms"],
        "n":             c["n"],
        "rhythm":        c["rhythm"],
        "trial":         c["trial"],
        "midi_set":      str(c["midi_set"]),
        "note_set":      ", ".join(midi_to_note(m) for m in c["midi_set"]),
        "presented":     str(c["presented"]),
        "raw_response":  raw,
        "count_pred":    pred,
        "count_correct": int(ok),
        "off_by":        off,
    }


SPEC = CatDSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="count",
    prompts_fn=prompts_for,
    record_fn=record_for,
    headline_metrics=("count",),
    record_extras=("duration_ms", "n", "rhythm", "trial"),
    label_fn=lambda j: (
        f"n={j['cond']['n']:>2} {j['cond']['rhythm']:>9} "
        f"{j['cond']['source']:>10} dur={j['cond']['duration_ms']:>5}ms "
        f"t={j['cond']['trial']}"
    ),
)


def preview() -> None:
    run_cat_d_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_d_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
