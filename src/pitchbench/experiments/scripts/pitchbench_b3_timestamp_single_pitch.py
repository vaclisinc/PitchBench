"""
b2 — Onset/offset detection of a single tone in silence.

Universal IVs: source, source_type, midi, duration_ms.
Experiment-specific IVs: pos_ms.

The model receives a single universal prompt asking for the onset and
offset of a single sustained tone embedded in a 60 s clip. Scoring uses
:func:`cat_b.score_timestamps` against the GT pair
``[pos_ms / 1000, (pos_ms + duration_ms) / 1000]`` with the tolerance from
``config.BENCHMARK_TIMESTAMP_TOLERANCE_MS`` — only ``correct`` flows into
the headline summary.

Usage::
    pitchbench --id b2 --preview
    pitchbench --id b2 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.cat_b import CatBSpec, run_cat_b_experiment

EXP_NAME     = Path(__file__).stem
SOURCES      = config.pitchbench_b3_SOURCES
PITCHES      = config.pitchbench_b3_PITCHES
DURATIONS_MS = config.pitchbench_b3_DURATIONS_MS
POSITIONS_MS = config.pitchbench_b3_POSITIONS_MS
TOTAL_DUR_MS = config.pitchbench_b3_TOTAL_DUR_MS

PROMPT = (
    "This audio is a 60-second clip that contains exactly ONE sustained "
    "musical note inside silence. Identify the onset and offset times of "
    "the note. Reply with ONLY two timestamps in MM:SS.cc format separated "
    "by a comma, e.g. '0:05.20, 0:08.50'. Nothing else. Output only the answer."
)


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        for midi in PITCHES:
            for dur in DURATIONS_MS:
                for pos in POSITIONS_MS:
                    if pos + dur > TOTAL_DUR_MS:
                        continue
                    rows.append({
                        "source":      src,
                        "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                        "midi":        midi,
                        "duration_ms": dur,
                        "pos_ms":      pos,
                    })
    return rows


def wav_for(c: dict) -> Path:
    path, _ = engine.clip_with_notes(
        [(c["midi"], c["pos_ms"], c["duration_ms"])],
        c["source"], TOTAL_DUR_MS,
        name_hint="b2",
    )
    return path


def gt_timestamps_for(c: dict) -> list[float]:
    on_s  = c["pos_ms"] / 1000.0
    off_s = (c["pos_ms"] + c["duration_ms"]) / 1000.0
    return [on_s, off_s]


SPEC = CatBSpec(
    exp_name=EXP_NAME,
    build_conditions_fn=build_conditions,
    wav_fn=wav_for,
    task_type="timing",
    prompt=PROMPT,
    gt_timestamps_fn=gt_timestamps_for,
    record_extras=("midi", "duration_ms", "pos_ms"),
)


def preview() -> None:
    run_cat_b_experiment(SPEC, mode="preview")


def run() -> dict | None:
    return run_cat_b_experiment(SPEC, mode="run")


if __name__ == "__main__":
    run()
