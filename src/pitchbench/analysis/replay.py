"""Replay Table 1 raw answers against the frozen official ground truth; no inference.

python -m pitchbench.analysis.replay --dataset-dir DIR --output-dir DIR
Add --baseline-input DIR for a complete 28-task, eight-model result bundle.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import math
import re
import platform
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pitchbench.config as config

config.AUDIO_DIR = config.GENERATED_DIR
from pitchbench.analysis.table1 import MODELS, TASKS, SEQUENCE_TASKS
from pitchbench.baselines.evaluation import (
    _condition_from_official_row,
    _timing_ground_truth,
)
from pitchbench.experiments.helpers.music import (
    standard_pitch_record,
    parse_mm_ss_cc,
    offline_scoring,
)
from pitchbench.experiments.helpers.cat_b import score_timestamps
from pitchbench.experiments.run import _id_to_name

REVISION = "f6c672608057cb877bfaacaeeff9bfb49eef7778"
SINGLE = {"A1", "A2", "A3", "B1", "B2", "D7a", "E1", "E2", "E3", "E4", "E5", "E6"}
FIELDS = {
    "B3": "correct",
    "B4": "correct",
    "B5": "correct",
    "C1": "count_correct",
    "C2": "interval_correct",
    "C3": "quality_correct",
    "C4": "any_correct",
    "D1": "count_correct",
    "D2": "answer_correct",
    "D3": "sequence_correct",
    "D4": "trajectory_correct",
    "D5": "answer_correct",
    "D6": "interval_correct",
}
GT_FIELDS = {
    "C1": "n",
    "C2": "interval_st",
    "C3": "chord_quality_gt",
    "C4": "midi_set",
    "D1": "n",
    "D2": "answer_gt",
    "D3": "pattern_gt",
    "D4": "trajectory_gt",
    "D5": "answer_gt",
    "D6": "signed_st",
    "D8": "midi_sequence_gt",
    "F1": "target_midi_gt",
    "F2": "target_midi_gt",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def identity(task, row, *, official=False, baseline=False):
    if task == "D4":
        return tuple(
            row[k]
            for k in ("source", "start_midi", "end_midi", "traj_name", "interval_st")
        )
    name = Path(row["audio"]["path"] if official else row["wav"]).name
    return re.sub(r"^\d{5}_", "", name) if baseline else name


def score_record(task, experiment, row, condition):
    """Use evaluation's actual parsers/scorers, with independently loaded GT."""
    if task in SINGLE:
        return standard_pitch_record(
            wav=row["wav"],
            source=condition["source"],
            source_type=row["source_type"],
            midi_gt=condition["midi"],
            **{f"raw_{f}": row[f"raw_{f}"] for f in ("midi", "spn", "hz")},
            raw_doremi="",
            prompt_doremi="",
            **{
                f"prompt_{f}": row.get(f"prompt_{f}", "") for f in ("midi", "spn", "hz")
            },
        )
    if task in {"B3", "B4", "B5"}:
        module = importlib.import_module(f"pitchbench.experiments.scripts.{experiment}")
        gt = _timing_ground_truth(task.lower(), condition)
        if task == "B5":
            encoded = [
                int(x) / 1000
                for onset, duration in re.findall(r"@(\d+)\+(\d+)", row["wav"])
                for x in (onset, int(onset) + int(duration))
            ]
            if encoded != gt:
                raise ValueError("B5 filename and official ground truth differ")
        elif [row["onset_s_gt"], row["offset_s_gt"]] != gt:
            raise ValueError(f"{task} saved and official ground truth differ")
        return {
            "correct": score_timestamps(
                gt,
                parse_mm_ss_cc(row["raw_response"]),
                module.SPEC.effective_tolerance_ms,
            )
        }
    module = importlib.import_module(f"pitchbench.experiments.scripts.{experiment}")
    responses = (
        {**{f: row[f"raw_{f}"] for f in ("midi", "spn", "hz")}, "doremi": ""}
        if task in {"C4", "D8", "F1", "F2"}
        else {"main": row["raw_response"]}
    )
    return module.SPEC.record_fn(condition, row["wav"], responses)


@offline_scoring()
def replay(repo, dataset, output, baseline_input=None):
    output.mkdir(parents=True, exist_ok=True)
    aggregates, items, sources, mismatches, gaps, datasets = [], [], [], [], [], []
    selected = list(MODELS) if baseline_input else list(MODELS)[:6]
    for task in TASKS:
        experiment = _id_to_name(task.lower())
        shard = dataset / experiment / "test-00000-of-00001.parquet"
        metadata = (
            dataset
            / ".cache/huggingface/download"
            / experiment
            / "test-00000-of-00001.parquet.metadata"
        )
        if metadata.read_text().splitlines()[0] != REVISION:
            raise ValueError(f"Unverified dataset revision: {shard}")
        file = pq.ParquetFile(shard)
        official = file.read(
            columns=[c for c in file.schema_arrow.names if c != "audio"]
            + ["audio.path"]
        ).to_pylist()
        index = {identity(task, r, official=True): r for r in official}
        if len(index) != len(official):
            raise ValueError(f"Duplicate official identities: {task}")
        datasets.append(
            dict(
                task=task,
                path=str(shard),
                sha256=digest(shard),
                n_samples=len(official),
            )
        )
        for model in selected:
            baseline = model.startswith("baseline/")
            if baseline:
                paths = []
                for path in baseline_input.glob(f"*/**/{experiment}/results_*.json"):
                    if json.loads(path.read_text())["metadata"]["model_name"] == model:
                        paths.append(path)
            else:
                paths = list(
                    (repo / "paper/evaluation" / ("_" + model) / experiment).glob(
                        "*/results_*.json"
                    )
                )
            if len(paths) != 1:
                raise ValueError(
                    f"Expected one result file: {model}/{task}, got {len(paths)}"
                )
            path = paths[0]
            payload = json.loads(path.read_text())
            rows = payload["results"]
            source = (
                str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)
            )
            sources.append(
                dict(model=model, task=task, path=source, sha256=digest(path))
            )
            seen = set()
            scores = []
            for i, row in enumerate(rows):
                key = identity(task, row, baseline=baseline)
                if key in seen:
                    raise ValueError(f"Duplicate result identity: {model}/{task}/{key}")
                if key not in index:
                    raise ValueError(f"Unknown result identity: {model}/{task}/{key}")
                seen.add(key)
                cond = _condition_from_official_row(task.lower(), index[key])
                scored = score_record(task, experiment, row, cond)
                gt = "midi_gt" if task in SINGLE else GT_FIELDS.get(task)
                if gt and scored[gt] != row[gt]:
                    raise ValueError(
                        f"Ground truth mismatch: {model}/{task}/{i}/{gt}: {row[gt]} != {scored[gt]}"
                    )
                field = (
                    "any_note_f1"
                    if task in SEQUENCE_TASKS
                    else "any_correct" if task in SINGLE else FIELDS[task]
                )
                value = float(scored[field])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid score: {value}")
                scores.append(value)
                # Sequence exact-match diagnostics intentionally changed when F1
                # was introduced; compare saved F1 only when it exists.
                checks = (
                    [f"{f}_note_f1" for f in ("midi", "spn", "hz", "any")]
                    if task in SEQUENCE_TASKS
                    else [
                        k
                        for k in scored
                        if not k.startswith("doremi")
                        and (k.endswith("_correct") or k == "correct")
                    ]
                )
                for k in checks:
                    if k in row and abs(float(scored[k]) - float(row[k])) > 1e-12:
                        mismatches.append(
                            dict(
                                model=model,
                                task=task,
                                item_index=i,
                                field=k,
                                saved=row[k],
                                recomputed=scored[k],
                            )
                        )
                items.append(
                    dict(
                        model=model,
                        task=task,
                        item_index=i,
                        stimulus=str(key),
                        score=value,
                    )
                )
            missing = set(index) - seen
            if missing:
                if (model, task, len(missing)) != (
                    "openrouter_openai_gpt_4o_audio_preview",
                    "B1",
                    2,
                ):
                    raise ValueError(
                        f"Unexpected missing responses: {model}/{task}: {len(missing)}"
                    )
                gaps.append(
                    dict(
                        model=model,
                        task=task,
                        expected=len(index),
                        observed=len(rows),
                        missing=sorted(missing),
                    )
                )
            if not scores:
                raise ValueError(f"No results for {model}/{task}")
            aggregates.append(
                dict(
                    model=model,
                    task=task,
                    metric="ordered_note_f1" if task in SEQUENCE_TASKS else "accuracy",
                    n_samples=len(scores),
                    n_expected=len(official),
                    score_sum=math.fsum(scores),
                    score=math.fsum(scores) / len(scores),
                )
            )
        print(task, "replayed", flush=True)
    write_csv(output / "metrics.csv", aggregates)
    write_csv(output / "item_scores.csv", items)
    write_csv(
        output / "score_mismatches.csv",
        mismatches,
        ["model", "task", "item_index", "field", "saved", "recomputed"],
    )
    overall = [
        dict(
            model=m,
            n_tasks=len(TASKS),
            score=math.fsum(r["score"] for r in aggregates if r["model"] == m)
            / len(TASKS),
        )
        for m in selected
    ]
    write_csv(output / "overall.csv", overall)
    receipt = dict(
        run_id="table1-recomputed",
        project="pitchbench",
        status="succeeded",
        complete_table=baseline_input is not None,
        git_commit=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        command=[
            "python",
            "-m",
            "pitchbench.analysis.replay",
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(output),
        ]
        + (["--baseline-input", str(baseline_input)] if baseline_input else []),
        output_path=str(output),
        dataset_ids=["pitchbench-authors/PitchBench@" + REVISION],
        new_model_queries=0,
        optional_llm_parsing="disabled explicitly by PITCHBENCH_OFFLINE_SCORING",
        pitch_formats=["midi", "spn", "hz"],
        aggregation="Per-item max(MIDI, SPN, Hz), mean over saved items, then equal mean over 28 tasks. Solfege excluded.",
        metric="D8/F1/F2 Ordered Note F1 (LCS); all others accuracy",
        hz_tolerance={
            "D8": "1 Hz",
            "F1/F2 and single pitch": "1%",
            "C4": "nearest MIDI",
        },
        python=platform.python_version(),
        scoring_files=[
            dict(
                path=str(path.relative_to(Path(__file__).resolve().parents[1])),
                sha256=digest(path),
            )
            for path in sorted(
                {Path(__file__).resolve()}
                | set(
                    (Path(__file__).resolve().parents[1] / "experiments/helpers").glob(
                        "*.py"
                    )
                )
                | set(
                    (Path(__file__).resolve().parents[1] / "experiments/scripts").glob(
                        "pitchbench_*.py"
                    )
                )
                | {Path(__file__).resolve().parents[1] / "baselines/evaluation.py"}
            )
        ],
        primary_score_mismatches=sum(
            m["field"]
            in {
                "any_correct",
                "midi_correct",
                "spn_correct",
                "hz_correct",
                "correct",
                *FIELDS.values(),
                "any_note_f1",
                "midi_note_f1",
                "spn_note_f1",
                "hz_note_f1",
            }
            for m in mismatches
        ),
        sources=sources,
        datasets=datasets,
        missing_responses=gaps,
        score_mismatches=len(mismatches),
        total_responses=len(items),
        cells=len(aggregates),
    )
    (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                cells=len(aggregates),
                items=len(items),
                mismatches=len(mismatches),
                gaps=gaps,
                overall=overall,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-input", type=Path)
    args = parser.parse_args()
    replay(
        args.repo_root.resolve(),
        args.dataset_dir.resolve(),
        args.output_dir.resolve(),
        args.baseline_input,
    )


if __name__ == "__main__":
    main()
