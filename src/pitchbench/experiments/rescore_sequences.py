"""Rescore saved D8/F1/F2 responses without model calls.

Run with ``python -m pitchbench.experiments.rescore_sequences --output-dir DIR``.
Use ``--baseline-input DIR`` to rescore and export saved DSP/Basic Pitch D8 answers.
Original evaluation files and the submission table remain read-only inputs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pitchbench.config as config

# Compatibility with the existing experiment modules' audio-engine import.
config.AUDIO_DIR = config.GENERATED_DIR

from pitchbench.experiments.helpers.cat_f import score_polyphonic_record
from pitchbench.experiments.helpers.music import offline_scoring
from pitchbench.experiments.scripts.pitchbench_d8_sequence_pitches import record_for


TASKS = ("pitchbench_d8_sequence_pitches", "pitchbench_f1_melodic_line_atonal",
         "pitchbench_f2_melodic_line_tonal")
FORMATS = ("midi", "spn", "doremi", "hz", "any")


def rescore_d8_record(row: dict) -> dict:
    cond = {"source": row["source"], "trial": row["trial"], "n_notes": row["n_notes"]}
    for name, field in (("midi", "midi"), ("note", "spn"),
                        ("doremi", "doremi"), ("hz", "hz")):
        cond[f"{name}_sequence"] = ast.literal_eval(row[f"{field}_sequence_gt"])
    return record_for(cond, row.get("wav", ""), {f: row[f"raw_{f}"] for f in FORMATS[:-1]})


@offline_scoring()
def rescore_d8_baselines(input_dir: Path, output: Path) -> None:
    """Export raw baseline answers and independently recomputed D8 metrics."""
    expected = {"baseline/dsp", "baseline/basic-pitch"}
    seen, aggregates, items, sources, payloads = set(), [], [], [], []
    for path in sorted(input_dir.rglob("results_*.json")):
        payload = json.loads(path.read_text())
        model = payload.get("metadata", {}).get("model_name")
        if model not in expected:
            continue
        if model in seen:
            raise ValueError(f"Duplicate baseline results: {model}")
        seen.add(model)
        rows = payload["results"]
        if len(rows) != 171:
            raise ValueError(f"Expected all 171 D8 stimuli for {model}; found {len(rows)}")
        identities = {(r["source"], r["n_notes"], r["trial"]) for r in rows}
        if len(identities) != 171:
            raise ValueError(f"Duplicate or missing D8 stimulus identities: {model}")
        rescored = []
        for index, row in enumerate(rows):
            scored = rescore_d8_record(row)
            for fmt in FORMATS:
                key = f"{fmt}_note_f1"
                if abs(scored[key] - row[key]) > 1e-12:
                    raise ValueError(f"Saved and recomputed score disagree: {model}/{index}/{key}")
            legacy_exact = int(any(
                all(value is True for value in ast.literal_eval(scored[f"{fmt}_per_pos"]))
                for fmt in ("midi", "spn", "hz")
            ))
            item = dict(model=model, item_index=index, source=row["source"], n_notes=row["n_notes"],
                        **{f"{f}_note_f1": scored[f"{f}_note_f1"] for f in FORMATS},
                        strict_any_accuracy=scored["any_sequence_correct"],
                        legacy_any_accuracy=legacy_exact)
            items.append(item)
            rescored.append(item)
        metrics = [f"{f}_note_f1" for f in FORMATS] + ["strict_any_accuracy", "legacy_any_accuracy"]
        aggregates.append(dict(model=model, n_samples=len(rows),
                               **{m: sum(r[m] for r in rescored) / len(rows) for m in metrics}))
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        payloads.append((model, payload))
    if seen != expected:
        raise ValueError(f"Missing baselines: {sorted(expected - seen)}")
    model_order = ("baseline/dsp", "baseline/basic-pitch")
    aggregates.sort(key=lambda r: model_order.index(r["model"]))
    items.sort(key=lambda r: (model_order.index(r["model"]), r["item_index"]))
    output.mkdir(parents=True, exist_ok=True)
    for model, payload in payloads:
        filename = "results_" + model.split("/")[1].replace("-", "_") + ".json"
        (output / filename).write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(output / "metrics.csv", aggregates)
    write_csv(output / "item_scores.csv", items)
    (output / "rescore.json").write_text(json.dumps(dict(
        metric="Ordered Note F1 = 2 * LCS / (n_gt + n_pred)",
        aggregation="Per-stimulus max(MIDI, SPN, Hz), then macro mean",
        hz_tolerance_hz=1.0, source_files=sources,
        validation="All saved F1 scores independently recomputed from raw answers",
    ), indent=2) + "\n")
    print(json.dumps(aggregates, indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@offline_scoring()
def rescore(repo: Path, output: Path) -> None:
    table = repo / "paper/figures-and-tables/accuracies_by_model_experiment.csv"
    with table.open() as handle:
        original = list(csv.DictReader(handle))
    models = sorted({r["model"] for r in original})
    expected = {(m, t) for m in models for t in TASKS}
    aggregates, items, sources = [], [], []
    seen = set()
    for task in TASKS:
        for path in sorted((repo / "paper/evaluation").glob(f"_*/{task}/*/results_*.json")):
            model = path.stem.removeprefix("results_")
            key = (model, task)
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate model/task: {key}")
            seen.add(key)
            source = str(path.relative_to(repo))
            sources.append(dict(path=source, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            rows = json.loads(path.read_text())["results"]
            if not rows:
                raise ValueError(f"Empty results: {path}")
            rescored = []
            for index, row in enumerate(rows):
                responses = {f: row[f"raw_{f}"] for f in FORMATS[:-1]}
                if task == TASKS[0]:
                    scored = rescore_d8_record(row)
                    exact = "any_sequence_correct"
                else:
                    scored = score_polyphonic_record(
                        {"target_pitches": ast.literal_eval(row["target_midi_gt"]),
                         "source": row.get("source", ""), "sources": row.get("sources", [])},
                        row.get("wav", ""), responses,
                    )
                    exact = "any_seq_correct"
                item = dict(model=model, task=task, source_json=source, item_index=index,
                            **{f"{f}_note_f1": scored[f"{f}_note_f1"] for f in FORMATS},
                            strict_any_accuracy=scored[exact],
                            legacy_any_accuracy=row[exact])
                rescored.append(item)
                items.append(item)
            metrics = [f"{f}_note_f1" for f in FORMATS] + ["strict_any_accuracy", "legacy_any_accuracy"]
            aggregates.append(dict(model=model, task=task, n_samples=len(rows),
                                   **{m: sum(r[m] for r in rescored) / len(rows) for m in metrics}))
    if seen != expected:
        raise ValueError(f"Missing model/task coverage: {sorted(expected - seen)}")
    replacements = {(r["model"], r["task"]): r["any_note_f1"] for r in aggregates}
    updated = []
    by_model = defaultdict(list)
    seen_table = set()
    for row in original:
        key = (row["model"], row["experiment"])
        if key in seen_table:
            raise ValueError(f"Duplicate table row: {key}")
        seen_table.add(key)
        value = replacements.get(key, float(row["accuracy"]))
        by_model[row["model"]].append(value)
        updated.append(dict(model=row["model"], experiment=row["experiment"], score=value))
    if not expected <= seen_table or any(len(v) != 28 for v in by_model.values()):
        raise ValueError("Expected 28 tasks per model including D8/F1/F2")
    overall = [dict(model=m, mean_score=sum(by_model[m]) / 28) for m in models]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", aggregates)
    write_csv(output / "item_scores.csv", items)
    write_csv(output / "scores_by_model_experiment.csv", updated)
    write_csv(output / "overall.csv", overall)
    sources.append(dict(path=str(table.relative_to(repo)),
                        sha256=hashlib.sha256(table.read_bytes()).hexdigest()))
    command = ["python", "-m", "pitchbench.experiments.rescore_sequences",
               "--repo-root", str(repo), "--output-dir", str(output)]
    receipt = dict(run_id="d8-ordered-note-f1", project="pitchbench", status="succeeded",
                   git_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                   command=command, dataset_ids=["PitchBench saved paper/evaluation responses"],
                   output_path=str(output), sources=sources, new_model_queries=0,
                   metric="Ordered Note F1 = 2 * LCS / (n_gt + n_pred)",
                   matching={"midi": "exact integer", "spn": "semitone distance zero",
                             "hz_d8": "absolute error <= 1 Hz", "hz_f1_f2": "relative error <= 1%",
                             "doremi": "pitch class only; excluded from ANY"},
                   aggregation="Per-stimulus max(MIDI, SPN, Hz), then macro mean over stimuli")
    (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    by_key = {(r["model"], r["task"]): r for r in aggregates}
    lines = ["# D8, F1 and F2 Ordered Note F1", "",
             "Rescored saved responses; no new model calls. Main scores use order-preserving "
             "one-to-one LCS matches: `2M/(N_gt+N_pred)`. ANY is the maximum of MIDI, SPN "
             "and Hz for each stimulus, then the macro mean. Solfège is excluded from ANY.", "",
             "D8 retains its ±1 Hz matching tolerance; F1/F2 retain ±1%. Full predicted "
             "sequences are scored, including extra notes. Strict exact match is auxiliary. "
             "Legacy D8 accuracy truncated extra predictions; it is reported separately for comparison.", "",
             "Scores below are percentages. Overall means replace D8/F1/F2 in the original "
             "28-task table; the other tasks are unchanged. Raw evaluation inputs remain unchanged.", "",
             "| Model | D8 legacy | D8 strict | D8 LCS | F1 LCS | F2 LCS | Overall |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in overall:
        m = row["model"]
        d8, f1, f2 = [by_key[(m, t)] for t in TASKS]
        values = [d8["legacy_any_accuracy"], d8["strict_any_accuracy"], d8["any_note_f1"],
                  f1["any_note_f1"], f2["any_note_f1"], row["mean_score"]]
        lines.append(f"| {m} | " + " | ".join(f"{100*v:.1f}" for v in values) + " |")
    lines += ["", "`metrics.csv` and `item_scores.csv` contain unrounded aggregate and per-stimulus "
              "evidence. `scores_by_model_experiment.csv` contains the updated 28-task table. "
              "`run.json` records the code commit, source hashes and reproduction command.", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(f"Rescored {len(items)} saved responses across {len(aggregates)} model/task pairs → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-input", type=Path,
                        help="Rescore/export complete DSP and Basic Pitch D8 result JSONs instead of LALMs")
    args = parser.parse_args()
    if args.baseline_input:
        rescore_d8_baselines(args.baseline_input.resolve(), args.output_dir.resolve())
    else:
        rescore(args.repo_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
