"""Build the paper's Table 1 from committed evidence, without inference.

PYTHONPATH=src python -m pitchbench.analysis.table1 [--check]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

MODELS = {
    "audio_flamingo_next_instruct": "AF-next-instruct",
    "openrouter_google_gemini_3_1_pro_preview": "Gemini 3.1 Pro",
    "openrouter_google_gemini_flash_latest": "Gemini 3 Flash",
    "openrouter_openai_gpt_4o_audio_preview": "GPT-4o audio",
    "dashscope_qwen3_5_omni_plus": "Qwen-3.5 omni plus",
    "dashscope_qwen3_5_omni_flash": "Qwen-3.5 omni flash",
    "baseline/dsp": "DSP",
    "baseline/basic-pitch": "Basic Pitch",
}
TASKS = {
    "A1": "Pitch ID",
    "A2": "Loudness",
    "A3": "Duration",
    "B1": "Silence",
    "B2": "At Time",
    "B3": "Time Pitch",
    "B4": "Time Spec.",
    "B5": "Time Multi.",
    "C1": "Count",
    "C2": "Interval",
    "C3": "Quality",
    "C4": "Chord P.",
    "D1": "Seq. Count",
    "D2": "High/Low",
    "D3": "Contour D.",
    "D4": "Contour C.",
    "D5": "Rank",
    "D6": "Seq. Int.",
    "D7a": "Ref. Pitch",
    "D8": "Seq. Pitch",
    "E1": "Effects",
    "E2": "Background",
    "E3": "Saturation",
    "E4": "Stretch",
    "E5": "Vibrato",
    "E6": "Off Pitch",
    "F1": "Atonal",
    "F2": "Tonal",
}
SEQUENCE_TASKS = {"D8", "F1", "F2"}
SOURCES = {
    "recomputed": "results/table1-recomputed/metrics.csv",
    "audit": "results/table1-recomputed/run.json",
    "unavailable": "results/table1-recomputed/unavailable.json",
}
NOTE = (
    "Scores are percentages. D8, F1 and F2 use Ordered Note F1; other tasks use "
    "accuracy. Mean weights each of the 28 tasks equally, using unrounded scores. "
    "D7a is the concatenated reference task (D7 in the paper); Y1 is excluded. "
    "GPT-4o B1 uses the 158 saved responses; two of the 160 responses are missing."
)


def unavailable_cells(repo: Path) -> dict[tuple[str, str], str]:
    """Explicit invalidations prevent stale audio answers from becoming scores."""
    path = repo / "results/table1-recomputed/unavailable.json"
    if not path.exists():
        return {}
    cells = {}
    for row in json.loads(path.read_text())["cells"]:
        key = row["model"], row["task"]
        if key in cells or key[0] not in MODELS or key[1] not in TASKS or not row["reason"]:
            raise ValueError(f"Invalid unavailable cell: {key}")
        cells[key] = row["reason"]
    return cells


def load_scores(repo: Path) -> dict[tuple[str, str], Decimal | None]:
    """Load the complete raw-answer replay; never blend rounded legacy tables."""
    relative = SOURCES["recomputed"]
    expected = {(model, task) for model in MODELS for task in TASKS}
    scores = {}
    unavailable = unavailable_cells(repo)
    with (repo / relative).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["model"], row["task"]
            if key in scores:
                raise ValueError(f"Duplicate score in {relative}: {key}")
            metric = "ordered_note_f1" if row["task"] in SEQUENCE_TASKS else "accuracy"
            if row["metric"] != metric:
                raise ValueError(f"Invalid score metric for {key}: {row['metric']}")
            if key in unavailable:
                if row["n_samples"] != "0" or row["score_sum"] or row["score"] or int(row["n_expected"]) <= 0:
                    raise ValueError(f"Unavailable cell contains a stale score: {key}")
                scores[key] = None
                continue
            n = int(row["n_samples"])
            total = Decimal(row["score_sum"])
            score = Decimal(row["score"])
            if (
                n <= 0
                or not total.is_finite()
                or not score.is_finite()
                or not 0 <= total <= n
                or not 0 <= score <= 1
                or abs(total / n - score) > Decimal("1e-15")
            ):
                raise ValueError(f"Invalid score in {relative}: {key}")
            if metric == "accuracy" and total != total.to_integral_value():
                raise ValueError(f"Invalid score count in {relative}: {key}")
            n_expected = int(row["n_expected"])
            if n != n_expected and (key, n, n_expected) != (
                ("openrouter_openai_gpt_4o_audio_preview", "B1"),
                158,
                160,
            ):
                raise ValueError(f"Coverage mismatch for {key}: {n}/{n_expected}")
            scores[key] = total * 100 / n
    if scores.keys() != expected:
        raise ValueError(
            f"Coverage mismatch in {relative}: "
            f"missing={sorted(expected - scores.keys())}, "
            f"extra={sorted(scores.keys() - expected)}"
        )
    return scores


def _display(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _ranked(values: list[Decimal | None], *, latex: bool = False) -> list[str]:
    displayed = [_display(v) for v in values]
    ranks = sorted({Decimal(v) for v in displayed if v != "—"}, reverse=True)
    result = []
    for value in displayed:
        if value == "—":
            result.append("--" if latex else value)
            continue
        if Decimal(value) == ranks[0]:
            value = rf"\textbf{{{value}}}" if latex else f"**{value}**"
        elif len(ranks) > 1 and Decimal(value) == ranks[1]:
            value = rf"\underline{{{value}}}" if latex else value
        result.append(value)
    return result


def render(repo: Path) -> dict[str, str]:
    scores = load_scores(repo)
    long_form = io.StringIO(newline="")
    long_writer = csv.writer(long_form, lineterminator="\n")
    long_writer.writerow(["model", "experiment", "metric", "n_samples", "accuracy", "accuracy_pct", "source_rows"])
    with (repo / SOURCES["recomputed"]).open(newline="") as handle:
        for row in csv.DictReader(handle):
            # Preserve the companion CSV's schema while making its metric explicit.
            name = next((Path(__file__).resolve().parents[1] / "experiments/scripts").glob(
                f"pitchbench_{row['task'].lower()}_*.py")).stem
            score = scores[row["model"], row["task"]]
            long_writer.writerow([row["model"], name, row["metric"], row["n_samples"],
                                  score / 100 if score is not None else "", _display(score) + "%" if score is not None else "—", 1 if score is not None else 0])
    pending_note = (" Unavailable corrected-audio results are shown as —; a model with any missing task has no overall mean or overall rank." if unavailable_cells(repo) else "")
    rows = [(task, [scores[model, task] for model in MODELS]) for task in TASKS]
    rows.append(
        (
            "Mean",
            [
                sum(scores[model, task] for task in TASKS) / len(TASKS)
                if all(scores[model, task] is not None for task in TASKS) else None
                for model in MODELS
            ],
        )
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["task", "metric", *MODELS])
    for task, values in rows:
        metric = (
            "macro_mean"
            if task == "Mean"
            else "ordered_note_f1" if task in SEQUENCE_TASKS else "accuracy"
        )
        writer.writerow([task, metric, *values])

    md = [
        "# Table 1 — PitchBench results",
        "",
        NOTE + pending_note,
        "",
        "Best displayed score per row is bold. All formats are generated from the same evidence; "
        "see [sources and reproduction](README.md).",
        "",
        "| Task | " + " | ".join(MODELS.values()) + " |",
        "| --- | " + " | ".join("---:" for _ in MODELS) + " |",
    ]
    for task, values in rows:
        label = task if task == "Mean" else f"{task} {TASKS[task]}"
        md.append(f"| {label} | " + " | ".join(_ranked(values)) + " |")

    tex = [
        r"% Generated by python -m pitchbench.analysis.table1; do not edit.",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{PitchBench results (\%). D8, F1 and F2 use Ordered Note F1; all other tasks use accuracy. "
        r"Best displayed value per row is bold and second-best underlined (including ties). "
        r"Mean is the unweighted average of 28 unrounded task scores. GPT-4o B1 has 158 saved responses out of 160 stimuli."
        + (r" -- denotes unavailable corrected-audio evidence; incomplete models have no mean." if pending_note else "") + "}",
        r"\label{tab:pitchbench_transposed}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll|c|cc|c|cc|cc}",
        r"\toprule",
        r"\textbf{Group} & \textbf{Task} & \textbf{Nvidia} & \multicolumn{2}{c|}{\textbf{Google}} & \textbf{OpenAI} & \multicolumn{2}{c|}{\textbf{Qwen}} & \multicolumn{2}{c}{\textbf{Baselines}} \\",
        "& & "
        + " & ".join(rf"\textit{{{label}}}" for label in MODELS.values())
        + r" \\",
    ]
    previous_group = None
    for task, values in rows:
        group = task[0]
        if group != previous_group:
            tex.append(r"\midrule")
        if task == "Mean":
            label = r"\multicolumn{2}{l|}{\textbf{Mean}}"
        else:
            count = sum(t.startswith(group) for t in TASKS)
            group_label = (
                rf"\multirow{{{count}}}{{*}}{{{group}}}"
                if group != previous_group
                else ""
            )
            paper_id = "d7" if task == "D7a" else task.lower()
            label = rf"{group_label} & {paper_id} \textit{{{TASKS[task]}}}"
        tex.append(label + " & " + " & ".join(_ranked(values, latex=True)) + r" \\")
        previous_group = group
    tex += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    manifest = {
        "units": "percent",
        "models": MODELS,
        "tasks": TASKS,
        "cell_sources": {
            "Available task/model cells": "recomputed:score_sum / n_samples * 100",
            "Unavailable cells": "explicit invalidation; no score or mean imputed",
            "Mean": "arithmetic mean of the 28 unrounded task scores",
        },
        "display": "one decimal, ROUND_HALF_UP; ranks use displayed values",
        "notes": NOTE + pending_note,
        "unavailable_cells": [dict(model=m, task=t, reason=reason) for (m, t), reason in unavailable_cells(repo).items()],
        "sources": {
            key: {
                "path": path,
                "sha256": hashlib.sha256((repo / path).read_bytes()).hexdigest(),
            }
            for key, path in SOURCES.items()
        },
    }
    return {
        "accuracies_by_model_experiment.csv": long_form.getvalue(),
        "table1.csv": stream.getvalue(),
        "table1.md": "\n".join(md) + "\n",
        "table1.tex": "\n".join(tex) + "\n",
        "table1.sources.json": json.dumps(manifest, indent=2, ensure_ascii=False)
        + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed outputs differ; write nothing",
    )
    args = parser.parse_args()
    output = args.output_dir or args.repo_root / "paper/figures-and-tables"
    rendered = render(args.repo_root)
    if args.check:
        stale = [
            name
            for name, content in rendered.items()
            if not (output / name).is_file()
            or (output / name).read_bytes() != content.encode()
        ]
        if stale:
            parser.exit(1, "Table 1 is out of date: " + ", ".join(stale) + "\n")
        print(
            "Table 1 verified: 28 tasks × 8 models, means, exports and source hashes."
        )
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in rendered.items():
            (output / name).write_bytes(content.encode())
        print(f"Table 1 written to {output}")


if __name__ == "__main__":
    main()
