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
    "A1": "Pitch ID", "A2": "Loudness", "A3": "Duration",
    "B1": "Silence", "B2": "At Time", "B3": "Time Pitch",
    "B4": "Time Spec.", "B5": "Time Multi.",
    "C1": "Count", "C2": "Interval", "C3": "Quality", "C4": "Chord P.",
    "D1": "Seq. Count", "D2": "High/Low", "D3": "Contour D.",
    "D4": "Contour C.", "D5": "Rank", "D6": "Seq. Int.",
    "D7a": "Ref. Pitch", "D8": "Seq. Pitch",
    "E1": "Effects", "E2": "Background", "E3": "Saturation",
    "E4": "Stretch", "E5": "Vibrato", "E6": "Off Pitch",
    "F1": "Atonal", "F2": "Tonal",
}
SEQUENCE_TASKS = {"D8", "F1", "F2"}
SOURCES = {
    "alm_accuracy": "paper/figures-and-tables/accuracies_by_model_experiment.csv",
    "alm_note_f1": "results/d8-ordered-note-f1/metrics.csv",
    "baseline_accuracy_and_f1": "results/d8-baselines-lcs/historical_table.csv",
    "baseline_d8": "results/d8-baselines-lcs/metrics.csv",
}
NOTE = (
    "Scores are percentages. D8, F1 and F2 use Ordered Note F1; other tasks use "
    "accuracy. Mean weights each of the 28 tasks equally. D7a is the concatenated "
    "reference task (D7 in the paper); Y1 is excluded. DSP and Basic Pitch means "
    "are approximate: their 27 non-D8 inputs were preserved only to 0.1 percentage "
    "points (mean rounding bound: ±0.0483 percentage points)."
)


def _task(experiment: str) -> str:
    value = experiment.split("_")[1].upper()
    return "D7a" if value == "D7A" else value


def c4_evidence(repo: Path) -> dict[str, dict]:
    """Recover three-format ANY; the legacy aggregate also counted solfège."""
    evidence = {}
    for model in MODELS:
        if model.startswith("baseline/"):
            continue
        paths = list((repo / "paper/evaluation" / ("_" + model)).glob(
            "pitchbench_c4_chord_pitches/*/results_*.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one saved C4 run for {model}; found {len(paths)}")
        path = paths[0]
        rows = json.loads(path.read_text())["results"]
        if len(rows) != 200:
            raise ValueError(f"Expected 200 C4 items for {model}")
        flags = [[row[f"{fmt}_correct"] for fmt in ("midi", "spn", "hz")] for row in rows]
        if any(flag not in (0, 1) for item in flags for flag in item):
            raise ValueError(f"Invalid C4 correctness flag for {model}")
        evidence[model] = {
            "path": str(path.relative_to(repo)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_samples": len(rows), "n_correct": sum(any(item) for item in flags),
        }
    return evidence


def load_scores(repo: Path) -> dict[tuple[str, str], Decimal]:
    """Require complete, unique coverage in every evidence source before merging."""
    alm_models = set(MODELS) - {"baseline/dsp", "baseline/basic-pitch"}
    expected = {
        "alm_accuracy": {(m, t) for m in alm_models for t in TASKS},
        "alm_note_f1": {(m, t) for m in alm_models for t in SEQUENCE_TASKS},
        "baseline_accuracy_and_f1": {(m, t) for m in MODELS if m.startswith("baseline/") for t in TASKS},
        "baseline_d8": {(m, "D8") for m in MODELS if m.startswith("baseline/")},
    }
    scores = {}
    for source, relative in SOURCES.items():
        current = {}

        def add(model: str, task: str, value: str, scale: int = 1) -> None:
            key = (model, task)
            if key in current:
                raise ValueError(f"Duplicate score in {relative}: {key}")
            score = Decimal(value) * scale
            if not score.is_finite() or not 0 <= score <= 100:
                raise ValueError(f"Invalid score in {relative}: {key} = {value}")
            current[key] = score

        with (repo / relative).open(newline="") as handle:
            for row in csv.DictReader(handle):
                if source == "baseline_accuracy_and_f1":
                    add("baseline/dsp", row["experiment"], row["dsp_displayed_pct"])
                    add("baseline/basic-pitch", row["experiment"], row["basic_pitch_displayed_pct"])
                elif source == "alm_accuracy":
                    add(row["model"], _task(row["experiment"]), row["accuracy"], 100)
                else:
                    task = "D8" if source == "baseline_d8" else _task(row["task"])
                    add(row["model"], task, row["any_note_f1"], 100)
        if current.keys() != expected[source]:
            missing = expected[source] - current.keys()
            extra = current.keys() - expected[source]
            raise ValueError(f"Coverage mismatch in {relative}: missing={sorted(missing)}, extra={sorted(extra)}")
        scores.update(current)
    for model, evidence in c4_evidence(repo).items():
        scores[model, "C4"] = Decimal(evidence["n_correct"]) * 100 / evidence["n_samples"]
    return scores


def _display(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _ranked(values: list[Decimal], *, latex: bool = False) -> list[str]:
    displayed = [_display(v) for v in values]
    ranks = sorted({Decimal(v) for v in displayed}, reverse=True)
    result = []
    for value in displayed:
        if Decimal(value) == ranks[0]:
            value = rf"\textbf{{{value}}}" if latex else f"**{value}**"
        elif len(ranks) > 1 and Decimal(value) == ranks[1]:
            value = rf"\underline{{{value}}}" if latex else value
        result.append(value)
    return result


def render(repo: Path) -> dict[str, str]:
    scores = load_scores(repo)
    rows = [(task, [scores[model, task] for model in MODELS]) for task in TASKS]
    rows.append(("Mean", [sum(scores[model, task] for task in TASKS) / len(TASKS) for model in MODELS]))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["task", "metric", *MODELS])
    for task, values in rows:
        metric = "macro_mean" if task == "Mean" else "ordered_note_f1" if task in SEQUENCE_TASKS else "accuracy"
        writer.writerow([task, metric, *values])

    md = ["# Table 1 — PitchBench results", "", NOTE, "",
          "Best displayed score per row is bold. All formats are generated from the same evidence; "
          "see [sources and reproduction](README.md).", "",
          "| Task | " + " | ".join(MODELS.values()) + " |",
          "| --- | " + " | ".join("---:" for _ in MODELS) + " |"]
    for task, values in rows:
        label = task if task == "Mean" else f"{task} {TASKS[task]}"
        md.append(f"| {label} | " + " | ".join(_ranked(values)) + " |")

    tex = [r"% Generated by python -m pitchbench.analysis.table1; do not edit.",
           r"\begin{table}[!htbp]", r"\centering",
           r"\caption{PitchBench results (\%). D8, F1 and F2 use Ordered Note F1; all other tasks use accuracy. "
           r"Best displayed value per row is bold and second-best underlined (including ties). "
           r"Mean is the unweighted average of 28 tasks; baseline means are approximate because non-D8 baseline scores were preserved to one decimal place.}",
           r"\label{tab:pitchbench_transposed}", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll|c|cc|c|cc|cc}", r"\toprule",
           r"\textbf{Group} & \textbf{Task} & \textbf{Nvidia} & \multicolumn{2}{c|}{\textbf{Google}} & \textbf{OpenAI} & \multicolumn{2}{c|}{\textbf{Qwen}} & \multicolumn{2}{c}{\textbf{Baselines}} \\",
           "& & " + " & ".join(rf"\textit{{{label}}}" for label in MODELS.values()) + r" \\"]
    previous_group = None
    for task, values in rows:
        group = task[0]
        if group != previous_group:
            tex.append(r"\midrule")
        if task == "Mean":
            label = r"\multicolumn{2}{l|}{\textbf{Mean}}"
        else:
            count = sum(t.startswith(group) for t in TASKS)
            group_label = rf"\multirow{{{count}}}{{*}}{{{group}}}" if group != previous_group else ""
            paper_id = "d7" if task == "D7a" else task.lower()
            label = rf"{group_label} & {paper_id} \textit{{{TASKS[task]}}}"
        tex.append(label + " & " + " & ".join(_ranked(values, latex=True)) + r" \\")
        previous_group = group
    tex += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    manifest = {
        "units": "percent", "models": MODELS, "tasks": TASKS,
        "cell_sources": {
            "ALM D8/F1/F2": "alm_note_f1:any_note_f1 * 100",
            "ALM C4": "alm_c4: per-item OR(midi_correct, spn_correct, hz_correct), then mean * 100",
            "ALM other tasks": "alm_accuracy:accuracy * 100 (excluding C4/D8/F1/F2)",
            "baseline D8": "baseline_d8:any_note_f1 * 100",
            "baseline other tasks": "baseline_accuracy_and_f1:*_displayed_pct",
            "Mean": "arithmetic mean of the 28 source scores, before display rounding",
        },
        "display": "one decimal, ROUND_HALF_UP; ranks use displayed values",
        "notes": NOTE,
        "sources": {key: {"path": path, "sha256": hashlib.sha256((repo / path).read_bytes()).hexdigest()}
                    for key, path in SOURCES.items()},
        "alm_c4": c4_evidence(repo),
    }
    return {"table1.csv": stream.getvalue(), "table1.md": "\n".join(md) + "\n",
            "table1.tex": "\n".join(tex) + "\n",
            "table1.sources.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Fail if committed outputs differ; write nothing")
    args = parser.parse_args()
    output = args.output_dir or args.repo_root / "paper/figures-and-tables"
    rendered = render(args.repo_root)
    if args.check:
        stale = [name for name, content in rendered.items()
                 if not (output / name).is_file() or (output / name).read_bytes() != content.encode()]
        if stale:
            parser.exit(1, "Table 1 is out of date: " + ", ".join(stale) + "\n")
        print("Table 1 verified: 28 tasks × 8 models, means, exports and source hashes.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in rendered.items():
            (output / name).write_bytes(content.encode())
        print(f"Table 1 written to {output}")


if __name__ == "__main__":
    main()
