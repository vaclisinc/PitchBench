"""Rescore saved Category-F responses with order-aware note F1.

The source evaluation JSON files are read-only inputs. Derived item scores,
aggregate scores, and an OpenReview-ready Markdown response are written beside
this script.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import pitchbench.config as pitchbench_config

pitchbench_config.AUDIO_DIR = pitchbench_config.GENERATED_DIR

from pitchbench.experiments.helpers.cat_f import score_polyphonic_record


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "paper" / "evaluation"
TABLE_INPUT = REPO_ROOT / "paper" / "figures-and-tables" / "accuracies_by_model_experiment.csv"
OUTPUT_JSON = EVAL_DIR / "category_f_ordered_note_f1_results.json"
OUTPUT_CSV = EVAL_DIR / "category_f_ordered_note_f1_results.csv"
OUTPUT_MD = EVAL_DIR / "category_f_ordered_note_f1_openreview.md"

MODEL_ORDER = (
    "audio_flamingo_next_instruct",
    "openrouter_google_gemini_3_1_pro_preview",
    "openrouter_google_gemini_flash_latest",
    "openrouter_openai_gpt_4o_audio_preview",
    "dashscope_qwen3_5_omni_plus",
    "dashscope_qwen3_5_omni_flash",
)
MODEL_LABELS = {
    "audio_flamingo_next_instruct": "AF-next-instruct",
    "openrouter_google_gemini_3_1_pro_preview": "Gemini 3.1 Pro",
    "openrouter_google_gemini_flash_latest": "Gemini 3 Flash",
    "openrouter_openai_gpt_4o_audio_preview": "GPT-4o audio",
    "dashscope_qwen3_5_omni_plus": "Qwen-3.5 omni plus",
    "dashscope_qwen3_5_omni_flash": "Qwen-3.5 omni flash",
}
TASK_LABELS = {
    "pitchbench_f1_melodic_line_atonal": "f1 Atonal",
    "pitchbench_f2_melodic_line_tonal": "f2 Tonal",
}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty score list")
    return sum(values) / len(values)


def _source_files() -> list[Path]:
    files = sorted(EVAL_DIR.glob("_*/pitchbench_f[12]_*/*/results_*.json"))
    if len(files) != 12:
        raise RuntimeError(f"expected 12 Category-F result JSON files, found {len(files)}")
    return files


def _rescore() -> tuple[list[dict], list[dict]]:
    item_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for source_path in _source_files():
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        model = source_path.stem.removeprefix("results_")
        task = source_path.parents[1].name
        key = (model, task)
        if key in seen:
            raise RuntimeError(f"duplicate result set for {model} / {task}")
        seen.add(key)

        rescored_records: list[dict] = []
        for item_index, record in enumerate(payload["results"]):
            target = ast.literal_eval(record["target_midi_gt"])
            rescored = score_polyphonic_record(
                {
                    "target_pitches": target,
                    "source": record.get("source", ""),
                    "sources": record.get("sources", []),
                },
                record.get("wav", ""),
                {
                    "midi": record.get("raw_midi", ""),
                    "spn": record.get("raw_spn", ""),
                    "doremi": record.get("raw_doremi", ""),
                    "hz": record.get("raw_hz", ""),
                },
            )
            rescored_records.append(rescored)
            item_rows.append(
                {
                    "model": model,
                    "task": task,
                    "source_json": str(source_path.relative_to(REPO_ROOT)),
                    "item_index": item_index,
                    "n_target": rescored["n_target"],
                    "midi_pred_count": rescored["midi_pred_count"],
                    "midi_note_f1": rescored["midi_note_f1"],
                    "spn_note_f1": rescored["spn_note_f1"],
                    "doremi_note_f1": rescored["doremi_note_f1"],
                    "hz_note_f1": rescored["hz_note_f1"],
                    "any_note_f1": rescored["any_note_f1"],
                    "any_seq_correct": rescored["any_seq_correct"],
                }
            )

        aggregate_rows.append(
            {
                "model": model,
                "task": task,
                "n_samples": len(rescored_records),
                "midi_note_f1": _mean([r["midi_note_f1"] for r in rescored_records]),
                "spn_note_f1": _mean([r["spn_note_f1"] for r in rescored_records]),
                "doremi_note_f1": _mean([r["doremi_note_f1"] for r in rescored_records]),
                "hz_note_f1": _mean([r["hz_note_f1"] for r in rescored_records]),
                "any_note_f1": _mean([r["any_note_f1"] for r in rescored_records]),
                "strict_any_accuracy": _mean(
                    [float(r["any_seq_correct"]) for r in rescored_records]
                ),
                "midi_count_mae": _mean(
                    [float(r["midi_count_abs_error"]) for r in rescored_records]
                ),
                "source_json": str(source_path.relative_to(REPO_ROOT)),
            }
        )

    expected = {(model, task) for model in MODEL_ORDER for task in TASK_LABELS}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise RuntimeError(f"unexpected model/task coverage; missing={missing}, extra={extra}")
    return aggregate_rows, item_rows


def _updated_overall_means(aggregate_rows: list[dict]) -> dict[str, float]:
    replacements = {
        (row["model"], row["task"]): row["any_note_f1"]
        for row in aggregate_rows
    }
    scores: dict[str, list[float]] = defaultdict(list)
    with TABLE_INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            experiment = row["experiment"]
            value = replacements.get((model, experiment), float(row["accuracy"]))
            scores[model].append(value)

    for model in MODEL_ORDER:
        if len(scores[model]) != 28:
            raise RuntimeError(
                f"expected 28 task scores for {model}, found {len(scores[model])}"
            )
    return {model: _mean(scores[model]) for model in MODEL_ORDER}


def _write_csv(aggregate_rows: list[dict], overall_means: dict[str, float]) -> None:
    fieldnames = [
        "model",
        "task",
        "n_samples",
        "midi_note_f1",
        "spn_note_f1",
        "doremi_note_f1",
        "hz_note_f1",
        "any_note_f1",
        "strict_any_accuracy",
        "midi_count_mae",
        "updated_overall_mean",
        "source_json",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted(
            aggregate_rows,
            key=lambda r: (MODEL_ORDER.index(r["model"]), r["task"]),
        ):
            writer.writerow({**row, "updated_overall_mean": overall_means[row["model"]]})


def _write_json(
    aggregate_rows: list[dict],
    item_rows: list[dict],
    overall_means: dict[str, float],
) -> None:
    receipt = {
        "metric": {
            "name": "Ordered Note F1 (MIDI-LCS)",
            "matching": "longest common subsequence with one-to-one ordered matches",
            "formula": "2 * matches / (n_ground_truth + n_prediction)",
            "formats": {
                "midi": "exact MIDI integer",
                "spn": "exact pitch after SPN normalization",
                "hz": "prediction within +/-1% of the ground-truth frequency",
                "doremi": "pitch-class match; reported only, excluded from any_note_f1",
            },
            "table_aggregation": (
                "max(midi_note_f1, spn_note_f1, hz_note_f1) per stimulus, "
                "then macro mean over stimuli"
            ),
        },
        "source": "saved raw responses in paper/evaluation Category-F result JSON files",
        "new_model_queries": 0,
        "aggregates": aggregate_rows,
        "updated_overall_means": overall_means,
        "item_scores": item_rows,
    }
    OUTPUT_JSON.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def _write_markdown(aggregate_rows: list[dict], overall_means: dict[str, float]) -> None:
    by_key = {(row["model"], row["task"]): row for row in aggregate_rows}
    best_f1 = max(
        by_key[(model, "pitchbench_f1_melodic_line_atonal")]["any_note_f1"]
        for model in MODEL_ORDER
    )
    best_f2 = max(
        by_key[(model, "pitchbench_f2_melodic_line_tonal")]["any_note_f1"]
        for model in MODEL_ORDER
    )

    lines = [
        "# Response on Category F scoring",
        "",
        "Thank you for raising this point. We agree that full-sequence exact match "
        "is too strict for the Category F transcription tasks because one missing "
        "or extra note can make an otherwise partially correct sequence score zero.",
        "",
        "We therefore rescored all saved Category F model responses using "
        "**Ordered Note F1**. For each response, we find the longest common "
        "subsequence of ground-truth and predicted pitches. If the alignment "
        "contains $M$ matched notes, the score is "
        "$2M/(N_{\\mathrm{gt}}+N_{\\mathrm{pred}})$. This gives partial credit "
        "while penalizing wrong, missing, extra, duplicated, and out-of-order notes.",
        "",
        "Following the aggregation used in our main table, we compute the score "
        "separately for MIDI, scientific pitch notation, and Hz, take the maximum "
        "of the three scores for each stimulus, and then macro-average over "
        "stimuli. Solfège is excluded because it does not encode octave. No models "
        "were queried again; the results below were computed from the original "
        "saved responses.",
        "",
        "| Model | f1 Atonal | f2 Tonal | Updated mean |",
        "|---|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        f1 = by_key[(model, "pitchbench_f1_melodic_line_atonal")]["any_note_f1"]
        f2 = by_key[(model, "pitchbench_f2_melodic_line_tonal")]["any_note_f1"]

        def fmt(value: float, best: float | None = None) -> str:
            rendered = f"{100 * value:.1f}"
            return f"**{rendered}**" if best is not None and value == best else rendered

        lines.append(
            f"| {MODEL_LABELS[model]} | {fmt(f1, best_f1)} | "
            f"{fmt(f2, best_f2)} | {fmt(overall_means[model])} |"
        )
    lines += [
        "",
        "We will replace the Category F exact-match values in the paper with these "
        "Ordered Note F1 scores, relabel the table values from accuracy to score "
        "where appropriate, and clarify the scoring and cross-format aggregation "
        "in the evaluation section.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    aggregate_rows, item_rows = _rescore()
    overall_means = _updated_overall_means(aggregate_rows)
    _write_csv(aggregate_rows, overall_means)
    _write_json(aggregate_rows, item_rows, overall_means)
    _write_markdown(aggregate_rows, overall_means)
    print(f"Wrote {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
