"""Summarize model accuracies from an aggregated combined CSV.

The input is expected to have columns like:

    model,experiment,metric,n_samples,value,value_pct

Selection rule per experiment:
1. Use ``accuracy`` if present.
2. Otherwise use ``accuracy.any`` if present.
3. Otherwise skip the experiment.

If the aggregated file contains multiple rows for the same model + experiment
for the selected metric, they are combined with a weighted average using
``n_samples``.

Outputs:
1. ``accuracies_by_model_experiment.csv``
   One row per model + experiment after combining duplicate runs.
2. ``accuracies_by_model_summary.csv``
   One row per model with per-experiment columns, simple mean across selected
   experiments, and overall weighted accuracy across all selected samples.
3. ``accuracies_by_instrument.csv`` + ``plot_accuracy_by_instrument.png``
   Weighted accuracy per instrument per model across all experiments.
4. ``accuracies_by_notation.csv`` + ``plot_accuracy_by_notation.png``
   Weighted accuracy per notation format (midi/spn/doremi/hz/any) per model.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

MODEL_SHORT_NAMES: dict[str, str] = {
    "audio_flamingo_next_instruct": "Audio Flamingo",
    "openrouter_google_gemini_3_1_pro_preview": "Gemini 2.5 Pro",
    "openrouter_openai_gpt_4o_audio_preview": "GPT-4o Audio",
}

MODEL_COLORS: dict[str, str] = {
    "audio_flamingo_next_instruct": "#41AB5D",               # green  (Flamingo)
    "openrouter_google_gemini_3_1_pro_preview": "#E07028",   # orange (Gemini)
    "openrouter_openai_gpt_4o_audio_preview": "#3182BD",     # blue   (GPT)
}

FORMAT_COLORS: dict[str, str] = {
    "midi": "#4C72B0",
    "spn": "#DD8452",
    "doremi": "#55A868",
    "hz": "#C44E52",
    "any": "#8172B2",
}

# Exclude waveform primitives and aggregate category labels from instrument plots
EXCLUDED_SOURCES: frozenset[str] = frozenset({
    "sine", "sawtooth", "square", "triangle",
    "brass_reeds_synth", "classical_quartet", "instrument",
    "mixed_timbres", "winds_and_strings",
})

_BY_SOURCE_RE = re.compile(r"^by_source\.([^.]+)\.any$")
_ACCURACY_FORMAT_RE = re.compile(r"^accuracy\.(midi|spn|doremi|hz|any)$")


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "accuracies_combined.csv"


def _parse_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    return int(value)


def _parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def _discover_selected_metrics(rows: list[dict[str, str]]) -> dict[str, str]:
    metrics_by_experiment: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        metrics_by_experiment[row["experiment"]].add(row["metric"])

    selected: dict[str, str] = {}
    for experiment, metrics in metrics_by_experiment.items():
        if "accuracy" in metrics:
            selected[experiment] = "accuracy"
        elif "accuracy.any" in metrics:
            selected[experiment] = "accuracy.any"
    return selected


def summarize_combined_accuracies(
    input_csv: str | Path,
    *,
    per_experiment_output: str = "accuracies_by_model_experiment.csv",
    per_model_output: str = "accuracies_by_model_summary.csv",
) -> tuple[Path, Path]:
    """Summarize an aggregated accuracies CSV into per-experiment and per-model outputs."""
    input_path = Path(input_csv).expanduser().resolve()
    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected_metric_by_experiment = _discover_selected_metrics(rows)

    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    experiment_order: list[str] = []
    seen_experiments: set[str] = set()
    for row in rows:
        experiment = row["experiment"]
        selected_metric = selected_metric_by_experiment.get(experiment)
        if selected_metric is None or row["metric"] != selected_metric:
            continue

        n_samples = _parse_int(row["n_samples"])
        value = _parse_float(row["value"])
        if n_samples is None or value is None:
            continue

        if experiment not in seen_experiments:
            experiment_order.append(experiment)
            seen_experiments.add(experiment)

        grouped[(row["model"], experiment)].append((n_samples, value))

    per_experiment_rows: list[dict[str, object]] = []
    model_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (model, experiment), values in sorted(grouped.items()):
        total_samples = sum(n_samples for n_samples, _ in values)
        weighted_accuracy = sum(n_samples * value for n_samples, value in values) / total_samples
        row = {
            "model": model,
            "experiment": experiment,
            "metric": selected_metric_by_experiment[experiment],
            "n_samples": total_samples,
            "accuracy": round(weighted_accuracy, 6),
            "accuracy_pct": f"{weighted_accuracy:.1%}",
            "source_rows": len(values),
        }
        per_experiment_rows.append(row)
        model_rows[model].append(row)

    per_experiment_path = input_path.parent / per_experiment_output
    with per_experiment_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "experiment",
                "metric",
                "n_samples",
                "accuracy",
                "accuracy_pct",
                "source_rows",
            ],
        )
        writer.writeheader()
        writer.writerows(per_experiment_rows)

    per_model_path = input_path.parent / per_model_output
    per_model_fieldnames = [
        "model",
        *experiment_order,
        "n_experiments",
        "mean_accuracy",
        "mean_accuracy_pct",
        "weighted_accuracy",
        "weighted_accuracy_pct",
        "total_samples",
    ]
    with per_model_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_model_fieldnames)
        writer.writeheader()
        for model in sorted(model_rows):
            rows_for_model = model_rows[model]
            wide_row: dict[str, object] = {"model": model}
            for experiment in experiment_order:
                wide_row[experiment] = ""
            total_samples = sum(int(row["n_samples"]) for row in rows_for_model)
            weighted_accuracy = sum(
                int(row["n_samples"]) * float(row["accuracy"])
                for row in rows_for_model
            ) / total_samples
            mean_accuracy = sum(float(row["accuracy"]) for row in rows_for_model) / len(rows_for_model)
            for row in rows_for_model:
                wide_row[str(row["experiment"])] = row["accuracy"]
            wide_row["n_experiments"] = len(rows_for_model)
            wide_row["mean_accuracy"] = round(mean_accuracy, 6)
            wide_row["mean_accuracy_pct"] = f"{mean_accuracy:.1%}"
            wide_row["weighted_accuracy"] = round(weighted_accuracy, 6)
            wide_row["weighted_accuracy_pct"] = f"{weighted_accuracy:.1%}"
            wide_row["total_samples"] = total_samples
            writer.writerow(wide_row)

    return per_experiment_path, per_model_path


def _short_model(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model)


def plot_accuracy_by_instrument(
    input_csv: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Weighted accuracy by instrument per model, aggregated across all experiments.

    Uses the ``by_source.<instrument>.any`` metrics so that "any-format correct"
    is the unit of measurement, making instruments comparable regardless of which
    notation format a model prefers.

    Returns (csv_path, plot_path).
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib and numpy are required for plotting")

    input_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir) if output_dir else input_path.parent / "paper" / datetime.now().strftime("%Y%m%d_%H%M%S")
    mkdir_kwargs = {"parents": True, "exist_ok": True}
    if hasattr(out_dir, "mkdir"):
        out_dir.mkdir(**mkdir_kwargs)

    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # model -> instrument -> [(n_samples, value)]
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        m = _BY_SOURCE_RE.match(row["metric"])
        if not m:
            continue
        instrument = m.group(1)
        n = _parse_int(row["n_samples"])
        v = _parse_float(row["value"])
        if n is None or v is None:
            continue
        if instrument not in EXCLUDED_SOURCES:
            grouped[row["model"]][instrument].append((n, v))

    # Weighted average per (model, instrument)
    results: dict[str, dict[str, tuple[int, float]]] = {}
    all_instrument_set: set[str] = set()
    for model, instruments in grouped.items():
        results[model] = {}
        for instrument, values in instruments.items():
            total_n = sum(n for n, _ in values)
            wav_acc = sum(n * v for n, v in values) / total_n
            results[model][instrument] = (total_n, wav_acc)
            all_instrument_set.add(instrument)

    all_instruments = sorted(all_instrument_set)
    models = sorted(results.keys())

    csv_rows = []
    for model in models:
        for instrument in all_instruments:
            if instrument not in results[model]:
                continue
            total_n, acc = results[model][instrument]
            csv_rows.append({
                "model": model,
                "instrument": instrument,
                "n_samples": total_n,
                "accuracy": round(acc, 6),
                "accuracy_pct": f"{acc:.1%}",
            })

    csv_path = out_dir / "accuracies_by_instrument.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "instrument", "n_samples", "accuracy", "accuracy_pct"])
        writer.writeheader()
        writer.writerows(csv_rows)

    n_instruments = len(all_instruments)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_instruments)

    fig, ax = plt.subplots(figsize=(max(14, n_instruments * 1.3), 6))
    for i, model in enumerate(models):
        accuracies = [
            results[model][inst][1] * 100 if inst in results[model] else 0.0
            for inst in all_instruments
        ]
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, accuracies, bar_width * 0.9,
               label=_short_model(model),
               color=MODEL_COLORS.get(model, f"C{i}"))

    tick_labels = []
    for instrument in all_instruments:
        counts = [results[m][instrument][0] for m in models if instrument in results[m]]
        n = max(counts) if counts else 0
        label = instrument.replace("_", " ").title()
        tick_labels.append(f"{label}\n({n})")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy by Instrument across All Experiments (any-format correct)")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    plot_path = out_dir / "plot_accuracy_by_instrument.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return csv_path, plot_path


def plot_accuracy_by_notation(
    input_csv: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Weighted accuracy by notation format per model, aggregated across experiments.

    Uses ``accuracy.<format>`` metrics (midi, spn, doremi, hz, any).  Only
    experiments that expose these per-format breakdowns are included; experiments
    with only a single aggregate metric (e.g. many b/c/d tasks) are excluded.

    Returns (csv_path, plot_path).
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib and numpy are required for plotting")

    input_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir) if output_dir else input_path.parent / "paper" / datetime.now().strftime("%Y%m%d_%H%M%S")
    mkdir_kwargs = {"parents": True, "exist_ok": True}
    if hasattr(out_dir, "mkdir"):
        out_dir.mkdir(**mkdir_kwargs)
        
    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    FORMATS = ["midi", "spn", "doremi", "hz", "any"]

    # model -> format -> [(n_samples, value)]
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        m = _ACCURACY_FORMAT_RE.match(row["metric"])
        if not m:
            continue
        fmt = m.group(1)
        n = _parse_int(row["n_samples"])
        v = _parse_float(row["value"])
        if n is None or v is None:
            continue
        grouped[row["model"]][fmt].append((n, v))

    results: dict[str, dict[str, tuple[int, float]]] = {}
    for model, formats in grouped.items():
        results[model] = {}
        for fmt, values in formats.items():
            total_n = sum(n for n, _ in values)
            wav_acc = sum(n * v for n, v in values) / total_n
            results[model][fmt] = (total_n, wav_acc)

    models = sorted(results.keys())

    csv_rows = []
    for model in models:
        for fmt in FORMATS:
            if fmt not in results.get(model, {}):
                continue
            total_n, acc = results[model][fmt]
            csv_rows.append({
                "model": model,
                "format": fmt,
                "n_samples": total_n,
                "accuracy": round(acc, 6),
                "accuracy_pct": f"{acc:.1%}",
            })

    csv_path = out_dir / "accuracies_by_notation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "format", "n_samples", "accuracy", "accuracy_pct"])
        writer.writeheader()
        writer.writerows(csv_rows)

    n_models = len(models)
    n_formats = len(FORMATS)
    bar_width = 0.8 / max(1, n_formats)
    x = np.arange(n_models)
    hatches = ["", "//", "..", "xx", "++"]

    fig, ax = plt.subplots(figsize=(max(8, n_models * 3), 6))
    for i, fmt in enumerate(FORMATS):
        accuracies = [
            results[model][fmt][1] * 100 if fmt in results.get(model, {}) else 0.0
            for model in models
        ]
        offset = (i - (n_formats - 1) / 2) * bar_width
        colors = [MODEL_COLORS.get(model, f"C{j}") for j, model in enumerate(models)]
        bars = ax.bar(
            x + offset,
            accuracies,
            bar_width * 0.9,
            label=fmt.upper(),
            color=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        for b in bars:
            b.set_hatch(hatches[i % len(hatches)])

    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in models], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy by Notation Format across All Experiments")
    ax.legend(loc="upper right", title="Format")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    plot_path = out_dir / "plot_accuracy_by_notation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Aggregated combined accuracies CSV",
    )
    args = parser.parse_args()

    per_experiment_path, per_model_path = summarize_combined_accuracies(args.input_csv)
    print(per_experiment_path)
    print(per_model_path)

    if _HAS_MATPLOTLIB:
        input_path = Path(args.input_csv).expanduser().resolve()
        plot_out_dir = input_path.parent / "paper" / datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_inst, plot_inst = plot_accuracy_by_instrument(args.input_csv, output_dir=plot_out_dir)
        print(csv_inst)
        print(plot_inst)

        csv_nota, plot_nota = plot_accuracy_by_notation(args.input_csv, output_dir=plot_out_dir)
        print(csv_nota)
        print(plot_nota)
    else:
        print("matplotlib/numpy not available — skipping plots")


if __name__ == "__main__":
    main()