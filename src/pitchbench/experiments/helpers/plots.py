"""
Standard plots used by experiments.

The basic helper produces grouped accuracy bar charts. Pitch experiments can also
generate target-vs-prediction line plots per model and per instrument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

VARIANT_COLORS: dict[str, str] = {
    "midi":    "#4C72B0",
    "abc":     "#DD8452",
    "solfege": "#55A868",
}

VARIANT_LABELS: dict[str, str] = {
    "midi":    "MIDI",
    "abc":     "ABC",
    "solfege": "Solfège",
}

MODEL_COLORS: list[str] = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _nearest_midi_for_pitch_class(gt_midi: int, pred_pc: int) -> int:
    base = gt_midi - (gt_midi % 12) + pred_pc
    candidates = [base - 12, base, base + 12]
    return min(candidates, key=lambda midi: abs(midi - gt_midi))


def _record_predicted_midi(record: dict[str, Any]) -> float | None:
    variant = record.get("prompt_variant")
    pred = record.get("pred")

    if variant == "midi":
        return float(pred) if isinstance(pred, int) else None

    if variant == "abc":
        try:
            from helpers.music import note_to_midi

            midi = note_to_midi(pred) if isinstance(pred, str) else None
        except Exception:
            midi = None
        return float(midi) if midi is not None else None

    if variant == "solfege" and isinstance(pred, int) and "midi" in record:
        return float(_nearest_midi_for_pitch_class(int(record["midi"]), pred))

    return None


def _sorted_task_values(records: list[dict[str, Any]], task_key: str) -> list[Any]:
    tasks = {r.get(task_key) for r in records if task_key in r}
    return sorted(tasks)


def _task_labels(records: list[dict[str, Any]], task_values: list[Any], task_key: str) -> list[str]:
    by_task = {r.get(task_key): r for r in records if task_key in r}

    if task_values and all(isinstance(v, int) for v in task_values):
        try:
            from helpers.music import midi_to_note

            return [midi_to_note(int(v)) for v in task_values]
        except Exception:
            return [str(v) for v in task_values]

    labels: list[str] = []
    for task in task_values:
        record = by_task.get(task, {})
        labels.append(str(record.get("note") or record.get("ground_truth_note") or task))
    return labels


def _mean_and_ci(values: list[float]) -> tuple[float, float, float]:
    import numpy as np

    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, mean, mean

    sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    delta = 1.96 * sem
    return mean, mean - delta, mean + delta


def _prepare_prediction_series(
    records: list[dict[str, Any]],
    source_key: str,
    task_key: str,
) -> tuple[list[Any], list[str], list[float], dict[str, list[float]], list[tuple[float, float, float]]]:
    import numpy as np

    task_values = _sorted_task_values(records, task_key)
    task_labels = _task_labels(records, task_values, task_key)

    actual_by_task: dict[Any, float] = {}
    for task in task_values:
        sub = [r for r in records if r.get(task_key) == task and "midi" in r]
        actual_by_task[task] = float(np.mean([int(r["midi"]) for r in sub])) if sub else float("nan")

    sources = sorted({r[source_key] for r in records if source_key in r})
    source_series: dict[str, list[float]] = {}
    aggregate_stats: list[tuple[float, float, float]] = []

    for source in sources:
        series: list[float] = []
        for task in task_values:
            preds = [
                pred for r in records
                if r.get(source_key) == source and r.get(task_key) == task
                for pred in [_record_predicted_midi(r)]
                if pred is not None
            ]
            series.append(float(np.mean(preds)) if preds else float("nan"))
        source_series[source] = series

    for task in task_values:
        preds = [
            pred for r in records
            if r.get(task_key) == task
            for pred in [_record_predicted_midi(r)]
            if pred is not None
        ]
        aggregate_stats.append(_mean_and_ci(preds) if preds else (float("nan"), float("nan"), float("nan")))

    actual_series = [actual_by_task[task] for task in task_values]
    return task_values, task_labels, actual_series, source_series, aggregate_stats


def _grouped_bar(
    ax,
    categories: list[str],
    data: dict[str, list[float]],   # variant → per-category accuracy (0–1)
    title: str,
    xlabel: str,
    ylabel: str = "Exact-match accuracy (%)",
) -> None:
    import numpy as np

    variants = [v for v in ("midi", "abc", "solfege") if v in data]
    n = len(categories)
    k = len(variants)
    width = 0.8 / k
    x = np.arange(n)

    for i, variant in enumerate(variants):
        vals = [v * 100 for v in data[variant]]
        ax.bar(
            x + (i - k / 2 + 0.5) * width,
            vals,
            width,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, None),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)


def save_accuracy_plots(
    records: list[dict[str, Any]],
    run_dir: Path,
    model_name: str,
    instrument_key: str = "instrument",
    pitch_key: str = "midi",
    prompt_key: str = "prompt_variant",
    accuracy_key: str = "exact_match",
) -> None:
    """
    Produce two bar-chart PNGs in run_dir:
      accuracy_per_instrument_<model>.png
      accuracy_per_pitch_<model>.png

    Args:
        records:        list of per-item result dicts
        run_dir:        directory to write plots into
        model_name:     used in filenames and titles
        instrument_key: field name holding instrument/waveform label
        pitch_key:      field name holding MIDI note number (int) or condition string
        prompt_key:     field name holding the prompt variant string
        accuracy_key:   field name holding 0/1 exact-match boolean/int
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    variants = sorted({r[prompt_key] for r in records if prompt_key in r})

    # ── Plot A: per instrument ────────────────────────────────────────────────

    instruments = sorted({r[instrument_key] for r in records if instrument_key in r})
    data_a: dict[str, list[float]] = {}
    for v in variants:
        accs = []
        for inst in instruments:
            sub = [r for r in records if r.get(instrument_key) == inst and r.get(prompt_key) == v]
            accs.append(np.mean([r[accuracy_key] for r in sub]) if sub else float("nan"))
        data_a[v] = accs

    fig, ax = plt.subplots(figsize=(max(8, len(instruments) * 0.8 + 2), 4))
    _grouped_bar(
        ax, instruments, data_a,
        title=f"Accuracy per instrument — {model_name}",
        xlabel="Instrument / waveform",
    )
    plt.tight_layout()
    p = run_dir / f"accuracy_per_instrument_{model_name}.png"
    plt.savefig(p, dpi=150)
    plt.close()

    # ── Plot B: per pitch ─────────────────────────────────────────────────────

    pitch_vals = sorted({r[pitch_key] for r in records if pitch_key in r})
    # convert MIDI ints to note names for labels if possible
    try:
        from helpers.music import midi_to_note
        pitch_labels = [midi_to_note(int(p)) if str(p).isdigit() else str(p)
                        for p in pitch_vals]
    except Exception:
        pitch_labels = [str(p) for p in pitch_vals]

    data_b: dict[str, list[float]] = {}
    for v in variants:
        accs = []
        for pv in pitch_vals:
            sub = [r for r in records if r.get(pitch_key) == pv and r.get(prompt_key) == v]
            accs.append(np.mean([r[accuracy_key] for r in sub]) if sub else float("nan"))
        data_b[v] = accs

    fig, ax = plt.subplots(figsize=(max(10, len(pitch_vals) * 0.45 + 2), 4))
    _grouped_bar(
        ax, pitch_labels, data_b,
        title=f"Accuracy per pitch — {model_name}",
        xlabel="Pitch (MIDI note)",
    )
    plt.tight_layout()
    p = run_dir / f"accuracy_per_pitch_{model_name}.png"
    plt.savefig(p, dpi=150)
    plt.close()

    print(f"Plots saved → {run_dir}/accuracy_per_{{instrument,pitch}}_{model_name}.png")


def save_pitch_prediction_plots(
    records: list[dict[str, Any]],
    run_dir: Path,
    model_name: str,
    source_key: str = "source",
    task_key: str = "midi",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    out_dir = run_dir / "prediction_plots" / "per_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [v for v in ("midi", "abc", "solfege") if any(r.get("prompt_variant") == v for r in records)]

    for variant in variants:
        sub = [r for r in records if r.get("prompt_variant") == variant]
        if not sub:
            continue

        _, task_labels, actual_series, source_series, aggregate_stats = _prepare_prediction_series(
            sub, source_key=source_key, task_key=task_key,
        )
        x = np.arange(len(task_labels))

        fig, ax = plt.subplots(figsize=(max(12, len(task_labels) * 0.35 + 3), 6))
        ax.plot(x, actual_series, color="#111111", linewidth=2.4, label="Actual MIDI")
        cmap = plt.get_cmap("tab20")
        for idx, (source, series) in enumerate(source_series.items()):
            ax.plot(
                x,
                series,
                linewidth=1.2,
                alpha=0.8,
                color=cmap(idx % 20),
                label=source,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Task")
        ax.set_ylabel("MIDI value")
        ax.set_title(f"Predicted vs actual MIDI by source — {VARIANT_LABELS.get(variant, variant)} — {model_name}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(out_dir / f"predicted_midi_by_source_{variant}_{_slug(model_name)}.png", dpi=150)
        plt.close()

        agg_mean = [m for m, _, _ in aggregate_stats]
        agg_low = [lo for _, lo, _ in aggregate_stats]
        agg_high = [hi for _, _, hi in aggregate_stats]

        fig, ax = plt.subplots(figsize=(max(12, len(task_labels) * 0.35 + 3), 6))
        ax.plot(x, actual_series, color="#111111", linewidth=2.4, label="Actual MIDI")
        ax.plot(x, agg_mean, color=VARIANT_COLORS.get(variant, "#4C72B0"), linewidth=2.0, label="Mean prediction")
        ax.fill_between(
            x,
            agg_low,
            agg_high,
            color=VARIANT_COLORS.get(variant, "#4C72B0"),
            alpha=0.2,
            label="95% CI",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Task")
        ax.set_ylabel("MIDI value")
        ax.set_title(f"Aggregate predicted vs actual MIDI — {VARIANT_LABELS.get(variant, variant)} — {model_name}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"predicted_midi_overall_{variant}_{_slug(model_name)}.png", dpi=150)
        plt.close()


def save_cross_model_pitch_plots(
    model_records: dict[str, list[dict[str, Any]]],
    run_dir: Path,
    source_key: str = "source",
    task_key: str = "midi",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if not model_records:
        return

    out_dir = run_dir / "prediction_plots" / "per_instrument"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = [record for records in model_records.values() for record in records]
    variants = [v for v in ("midi", "abc", "solfege") if any(r.get("prompt_variant") == v for r in all_records)]
    sources = sorted({r[source_key] for r in all_records if source_key in r})

    for variant in variants:
        for source in sources:
            variant_records = [r for r in all_records if r.get("prompt_variant") == variant and r.get(source_key) == source]
            if not variant_records:
                continue

            task_values = _sorted_task_values(variant_records, task_key)
            task_labels = _task_labels(variant_records, task_values, task_key)
            x = np.arange(len(task_values))

            actual_series: list[float] = []
            for task in task_values:
                sub = [r for r in variant_records if r.get(task_key) == task and "midi" in r]
                actual_series.append(float(np.mean([int(r["midi"]) for r in sub])) if sub else float("nan"))

            fig, ax = plt.subplots(figsize=(max(12, len(task_labels) * 0.35 + 3), 6))
            ax.plot(x, actual_series, color="#111111", linewidth=2.4, label="Actual MIDI")

            for idx, model_name in enumerate(sorted(model_records)):
                model_variant_records = [
                    r for r in model_records[model_name]
                    if r.get("prompt_variant") == variant and r.get(source_key) == source
                ]
                if not model_variant_records:
                    continue

                series: list[float] = []
                for task in task_values:
                    preds = [
                        pred for r in model_variant_records
                        if r.get(task_key) == task
                        for pred in [_record_predicted_midi(r)]
                        if pred is not None
                    ]
                    series.append(float(np.mean(preds)) if preds else float("nan"))

                ax.plot(
                    x,
                    series,
                    linewidth=1.6,
                    marker="o",
                    markersize=3,
                    color=MODEL_COLORS[idx % len(MODEL_COLORS)],
                    label=model_name,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Task")
            ax.set_ylabel("MIDI value")
            ax.set_title(f"Per-instrument predictions by model — {source} — {VARIANT_LABELS.get(variant, variant)}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"predicted_midi_by_model_{variant}_{_slug(source)}.png", dpi=150)
            plt.close()
