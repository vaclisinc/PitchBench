"""
Standard plots used by experiments.

The basic helper produces grouped accuracy bar charts. Pitch experiments can also
generate target-vs-prediction line plots per model and per instrument.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import re


def _json_safe(obj: Any) -> Any:
    """Recursively convert a structure to JSON-serialisable form (NaN → None)."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "tolist"):  # numpy arrays / scalars
        return _json_safe(obj.tolist())
    return obj


def save_plot_data_json(png_path: Path, data: dict[str, Any]) -> Path:
    """Write a JSON sidecar next to ``png_path`` (same stem, ``.json``).

    The JSON captures the data points plotted so reviewers can re-render or
    audit the values without OCR-ing the PNG.
    """
    json_path = png_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_safe(data), indent=2))
    return json_path

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
            from pitchbench.experiments.helpers.music import note_to_midi
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
            from pitchbench.experiments.helpers.music import midi_to_note
            return [f"{v}\n({midi_to_note(int(v))})" for v in task_values]
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


def _hz_iou(pred: Any, gt: Any) -> float:
    """Ratio metric for Hz predictions: min/max in [0, 1]. 1 = exact."""
    try:
        p, g = float(pred), float(gt)
        if p > 0 and g > 0:
            return min(p, g) / max(p, g)
    except (TypeError, ValueError):
        pass
    return 0.0


def _iv_labels(iv_values: list[Any], iv_key: str) -> list[str]:
    """Format IV values for x-axis; MIDI keys get '69\\n(A4)' style labels."""
    if "midi" in iv_key.lower() and iv_values and all(isinstance(v, int) for v in iv_values):
        try:
            from pitchbench.experiments.helpers.music import midi_to_note
            return [f"{v}\n({midi_to_note(v)})" for v in iv_values]
        except Exception:
            pass
    return [str(v) for v in iv_values]


FORMAT_METRIC: dict[str, str] = {
    "midi":   "Exact-match accuracy (%)",
    "spn":    "Exact-match accuracy (%)",
    "doremi": "Exact-match accuracy (%)",
    "hz":     "Hz accuracy (%, ±1% tolerance)",
    "any":    "Any-format accuracy (%)",
}

FORMAT_CORRECT_KEY: dict[str, str | None] = {
    "midi":   "midi_correct",
    "spn":    "spn_correct",
    "doremi": "doremi_correct",
    "hz":     "hz_correct",
    "any":    "any_correct",
}

FORMAT_COLORS: dict[str, str] = {
    "midi":   "#4C72B0",
    "spn":    "#DD8452",
    "doremi": "#55A868",
    "hz":     "#C44E52",
    "any":    "#8172B3",
}

FORMAT_DISPLAY: dict[str, str] = {
    "midi":   "MIDI",
    "spn":    "SPN / ABC",
    "doremi": "Doremi",
    "hz":     "Hz",
    "any":    "Any",
}


def _format_score(record: dict[str, Any], fmt: str) -> float | None:
    """Return the per-record score for a given format.

    Hz uses the binary ``hz_correct`` (1 iff pred / gt ∈ [0.99, 1.01]); the
    same shape as the other three formats so all four can be plotted on one
    accuracy axis.
    """
    key = FORMAT_CORRECT_KEY.get(fmt)
    if key is None:
        return None
    val = record.get(key)
    return float(val) if val is not None else None


def save_per_format_iv_plots(
    records: list[dict[str, Any]],
    run_dir: Path,
    model_name: str,
    iv_key: str,
    iv_label: str = "",
    group_by_source: bool = True,
) -> None:
    """One bar-chart PNG per notation format (midi/spn/doremi/hz).

    By default the bars are grouped by source within each IV bucket
    (used by a1 where the source dimension is part of the design).
    Pass ``group_by_source=False`` to draw a single bar per IV value —
    appropriate when the IV (e.g. duration, condition) is the primary axis
    and the source dimension is incidental.

    Hz bars show IoU instead of exact-match accuracy.
    Files: per_format/<fmt>_by_<iv_key>_<model>.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    out_dir = run_dir / "per_format"
    out_dir.mkdir(parents=True, exist_ok=True)

    iv_vals   = sorted({r[iv_key] for r in records if iv_key in r})
    iv_lbls   = _iv_labels(iv_vals, iv_key)
    sources   = sorted({r.get("source", r.get("instrument", "")) for r in records})
    x         = np.arange(len(iv_vals))
    x_label   = iv_label or iv_key

    for fmt in ("midi", "spn", "doremi", "hz"):
        # series: source → per-IV-value mean score
        series: dict[str, list[float]] = {}
        if group_by_source:
            for src in sources:
                vals = []
                for iv in iv_vals:
                    sub = [r for r in records
                           if r.get(iv_key) == iv
                           and r.get("source", r.get("instrument", "")) == src]
                    scores = [s for r in sub for s in [_format_score(r, fmt)] if s is not None]
                    vals.append(float(np.mean(scores) * 100) if scores else float("nan"))
                series[src] = vals
        else:
            # Single bar per IV value — collapse over source.
            vals = []
            for iv in iv_vals:
                sub    = [r for r in records if r.get(iv_key) == iv]
                scores = [s for r in sub for s in [_format_score(r, fmt)] if s is not None]
                vals.append(float(np.mean(scores) * 100) if scores else float("nan"))
            series["any"] = vals

        cmap  = plt.get_cmap("tab20")

        fig, ax = plt.subplots(figsize=(max(10, len(iv_vals) * 0.45 + 2), 4))
        if group_by_source:
            k = len(sources)
            width = min(0.8 / max(k, 1), 0.35)
            for i, src in enumerate(sources):
                ax.bar(
                    x + (i - k / 2 + 0.5) * width,
                    series[src],
                    width,
                    label=src,
                    color=cmap(i % 20),
                    alpha=0.85,
                )
            ax.legend(fontsize=7, ncol=max(1, k // 4 + 1))
        else:
            ax.bar(x, series["any"], 0.7, color=FORMAT_COLORS[fmt], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(iv_lbls, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_xlabel(x_label)
        ax.set_ylabel(FORMAT_METRIC[fmt])
        ax.set_title(f"{FORMAT_DISPLAY[fmt]} — {model_name}")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fname = out_dir / f"{fmt}_by_{_slug(iv_key)}_{_slug(model_name)}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        save_plot_data_json(fname, {
            "experiment_iv":    iv_key,
            "iv_values":        iv_vals,
            "iv_labels":        iv_lbls,
            "format":           fmt,
            "grouped_by_source": group_by_source,
            "metric":           FORMAT_METRIC[fmt],
            "sources":          sources,
            "series_pct":       series,
            "model_name":       model_name,
        })

    print(f"Per-format plots → {out_dir}/")


def save_combined_iv_plot(
    records: list[dict[str, Any]],
    run_dir: Path,
    model_name: str,
    iv_key: str,
    iv_label: str = "",
) -> None:
    """One grouped bar chart comparing notation formats on accuracy vs the main IV.

    Hz is omitted: it uses an IoU-style ratio metric, not exact-match
    accuracy, so plotting it on the same y-axis would mislead.

    File: combined_iv_<iv_key>_<model>.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    iv_vals = sorted({r[iv_key] for r in records if iv_key in r})
    iv_lbls = _iv_labels(iv_vals, iv_key)
    x       = np.arange(len(iv_vals))
    x_label = iv_label or iv_key

    formats = ("midi", "spn", "doremi")   # Hz omitted — different score type
    fig, ax = plt.subplots(figsize=(max(10, len(iv_vals) * 0.45 + 2), 4))
    series_pct: dict[str, list[float]] = {}
    width = min(0.8 / len(formats), 0.28)
    for i, fmt in enumerate(formats):
        means = []
        for iv in iv_vals:
            sub    = [r for r in records if r.get(iv_key) == iv]
            scores = [s for r in sub for s in [_format_score(r, fmt)] if s is not None]
            means.append(float(np.mean(scores) * 100) if scores else float("nan"))
        series_pct[fmt] = means
        offset = (i - (len(formats) - 1) / 2) * width
        ax.bar(x + offset, means, width,
               color=FORMAT_COLORS[fmt],
               label=FORMAT_DISPLAY[fmt],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(iv_lbls, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs {x_label} — {model_name}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fname = run_dir / f"combined_iv_{_slug(iv_key)}_{_slug(model_name)}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    save_plot_data_json(fname, {
        "experiment_iv": iv_key,
        "iv_values":     iv_vals,
        "iv_labels":     iv_lbls,
        "formats":       list(formats),
        "metric":        "Accuracy (%)",
        "series_pct":    series_pct,
        "model_name":    model_name,
    })
    print(f"Combined IV plot → {fname}")


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
    p = run_dir / f"accuracy_per_instrument_{_slug(model_name)}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    save_plot_data_json(p, {
        "axis":         "instrument",
        "instruments":  instruments,
        "variants":     variants,
        "accuracy_pct": {v: [a * 100 for a in accs] for v, accs in data_a.items()},
        "model_name":   model_name,
    })

    # ── Plot B: per pitch ─────────────────────────────────────────────────────

    pitch_vals = sorted({r[pitch_key] for r in records if pitch_key in r})
    try:
        from pitchbench.experiments.helpers.music import midi_to_note
        pitch_labels = [f"{p}\n({midi_to_note(int(p))})" if isinstance(p, int) else str(p)
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
    p = run_dir / f"accuracy_per_pitch_{_slug(model_name)}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    save_plot_data_json(p, {
        "axis":         "pitch",
        "pitch_values": pitch_vals,
        "pitch_labels": pitch_labels,
        "variants":     variants,
        "accuracy_pct": {v: [a * 100 for a in accs] for v, accs in data_b.items()},
        "model_name":   model_name,
    })

    print(f"Plots saved → {run_dir}/accuracy_per_{{instrument,pitch}}_{_slug(model_name)}.png")


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
        png_by_source = out_dir / f"predicted_midi_by_source_{variant}_{_slug(model_name)}.png"
        plt.savefig(png_by_source, dpi=150)
        plt.close()
        save_plot_data_json(png_by_source, {
            "kind":          "predicted_midi_by_source",
            "variant":       variant,
            "task_labels":   task_labels,
            "actual_series": actual_series,
            "source_series": source_series,
            "model_name":    model_name,
        })

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
        png_overall = out_dir / f"predicted_midi_overall_{variant}_{_slug(model_name)}.png"
        plt.savefig(png_overall, dpi=150)
        plt.close()
        save_plot_data_json(png_overall, {
            "kind":          "predicted_midi_overall",
            "variant":       variant,
            "task_labels":   task_labels,
            "actual_series": actual_series,
            "agg_mean":      agg_mean,
            "agg_low":       agg_low,
            "agg_high":      agg_high,
            "model_name":    model_name,
        })


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
            png = out_dir / f"predicted_midi_by_model_{variant}_{_slug(source)}.png"
            plt.savefig(png, dpi=150)
            plt.close()
            save_plot_data_json(png, {
                "kind":          "predicted_midi_by_model",
                "variant":       variant,
                "source":        source,
                "task_labels":   task_labels,
                "actual_series": actual_series,
                "models":        sorted(model_records),
            })


# ─── Uniform per-experiment plots (apply to every experiment) ─────────────────

# Ordered list of primary-score field candidates. The first one present in
# every record is used as that experiment's primary score.
_PRIMARY_SCORE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("midi_correct",        "MIDI accuracy (%)"),
    ("answer_correct",      "Answer accuracy (%)"),
    ("interval_correct",    "Interval accuracy (%)"),
    ("count_correct",       "Count accuracy (%)"),
    ("quality_correct",     "Quality accuracy (%)"),
    ("trajectory_correct",  "Trajectory accuracy (%)"),
    ("sequence_correct",    "Sequence accuracy (%)"),
    ("exact_match",         "Exact-match accuracy (%)"),
    ("mean_iou",            "Mean IoU (%)"),
    ("iou",                 "IoU (%)"),
    ("midi_sequence_correct", "Sequence accuracy (%)"),
    ("midi_seq_correct",    "Sequence accuracy (%)"),
)


def _detect_primary_score(records: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Return ``(key, label)`` for the first known scalar score field present
    in every record, or ``None`` if no recognised score field is found.
    """
    if not records:
        return None
    for key, label in _PRIMARY_SCORE_CANDIDATES:
        if all(isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
               for r in records):
            return key, label
    return None


def _bar_plot_by_key(
    records: list[dict[str, Any]],
    group_key: str,
    score_key: str,
    score_label: str,
    title: str,
    xlabel: str,
    out_path: Path,
    *,
    label_formatter=None,
) -> dict[str, Any] | None:
    """Render a bar plot of ``mean(records[score_key]) * 100`` grouped by
    ``records[group_key]`` and save PNG + JSON sidecar.

    Returns the data dict written to JSON, or ``None`` if matplotlib is missing
    or no buckets had data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    buckets: dict[Any, list[float]] = {}
    for r in records:
        if group_key not in r or not isinstance(r.get(score_key), (int, float)):
            continue
        if isinstance(r.get(score_key), bool):
            continue
        buckets.setdefault(r[group_key], []).append(float(r[score_key]))
    if not buckets:
        return None

    keys = sorted(buckets.keys(), key=lambda v: (str(type(v)), v))
    means = [float(np.mean(buckets[k]) * 100) for k in keys]
    counts = [len(buckets[k]) for k in keys]
    labels = [label_formatter(k) for k in keys] if label_formatter else [str(k) for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.55 + 2), 4))
    ax.bar(range(len(keys)), means, color="#4C72B0", alpha=0.85)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(score_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate bars with n
    for i, (m, n) in enumerate(zip(means, counts)):
        ax.text(i, min(m + 2, 105), f"n={n}", ha="center", fontsize=7, color="#555")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    data = {
        "group_key":   group_key,
        "score_key":   score_key,
        "score_label": score_label,
        "x_values":    keys,
        "x_labels":    labels,
        "mean_pct":    means,
        "n_samples":   counts,
    }
    save_plot_data_json(out_path, data)
    return data


# Public alias — the bar-plot helper is generally useful and callable from
# experiment scripts that want a single-metric ad-hoc bar chart with a JSON
# sidecar (e.g. the explicit MIDI-only plots in a1).
save_bar_plot_by_key = _bar_plot_by_key


def _is_four_format_pitch_record(records: list[dict[str, Any]]) -> bool:
    """A record is a 4-format pitch record iff every entry carries the four
    standard correctness fields. Such experiments produce their own per-format
    + combined plots, so the universal pipeline below is skipped to avoid
    instrument/pitch-centred plots the experiment explicitly didn't ask for."""
    if not records:
        return False
    return all(
        all(k in r for k in ("midi_correct", "doremi_correct", "hz_correct"))
        for r in records
    )


def save_uniform_plots(
    records: list[dict[str, Any]],
    run_dir: Path,
    model_name: str,
    exp_name: str,
) -> None:
    """Generate the standard set of per-experiment plots:

    - ``by_source_<model>.png``           — accuracy grouped by source
    - ``by_<iv>_<model>.png`` (one per IV) — accuracy grouped by each major IV

    Major IVs come from ``config.EXPERIMENT_DEFAULTS[exp_name]["strata"]``.
    The primary score is auto-detected from the records (first matching
    scalar field in :data:`_PRIMARY_SCORE_CANDIDATES`); if none match, this
    function quietly returns.

    Four-format pitch experiments (a1-a5, b1, b4, e1-e3, z1) are skipped:
    they call ``save_per_format_iv_plots`` and ``save_combined_iv_plot``
    explicitly, and the user has opted out of the instrument/pitch-centred
    plots this helper would otherwise add for them.
    """
    if _is_four_format_pitch_record(records):
        return
    score = _detect_primary_score(records)
    if score is None:
        return
    score_key, score_label = score

    out_dir = run_dir / "uniform"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) by source (universal axis)
    if any("source" in r for r in records):
        _bar_plot_by_key(
            records,
            group_key="source",
            score_key=score_key,
            score_label=score_label,
            title=f"{score_label} by source — {model_name}",
            xlabel="Source",
            out_path=out_dir / f"by_source_{_slug(model_name)}.png",
        )

    # 2) one bar plot per major IV (from EXPERIMENT_DEFAULTS strata)
    try:
        from pitchbench import config
        strata = tuple(config.EXPERIMENT_DEFAULTS.get(exp_name, {}).get("strata", ()))
    except Exception:
        strata = ()

    for iv in strata:
        if iv == "source":   # already plotted above
            continue
        # Records may rename the strata key (e.g. strata says "midi" but
        # 4-format records carry "midi_gt"). Try a "_gt" fallback.
        if any(iv in r for r in records):
            iv_field = iv
        elif any(f"{iv}_gt" in r for r in records):
            iv_field = f"{iv}_gt"
        else:
            continue
        formatter = None
        if iv in ("midi", "midi_gt"):
            try:
                from pitchbench.experiments.helpers.music import midi_to_note
                formatter = lambda v: f"{v}\n({midi_to_note(int(v))})" if isinstance(v, int) else str(v)
            except Exception:
                formatter = None
        _bar_plot_by_key(
            records,
            group_key=iv_field,
            score_key=score_key,
            score_label=score_label,
            title=f"{score_label} by {iv} — {model_name}",
            xlabel=iv,
            out_path=out_dir / f"by_{_slug(iv)}_{_slug(model_name)}.png",
            label_formatter=formatter,
        )
