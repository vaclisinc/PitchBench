"""Redraw manuscript Figures 2–4 from saved evidence, without model calls.

PYTHONPATH=src python -m pitchbench.analysis.publication --output-dir OUTPUT

Pass --qa-scripts to the nature-figure skill's scripts directory to require
its panel-alignment gate before export. The original results and figures are
never overwritten. See results/publication-figures/README.md for provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pitchbench.analysis.a1 import _predicted_midi
from pitchbench.experiments.helpers.music import midi_to_note


REPO = Path(__file__).resolve().parents[3]
WIDTH_IN = 5.5  # Actual NeurIPS manuscript text width: 139.7 mm.
MODELS = (
    "audio_flamingo_next_instruct",
    "openrouter_google_gemini_3_1_pro_preview",
    "openrouter_google_gemini_flash_latest",
    "openrouter_openai_gpt_4o_audio_preview",
    "dashscope_qwen3_5_omni_plus",
    "dashscope_qwen3_5_omni_flash",
)
NAMES = (
    "Audio Flamingo Next",
    "Gemini 3.1 Pro",
    "Gemini Flash",
    "GPT-4o Audio",
    "Qwen 3.5 Omni Plus",
    "Qwen 3.5 Omni Flash",
)
COLORS = ("#287C80", "#9F6A2C", "#B39A3F", "#4477AA", "#705098", "#BA667F")
MARKERS = ("o", "s", "^", "D", "P", "X")
FORMATS = ("midi", "spn", "hz")
PITCH_MIN, PITCH_MAX = 29, 89
INK = "#252C34"
CMAP = LinearSegmentedColormap.from_list(
    "pitch_probability", ["#FFFFFF", "#DCE9F1", "#A7C7DD", "#5792B6", "#174769"]
)


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.55,
        "axes.edgecolor": "#727B84",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> dict:
    return {"path": path.relative_to(REPO).as_posix(), "sha256": sha(path)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_aggregate(rows: list[dict], formats: tuple[str, ...], pitches=None) -> None:
    """Fail rather than silently drop, duplicate, or fabricate plotted cells."""
    keys = []
    for row in rows:
        value = float(row["accuracy"])
        if not math.isfinite(value) or not 0 <= value <= 1 or int(row["n_samples"]) <= 0:
            raise ValueError(f"Invalid aggregate: {row}")
        key = (row["model"], row["format"])
        keys.append((*key, int(row["pitch"])) if pitches is not None else key)
    expected = {(m, f) for m in MODELS for f in formats}
    if pitches is not None:
        expected = {(m, f, p) for m, f in expected for p in pitches}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Aggregate coverage differs from the paper figure contract")


def export(fig, out: Path, name: str, qa_scripts: Path | None, **alignment) -> dict:
    """Measure the final layout; retain editable vectors and a 600-dpi preview."""
    fig.canvas.draw()
    geometry = {
        "size_inches": list(fig.get_size_inches()),
        "axes_rectangles": [list(ax.get_position().bounds) for ax in alignment["axes"]],
        "alignment": "not audited",
    }
    if qa_scripts is not None:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        report = require_matplotlib_panel_alignment(
            fig, json_out=out / f"{name}.alignment.json",
            overlay_svg=out / f"{name}.alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **alignment,
        )
        geometry["alignment"] = report.get("verdict", report.get("status", "see audit"))
    # No tight cropping: exported physical dimensions match the audit exactly.
    fig.savefig(out / f"{name}.pdf", dpi=600)
    fig.savefig(out / f"{name}.svg", dpi=600)
    fig.savefig(out / f"{name}.png", dpi=600)
    plt.close(fig)
    return geometry


def a1_evidence() -> tuple[dict, list[dict], list[dict], list[dict]]:
    matrices, coverage, sparse, inputs = {}, [], [], []
    size = PITCH_MAX - PITCH_MIN + 1
    for model in MODELS:
        paths = list((REPO / "paper/evaluation" / ("_" + model)).glob(
            "pitchbench_a1_single_pitch_id/*/results_*.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one saved A1 run for {model}")
        path = paths[0]
        inputs.append(source(path))
        rows = read_csv(path)
        gt_counts = Counter(int(float(r["midi_gt"])) for r in rows)
        if gt_counts != Counter({pitch: 19 for pitch in range(PITCH_MIN, PITCH_MAX + 1)}):
            raise ValueError(f"Unexpected A1 ground-truth coverage for {model}")
        for fmt in FORMATS:
            counts = np.zeros((size, size), dtype=int)
            statuses = Counter()
            for row in rows:
                gt = int(float(row["midi_gt"]))
                pred = _predicted_midi(row, fmt, gt)
                if pred is None or not math.isfinite(pred):
                    statuses["invalid"] += 1
                elif not PITCH_MIN <= round(pred) <= PITCH_MAX:
                    statuses["outside_range"] += 1
                else:
                    counts[round(pred) - PITCH_MIN, gt - PITCH_MIN] += 1
                    statuses["in_range"] += 1
            if sum(statuses.values()) != len(rows) or counts.sum() != statuses["in_range"]:
                raise AssertionError("A1 accounting mismatch")
            denominator = counts.sum(axis=0)
            # Preserve the published plot's conditional, column-wise normalization.
            matrix = np.divide(counts, denominator, out=np.zeros_like(counts, dtype=float),
                               where=denominator > 0)
            matrices[model, fmt] = matrix
            coverage.append({
                "model": model, "format": fmt, "n_total": len(rows),
                "n_in_range": statuses["in_range"],
                "n_outside_range": statuses["outside_range"],
                "n_invalid": statuses["invalid"],
                "empty_gt_columns": int((denominator == 0).sum()),
            })
            for pred_idx, gt_idx in zip(*np.nonzero(counts)):
                sparse.append({
                    "model": model, "format": fmt,
                    "midi_gt": int(gt_idx + PITCH_MIN),
                    "midi_pred_rounded": int(pred_idx + PITCH_MIN),
                    "count": int(counts[pred_idx, gt_idx]),
                    "n_in_range_at_gt": int(denominator[gt_idx]),
                    "n_total_at_gt": gt_counts[gt_idx + PITCH_MIN],
                })
    return matrices, coverage, sparse, inputs


def figure2(out: Path, qa_scripts: Path | None) -> dict:
    matrices, coverage, sparse, inputs = a1_evidence()
    write_csv(out / "figure2_response_coverage.csv", coverage)
    write_csv(out / "figure2_prediction_counts.csv", sparse)
    covered = {(r["model"], r["format"]): r["n_in_range"] / r["n_total"] for r in coverage}
    fig = plt.figure(figsize=(WIDTH_IN, 3.62))
    # Eighteen equal axes; model titles sit above, never on the heatmap.
    grid = fig.add_gridspec(3, 6, left=.075, right=.985, bottom=.155, top=.905,
                           wspace=.24, hspace=.69)
    axes = []
    for idx, (model, name) in enumerate(zip(MODELS, NAMES)):
        row, pair = divmod(idx, 2)
        group_axes = []
        for j, fmt in enumerate(FORMATS):
            ax = fig.add_subplot(grid[row, pair * 3 + j])
            axes.append(ax)
            group_axes.append(ax)
            ax.imshow(matrices[model, fmt], origin="lower", aspect="auto", cmap=CMAP,
                      vmin=0, vmax=1, interpolation="nearest",
                      extent=(28.5, 89.5, 28.5, 89.5), rasterized=True)
            ax.set(xticks=[36, 60, 84], yticks=[36, 60, 84], xlim=(28.5, 89.5), ylim=(28.5, 89.5))
            ax.tick_params(pad=1.5, length=1.5, labelsize=5.3)
            if j != 0:
                ax.tick_params(labelleft=False)
            if row != 2:
                ax.tick_params(labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.35)
                spine.set_color("#AFBAC4")
            ax.set_title(f"{fmt.upper()} · {covered[model, fmt]:.0%}", fontsize=5.7, pad=3)
        box0, box1 = group_axes[0].get_position(), group_axes[-1].get_position()
        fig.text(box0.x0, box0.y1 + .060, chr(97 + idx), weight="bold", fontsize=8)
        fig.text((box0.x0 + box1.x1) / 2 + .015, box0.y1 + .060,
                 name, fontsize=6.7, ha="center")
    fig.text(.013, .545, "Predicted pitch (MIDI)", rotation=90,
             rotation_mode="anchor", ha="center", fontsize=7)
    fig.text(.53, .087, "Ground-truth pitch (MIDI)", ha="center", fontsize=7)
    cax = fig.add_axes([.745, .038, .24, .021])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=cax,
                      orientation="horizontal", ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=5.3, pad=1, length=1.5)
    cb.outline.set_visible(False)
    fig.text(.075, .040, "Titles: % of responses within MIDI 29–89", fontsize=5.6)
    fig.text(.735, .045, "Probability", ha="right", fontsize=5.6)
    geometry = export(fig, out, "figure2_pitch_confusion", qa_scripts,
                      axes=axes, panel_ids=[f"{chr(97+i//3)}-{FORMATS[i%3]}" for i in range(18)])
    return {
        "inputs": inputs, "geometry": geometry,
        "normalization": "P(rounded predicted MIDI | ground-truth MIDI, valid prediction in 29–89)",
        "exclusions": coverage,
        "all_response_counts": "figure2_response_coverage.csv",
        "nonzero_heatmap_counts": "figure2_prediction_counts.csv",
        "note": "Zero-count cells are implicit. No predictions were downsampled; excluded answers are disclosed.",
    }


def figure3(out: Path, qa_scripts: Path | None) -> dict:
    path = REPO / "paper/figures-and-tables/accuracies_by_pitch.csv"
    rows = read_csv(path)
    pitches = tuple(range(48, 73))
    check_aggregate(rows, ("midi", "spn"), pitches)
    values = {(r["model"], r["format"], int(r["pitch"])): float(r["accuracy"]) * 100 for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.08), sharey=True)
    fig.subplots_adjust(left=.082, right=.987, bottom=.24, top=.715, wspace=.12)
    for j, (ax, fmt) in enumerate(zip(axes, ("midi", "spn"))):
        ax.axvspan(68.6, 69.4, color="#EAE7F0", zorder=0)
        for model, color, marker in zip(MODELS, COLORS, MARKERS):
            y = [values[model, fmt, p] for p in pitches]
            ax.plot(pitches, y, color=color, marker=marker, linewidth=.9,
                    markersize=2.4, markeredgewidth=.35, zorder=3)
        ax.set(xlim=(47.6, 72.4), ylim=(-2, 104), yticks=[0, 25, 50, 75, 100], xticks=pitches)
        ax.set_xticklabels([midi_to_note(p) for p in pitches], rotation=90,
                           rotation_mode="anchor", ha="right", va="center", fontsize=5.6)
        ax.tick_params(axis="x", pad=3, length=1.8)
        ax.grid(axis="y", color="#E6E9ED", linewidth=.4, zorder=0)
        ax.set_title(fmt.upper(), loc="left", fontsize=7.5, pad=6)
        ax.text(-.075, 1.10, chr(97+j), transform=ax.transAxes, fontsize=8, weight="bold")
        ax.get_xticklabels()[69 - 48].set_weight("bold")
    axes[0].set_ylabel("Accuracy (%)", labelpad=3)
    handles = [Line2D([], [], color=c, marker=m, linewidth=1, markersize=3)
               for c, m in zip(COLORS, MARKERS)]
    fig.legend(handles, NAMES, loc="upper center", bbox_to_anchor=(.54, 1.0),
               ncol=3, columnspacing=1.35, handlelength=1.8, handletextpad=.5,
               labelspacing=.65, borderaxespad=0)
    fig.text(.54, .022, "Ground-truth pitch", ha="center", fontsize=7)
    geometry = export(fig, out, "figure3_pitch_accuracy", qa_scripts,
                      axes=list(axes), panel_ids=["a", "b"], require_panel_labels=True)
    return {"inputs": [source(path)], "geometry": geometry, "n_cells": len(rows),
            "aggregation": "Unmodified saved sample-weighted accuracy for each model, notation and pitch.",
            "uncertainty": "Not supplied in the source aggregate; no intervals inferred.",
            "exclusions": "None: all 300 source rows retained, covering MIDI/SPN and C3–C5."}


def figure4(out: Path, qa_scripts: Path | None) -> dict:
    path = REPO / "paper/figures-and-tables/accuracies_by_notation.csv"
    rows = read_csv(path)
    formats = ("midi", "spn", "doremi", "hz", "any")
    check_aggregate(rows, formats)
    values = {(r["model"], r["format"]): float(r["accuracy"]) * 100 for r in rows}
    matrix = np.array([[values[m, f] for f in formats[:4]] for m in MODELS])
    fig = plt.figure(figsize=(WIDTH_IN, 2.05))
    grid = fig.add_gridspec(1, 5, left=.278, right=.977, bottom=.235, top=.85, wspace=.32)
    ax = fig.add_subplot(grid[0, :4])
    aggregate = fig.add_subplot(grid[0, 4])
    ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set(xticks=range(4), xticklabels=["MIDI", "SPN", "DoReMi", "Hz"],
           yticks=range(6), yticklabels=NAMES)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=5, labelsize=6.4)
    for label, color in zip(ax.get_yticklabels(), COLORS):
        label.set_color(color)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(6):
        for j in range(4):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if value >= 75 else INK,
                    weight="bold" if value == matrix[i].max() else "normal")
    aggregate.set(xlim=(-.5, .5), ylim=(5.5, -.5), yticks=[], xticks=[0], xticklabels=["ANY"])
    aggregate.xaxis.tick_top()
    aggregate.tick_params(axis="x", length=0, pad=5, labelsize=6.4)
    for i, model in enumerate(MODELS):
        aggregate.add_patch(Rectangle((-.5, i-.5), 1, 1, facecolor="#F0F2F4", edgecolor="white", linewidth=1.2))
        aggregate.text(0, i, f"{values[model, 'any']:.1f}", ha="center", va="center", fontsize=7)
    for axis in (ax, aggregate):
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.text(.278, .945, "Requested output representation", fontsize=7)
    fig.text(.91, .945, "Aggregate", ha="center", fontsize=6.4)
    fig.text(.02, .14, "Bold: best requested format for each model", fontsize=5.8)
    cax = fig.add_axes([.70, .10, .27, .025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap=CMAP), cax=cax,
                      orientation="horizontal", ticks=[0, 50, 100])
    cb.ax.tick_params(labelsize=5.5, pad=1, length=1.5)
    cb.outline.set_visible(False)
    fig.text(.69, .109, "Accuracy (%)", ha="right", fontsize=5.8)
    geometry = export(fig, out, "figure4_notation_accuracy", qa_scripts,
                      axes=[ax, aggregate], panel_ids=["formats", "aggregate"])
    return {"inputs": [source(path)], "geometry": geometry, "n_cells": len(rows),
            "aggregation": "All saved values retained. ANY is a historical aggregate, not a fifth response notation; denominators differ.",
            "uncertainty": "Not supplied; no intervals inferred.", "exclusions": "None: all 30 rows retained."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path, help="Run the nature-figure alignment gate before exporting")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    manifest = {
        "generator": source(Path(__file__)),
        "prediction_parser": source(REPO / "src/pitchbench/analysis/a1.py"),
        "music_parser": source(REPO / "src/pitchbench/experiments/helpers/music.py"),
        "backend": "Python/matplotlib", "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__, "width_mm": WIDTH_IN * 25.4,
        "figure2": figure2(args.output_dir, args.qa_scripts),
        "figure3": figure3(args.output_dir, args.qa_scripts),
        "figure4": figure4(args.output_dir, args.qa_scripts),
    }
    (args.output_dir / "figure_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported Figures 2–4 and source/coverage records to {args.output_dir}")


if __name__ == "__main__":
    main()
