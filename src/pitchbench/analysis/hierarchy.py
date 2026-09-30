"""Editable manuscript Figure 1: the complete PitchBench task hierarchy.

PYTHONPATH=src python -m pitchbench.analysis.hierarchy --output-dir OUTPUT \
    --qa-scripts /PATH/TO/nature-figure/scripts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parents[3]


# Short display labels preserve the task meanings in EXPERIMENTS.md and the
# manuscript Task Hierarchy. These are a taxonomy, not simulated observations.
CATEGORIES = (
    ("A", "Single-note identification", 1, (
        "Single pitch", "Loudness variation", "Duration variation")),
    ("B", "Temporal localization", 2, (
        "Pitch within silence", "Pitch at a timestamp", "Timing of a single tone",
        "Timing of a named pitch", "Timing of every note")),
    ("C", "Chordal structure", 2, (
        "Count simultaneous pitches", "Identify a dyad interval",
        "Identify chord quality", "Name all pitches in a chord")),
    ("D", "Sequential structure", 2, (
        "Count pitches in sequence", "Higher / lower judgement",
        "Discrete melodic contour", "Continuous pitch trajectory",
        "Rank pitches by height", "Signed interval between notes",
        "Pitch with a reference tone", "All pitches in sequence")),
    ("E", "Acoustic variations", 2, (
        "Audio effects", "Background interference", "Harmonic saturation",
        "Time stretching / resampling", "Vibrato", "Slight detuning")),
    ("F", "Melody in polyphony", 3, (
        "Voice in synthetic mixtures", "Voice in Bach chorales")),
)
LEVEL_COLORS = {1: "#526C80", 2: "#376E91", 3: "#287C80"}
LEVEL_FILLS = {1: "#F0F4F7", 2: "#F1F6F9", 3: "#EDF5F4"}
NAME = "figure1_task_hierarchy"


def source(path: Path) -> dict:
    return {"path": path.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def render(output_dir: Path, qa_scripts: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(task=f"{letter}{i}", category=title, level=level, label=label)
            for letter, title, level, labels in CATEGORIES
            for i, label in enumerate(labels, 1)]
    expected_counts = {"A": 3, "B": 5, "C": 4, "D": 8, "E": 6, "F": 2}
    assert {c: len(labels) for c, _, _, labels in CATEGORIES} == expected_counts
    assert len(rows) == len({r["task"] for r in rows}) == 28
    with (output_dir / f"{NAME}.tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "text.color": "#252C34",
                         "figure.facecolor": "white", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.dpi": 600})
    # 139.7 mm equals the manuscript text width. No rescaling or tight crop.
    fig = plt.figure(figsize=(5.5, 3.8))
    grid = fig.add_gridspec(4, 2, left=.225, right=.992, bottom=.025, top=.985,
                           height_ratios=[.53, .94, 1.36, .53],
                           hspace=.15, wspace=.06)
    specs = [grid[0, :], grid[1, 0], grid[1, 1], grid[2, 0], grid[2, 1], grid[3, :]]
    axes = []
    for (letter, title, level, labels), spec in zip(CATEGORIES, specs):
        ax = fig.add_subplot(spec)
        axes.append(ax)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        accent = LEVEL_COLORS[level]
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=LEVEL_FILLS[level], edgecolor="none"))
        # All distances are physical points, so type and padding stay consistent.
        width = ax.get_position().width * fig.get_figwidth() * 72
        height = ax.get_position().height * fig.get_figheight() * 72
        tx, ty = lambda pt: pt / width, lambda pt: 1 - pt / height
        ax.add_patch(Rectangle((0, 0), tx(2), 1, facecolor=accent, edgecolor="none"))
        ax.text(tx(9), ty(11), letter, fontsize=8, weight="bold", color=accent,
                va="center")
        ax.text(tx(22), ty(11), title, fontsize=7.1, weight="bold", va="center")
        if letter in ("A", "F"):
            # Compact horizontal tasks for the atomic and melodic end levels.
            available = width - 18
            for i, label in enumerate(labels):
                x = 9 + i * available / len(labels)
                ax.text(tx(x), ty(26), f"{letter}{i+1}", fontsize=6.1,
                        weight="bold", color=accent, va="center")
                ax.text(tx(x + 14), ty(26), label, fontsize=6.1, va="center")
        else:
            for i, label in enumerate(labels):
                y = 26 + i * 8.4
                ax.text(tx(9), ty(y), f"{letter}{i+1}", fontsize=6.3,
                        weight="bold", color=accent, va="center")
                ax.text(tx(24), ty(y), label, fontsize=6.3, va="center")

    # A single vertical rail encodes the conceptual progression, not task scores.
    centres = [sum(axes[0].get_position().intervaly) / 2,
               (axes[1].get_position().y1 + axes[4].get_position().y0) / 2,
               sum(axes[-1].get_position().intervaly) / 2]
    for upper, lower in zip(centres, centres[1:]):
        fig.add_artist(FancyArrowPatch((.032, upper-.023), (.032, lower+.023),
                                      transform=fig.transFigure, arrowstyle="-|>",
                                      mutation_scale=6, linewidth=.65, color="#B6C4CE"))
    for level, title, y in zip((1, 2, 3), ("Atomic", "Contextual", "Melodic"), centres):
        color = LEVEL_COLORS[level]
        # Ellipse compensated for figure aspect, yielding a physical circle.
        fig.add_artist(Ellipse((.032, y), .036, .036 * 5.5 / 3.8,
                               transform=fig.transFigure, facecolor=color, edgecolor="none"))
        fig.text(.032, y, str(level), ha="center", va="center", fontsize=6.5,
                 weight="bold", color="white")
        fig.text(.065, y+.033, f"LEVEL {level}", fontsize=5.6, color=color)
        fig.text(.065, y-.005, title, fontsize=8.3, weight="bold", color=color)
        fig.text(.065, y-.039, "pitch perception", fontsize=5.6, color="#56636D")

    fig.canvas.draw()
    geometry = {"size_inches": [5.5, 3.8], "alignment": "not audited"}
    if qa_scripts is not None:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        report = require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=[category[0] for category in CATEGORIES],
            json_out=output_dir / f"{NAME}.alignment.json",
            overlay_svg=output_dir / f"{NAME}.alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        geometry["alignment"] = report.get("verdict", report.get("status", "see audit"))
    fig.savefig(output_dir / f"{NAME}.pdf", dpi=600)
    fig.savefig(output_dir / f"{NAME}.svg", dpi=600)
    fig.savefig(output_dir / f"{NAME}.png", dpi=600)
    plt.close(fig)
    manifest = {
        "generator": source(Path(__file__)),
        "task_definitions": source(REPO / "EXPERIMENTS.md"),
        "geometry": geometry,
        "category_counts": expected_counts,
        "level_counts": {str(level): sum(r["level"] == level for r in rows) for level in (1, 2, 3)},
        "n_tasks": 28,
        "meaning": "Conceptual taxonomy; no empirical scores, difficulty scale, or uncertainty encoded.",
        "exclusions": "None: all 28 tasks and six categories retained.",
        "exports": "PDF and SVG with editable text; 600-dpi PNG.",
    }
    (output_dir / f"{NAME}.sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {NAME} with all 28 tasks to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path)
    args = parser.parse_args()
    render(args.output_dir, args.qa_scripts)


if __name__ == "__main__":
    main()
