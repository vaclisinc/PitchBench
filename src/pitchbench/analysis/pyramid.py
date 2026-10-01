"""Redraw Figure 1 while retaining the original three-level pyramid.

PYTHONPATH=src python -m pitchbench.analysis.pyramid --output-dir OUTPUT \
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
from matplotlib.patches import Polygon
from matplotlib.transforms import Bbox

REPO = Path(__file__).resolve().parents[3]
NAME = "figure1_pitch_pyramid"
INK = "#253B4B"
COLORS = {"A": "#626575", "B": "#527A99", "C": "#7C7193",
          "D": "#627D66", "E": "#9A755E", "F": "#527D79"}
FILLS = {"A": "#E8E8EB", "B": "#DAE7F1", "C": "#E6E1EE",
         "D": "#DDE8DD", "E": "#F3E4D9", "F": "#D9E9E7"}
TASKS = (
    ("A", "Single note", 1, ("Pitch identification", "Loudness variation", "Duration variation")),
    ("B", "Temporal localization", 2, (
        "Pitch within silence", "Pitch at a timestamp", "Single-tone onset / offset",
        "Timing of a named pitch", "Timing of every note")),
    ("C", "Chordal structure", 2, (
        "Count pitches in a chord", "Identify a dyad interval", "Identify chord quality",
        "Name all pitches in a chord")),
    ("D", "Sequence", 2, (
        "Count pitches", "Higher / lower", "Discrete contour",
        "Continuous contour", "Rank pitches", "Signed interval",
        "Reference pitch", "All pitches in order")),
    ("E", "Acoustics", 2, (
        "Audio effects", "Background sounds", "Harmonic saturation",
        "Stretch / resample", "Vibrato", "Slight detuning")),
    ("F", "Polyphonic music", 3, ("Melody in synthetic mixtures", "Melody in Bach chorales")),
)


def source(path: Path) -> dict:
    return {"path": path.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def render(out: Path, qa_scripts: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(task=f"{letter}{i}", category=title, level=level, label=label)
            for letter, title, level, labels in TASKS for i, label in enumerate(labels, 1)]
    assert len(rows) == len({r["task"] for r in rows}) == 28
    assert [len(t[3]) for t in TASKS] == [3, 5, 4, 8, 6, 2]
    with (out / f"{NAME}.tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "text.color": INK, "figure.facecolor": "white",
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 600})
    # One coordinate unit is one export point. The manuscript scales to 5.5 in.
    # Extra room on the right lets D/E sit beside the contextual tier.
    figure_width = 414 / 72
    fig = plt.figure(figsize=(figure_width, 2.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 414), ylim=(208.8, 0))
    ax.set_axis_off()

    def text(x, y, label, size=5.7, color=INK, weight="normal", ha="left"):
        return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                       ha=ha, va="center")

    def poly(points, color):
        ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="white",
                             linewidth=.8, joinstyle="miter"))

    # All three tiers share the same two outer slopes, with open gaps between.
    centre, apex_y, slope = 238, 7, .64

    def edges(y):
        half = (y - apex_y) * slope
        return centre - half, centre + half

    # Level 3: melodic pitch perception at the apex.
    left, right = edges(40)
    poly([(centre, apex_y), (right, 40), (left, 40)], FILLS["F"])
    text(centre, 29, "F", size=10, color=COLORS["F"], weight="bold", ha="center")

    # Level 2: preserve the original four side-by-side category wedges.
    top, bottom = 47, 164
    top_left, top_right = edges(top)
    bottom_left, bottom_right = edges(bottom)
    for i, (letter, label) in enumerate(zip("BCDE", ("Temporal", "Chord", "Sequence", "Acoustic"))):
        u, v = i / 4, (i + 1) / 4
        poly([(top_left + u * (top_right-top_left), top),
              (top_left + v * (top_right-top_left), top),
              (bottom_left + v * (bottom_right-bottom_left), bottom),
              (bottom_left + u * (bottom_right-bottom_left), bottom)], FILLS[letter])
        label_left, label_right = edges(141)
        x = label_left + (i + .5) / 4 * (label_right-label_left)
        text(x, 134, letter, size=11, color=COLORS[letter], weight="bold", ha="center")
        text(x, 147, label, size=5.4, color=COLORS[letter], ha="center")

    # Level 1: A1 foundation, with A2 and A3 as the two variation blocks above.
    upper, split, lower = 171, 183, 203
    ul, ur = edges(upper)
    sl, sr = edges(split)
    ll, lr = edges(lower)
    poly([(ul, upper), (centre, upper), (centre, split), (sl, split)], "#F3F3F5")
    poly([(centre, upper), (ur, upper), (sr, split), (centre, split)], "#F3F3F5")
    poly([(sl, split), (sr, split), (lr, lower), (ll, lower)], FILLS["A"])
    text((ul+centre)/2, 177, "A2  Loudness", size=6.0, color=COLORS["A"], ha="center")
    text((ur+centre)/2, 177, "A3  Duration", size=6.0, color=COLORS["A"], ha="center")
    text(centre, 193, "A1  Single pitch", size=6.0, color=COLORS["A"], ha="center")

    # Move the level labels to the side; no title-only space above/below tiers.
    for level, title, y0, y1, color in (
        (3, "Melodic", 7, 40, COLORS["F"]),
        (2, "Contextual", 47, 164, INK),
        (1, "Atomic", 171, 203, COLORS["A"]),
    ):
        middle = (y0+y1)/2
        ax.plot([38, 35, 35, 38], [y0, y0, y1, y1], color="#B5C1CA", linewidth=.6)
        text(31, middle-4.5, f"LEVEL {level}", size=5.3, color=color, weight="bold", ha="right")
        text(31, middle+4.5, title, size=5.3, color=color, ha="right")

    # Original side descriptions, aligned and kept clear of the pyramid slopes.
    positions = {"F": (43, 10), "B": (43, 52), "C": (43, 111), "A": (43, 173),
                 "D": (339, 30), "E": (339, 105)}
    legend_texts = []
    for letter, title, level, labels in TASKS:
        x, y = positions[letter]
        legend_texts.append(text(x, y, letter, size=8, color=COLORS[letter], weight="bold"))
        legend_texts.append(text(x+12, y, title, size=6.2, weight="bold"))
        rule_width = 78 if letter == "A" else (67 if letter in "DE" else 96)
        ax.plot([x, x+rule_width], [y+6, y+6], color=COLORS[letter], linewidth=.55, alpha=.55)
        for i, label in enumerate(labels):
            legend_texts.append(text(x, y+13+i*7.2, f"{letter}{i+1}", size=5.6, color=COLORS[letter], weight="bold"))
            legend_texts.append(text(x+13, y+13+i*7.2, label, size=5.6))

    fig.canvas.draw()
    # Bounding rectangles of neighbouring wedges overlap. Check the actual
    # polygons, with a physical safety margin, for every exterior legend label.
    renderer = fig.canvas.get_renderer()
    clearance_pt = 2.5
    for label in legend_texts:
        bounds = label.get_window_extent(renderer).transformed(ax.transData.inverted())
        x0, x1 = sorted((bounds.x0, bounds.x1))
        y0, y1 = sorted((bounds.y0, bounds.y1))
        padded = Bbox.from_extents(x0-clearance_pt, y0-clearance_pt,
                                   x1+clearance_pt, y1+clearance_pt)
        for patch in ax.patches:
            if patch.get_path().intersects_bbox(padded, filled=True):
                raise ValueError(f"Legend too close to pyramid: {label.get_text()}")
    geometry = {"size_inches": [figure_width, 2.9], "alignment": "not audited",
                "manuscript_width_inches": 5.5,
                "manuscript_scale": 5.5 / figure_width,
                "pyramid_width_pt": 2*(lower-apex_y)*slope,
                "legend_polygon_clearance_pt": clearance_pt,
                "legend_text_boxes_checked": len(legend_texts)}
    if qa_scripts is not None:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        report = require_matplotlib_panel_alignment(
            fig, axes=[ax], panel_ids=["pyramid"],
            json_out=out / f"{NAME}.alignment.json", tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True)
        geometry["alignment"] = report.get("verdict", report.get("status", "see audit"))
    fig.savefig(out / f"{NAME}.pdf", dpi=600)
    fig.savefig(out / f"{NAME}.svg", dpi=600)
    fig.savefig(out / f"{NAME}.png", dpi=600)
    plt.close(fig)
    manifest = {"generator": source(Path(__file__)), "definitions": source(REPO / "EXPERIMENTS.md"),
                "geometry": geometry, "n_tasks": 28, "category_counts": dict(zip("ABCDEF", [3,5,4,8,6,2])),
                "structure": "A base with A1/A2/A3; B/C/D/E middle wedges; F apex.",
                "meaning": "Conceptual taxonomy; areas do not encode counts or measured difficulty.",
                "exclusions": "None. All 28 tasks retained in side legends; A1–A3 repeat within the pyramid, and the apex shows only F."}
    (out / f"{NAME}.sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {NAME} to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path)
    args = parser.parse_args()
    render(args.output_dir, args.qa_scripts)


if __name__ == "__main__":
    main()
