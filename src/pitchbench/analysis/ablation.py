"""
Ablation summary CSV and line plots for the analysis results.

Usage::

    pitchbench ablation --kind csv      # write ablation_summary.csv to paper/
    pitchbench ablation --kind ablation # ablation line plots (rows=models, cols=conditions)
    pitchbench ablation --kind format   # format/mode line plots (baseline, d7a, d7b)
    pitchbench ablation --kind all      # all of the above (default)

Models are discovered automatically from results/analysis/<model_label>/ directories.
The most recent timestamp run for each model is used.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from pitchbench.configs.plot_config import MODEL_COLORS, MODEL_SHORT_NAMES

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis"
OUT_DIR      = REPO_ROOT / "paper"
COMBINED_CSV = REPO_ROOT / "paper" / "analysis" / "accuracies_combined.csv"

_TS_RE = re.compile(r"^\d{8}_\d{6}$")

# ── experiment / condition definitions ────────────────────────────────────────

_EFFECT_SHORT = {
    "highpass_above_f0": "highpass",
    "lowpass_at_f0":     "lowpass",
    "bitcrush_4bit":     "bitcrush",
    "distortion_heavy":  "distort",
    "reverb_long":       "reverb",
    "chorus_heavy":      "chorus",
}
_BG_SHORT = {
    "white_noise":                      "white noise",
    "church-bells":                     "bells",
    "crowd-noise":                      "crowd",
    "rain":                             "rain",
    "street-noise":                     "street",
    "oscillating-high-and-low-pitches": "osc. pitch",
}

# Canonical IV name → raw column name in results_*.csv
_CANONICAL_TO_RAW: dict[str, str] = {
    "loudness":   "loudness_db",
    "duration":   "duration_ms",
    "position":   "pos_ms",
    "saturation": "saturation_level",
}

# Each entry: (exp_name, cond_col, label_fn, group_title)
# cond_col: the canonical IV name as it appears after "by_" in accuracies_combined.csv.
#           None means a single-condition experiment (use overall accuracy.any).
ABLATIONS: list[tuple] = [
    ("pitchbench_a1_single_pitch_id", None,
     lambda v: "clean signal (open-ended)", "Baseline"),
    ("pitchbench_y1_single_pitch_id_mcq", "semitone_step",
     lambda v: f"\u00b1{v} semitones", "MCQ"),
    ("pitchbench_a2_single_pitch_by_loudness", "loudness",
     lambda v: f"{'+' if float(v) > 0 else ''}{v} dB", "Loudness"),
    ("pitchbench_a3_single_pitch_by_duration", "duration",
     lambda v: f"{v} ms", "Duration"),
    ("pitchbench_b1_single_pitch_within_silence", "position",
     lambda v: f"{int(float(v)) // 1000} s", "Position"),
    ("pitchbench_b2_pitch_at_timestamp", "n_notes",
     lambda v: f"{v} notes", "N-notes"),
    ("pitchbench_e1_audio_effects", "effect",
     lambda v: _EFFECT_SHORT.get(v, v), "Effects"),
    ("pitchbench_e2_background", "background",
     lambda v: _BG_SHORT.get(v, v), "Background"),
    ("pitchbench_e3_harmonic_saturation", "saturation",
     lambda v: v.replace("_", " "), "Saturation"),
    ("pitchbench_e4_time_stretching", "condition",
     lambda v: v.replace("_", " "), "Time stretch"),
    ("pitchbench_e6_slightly_off", None,
     lambda v: "overall", "Detune"),
]

EXPERIMENTS: list[tuple] = [
    ("pitchbench_a1_single_pitch_id", None,
     lambda v: "baseline", "Baseline"),
    ("pitchbench_d7a_pitch_with_reference", "ref_midi",
     lambda v: f"ref MIDI {v}", "Ref pitch\n(concat)"),
    ("pitchbench_d7b_pitch_with_reference_split", "ref_midi",
     lambda v: f"ref MIDI {v}", "Ref pitch\n(split)"),
]

# ── dynamic model discovery ───────────────────────────────────────────────────

def _discover_runs(analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Path]:
    """Return {model_label: latest_run_dir} for every model found in analysis_dir."""
    result: dict[str, Path] = {}
    if not analysis_dir.exists():
        return result
    for model_dir in sorted(analysis_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        run_dirs = sorted(
            d for d in model_dir.iterdir()
            if d.is_dir() and _TS_RE.match(d.name)
        )
        if run_dirs:
            result[model_dir.name] = run_dirs[-1]
    return result


def _order_models(labels: Iterable[str]) -> list[str]:
    """Order labels by plot_config.MODEL_SHORT_NAMES insertion order; unknowns sorted after."""
    known_order = list(MODEL_SHORT_NAMES.keys())
    label_set   = set(labels)
    known = [m for m in known_order if m in label_set]
    extra = sorted(label_set - set(known_order))
    return known + extra


_FALLBACK_COLORS = ["#7F8C8D", "#95A5A6", "#BDC3C7", "#ABB2B9"]


def _model_color(label: str, fallback_idx: int = 0) -> str:
    return MODEL_COLORS.get(label, _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)])


def _model_display(label: str) -> str:
    return MODEL_SHORT_NAMES.get(label, label)


# ── accuracy CSV loading ──────────────────────────────────────────────────────

def _load_accuracies(run_dir: Path, exp_name: str) -> dict[str, float]:
    """Load {metric: value} from accuracies_*.csv for one experiment in a run."""
    exp_dir = run_dir / exp_name
    if not exp_dir.exists():
        return {}
    csvs = list(exp_dir.glob("accuracies_*.csv"))
    if not csvs:
        return {}
    result: dict[str, float] = {}
    with csvs[0].open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row.get("value", "").strip()
            if not val:
                continue
            try:
                result[row["metric"]] = float(val)
            except (ValueError, KeyError):
                pass
    return result


def _pct(val: float | None) -> str:
    return "" if val is None else f"{val * 100:.1f}%"


def _cond_value(metric: str) -> str:
    """Extract condition value from 'by_X.VALUE.any' → 'VALUE'."""
    parts = metric.split(".")
    return ".".join(parts[1:-1])


def _sort_key(val: str) -> tuple:
    try:
        return (0, float(val))
    except ValueError:
        return (1, val)


# ── ablation summary CSV ──────────────────────────────────────────────────────

def build_ablation_csv(
    combined_csv: Path = COMBINED_CSV,
) -> tuple[list[str], list[list[str]]]:
    """Build ablation summary as (header, rows).

    Reads from paper/analysis/accuracies_combined.csv.
    Rows: one per (experiment, condition_value).
    Columns: group, experiment, condition, <model_display_name>, ...
    """
    # Load combined CSV into {model: {exp_name: {metric: value}}}
    all_accs: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with combined_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row.get("value", "").strip()
            if not val:
                continue
            try:
                all_accs[row["model"]][row["experiment"]][row["metric"]] = float(val)
            except (ValueError, KeyError):
                pass

    model_labels = _order_models(all_accs.keys())
    header = ["group", "experiment", "condition",
              *(_model_display(m) for m in model_labels)]
    rows: list[list[str]] = []

    for exp_name, cond_col, label_fn, group_title in ABLATIONS:
        model_accs: dict[str, dict[str, float]] = {
            ml: all_accs.get(ml, {}).get(exp_name, {})
            for ml in model_labels
        }

        if cond_col is None:
            row = [group_title, exp_name, label_fn(None)]
            for ml in model_labels:
                row.append(_pct(model_accs.get(ml, {}).get("accuracy.any")))
            rows.append(row)
        else:
            # Discover per-condition breakdowns restricted to the expected
            # canonical IV column (e.g. by_loudness.* not by_source.*).
            expected_prefix = f"by_{cond_col}."
            all_cond_metrics: set[str] = set()
            for accs in model_accs.values():
                for m in accs:
                    if m.startswith(expected_prefix) and m.endswith(".any"):
                        all_cond_metrics.add(m)

            if all_cond_metrics:
                cond_values = sorted(
                    {_cond_value(m) for m in all_cond_metrics},
                    key=_sort_key,
                )
                for cond_val in cond_values:
                    metric = next(
                        (m for m in all_cond_metrics if _cond_value(m) == cond_val),
                        None,
                    )
                    if metric is None:
                        continue
                    row = [group_title, exp_name, label_fn(cond_val)]
                    for ml in model_labels:
                        row.append(_pct(model_accs.get(ml, {}).get(metric)))
                    rows.append(row)

            # Overall row: always present (either as sole row or as category average).
            overall_row = [group_title, exp_name, "overall"]
            for ml in model_labels:
                overall_row.append(_pct(model_accs.get(ml, {}).get("accuracy.any")))
            rows.append(overall_row)

            # MCQ summary rows: Mean across conditions + Gain vs A1 baseline.
            if exp_name == "pitchbench_y1_single_pitch_id_mcq" and all_cond_metrics:
                mean_vals: dict[str, float | None] = {}
                for ml in model_labels:
                    accs = model_accs[ml]
                    vals = [accs[m] for m in all_cond_metrics if m in accs]
                    mean_vals[ml] = sum(vals) / len(vals) if vals else None

                mean_row = [group_title, exp_name, "Mean"]
                for ml in model_labels:
                    mean_row.append(_pct(mean_vals[ml]))
                rows.append(mean_row)

                gain_row = [group_title, exp_name, "Gain (MCQ \u2212 baseline)"]
                for ml in model_labels:
                    baseline = all_accs.get(ml, {}).get(
                        "pitchbench_a1_single_pitch_id", {}
                    ).get("accuracy.any")
                    mean_v = mean_vals[ml]
                    if baseline is not None and mean_v is not None:
                        gain = mean_v - baseline
                        sign = "+" if gain >= 0 else ""
                        gain_row.append(f"{sign}{gain * 100:.1f}%")
                    else:
                        gain_row.append("")
                rows.append(gain_row)

                gain_mean_row = [group_title, exp_name, "Gain (mean MCQ \u2212 baseline)"]
                for ml in model_labels:
                    baseline = all_accs.get(ml, {}).get(
                        "pitchbench_a1_single_pitch_id", {}
                    ).get("accuracy.any")
                    accs = model_accs[ml]
                    vals = [accs[m] for m in all_cond_metrics if m in accs]
                    if baseline is not None and vals:
                        mean_with_baseline = (sum(vals) + baseline) / (len(vals) + 1)
                        gain = mean_with_baseline - baseline
                        sign = "+" if gain >= 0 else ""
                        gain_mean_row.append(f"{sign}{gain * 100:.1f}%")
                    else:
                        gain_mean_row.append("")
                rows.append(gain_mean_row)

    return header, rows


def write_ablation_csv(out_dir: Path = OUT_DIR, combined_csv: Path = COMBINED_CSV) -> Path:
    """Write ablation_summary.csv to out_dir and return the path."""
    header, rows = build_ablation_csv(combined_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{ts}_ablation_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  → {path}")
    return path


# ── results CSV loading (for line plots) ─────────────────────────────────────

def _find_results_csv(model_label: str, exp_name: str,
                      analysis_dir: Path = ANALYSIS_DIR) -> Path | None:
    """Find results_*.csv for model_label/exp_name in the latest analysis run."""
    model_dir = analysis_dir / model_label
    if not model_dir.exists():
        return None
    run_dirs = sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and _TS_RE.match(d.name)
    )
    if not run_dirs:
        return None
    exp_dir = run_dirs[-1] / exp_name
    if not exp_dir.exists():
        return None
    csvs = list(exp_dir.glob("results_*.csv"))
    return csvs[0] if csvs else None


# ── SPN helpers ───────────────────────────────────────────────────────────────

_NOTE_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SPN_RE   = re.compile(r"^([A-Ga-g])([#b♯♭]?)(-?\d+)$")


def spn_to_midi(spn: str) -> int | None:
    if not spn:
        return None
    m = _SPN_RE.match(spn.strip())
    if not m:
        return None
    note, acc, octave = m.groups()
    semitone = _NOTE_MAP.get(note.upper())
    if semitone is None:
        return None
    if acc in ("#", "♯"):
        semitone += 1
    elif acc in ("b", "♭"):
        semitone -= 1
    midi = (int(octave) + 1) * 12 + semitone
    return midi if 0 <= midi <= 127 else None


# ── shared plot helpers ───────────────────────────────────────────────────────

def _make_shades(base_hex: str, n: int) -> list[tuple[float, float, float]]:
    if n == 1:
        return [mcolors.to_rgb(base_hex)]
    r, g, b = mcolors.to_rgb(base_hex)
    light = (r + (1 - r) * 0.65, g + (1 - g) * 0.65, b + (1 - b) * 0.65)
    dark  = (r * 0.30, g * 0.30, b * 0.30)
    return [
        tuple(light[c] * (1 - i / (n - 1)) + dark[c] * (i / (n - 1)) for c in range(3))  # type: ignore[misc]
        for i in range(n)
    ]


def _l1(xs: list[int], ys: list[float]) -> float:
    return float(np.mean([abs(y - x) for x, y in zip(xs, ys)]))


def _draw_placeholder(ax: plt.Axes) -> None:
    ax.set_facecolor("#E0E0E0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#9E9E9E")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#757575", fontstyle="italic")


# ── ablation data loading (for line plots) ────────────────────────────────────

def _add_detune_level(rows: list[dict]) -> None:
    by_pitch: dict[str, set[float]] = defaultdict(set)
    for r in rows:
        by_pitch[r["midi_gt"]].add(float(r["detune_hz"]))
    level_of: dict[tuple, str] = {}
    for pitch, detunes in by_pitch.items():
        for rank, d in enumerate(sorted(detunes), start=1):
            level_of[(pitch, d)] = str(rank)
    for r in rows:
        r["_detune_level"] = level_of[(r["midi_gt"], float(r["detune_hz"]))]


def _load_ablation_data(
    model_label: str,
    exp_name: str,
    cond_col: str | None,
    analysis_dir: Path = ANALYSIS_DIR,
) -> dict[str, dict[int, list[int]]]:
    path = _find_results_csv(model_label, exp_name, analysis_dir)
    if path is None:
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {}
    if "e6" in exp_name:
        _add_detune_level(rows)
    result: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        raw_col = _CANONICAL_TO_RAW.get(cond_col, cond_col) if cond_col else None
        cond = r[raw_col] if raw_col else "baseline"
        try:
            x = int(r["midi_gt"])
        except (KeyError, ValueError):
            continue
        try:
            result[cond][x].append(int(float(r["midi_pred"])))
        except (KeyError, ValueError, TypeError):
            pass
        mp = spn_to_midi(r.get("spn_pred", ""))
        if mp is not None:
            result[cond][x].append(mp)
    return {k: dict(v) for k, v in result.items()}


def _load_format_data(
    model_label: str,
    exp_name: str,
    cond_col: str | None,
    analysis_dir: Path = ANALYSIS_DIR,
) -> dict[str, dict[int, list[int]]] | None:
    path = _find_results_csv(model_label, exp_name, analysis_dir)
    if path is None:
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    result: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        raw_col = _CANONICAL_TO_RAW.get(cond_col, cond_col) if cond_col else None
        cond = r[raw_col] if raw_col else "baseline"
        try:
            x = int(r["midi_gt"])
        except (KeyError, ValueError):
            continue
        try:
            result[cond][x].append(int(float(r["midi_pred"])))
        except (KeyError, ValueError, TypeError):
            pass
        mp = spn_to_midi(r.get("spn_pred", ""))
        if mp is not None:
            result[cond][x].append(mp)
    return {k: dict(v) for k, v in result.items()} if result else None


# ── plot: ablation lines ──────────────────────────────────────────────────────

def plot_ablation_lines(
    analysis_dir: Path = ANALYSIS_DIR,
    out_dir: Path = OUT_DIR,
) -> tuple[Path, Path]:
    """Ablation study plot: rows=models (discovered), cols=ablation conditions."""
    runs         = _discover_runs(analysis_dir)
    model_order  = _order_models(runs.keys())
    if not model_order:
        print("  [SKIP] ablation lines: no models found in analysis dir")
        raise SystemExit(1)

    n_rows = len(model_order)
    n_cols = len(ABLATIONS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.6 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")

    for row_i, model_label in enumerate(model_order):
        base_color = _model_color(model_label, row_i)
        for col_i, (exp_name, cond_col, label_fn, title) in enumerate(ABLATIONS):
            ax = axes[row_i][col_i]
            data = _load_ablation_data(model_label, exp_name, cond_col, analysis_dir)
            if not data:
                ax.set_visible(False)
                continue
            all_xs = sorted({x for cd in data.values() for x in cd})
            if len(all_xs) < 2:
                ax.set_visible(False)
                continue
            xmin, xmax = all_xs[0], all_xs[-1]
            ax.plot([xmin, xmax], [xmin, xmax],
                    color="#BDBDBD", lw=1.0, ls="--", zorder=0, label="_nolegend_")
            line_data: dict[str, tuple[list[int], list[float], float]] = {}
            for cond, xdict in data.items():
                xs = [x for x in all_xs if x in xdict and xdict[x]]
                if len(xs) < 2:
                    continue
                ys = [float(np.mean(xdict[x])) for x in xs]
                line_data[cond] = (xs, ys, _l1(xs, ys))
            if not line_data:
                ax.set_visible(False)
                continue
            sorted_conds = sorted(line_data, key=lambda c: line_data[c][2])
            shades = _make_shades(base_color, len(sorted_conds))
            for shade_i, cond in enumerate(sorted_conds):
                xs, ys, l1 = line_data[cond]
                ax.plot(xs, ys, color=shades[shade_i], lw=1.5, marker="o", ms=3,
                        zorder=shade_i + 1, label=f"{label_fn(cond)}  L1={l1:.1f}")
            pad = max(3, (xmax - xmin) * 0.06)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(0, 127)
            ax.tick_params(labelsize=6)
            ax.set_xticks(all_xs[::max(1, len(all_xs) // 4)])
            ax.tick_params(axis="x", labelrotation=45)
            if row_i == 0:
                ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
            if col_i == 0:
                ax.set_ylabel(_model_display(model_label), fontsize=8,
                              fontweight="bold", labelpad=4)
            if row_i == n_rows - 1:
                ax.set_xlabel("GT MIDI", fontsize=6)
            ax.legend(fontsize=5, loc="upper left", framealpha=0.75,
                      handlelength=1.2, borderpad=0.4, labelspacing=0.3)

    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.4)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{ts}_ablation_lines.pdf"
    png_path = out_dir / f"{ts}_ablation_lines.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")
    plt.close(fig)
    return pdf_path, png_path


# ── plot: format/mode lines ───────────────────────────────────────────────────

def plot_format_lines(
    analysis_dir: Path = ANALYSIS_DIR,
    out_dir: Path = OUT_DIR,
) -> tuple[Path, Path]:
    """Format/mode plot: rows=models (discovered), cols=format variants."""
    runs        = _discover_runs(analysis_dir)
    model_order = _order_models(runs.keys())
    if not model_order:
        print("  [SKIP] format lines: no models found in analysis dir")
        raise SystemExit(1)

    n_rows = len(model_order)
    n_cols = len(EXPERIMENTS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 3.0 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")

    for row_i, model_label in enumerate(model_order):
        base_color = _model_color(model_label, row_i)
        for col_i, (exp_name, cond_col, label_fn, title) in enumerate(EXPERIMENTS):
            ax = axes[row_i][col_i]
            data = _load_format_data(model_label, exp_name, cond_col, analysis_dir)
            if data is None:
                _draw_placeholder(ax)
                if row_i == 0:
                    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
                if col_i == 0:
                    ax.set_ylabel(_model_display(model_label), fontsize=9,
                                  fontweight="bold", labelpad=4)
                continue
            all_xs = sorted({x for cd in data.values() for x in cd})
            if len(all_xs) < 2:
                _draw_placeholder(ax)
                continue
            xmin, xmax = all_xs[0], all_xs[-1]
            ax.plot([xmin, xmax], [xmin, xmax],
                    color="#BDBDBD", lw=1.2, ls="--", zorder=0, label="_nolegend_")
            line_data: dict[str, tuple[list[int], list[float], float]] = {}
            for cond, xdict in data.items():
                xs = [x for x in all_xs if x in xdict and xdict[x]]
                if not xs:
                    continue
                ys = [float(np.mean(xdict[x])) for x in xs]
                line_data[cond] = (xs, ys, _l1(xs, ys))
            if not line_data:
                _draw_placeholder(ax)
                continue
            sorted_conds = sorted(line_data, key=lambda c: line_data[c][2])
            shades = _make_shades(base_color, len(sorted_conds))
            for shade_i, cond in enumerate(sorted_conds):
                xs, ys, l1 = line_data[cond]
                ax.plot(xs, ys, color=shades[shade_i], lw=1.5, marker="o", ms=3.5,
                        zorder=shade_i + 1, label=f"{label_fn(cond)}  L1={l1:.1f}")
            pad = max(3, (xmax - xmin) * 0.06)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(0, 127)
            ax.tick_params(labelsize=6)
            ax.set_xticks(all_xs[::max(1, len(all_xs) // 5)])
            ax.tick_params(axis="x", labelrotation=45)
            if row_i == 0:
                ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
            if col_i == 0:
                ax.set_ylabel(_model_display(model_label), fontsize=9,
                              fontweight="bold", labelpad=4)
            if row_i == n_rows - 1:
                ax.set_xlabel("GT MIDI", fontsize=7)
            n_legend_cols = 2 if len(sorted_conds) > 6 else 1
            ax.legend(fontsize=5, loc="upper left", framealpha=0.75,
                      handlelength=1.2, borderpad=0.4, labelspacing=0.25,
                      ncol=n_legend_cols)

    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.5)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{ts}_format_lines.pdf"
    png_path = out_dir / f"{ts}_format_lines.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")
    plt.close(fig)
    return pdf_path, png_path


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--kind",
        choices=["csv", "ablation", "format", "all"],
        default="all",
        help="Which output to produce (default: all)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ANALYSIS_DIR,
        metavar="DIR",
        help=f"Root of analysis results (default: {ANALYSIS_DIR})",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=COMBINED_CSV,
        metavar="CSV",
        help=f"Combined accuracies CSV for the ablation table (default: {COMBINED_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        metavar="DIR",
        help=f"Output directory for plots and CSV (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    if args.kind in ("csv", "all"):
        write_ablation_csv(out_dir=args.out_dir, combined_csv=args.combined_csv)
    if args.kind in ("ablation", "all"):
        plot_ablation_lines(analysis_dir=args.analysis_dir, out_dir=args.out_dir)
    if args.kind in ("format", "all"):
        plot_format_lines(analysis_dir=args.analysis_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
