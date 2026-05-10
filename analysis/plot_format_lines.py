"""
Format/mode line plots: mean predicted MIDI vs ground-truth MIDI.

Columns: Baseline (a1) | MCQ (y1, 3 step sizes) | Ref pitch concat (d7a) | Ref pitch split (d7b)
Rows:    one per model

Within each subplot:
  - grey dashed x=y diagonal (perfect prediction)
  - one line per condition variable, shaded by L1 (lighter=lower L1)
  - greyed-out placeholder box when data is unavailable

Output: paper/<YYYYMMDD_HHMMSS>_format_lines.{pdf,png}
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis"
OUT_DIR      = REPO_ROOT / "paper"

# ── model config ──────────────────────────────────────────────────────────────
MODEL_DIR = {
    "openrouter_google_gemini_3_1_pro_preview": "_gemini-pro",
    "openrouter_google_gemini_flash_latest":    "_gemini-flash",
    "audio_flamingo_next_instruct":             "_flamingo",
    "openrouter_openai_gpt_4o_audio_preview":   "_gpt",
}
MODEL_DISPLAY = {
    "openrouter_google_gemini_3_1_pro_preview": "Gemini Pro",
    "openrouter_google_gemini_flash_latest":    "Gemini Flash",
    "audio_flamingo_next_instruct":             "Audio Flamingo",
    "openrouter_openai_gpt_4o_audio_preview":   "GPT-4o",
}
MODEL_ORDER = [
    "openrouter_google_gemini_3_1_pro_preview",
    "openrouter_google_gemini_flash_latest",
    "audio_flamingo_next_instruct",
    "openrouter_openai_gpt_4o_audio_preview",
]
MODEL_COLOR = {
    "openrouter_google_gemini_3_1_pro_preview": "#1565C0",
    "openrouter_google_gemini_flash_latest":    "#00838F",
    "audio_flamingo_next_instruct":             "#E65100",
    "openrouter_openai_gpt_4o_audio_preview":   "#2E7D32",
}

# ── experiment definitions ────────────────────────────────────────────────────
# (exp_name, condition_col, label_fn, title)
# condition_col=None → single "baseline" line
EXPERIMENTS: list[tuple] = [
    (
        "pitchbench_a1_single_pitch_id", None,
        lambda v: "baseline",
        "Baseline",
    ),
    (
        "pitchbench_d7a_pitch_with_reference", "ref_midi",
        lambda v: f"ref MIDI {v}",
        "Ref pitch\n(concat)",
    ),
    (
        "pitchbench_d7b_pitch_with_reference_split", "ref_midi",
        lambda v: f"ref MIDI {v}",
        "Ref pitch\n(split)",
    ),
]

# ── SPN → MIDI ────────────────────────────────────────────────────────────────
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

# ── data loading ──────────────────────────────────────────────────────────────
def find_results_csv(model_id: str, exp_name: str) -> Path | None:
    base = ANALYSIS_DIR / MODEL_DIR[model_id] / exp_name
    if not base.exists():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        return None
    csvs = list(runs[-1].glob("results_*.csv"))
    return csvs[0] if csvs else None


def load_data(
    model_id: str,
    exp_name: str,
    cond_col: str | None,
) -> dict[str, dict[int, list[int]]] | None:
    """Return {condition: {midi_gt: [predicted_midi_values]}} or None if missing."""
    path = find_results_csv(model_id, exp_name)
    if path is None:
        return None

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None

    result: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        cond = r[cond_col] if cond_col else "baseline"
        try:
            x = int(r["midi_gt"])
        except (KeyError, ValueError):
            continue

        # MIDI-format prediction
        try:
            result[cond][x].append(int(float(r["midi_pred"])))
        except (KeyError, ValueError, TypeError):
            pass

        # SPN-format prediction → MIDI
        mp = spn_to_midi(r.get("spn_pred", ""))
        if mp is not None:
            result[cond][x].append(mp)

    return {k: dict(v) for k, v in result.items()} if result else None

# ── colour utilities ───────────────────────────────────────────────────────────
def _make_shades(
    base_hex: str, n: int
) -> list[tuple[float, float, float]]:
    """n shades: index 0 = lightest (low L1), index n-1 = darkest (high L1)."""
    if n == 1:
        return [mcolors.to_rgb(base_hex)]
    r, g, b = mcolors.to_rgb(base_hex)
    light = (r + (1 - r) * 0.65, g + (1 - g) * 0.65, b + (1 - b) * 0.65)
    dark  = (r * 0.30, g * 0.30, b * 0.30)
    shades = []
    for i in range(n):
        t = i / (n - 1)
        shades.append(tuple(light[c] * (1 - t) + dark[c] * t for c in range(3)))
    return shades  # type: ignore[return-value]

# ── L1 ────────────────────────────────────────────────────────────────────────
def _l1(xs: list[int], ys: list[float]) -> float:
    return float(np.mean([abs(y - x) for x, y in zip(xs, ys)]))

# ── placeholder ───────────────────────────────────────────────────────────────
def _draw_placeholder(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#E0E0E0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#9E9E9E")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5, 0.5, "No data",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=9, color="#757575",
        fontstyle="italic",
    )

# ── main plot ─────────────────────────────────────────────────────────────────
def plot_all() -> None:
    n_rows = len(MODEL_ORDER)
    n_cols = len(EXPERIMENTS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_i, model_id in enumerate(MODEL_ORDER):
        base_color = MODEL_COLOR[model_id]

        for col_i, (exp_name, cond_col, label_fn, title) in enumerate(EXPERIMENTS):
            ax = axes[row_i][col_i]

            data = load_data(model_id, exp_name, cond_col)

            # ── placeholder ──────────────────────────────────────────────────
            if data is None:
                _draw_placeholder(ax, title)
                if row_i == 0:
                    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
                if col_i == 0:
                    ax.set_ylabel(MODEL_DISPLAY[model_id], fontsize=9,
                                  fontweight="bold", labelpad=4)
                continue

            all_xs = sorted({x for cd in data.values() for x in cd})
            if len(all_xs) < 2:
                _draw_placeholder(ax, title)
                continue

            xmin, xmax = all_xs[0], all_xs[-1]

            # diagonal
            ax.plot([xmin, xmax], [xmin, xmax],
                    color="#BDBDBD", lw=1.2, ls="--", zorder=0,
                    label="_nolegend_")

            # per-condition lines + L1
            line_data: dict[str, tuple[list[int], list[float], float]] = {}
            for cond, xdict in data.items():
                xs = [x for x in all_xs if x in xdict and xdict[x]]
                if not xs:
                    continue
                ys = [float(np.mean(xdict[x])) for x in xs]
                line_data[cond] = (xs, ys, _l1(xs, ys))

            if not line_data:
                _draw_placeholder(ax, title)
                continue

            # sort ascending L1; lower → lighter shade
            sorted_conds = sorted(line_data, key=lambda c: line_data[c][2])
            shades = _make_shades(base_color, len(sorted_conds))

            for shade_i, cond in enumerate(sorted_conds):
                xs, ys, l1 = line_data[cond]
                ax.plot(
                    xs, ys,
                    color=shades[shade_i],
                    lw=1.5, marker="o", ms=3.5, zorder=shade_i + 1,
                    label=f"{label_fn(cond)}  L1={l1:.1f}",
                )

            pad = max(3, (xmax - xmin) * 0.06)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(0, 127)
            ax.tick_params(labelsize=6)
            ax.set_xticks(all_xs[::max(1, len(all_xs) // 5)])
            ax.tick_params(axis="x", labelrotation=45)

            if row_i == 0:
                ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
            if col_i == 0:
                ax.set_ylabel(MODEL_DISPLAY[model_id], fontsize=9,
                              fontweight="bold", labelpad=4)
            if row_i == n_rows - 1:
                ax.set_xlabel("GT MIDI", fontsize=7)

            n_legend_cols = 2 if len(sorted_conds) > 6 else 1
            ax.legend(
                fontsize=5,
                loc="upper left",
                framealpha=0.75,
                handlelength=1.2,
                borderpad=0.4,
                labelspacing=0.25,
                ncol=n_legend_cols,
            )

    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.5)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUT_DIR / f"{ts}_format_lines.pdf"
    png_path = OUT_DIR / f"{ts}_format_lines.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_all()
