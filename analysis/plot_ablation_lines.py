"""
Ablation-study line plots: mean predicted MIDI vs ground-truth MIDI.

Layout: rows = models, columns = ablation studies.
Within each subplot:
  - grey dashed x=y diagonal (perfect prediction)
  - one line per condition variable, colored in shades of the model colour
  - lighter shade → lower L1, darker shade → higher L1

Source CSVs: results/analysis/_<model>/<exp>/run_*/results_*.csv
Output:      paper/<YYYYMMDD_HHMMSS>_ablation_lines.{pdf,png}

Predictions pooled from MIDI-format answer (midi_pred) and SPN-format answer
(spn_pred parsed to MIDI), averaging across all sources.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
# One saturated base colour per model; shades derived programmatically
MODEL_COLOR = {
    "openrouter_google_gemini_3_1_pro_preview": "#1565C0",  # blue
    "openrouter_google_gemini_flash_latest":    "#00838F",  # teal
    "audio_flamingo_next_instruct":             "#E65100",  # deep orange
    "openrouter_openai_gpt_4o_audio_preview":   "#2E7D32",  # dark green
}

# ── short labels for verbose condition names ───────────────────────────────────
_EFFECT_SHORT = {
    "highpass_above_f0": "highpass",
    "lowpass_at_f0":     "lowpass",
    "bitcrush_4bit":     "bitcrush",
    "distortion_heavy":  "distort",
    "reverb_long":       "reverb",
    "chorus_heavy":      "chorus",
}
_BG_SHORT = {
    "white_noise":                        "white noise",
    "church-bells":                       "bells",
    "crowd-noise":                        "crowd",
    "rain":                               "rain",
    "street-noise":                       "street",
    "oscillating-high-and-low-pitches":   "osc. pitch",
}

# ── ablation study config ─────────────────────────────────────────────────────
# (exp_name, condition_col_in_csv, label_fn, subplot_title)
# condition_col=None → single "baseline" line
ABLATIONS: list[tuple] = [
    (
        "pitchbench_a1_single_pitch_id", None,
        lambda v: "baseline",
        "Baseline",
    ),
    (
        "pitchbench_a2_single_pitch_by_loudness", "loudness_db",
        lambda v: f"{'+' if float(v) > 0 else ''}{v} dB",
        "Loudness",
    ),
    (
        "pitchbench_a3_single_pitch_by_duration", "duration_ms",
        lambda v: f"{v} ms",
        "Duration",
    ),
    (
        "pitchbench_b1_single_pitch_within_silence", "pos_ms",
        lambda v: f"{int(v)//1000} s",
        "Position",
    ),
    (
        "pitchbench_b2_pitch_at_timestamp", "n_notes",
        lambda v: f"{v} notes",
        "N-notes",
    ),
    (
        "pitchbench_e1_audio_effects", "effect",
        lambda v: _EFFECT_SHORT.get(v, v),
        "Effects",
    ),
    (
        "pitchbench_e2_background", "background",
        lambda v: _BG_SHORT.get(v, v),
        "Background",
    ),
    (
        "pitchbench_e3_harmonic_saturation", "saturation_level",
        lambda v: v.replace("_", " "),
        "Saturation",
    ),
    (
        "pitchbench_e4_time_stretching", "condition",
        lambda v: v.replace("_", " "),
        "Time stretch",
    ),
    (
        "pitchbench_e6_slightly_off", "_detune_level",   # synthetic column added below
        lambda v: f"level {v}",
        "Detune",
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


def _add_detune_level(rows: list[dict]) -> None:
    """Add synthetic '_detune_level' column (1=most negative, 6=most positive)."""
    by_pitch: dict[str, set[float]] = defaultdict(set)
    for r in rows:
        by_pitch[r["midi_gt"]].add(float(r["detune_hz"]))
    level_of: dict[tuple, str] = {}
    for pitch, detunes in by_pitch.items():
        for rank, d in enumerate(sorted(detunes), start=1):
            level_of[(pitch, d)] = str(rank)
    for r in rows:
        r["_detune_level"] = level_of[(r["midi_gt"], float(r["detune_hz"]))]


def load_data(
    model_id: str,
    exp_name: str,
    cond_col: str | None,
) -> dict[str, dict[int, list[int]]]:
    """Return {condition: {midi_gt: [predicted_midi_values]}}."""
    path = find_results_csv(model_id, exp_name)
    if path is None:
        return {}

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {}

    if "e6" in exp_name:
        _add_detune_level(rows)

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

        # SPN-format prediction converted to MIDI
        mp = spn_to_midi(r.get("spn_pred", ""))
        if mp is not None:
            result[cond][x].append(mp)

    return {k: dict(v) for k, v in result.items()}

# ── colour utilities ───────────────────────────────────────────────────────────
def _make_shades(
    base_hex: str, n: int
) -> list[tuple[float, float, float]]:
    """n shades: index 0 = lightest, index n-1 = darkest."""
    if n == 1:
        return [mcolors.to_rgb(base_hex)]
    r, g, b = mcolors.to_rgb(base_hex)
    light = (r + (1 - r) * 0.65, g + (1 - g) * 0.65, b + (1 - b) * 0.65)
    dark  = (r * 0.30, g * 0.30, b * 0.30)
    shades = []
    for i in range(n):
        t = i / (n - 1)  # 0=lightest, 1=darkest
        shades.append(tuple(light[c] * (1 - t) + dark[c] * t for c in range(3)))
    return shades  # type: ignore[return-value]

# ── L1 computation ────────────────────────────────────────────────────────────
def _l1(xs: list[int], ys: list[float]) -> float:
    return float(np.mean([abs(y - x) for x, y in zip(xs, ys)]))

# ── main plot ─────────────────────────────────────────────────────────────────
def plot_all() -> None:
    n_rows = len(MODEL_ORDER)
    n_cols = len(ABLATIONS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_i, model_id in enumerate(MODEL_ORDER):
        base_color = MODEL_COLOR[model_id]

        for col_i, (exp_name, cond_col, label_fn, title) in enumerate(ABLATIONS):
            ax = axes[row_i][col_i]

            data = load_data(model_id, exp_name, cond_col)

            if not data:
                ax.set_visible(False)
                continue

            all_xs = sorted({x for cd in data.values() for x in cd})
            if len(all_xs) < 2:
                ax.set_visible(False)
                continue

            xmin, xmax = all_xs[0], all_xs[-1]

            # reference diagonal
            ax.plot([xmin, xmax], [xmin, xmax],
                    color="#BDBDBD", lw=1.0, ls="--", zorder=0, label="_nolegend_")

            # build per-condition lines and their L1
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

            # sort by L1 ascending; lower L1 → lighter shade
            sorted_conds = sorted(line_data, key=lambda c: line_data[c][2])
            shades = _make_shades(base_color, len(sorted_conds))

            for shade_i, cond in enumerate(sorted_conds):
                xs, ys, l1 = line_data[cond]
                ax.plot(
                    xs, ys,
                    color=shades[shade_i],
                    lw=1.5, marker="o", ms=3, zorder=shade_i + 1,
                    label=f"{label_fn(cond)}  L1={l1:.1f}",
                )

            # axes
            pad = max(3, (xmax - xmin) * 0.06)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(0, 127)
            ax.tick_params(labelsize=6)
            ax.set_xticks(all_xs[::max(1, len(all_xs)//4)])
            ax.tick_params(axis="x", labelrotation=45)

            # column header (top row only)
            if row_i == 0:
                ax.set_title(title, fontsize=8, fontweight="bold", pad=3)

            # row label (left column only)
            if col_i == 0:
                ax.set_ylabel(MODEL_DISPLAY[model_id], fontsize=8,
                              fontweight="bold", labelpad=4)

            # x label (bottom row only)
            if row_i == n_rows - 1:
                ax.set_xlabel("GT MIDI", fontsize=6)

            # legend inside subplot
            ax.legend(
                fontsize=5,
                loc="upper left",
                framealpha=0.75,
                handlelength=1.2,
                borderpad=0.4,
                labelspacing=0.3,
            )

    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.4)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUT_DIR / f"{ts}_ablation_lines.pdf"
    png_path = OUT_DIR / f"{ts}_ablation_lines.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_all()
