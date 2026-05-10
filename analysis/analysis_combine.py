"""
analysis_combine.py — Build paper-ready LaTeX tables from accuracies_combined.csv.

Usage::
    python analysis/analysis_combine.py results/analysis/accuracies_combined.csv
    python analysis/analysis_combine.py results/analysis/accuracies_combined.csv --out paper/tables.tex
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model ordering & display names  (matches LaTeX header layout)
# ---------------------------------------------------------------------------

MODELS: list[tuple[str, str]] = [
    ("audio_flamingo_next_instruct",          "AF-next-instruct"),
    ("openrouter_google_gemini_3_1_pro_preview", "Gemini 3.1 Pro"),
    ("openrouter_google_gemini_flash_latest", "Gemini 3 Flash"),
    ("openrouter_openai_gpt_4o_audio_preview","GPT-4o audio"),
    ("dashscope_qwen3_5_omni_plus",           "Qwen-3.5 omni plus"),
    ("dashscope_qwen3_5_omni_flash",          "Qwen-3.5 omni flash"),
]

MODEL_KEYS = [m[0] for m in MODELS]

# Some experiments were re-run under an alternate slug (e.g. dashscope2_*).
# Map primary slug → list of fallback slugs tried in order when data is absent.
MODEL_SLUG_FALLBACKS: dict[str, list[str]] = {
    "dashscope_qwen3_5_omni_plus": ["dashscope2_qwen3_5_omni_plus"],
}

# ---------------------------------------------------------------------------
# Table row definitions: (group_label, row_label, experiment, metric)
# ---------------------------------------------------------------------------

MAIN_TABLE_ROWS: list[tuple[str, str, str, str]] = [
    # A1 baseline
    ("A1 (baseline)", "clean signal",
     "pitchbench_a1_single_pitch_id", "accuracy.any"),

    # A2 loudness
    ("A2 (loudness)", r"\(-30\) dB",
     "pitchbench_a2_single_pitch_by_loudness", "by_loudness.-30.any"),
    ("",              r"\(-20\) dB",
     "pitchbench_a2_single_pitch_by_loudness", "by_loudness.-20.any"),
    ("",              r"\(-12\) dB",
     "pitchbench_a2_single_pitch_by_loudness", "by_loudness.-12.any"),
    ("",              r"\(+6\) dB",
     "pitchbench_a2_single_pitch_by_loudness", "by_loudness.6.any"),
    ("",              r"\(+12\) dB",
     "pitchbench_a2_single_pitch_by_loudness", "by_loudness.12.any"),

    # A3 duration
    ("A3 (duration)", "50 ms",
     "pitchbench_a3_single_pitch_by_duration", "by_duration.50.any"),
    ("",              "250 ms",
     "pitchbench_a3_single_pitch_by_duration", "by_duration.250.any"),
    ("",              "1 s",
     "pitchbench_a3_single_pitch_by_duration", "by_duration.1000.any"),
    ("",              "4 s",
     "pitchbench_a3_single_pitch_by_duration", "by_duration.4000.any"),
    ("",              "15 s",
     "pitchbench_a3_single_pitch_by_duration", "by_duration.15000.any"),
    ("",              "60 s",
     "pitchbench_a3_single_pitch_by_duration", "by_duration.60000.any"),

    # B1 temporal offset
    ("B1 (temporal offset)", "onset at 10 s",
     "pitchbench_b1_single_pitch_within_silence", "by_position.10000.any"),
    ("",                     "onset at 30 s",
     "pitchbench_b1_single_pitch_within_silence", "by_position.30000.any"),
    ("",                     "onset at 50 s",
     "pitchbench_b1_single_pitch_within_silence", "by_position.50000.any"),

    # E1 audio effects
    ("E1 (audio effects)", "High-pass filtering",
     "pitchbench_e1_audio_effects", "by_effect.highpass_above_f0.any"),
    ("",                   "Low-pass filtering",
     "pitchbench_e1_audio_effects", "by_effect.lowpass_at_f0.any"),
    ("",                   "Distortion",
     "pitchbench_e1_audio_effects", "by_effect.distortion_heavy.any"),
    ("",                   "Reverb",
     "pitchbench_e1_audio_effects", "by_effect.reverb_long.any"),
    ("",                   "Chorus",
     "pitchbench_e1_audio_effects", "by_effect.chorus_heavy.any"),

    # E2 background
    ("E2 (background)", "white noise",
     "pitchbench_e2_background", "by_background.white_noise.any"),
    ("",                "bells",
     "pitchbench_e2_background", "by_background.church-bells.any"),
    ("",                "crowd noise",
     "pitchbench_e2_background", "by_background.crowd-noise.any"),
    ("",                "rain",
     "pitchbench_e2_background", "by_background.rain.any"),
    ("",                "street ambience",
     "pitchbench_e2_background", "by_background.street-noise.any"),
    ("",                "oscillating noise",
     "pitchbench_e2_background", "by_background.oscillating-high-and-low-pitches.any"),

    # E3 saturation
    ("E3 (saturation)", "saturation",
     "pitchbench_e3_harmonic_saturation", "accuracy.any"),

    # E4 time stretch
    ("E4 (time stretch)", "time stretching",
     "pitchbench_e4_time_stretching", "accuracy.any"),

    # E6 detuning
    ("E6 (detuning)", "detuning",
     "pitchbench_e6_slightly_off", "accuracy.any"),
]

MCQ_TABLE_ROWS: list[tuple[str, str, str, str]] = [
    ("A1 (baseline)",    "clean signal (open-ended)",
     "pitchbench_a1_single_pitch_id", "accuracy.any"),
    ("A1 MCQ",           r"\(\pm\)2 semitones",
     "pitchbench_y1_single_pitch_id_mcq", "by_semitone_step.2.any"),
    ("",                 r"\(\pm\)4 semitones",
     "pitchbench_y1_single_pitch_id_mcq", "by_semitone_step.4.any"),
    ("",                 r"\(\pm\)6 semitones",
     "pitchbench_y1_single_pitch_id_mcq", "by_semitone_step.6.any"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: str | Path) -> dict[tuple[str, str, str], float]:
    """Return {(model, experiment, metric): value} from accuracies_combined.csv."""
    data: dict[tuple[str, str, str], float] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val = row.get("value", "").strip()
            if not val:
                continue
            try:
                data[(row["model"], row["experiment"], row["metric"])] = float(val)
            except (ValueError, KeyError):
                pass
    return data


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float], bold: bool = False, underline: bool = False) -> str:
    if val is None:
        return r"--"
    s = f"{val * 100:.1f}"
    if bold and underline:
        s = rf"\textbf{{\underline{{{s}}}}}"
    elif bold:
        s = rf"\textbf{{{s}}}"
    elif underline:
        s = rf"\underline{{{s}}}"
    return s


def _multirow(text: str, n: int) -> str:
    if n <= 1:
        return text
    return rf"\multirow{{{n}}}{{*}}{{{text}}}"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _lookup(
    data: dict[tuple[str, str, str], float],
    model: str,
    exp: str,
    metric: str,
) -> Optional[float]:
    """Return value for (model, exp, metric), trying fallback slugs if needed."""
    key = (model, exp, metric)
    if key in data:
        return data[key]
    for alt in MODEL_SLUG_FALLBACKS.get(model, []):
        alt_key = (alt, exp, metric)
        if alt_key in data:
            return data[alt_key]
    return None


def _build_rows(
    rows: list[tuple[str, str, str, str]],
    data: dict[tuple[str, str, str], float],
    highlight_groups: bool = False,
) -> list[str]:
    """Render data rows, collapsing repeated group labels into \\multirow.

    If *highlight_groups* is True, bold the max and underline the min value
    within each group for every model column (groups with a single row are
    left unformatted).
    """
    # Pre-compute group spans for \multirow and group membership
    group_spans: dict[int, int] = {}
    # row_index -> (group_start, group_end_exclusive)
    row_group: dict[int, tuple[int, int]] = {}
    i = 0
    while i < len(rows):
        g = rows[i][0]
        if g:
            span = 1
            j = i + 1
            while j < len(rows) and rows[j][0] == "":
                span += 1
                j += 1
            group_spans[i] = span
            for k in range(i, i + span):
                row_group[k] = (i, i + span)
        i += 1

    # Pre-compute per-group per-model min/max (only when highlight_groups)
    # group_stats[(group_start, model)] = (min_val, max_val)
    group_stats: dict[tuple[int, str], tuple[float, float]] = {}
    if highlight_groups:
        for g_start, span in group_spans.items():
            if span <= 1:
                continue
            for model in MODEL_KEYS:
                vals = [
                    v for k in range(g_start, g_start + span)
                    if (v := _lookup(data, model, rows[k][2], rows[k][3])) is not None
                ]
                if vals:
                    group_stats[(g_start, model)] = (min(vals), max(vals))

    lines: list[str] = []
    for idx, (group, task, exp, metric) in enumerate(rows):
        g_start, g_end = row_group.get(idx, (idx, idx + 1))
        span = group_spans.get(g_start, 1)
        cells: list[str] = []
        for model in MODEL_KEYS:
            val = _lookup(data, model, exp, metric)
            if highlight_groups and span > 1 and (g_start, model) in group_stats:
                lo, hi = group_stats[(g_start, model)]
                is_max = val is not None and val == hi
                is_min = val is not None and val == lo
                cells.append(_fmt(val, bold=is_max, underline=is_min))
            else:
                cells.append(_fmt(val))
        group_cell = _multirow(group, span) if idx in group_spans else ""
        lines.append(
            f" {group_cell} & {task} & "
            + " & ".join(cells)
            + r" \\"
        )
    return lines


def _mean_row(
    rows: list[tuple[str, str, str, str]],
    data: dict[tuple[str, str, str], float],
) -> str:
    means: list[str] = []
    for model in MODEL_KEYS:
        vals = [
            _lookup(data, model, exp, metric)
            for (_, _, exp, metric) in rows
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            means.append(f"{sum(vals) / len(vals) * 100:.1f}")
        else:
            means.append(r"--")
    return r"\multicolumn{2}{l|}{\textbf{Mean}} & " + " & ".join(means) + r" \\"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

HEADER = r"""\begin{table}[!htbp]
\centering
\caption{Unified PitchBench: robustness across loudness, temporal structure, audio effects, and background conditions.}
\label{tab:pitchbench_unified}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll|c|cc|c|cc}
\toprule
\textbf{Group} & \textbf{Task}
  & \multicolumn{1}{c|}{\textbf{Nvidia}}
  & \multicolumn{2}{c|}{\textbf{Google}}
  & \multicolumn{1}{c|}{\textbf{OpenAI}}
  & \multicolumn{2}{c}{\textbf{Qwen}} \\
 &
  & \textit{AF-next-instruct}
  & \textit{Gemini 3.1 Pro}
  & \textit{Gemini 3 Flash}
  & \textit{GPT-4o audio}
  & \textit{Qwen-3.5 omni plus}
  & \textit{Qwen-3.5 omni flash} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}%
}
\end{table}"""

MCQ_HEADER = r"""\begin{table}[!htbp]
\centering
\caption{PitchBench MCQ: 3-option pitch identification by semitone distractor distance.}
\label{tab:pitchbench_mcq}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll|c|cc|c|cc}
\toprule
\textbf{Group} & \textbf{Task}
  & \multicolumn{1}{c|}{\textbf{Nvidia}}
  & \multicolumn{2}{c|}{\textbf{Google}}
  & \multicolumn{1}{c|}{\textbf{OpenAI}}
  & \multicolumn{2}{c}{\textbf{Qwen}} \\
 &
  & \textit{AF-next-instruct}
  & \textit{Gemini 3.1 Pro}
  & \textit{Gemini 3 Flash}
  & \textit{GPT-4o audio}
  & \textit{Qwen-3.5 omni plus}
  & \textit{Qwen-3.5 omni flash} \\
\midrule"""


def build_robustness_table(csv_path: str | Path) -> str:
    """Return the LaTeX string for the main robustness table."""
    data = load_csv(csv_path)

    # Group the rows by their first group-change point to insert \midrule\midrule
    # after the first A1 group and plain \midrule between subsequent groups.
    rows_tex = _build_rows(MAIN_TABLE_ROWS, data, highlight_groups=True)

    # Insert separators: double-midrule after A1 (index 0), single between others
    group_starts = [i for i, (g, *_) in enumerate(MAIN_TABLE_ROWS) if g]
    # first separator (after A1) is double; rest are single
    lines: list[str] = [HEADER]
    for idx, row_tex in enumerate(rows_tex):
        lines.append(row_tex)
        if idx in group_starts[1:]:  # next group starts after this row
            # find next group start to decide separator before it
            pass
    # Rebuild: emit row, then midrule BEFORE the next group start
    lines = [HEADER]
    for idx, row_tex in enumerate(rows_tex):
        lines.append(row_tex)
        # is the NEXT row a new group?
        next_idx = idx + 1
        if next_idx < len(MAIN_TABLE_ROWS) and MAIN_TABLE_ROWS[next_idx][0]:
            if idx == 0:
                lines.append(r"\midrule\midrule")
            else:
                lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(_mean_row(MAIN_TABLE_ROWS, data))
    lines.append(FOOTER)
    return "\n".join(lines)


def _gain_row(
    baseline_exp: str,
    baseline_metric: str,
    all_rows: list[tuple[str, str, str, str]],
    data: dict[tuple[str, str, str], float],
) -> str:
    """Render a Gain row: Mean (over all_rows) minus the baseline value."""
    cells: list[str] = []
    for model in MODEL_KEYS:
        base = _lookup(data, model, baseline_exp, baseline_metric)
        vals = [
            _lookup(data, model, exp, metric)
            for (_, _, exp, metric) in all_rows
        ]
        vals = [v for v in vals if v is not None]
        if base is not None and vals:
            gain = sum(vals) / len(vals) - base
            sign = "+" if gain >= 0 else "\u2212"
            cells.append(rf"{sign}{abs(gain * 100):.1f}")
        else:
            cells.append(r"--")
    return r"\multicolumn{2}{l|}{\textbf{Gain (MCQ \(-\) baseline)}} & " + " & ".join(cells) + r" \\"


def build_mcq_table(csv_path: str | Path) -> str:
    """Return the LaTeX string for the MCQ semitone-step table."""
    data = load_csv(csv_path)
    rows_tex = _build_rows(MCQ_TABLE_ROWS, data)
    # Separate baseline row from MCQ condition rows
    baseline_row = MCQ_TABLE_ROWS[0]
    lines = [MCQ_HEADER]
    for idx, row_tex in enumerate(rows_tex):
        lines.append(row_tex)
        next_idx = idx + 1
        if next_idx < len(MCQ_TABLE_ROWS) and MCQ_TABLE_ROWS[next_idx][0]:
            if idx == 0:
                lines.append(r"\midrule\midrule")
            else:
                lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(_mean_row(MCQ_TABLE_ROWS, data))
    lines.append(_gain_row(
        baseline_row[2], baseline_row[3],
        MCQ_TABLE_ROWS,
        data,
    ))
    lines.append(FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from accuracies_combined.csv"
    )
    parser.add_argument("csv", help="Path to accuracies_combined.csv")
    parser.add_argument("--out", default=None,
                        help="Write output to this .tex file instead of stdout")
    args = parser.parse_args(argv)

    main_tex = build_robustness_table(args.csv)
    mcq_tex  = build_mcq_table(args.csv)
    output   = main_tex + "\n\n" + mcq_tex

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
