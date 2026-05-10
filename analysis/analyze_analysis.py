"""
Build three summary CSVs matching the paper's LaTeX tables.

  Table 1 – Sound attributes:   loudness × duration × time-placement × n-notes
  Table 2 – Environmental:      effects × background × saturation × stretch × detune
  Table 3 – Format/mode:        open-ended baseline vs. MCQ vs. reference-pitch (d7a/d7b)

Source: results/analysis/accuracies_combined.csv
Output: results/analysis/summary/<YYYYMMDD_HHMMSS>_table{1,2,3}.csv
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "results" / "analysis" / "accuracies_combined.csv"
OUT_DIR   = REPO_ROOT / "results" / "analysis" / "summary"

# ── display names and order ───────────────────────────────────────────────────
MODEL_DISPLAY: dict[str, str] = {
    "audio_flamingo_next_instruct":             "Audio Flamingo",
    "openrouter_google_gemini_3_1_pro_preview": "Gemini Pro",
    "openrouter_google_gemini_flash_latest":    "Gemini Flash",
    "openrouter_openai_gpt_4o_audio_preview":   "GPT-4o",
}

MODEL_ORDER = [
    "openrouter_google_gemini_3_1_pro_preview",
    "openrouter_google_gemini_flash_latest",
    "audio_flamingo_next_instruct",
    "openrouter_openai_gpt_4o_audio_preview",
]

D7_INTERVAL_STEPS = [-12, -7, -5, -4, -3, -1, 0, 1, 3, 4, 5, 7, 12]


# ── data loading ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> dict[tuple[str, str, str], float]:
    """Return {(model, experiment, metric): value} from the combined CSV."""
    data: dict[tuple[str, str, str], float] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val_str = row.get("value", "")
            if not val_str:
                continue
            try:
                data[(row["model"], row["experiment"], row["metric"])] = float(val_str)
            except ValueError:
                pass
    return data


def get_models(data: dict) -> list[str]:
    all_models = {m for m, _, _ in data}
    ordered    = [m for m in MODEL_ORDER if m in all_models]
    extra      = sorted(all_models - set(ordered))
    return ordered + extra


# ── formatting ────────────────────────────────────────────────────────────────

def pct(val: float | None) -> str:
    return "—" if val is None else f"{val * 100:.1f}%"


def get(
    data: dict[tuple[str, str, str], float],
    model: str,
    exp: str,
    metric: str,
) -> float | None:
    return data.get((model, exp, metric))


# ── table builders ────────────────────────────────────────────────────────────

def build_table1(
    data: dict[tuple[str, str, str], float],
    models: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Table 1 — Sound attributes: loudness, duration, time placement, n-notes."""
    header = [
        "Model", "Baseline (a1)",
        # a2 – loudness
        "Loudness -30 dB", "Loudness -20 dB", "Loudness -12 dB",
        "Loudness +6 dB",  "Loudness +12 dB",
        # a3 – duration
        "Duration 50 ms",   "Duration 250 ms",   "Duration 1000 ms",
        "Duration 4000 ms", "Duration 15000 ms", "Duration 60000 ms",
        # b1 – time placement within silence
        "Position 10000 ms", "Position 30000 ms", "Position 50000 ms",
        # b2 – n-notes at timestamp
        "N-notes 5", "N-notes 10",
    ]
    rows: list[list[str]] = []
    for m in models:
        rows.append([
            MODEL_DISPLAY.get(m, m),
            pct(get(data, m, "pitchbench_a1_single_pitch_id", "accuracy.any")),
            # loudness
            pct(get(data, m, "pitchbench_a2_single_pitch_by_loudness", "by_loudness.-30.any")),
            pct(get(data, m, "pitchbench_a2_single_pitch_by_loudness", "by_loudness.-20.any")),
            pct(get(data, m, "pitchbench_a2_single_pitch_by_loudness", "by_loudness.-12.any")),
            pct(get(data, m, "pitchbench_a2_single_pitch_by_loudness", "by_loudness.6.any")),
            pct(get(data, m, "pitchbench_a2_single_pitch_by_loudness", "by_loudness.12.any")),
            # duration
            pct(get(data, m, "pitchbench_a3_single_pitch_by_duration", "by_duration.50.any")),
            pct(get(data, m, "pitchbench_a3_single_pitch_by_duration", "by_duration.250.any")),
            pct(get(data, m, "pitchbench_a3_single_pitch_by_duration", "by_duration.1000.any")),
            pct(get(data, m, "pitchbench_a3_single_pitch_by_duration", "by_duration.4000.any")),
            pct(get(data, m, "pitchbench_a3_single_pitch_by_duration", "by_duration.15000.any")),
            pct(get(data, m, "pitchbench_a3_single_pitch_by_duration", "by_duration.60000.any")),
            # time placement
            pct(get(data, m, "pitchbench_b1_single_pitch_within_silence", "by_position.10000.any")),
            pct(get(data, m, "pitchbench_b1_single_pitch_within_silence", "by_position.30000.any")),
            pct(get(data, m, "pitchbench_b1_single_pitch_within_silence", "by_position.50000.any")),
            # n-notes at timestamp
            pct(get(data, m, "pitchbench_b2_pitch_at_timestamp", "by_n_notes.5.any")),
            pct(get(data, m, "pitchbench_b2_pitch_at_timestamp", "by_n_notes.10.any")),
        ])
    return header, rows


def build_table2(
    data: dict[tuple[str, str, str], float],
    models: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Table 2 — Environmental complexity: effects, background, saturation, stretch, detune."""
    header = [
        "Model", "Baseline (a1)",
        # e1 – audio effects
        "Effect: highpass", "Effect: lowpass",  "Effect: bitcrush",
        "Effect: distortion", "Effect: reverb", "Effect: chorus",
        # e2 – background noise
        "BG: white noise", "BG: bells",  "BG: crowd",
        "BG: rain",        "BG: street", "BG: osc. pitch",
        # e3, e4, e6 – single overall accuracy
        "Saturation (overall)",
        "Time stretch (overall)",
        "Detune (overall)",
    ]
    rows: list[list[str]] = []
    for m in models:
        rows.append([
            MODEL_DISPLAY.get(m, m),
            pct(get(data, m, "pitchbench_a1_single_pitch_id", "accuracy.any")),
            # effects
            pct(get(data, m, "pitchbench_e1_audio_effects", "by_effect.highpass_above_f0.any")),
            pct(get(data, m, "pitchbench_e1_audio_effects", "by_effect.lowpass_at_f0.any")),
            pct(get(data, m, "pitchbench_e1_audio_effects", "by_effect.bitcrush_4bit.any")),
            pct(get(data, m, "pitchbench_e1_audio_effects", "by_effect.distortion_heavy.any")),
            pct(get(data, m, "pitchbench_e1_audio_effects", "by_effect.reverb_long.any")),
            pct(get(data, m, "pitchbench_e1_audio_effects", "by_effect.chorus_heavy.any")),
            # backgrounds
            pct(get(data, m, "pitchbench_e2_background", "by_background.white_noise.any")),
            pct(get(data, m, "pitchbench_e2_background", "by_background.church-bells.any")),
            pct(get(data, m, "pitchbench_e2_background", "by_background.crowd-noise.any")),
            pct(get(data, m, "pitchbench_e2_background", "by_background.rain.any")),
            pct(get(data, m, "pitchbench_e2_background", "by_background.street-noise.any")),
            pct(get(data, m, "pitchbench_e2_background", "by_background.oscillating-high-and-low-pitches.any")),
            # single-column summaries
            pct(get(data, m, "pitchbench_e3_harmonic_saturation", "accuracy.any")),
            pct(get(data, m, "pitchbench_e4_time_stretching",     "accuracy.any")),
            pct(get(data, m, "pitchbench_e6_slightly_off",        "accuracy.any")),
        ])
    return header, rows


def build_table3(
    data: dict[tuple[str, str, str], float],
    models: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Table 3 — Open-ended baseline vs. MCQ vs. reference-pitch (d7a / d7b)."""
    d7a_cols = [f"d7a {step:+d} st" for step in D7_INTERVAL_STEPS]
    d7b_cols = [f"d7b {step:+d} st" for step in D7_INTERVAL_STEPS]

    header = [
        "Model",
        "Baseline (a1)",
        "MCQ (y1)",
        *d7a_cols,
        *d7b_cols,
    ]
    rows: list[list[str]] = []
    for m in models:
        row = [
            MODEL_DISPLAY.get(m, m),
            pct(get(data, m, "pitchbench_a1_single_pitch_id",           "accuracy.any")),
            pct(get(data, m, "pitchbench_y1_single_pitch_id_mcq",       "accuracy.any")),
        ]
        row.extend(
            pct(get(data, m, "pitchbench_d7a_pitch_with_reference", f"by_interval.{step}.any"))
            for step in D7_INTERVAL_STEPS
        )
        row.extend(
            pct(get(data, m, "pitchbench_d7b_pitch_with_reference_split", f"by_interval.{step}.any"))
            for step in D7_INTERVAL_STEPS
        )
        rows.append(row)
    return header, rows


# ── output ────────────────────────────────────────────────────────────────────

def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data   = load_csv(INPUT_CSV)
    models = get_models(data)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Models: {[MODEL_DISPLAY.get(m, m) for m in models]}")

    h1, r1 = build_table1(data, models)
    h2, r2 = build_table2(data, models)
    h3, r3 = build_table3(data, models)

    write_csv(OUT_DIR / f"{ts}_table1_sound_attributes.csv",        h1, r1)
    write_csv(OUT_DIR / f"{ts}_table2_environmental_complexity.csv", h2, r2)
    write_csv(OUT_DIR / f"{ts}_table3_mcq_vs_reference.csv",        h3, r3)


if __name__ == "__main__":
    main()
