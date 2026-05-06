#!/usr/bin/env python3
"""
Analyze pitchbench_a1_single_pitch_id results across models.

Extracts MIDI accuracy data from all a1 experiments in a directory,
aggregates by MIDI pitch and model, and creates a visualization.

Usage:
    python analyze_a1.py results/eval
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from pitchbench.experiments.helpers.music import extract_freq, extract_solfege, note_to_midi, solfege_pc_to_midi


FORMATS = ('midi', 'spn', 'hz')


def extract_a1_data(directory):
    """Extract A1 predicted MIDI values by format, model, and GT MIDI.

    Returns {format: {model: {midi_gt: [midi_pred]}}}
    """
    data = {fmt: defaultdict(lambda: defaultdict(list)) for fmt in FORMATS}

    for csv_file in Path(directory).rglob('results_*.csv'):
        csv_path = csv_file.as_posix()

        # Prefer model-first layout: results/eval/<model_bucket>/pitchbench_a1_single_pitch_id/.../accuracies_<model>.csv
        # Also supports older layouts that may rely on an experiment column inside CSV rows.
        is_a1_path = '/pitchbench_a1_single_pitch_id/' in csv_path

        # Extract model name from filename
        model = csv_file.stem.replace('accuracies_', '').replace('results_', '')

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp = row.get('experiment', '')

                # Accept A1 if path indicates it, or if legacy CSV includes experiment field.
                is_a1_row = is_a1_path or ('pitchbench_a1' in exp)
                if not is_a1_row:
                    continue

                try:
                    midi_gt = int(float(row.get('midi_gt', '')))
                except (ValueError, IndexError, TypeError):
                    continue

                for fmt in FORMATS:
                    midi_pred = _predicted_midi(row, fmt, midi_gt)
                    if midi_pred is None:
                        continue
                    data[fmt][model][midi_gt].append(midi_pred)

    return data


def _predicted_midi(row, fmt, midi_gt):
    """Convert a row prediction in `fmt` into MIDI float, or None."""
    if fmt == 'midi':
        try:
            return float(row.get('midi_pred', ''))
        except (ValueError, TypeError):
            return None

    if fmt == 'spn':
        spn = row.get('spn_pred', '')
        if not isinstance(spn, str) or not spn.strip():
            return None
        midi = note_to_midi(spn.strip())
        return float(midi) if midi is not None else None

    if fmt == 'doremi':
        raw = row.get('raw_doremi') or row.get('doremi_pred') or ''
        if not isinstance(raw, str) or not raw.strip():
            return None
        pc = extract_solfege(raw)
        if pc is None:
            return None
        return float(solfege_pc_to_midi(int(midi_gt), int(pc)))

    if fmt == 'hz':
        raw_hz = row.get('raw_hz') or row.get('hz_pred') or ''
        hz = extract_freq(str(raw_hz))
        if hz is None or hz <= 0:
            return None
        return float(12.0 * math.log2(float(hz) / 440.0) + 69.0)

    return None


def compute_stats(data) -> dict[str, dict[str, dict[int, tuple[float, float, float, int]]]]:
    """Compute mean and bootstrap CI by format/model/GT MIDI.

    Returns {format: {model: {midi_gt: (mean_pred, ci_lower, ci_upper, n)}}}
    """
    stats = {fmt: {} for fmt in FORMATS}

    for fmt in FORMATS:
        for model, midi_data in data[fmt].items():
            stats[fmt][model] = {}
            for midi_gt, preds in midi_data.items():
                if not preds:
                    continue

                mean, ci_lower, ci_upper = _bootstrap_ci(preds)

                stats[fmt][model][midi_gt] = (mean, ci_lower, ci_upper, len(preds))

    return stats


def _bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    """Return (mean, ci_lo, ci_hi); empty input returns NaNs."""
    if not values:
        return float('nan'), float('nan'), float('nan')

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def compute_l1_scores(stats):
    """Compute per-model L1 distance from the ideal y=x line.

    Returns {model: l1_score}, where l1_score is mean absolute MIDI error
    between per-pitch mean predicted MIDI and GT MIDI.
    """
    l1_scores = {fmt: {} for fmt in FORMATS}

    for fmt in FORMATS:
        for model, model_stats in stats[fmt].items():
            if not model_stats:
                continue

            errors = [abs(vals[0] - midi_gt) for midi_gt, vals in model_stats.items()]
            l1_scores[fmt][model] = float(np.mean(errors))

    return l1_scores


def pretty_model_name(model):
    """Map internal model IDs to human-friendly labels for plotting."""
    m = model.lower()

    if 'audio_flamingo_next' in m:
        return 'Audio Flamingo Next'
    if 'gemini' in m:
        return 'Google Gemini'
    if 'gpt_4o_audio' in m or 'openai_gpt_4o_audio' in m:
        return 'OpenAI GPT-4o Audio'
    if 'openrouter_' in m:
        # Fallback: strip provider prefix and title-case.
        m = m.replace('openrouter_', '')
    return m.replace('_', ' ').replace('-', ' ').title()


def _axis_limits_for_format(format_data):
    """Compute axis limits from GT and predicted MIDI values for one format."""
    gt_values = []
    pred_values = []

    for model_data in format_data.values():
        for midi_gt, preds in model_data.items():
            gt_values.append(float(midi_gt))
            pred_values.extend(float(p) for p in preds)

    if not gt_values:
        return 0.0, 127.0

    min_val = min(gt_values + pred_values) if pred_values else min(gt_values)
    max_val = max(gt_values + pred_values) if pred_values else max(gt_values)
    lo = math.floor(min_val) - 1
    hi = math.ceil(max_val) + 1
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def plot_a1_results(
    stats: dict[str, dict[str, dict[int, tuple[float, float, float, int]]]],
    raw_data: Any,
    l1_scores: Any = None,
    output_file: str = 'a1_results.png',
) -> None:
    """Create a horizontal row of A1 predicted-vs-truth MIDI subplots."""
    plt.style.use('seaborn-v0_8-whitegrid')

    ranges = []
    for fmt in FORMATS:
        lo, hi = _axis_limits_for_format(raw_data[fmt])
        ranges.append(hi - lo)
    max_range = max(ranges) if ranges else 60.0
    scale = max(1.0, min(1.35, max_range / 60.0))

    fig, axes = plt.subplots(1, 3, figsize=(18 * scale, 6.6 * scale), sharex=False, sharey=False)
    axes_flat = list(axes)

    model_colors = {
        'audio_flamingo_next_instruct': '#41AB5D',               # green  (Flamingo)
        'openrouter_google_gemini_3_1_pro_preview': '#E07028',   # orange (Gemini)
        'openrouter_openai_gpt_4o_audio_preview': '#3182BD',     # blue   (GPT)
    }
    _fallback_palette = ['#7570b3', '#e7298a', '#e6ab02', '#a6761d', '#b2df8a']

    if l1_scores is None:
        l1_scores = {fmt: {} for fmt in FORMATS}  # type: ignore[assignment]

    for subplot_idx, fmt in enumerate(FORMATS):
        ax = axes_flat[subplot_idx]
        format_stats = stats[fmt]
        lo_lim, hi_lim = 28.0, 100.0

        all_midis = sorted(set(midi for model_data in format_stats.values() for midi in model_data.keys()))

        for idx, (model, model_stats) in enumerate(sorted(format_stats.items())):
            color = model_colors.get(model, _fallback_palette[idx % len(_fallback_palette)])
            midis = sorted(model_stats.keys())
            means = [model_stats[m][0] for m in midis]
            lower = [model_stats[m][1] for m in midis]
            upper = [model_stats[m][2] for m in midis]

            display = pretty_model_name(model)
            label = display
            if model in l1_scores.get(fmt, {}):
                label = f'{display} (L1={l1_scores[fmt][model]:.2f})'

            valid = [i for i, m in enumerate(means) if not np.isnan(m)]
            if not valid:
                continue

            xs = [midis[i] for i in valid]
            ys = [means[i] for i in valid]
            low_vals = [lower[i] for i in valid]
            high_vals = [upper[i] for i in valid]

            ax.fill_between(xs, low_vals, high_vals, color=color, alpha=0.14)
            ax.plot(xs, ys, marker='o', markersize=4.0, linewidth=2.0, color=color, label=label)

        if all_midis:
            diag = np.linspace(lo_lim, hi_lim, 200)
            ax.plot(diag, diag, color='#6c757d', linewidth=1.6, linestyle='--', label='y = x (perfect)')

            ax.set_xlim([lo_lim, hi_lim])
            ax.set_ylim([lo_lim, hi_lim])

            tick_step = 12 if (hi_lim - lo_lim) > 72 else 6
            tick_start = int(math.ceil(lo_lim / tick_step) * tick_step)
            tick_end = int(math.floor(hi_lim / tick_step) * tick_step)
            ticks = list(range(tick_start, tick_end + 1, tick_step))
            if ticks:
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='major', alpha=0.25, linewidth=0.8)
        ax.grid(False, which='minor')
        ax.set_title(f'{fmt.upper()} Input', fontsize=12, fontweight='bold')

        ax.set_xlabel('Ground-Truth MIDI Pitch', fontsize=11, fontweight='bold')
        if subplot_idx == 0:
            ax.set_ylabel('Predicted MIDI Pitch', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')

        ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=8)

    fig.suptitle('A1 Single-Pitch Identification: Predicted vs Ground Truth by Input Format', fontsize=16, fontweight='bold', y=0.98)

    out_path = Path(output_file)
    if out_path.parent != Path('.'):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def save_summary_csv(stats, output_file='analyze_a1.csv'):
    """Save aggregated predicted-MIDI stats to CSV for inspection."""
    out_path = Path(output_file)
    if out_path.parent != Path('.'):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['format', 'model', 'midi_gt', 'pred_midi_mean', 'ci_lower', 'ci_upper', 'n_samples'])

        for fmt in FORMATS:
            for model in sorted(stats[fmt].keys()):
                for midi in sorted(stats[fmt][model].keys()):
                    mean, ci_lower, ci_upper, n = stats[fmt][model][midi]
                    writer.writerow([fmt, model, midi, f'{mean:.4f}', f'{ci_lower:.4f}', f'{ci_upper:.4f}', n])

    print(f"Saved summary to {output_file}")


def save_l1_csv(l1_scores, stats, output_file='analyze_a1_l1.csv'):
    """Save per-model L1 scores to CSV."""
    out_path = Path(output_file)
    if out_path.parent != Path('.'):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['format', 'model', 'l1_score', 'n_midis'])

        for fmt in FORMATS:
            for model in sorted(l1_scores[fmt].keys()):
                writer.writerow([fmt, model, f'{l1_scores[fmt][model]:.6f}', len(stats[fmt].get(model, {}))])

    print(f"Saved L1 scores to {output_file}")


def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = 'results/eval'

    data_dir = Path(directory)
    if not data_dir.exists():
        print(f"Error: {directory} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {directory} for a1 results...")
    data = extract_a1_data(directory)

    if not any(data[fmt] for fmt in FORMATS):
        print("No a1 results found", file=sys.stderr)
        sys.exit(1)

    print(f"Found data for {len(data['midi'])} models")
    for model in sorted(data['midi'].keys()):
        n_midis = len(data['midi'][model])
        n_samples = sum(len(accs) for accs in data['midi'][model].values())
        print(f"  {model}: {n_midis} MIDI pitches, {n_samples} total samples")

    stats = compute_stats(data)
    l1_scores = compute_l1_scores(stats)

    print('Per-model L1 (mean absolute MIDI error from y=x):')
    for fmt in FORMATS:
        print(f'  [{fmt}]')
        for model in sorted(l1_scores[fmt].keys()):
            print(f"    {model}: {l1_scores[fmt][model]:.6f}")

    save_summary_csv(stats, 'paper/analyze_a1.csv')
    save_l1_csv(l1_scores, stats, 'paper/analyze_a1_l1.csv')
    plot_a1_results(stats, data, l1_scores=l1_scores, output_file='paper/a1_results_combined.png')


if __name__ == '__main__':
    main()
