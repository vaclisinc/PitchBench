"""
Write experiment results in three forms per model:
  <run_dir>/results_<model>.json   full reproducible record
  <run_dir>/results_<model>.txt    human-readable summary
  <run_dir>/results_<model>.csv    one row per test item

And when multiple models finish, a cross-model comparison:
  <run_dir>/comparison.json / .csv / .txt
"""

import csv
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pitchbench.config as config

RESULTS_ROOT = config.RESULTS_DIR
DATA_ROOT    = config.DATA_DIR


# ── Run-log tee ───────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    """Duplicate writes to two streams (stdout + a log file)."""
    def __init__(self, primary: io.TextIOBase, secondary: io.TextIOBase) -> None:
        self._p = primary
        self._s = secondary

    def write(self, s: str) -> int:
        self._p.write(s)
        self._s.write(s)
        return len(s)

    def flush(self) -> None:
        self._p.flush()
        self._s.flush()


# ── Metadata ──────────────────────────────────────────────────────────────────

def get_run_metadata(**extra) -> dict[str, Any]:
    """Return base metadata: timestamp + git commit + any extra kwargs."""
    meta: dict[str, Any] = {"timestamp": datetime.now().isoformat()}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=Path.cwd(), stderr=subprocess.DEVNULL,
        ) != 0
        meta["git_commit"] = commit
        meta["git_dirty"]  = dirty
    except Exception:
        meta["git_commit"] = None
        meta["git_dirty"]  = None
    meta.update(extra)
    return meta


# ── Directories ───────────────────────────────────────────────────────────────

def exp_data_dir(exp_name: str) -> Path:
    """Return (and create) the data directory for an experiment."""
    d = DATA_ROOT / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_run_dir(exp_name: str) -> Path:
    """Create and return a new sequentially-ID'd run directory.

    Format: results/<exp_name>/run_<NNN>_<YYYYMMDD_HHMMSS>/
    The NNN counter is based on existing run_* subdirectories.
    Also starts teeing stdout to run_log.txt inside the new directory.
    """
    exp_dir = RESULTS_ROOT / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_id   = len(existing) + 1
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = exp_dir / f"run_{run_id:03d}_{ts}"
    run_dir.mkdir()
    log_file = open(run_dir / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)  # type: ignore[assignment]
    return run_dir


# ── CSV field ordering ────────────────────────────────────────────────────────

# Ground-truth + IV anchor columns, in priority order — used by
# _ordered_csv_fields and summarise_marginals to pick stable layouts.
_GT_ANCHORS = [
    # Universal IVs (PitchBench v2)
    "duration_ms", "midi", "source", "source_type",
    # Common per-experiment IVs
    "gap_ms", "interval", "interval_name", "interval_st", "direction",
    "ref_midi", "ref_note", "condition",
    "vibrato_rate_hz", "vibrato_depth_cents",
    "detune_hz",
    "pos_ms", "onset_s_gt", "offset_s_gt", "query_time_s",
    "n_notes", "n_distractors", "n_transitions", "step_size_st",
    "note_duration_ms", "glide_speed_ms_per_st",
    "chord_quality_gt", "same_instrument", "task",
    "loudness_db", "effect", "background", "snr_db",
    "rhythm", "seed",
    # Legacy ground-truth aliases
    "note", "frequency", "ground_truth_note", "ground_truth_freq",
    "note_1", "note_2",
]


def _ordered_csv_fields(fieldnames: list[str]) -> list[str]:
    """Push prompt_* fields to the end; keep all other fields in original order."""
    prompt_fields = [f for f in fieldnames if f.startswith("prompt")]
    other_fields  = [f for f in fieldnames if not f.startswith("prompt")]
    return other_fields + prompt_fields


# ── Marginal-accuracy summary (v2) ────────────────────────────────────────────

# Columns we never aggregate over: raw responses, prompts, predictions, scoring,
# audio path. Anything else that has at least 2 distinct values *across the
# records* is treated as an IV and gets a marginal-accuracy block.
_NON_IV_PREFIXES: tuple[str, ...] = (
    "prompt_", "raw_", "audio_", "wav",
)
_NON_IV_SUFFIXES: tuple[str, ...] = (
    "_pred", "_correct", "_within_1", "_pc_correct", "_octave_correct",
    "_abs_error", "_gt", "_err",
)
_NON_IV_EXACT: set[str] = {
    "wav", "audio_url", "audio_file",
    "raw_response", "raw_midi", "raw_spn", "raw_abc", "raw_doremi", "raw_hz",
    "elapsed_s", "model_name", "model_info",
}


def _is_iv_column(name: str) -> bool:
    if name in _NON_IV_EXACT:
        return False
    if any(name.startswith(p) for p in _NON_IV_PREFIXES):
        return False
    if any(name.endswith(s) for s in _NON_IV_SUFFIXES):
        return False
    return True


def summarise_marginals(
    records: list[dict[str, Any]],
    formats: tuple[str, ...] = ("midi", "spn", "doremi", "hz"),
    *,
    extra_metrics: tuple[str, ...] = (),
) -> dict[str, dict[Any, dict[str, Any]]]:
    """Aggregate per-stimulus accuracy across every IV that varies in the records.

    Returns ``{variable: {value: {metric: rate, ..., "n": int}}}`` where
    ``metric`` is each of the ``<format>_correct`` columns plus any names
    listed in ``extra_metrics`` (used by non-pitch-ID experiments — e.g.
    ``("trajectory_correct",)`` or ``("iou", "within_500ms_on")``).

    Only columns that appear with at least 2 distinct values across the records
    are reported, so a CSV-detected IV that turns out to have been held
    constant in this run is silently dropped from the summary.
    """
    if not records:
        return {}

    # Discover IV columns + count distinct values
    candidates: dict[str, set] = {}
    for r in records:
        for k, v in r.items():
            if not _is_iv_column(k):
                continue
            try:
                hash(v)
            except TypeError:                       # unhashable (list, dict)
                continue
            candidates.setdefault(k, set()).add(v)

    iv_cols = [k for k, vs in candidates.items() if len(vs) >= 2]

    # Order: prefer the order in _GT_ANCHORS, append remaining.
    anchors    = [c for c in _GT_ANCHORS if c in iv_cols]
    remaining  = [c for c in iv_cols if c not in anchors]
    ordered    = anchors + remaining

    metric_cols = [f"{fmt}_correct" for fmt in formats] + list(extra_metrics)
    metric_cols = [m for m in metric_cols if any(m in r for r in records)]

    out: dict[str, dict[Any, dict[str, Any]]] = {}
    for col in ordered:
        groups: dict[Any, list[dict[str, Any]]] = {}
        for r in records:
            v = r.get(col)
            try:
                hash(v)
            except TypeError:
                continue
            groups.setdefault(v, []).append(r)

        col_summary: dict[Any, dict[str, Any]] = {}
        for v, sub in groups.items():
            entry: dict[str, Any] = {"n": len(sub)}
            for m in metric_cols:
                vals = [r[m] for r in sub if m in r and isinstance(r[m], (int, float, bool))]
                entry[m] = round(sum(vals) / len(vals), 4) if vals else None
            col_summary[v] = entry
        if col_summary:
            out[col] = col_summary
    return out


def _format_marginals_block(
    marginals: dict[str, dict[Any, dict[str, Any]]],
    formats: tuple[str, ...],
    extra_metrics: tuple[str, ...] = (),
) -> list[str]:
    """Render the marginal-accuracy dict as fixed-width text blocks for the .txt file."""
    if not marginals:
        return []

    metric_cols = [f"{fmt}_correct" for fmt in formats] + list(extra_metrics)
    headers     = list(formats) + [m for m in extra_metrics]
    out: list[str] = ["", "== Marginal accuracy ==", ""]

    def _val_str(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:>5.2%}" if 0.0 <= v <= 1.0 else f"{v:>6.2f}"
        return f"{v!s:>6}"

    name_w = 14
    for col, group in marginals.items():
        out.append(f"{col:<{name_w}}  " + "  ".join(f"{h:>7}" for h in headers) + "    n")
        out.append(f"{'─' * name_w}  " + "  ".join("─" * 7 for _ in headers) + "  ────")
        for v in sorted(group.keys(), key=lambda x: (str(type(x).__name__), x)):
            row = group[v]
            cells: list[str] = []
            for m in metric_cols:
                val = row.get(m)
                if val is None:
                    cells.append(f"{'—':>7}")
                elif isinstance(val, float) and 0.0 <= val <= 1.0:
                    cells.append(f"{val:>7.2%}")
                else:
                    cells.append(f"{val!s:>7}")
            out.append(f"{str(v):<{name_w}}  " + "  ".join(cells) + f"  {row['n']:>4}")
        out.append("")
    return out


# ── Per-model results ─────────────────────────────────────────────────────────

def save_format_accuracy_csv(
    run_dir: Path,
    model_name: str,
    per_format: dict[str, float],
) -> None:
    """Write format_accuracy_<model>.csv — one row per prompt format.

    Columns: Format, Accuracy  (e.g. "MIDI", "19.8%")
    """
    path = run_dir / f"format_accuracy_{model_name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Format", "Accuracy"])
        for fmt, acc in per_format.items():
            writer.writerow([fmt.upper(), f"{acc:.1%}"])


def save_results(
    exp_name: str,
    model_name: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
    summary_lines: list[str] | None = None,
    run_dir: Path | None = None,
    *,
    formats: tuple[str, ...] = ("midi", "spn", "doremi", "hz"),
    extra_metrics: tuple[str, ...] = (),
) -> Path:
    """
    Write JSON + TXT + CSV for one model's results; return the run directory.

    Args:
        exp_name:      e.g. "exp_3_nsynth"
        model_name:    config slug used as file stem, e.g. "music_flamingo"
        records:       list of per-item dicts → CSV rows + JSON results
        summary:       aggregate stats dict
        metadata:      run config / timestamps
        summary_lines: pre-formatted lines for the TXT block (auto-generated if None)
        run_dir:       directory to write into; created via make_run_dir() if None
    """
    if run_dir is None:
        run_dir = make_run_dir(exp_name)

    stem = f"results_{model_name}"

    # ── Marginal-accuracy summary (added automatically; included in JSON) ─────
    marginals = summarise_marginals(records, formats=formats, extra_metrics=extra_metrics)
    summary   = {**summary, "marginals": marginals}

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {"metadata": metadata, "summary": summary, "results": records}
    (run_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))

    # ── TXT ───────────────────────────────────────────────────────────────────
    lines: list[str] = [
        f"Experiment : {exp_name}",
        f"Model      : {model_name}",
    ]
    for k, v in metadata.items():
        lines.append(f"{k:10} : {v}")
    lines += ["", "SUMMARY", "=" * 50]
    if summary_lines:
        lines.extend(summary_lines)
    else:
        for k, v in summary.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                lines.append(f"  {k:<30} {v}")
    lines.extend(_format_marginals_block(marginals, formats=formats, extra_metrics=extra_metrics))
    (run_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if records:
        csv_records: list[dict[str, Any]] = []
        for r in records:
            row = dict(r)
            # Rename wav → audio_url and add audio_file (basename)
            if "wav" in row:
                wav_path = str(row.pop("wav"))
                row["audio_url"]  = wav_path
                row["audio_file"] = Path(wav_path).name
            elif "audio_url" not in row and "audio_file" not in row:
                pass
            csv_records.append(row)

        fieldnames = _ordered_csv_fields(list(csv_records[0].keys()))
        # Place audio_file right after audio_url
        if "audio_file" in fieldnames and "audio_url" in fieldnames:
            fieldnames = [f for f in fieldnames if f != "audio_file"]
            fieldnames.insert(fieldnames.index("audio_url") + 1, "audio_file")

        with open(run_dir / f"{stem}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_records)

    # ── Per-format accuracy summary ───────────────────────────────────────────
    per_format = summary.get("per_format")
    if per_format:
        save_format_accuracy_csv(run_dir, model_name, per_format)

    print(f"\nResults saved → {run_dir}/{stem}.*")
    return run_dir


# ── Cross-model comparison ────────────────────────────────────────────────────

def save_comparison(
    run_dir: Path,
    model_summaries: dict[str, dict[str, Any]],
    exp_name: str,
) -> None:
    """
    Write comparison.{json,csv,txt} aggregating all models' summaries.

    Only scalar (int/float) metrics are included in the flat table.
    Nested dicts (per_delta_accuracy, per_instrument, etc.) appear only in JSON.
    """
    if len(model_summaries) < 2:
        return

    models = list(model_summaries.keys())

    # collect flat numeric metrics, preserving insertion order
    flat_keys: list[str] = []
    seen: set[str] = set()
    for summary in model_summaries.values():
        for k, v in summary.items():
            if k not in seen and isinstance(v, (int, float)) and not isinstance(v, bool):
                flat_keys.append(k)
                seen.add(k)

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {
        "exp_name":       exp_name,
        "run_dir":        str(run_dir),
        "models":         models,
        "metrics":        flat_keys,
        "comparison":     {
            m: {k: model_summaries[m].get(k) for k in flat_keys}
            for m in models
        },
        "full_summaries": model_summaries,
    }
    (run_dir / "comparison.json").write_text(json.dumps(payload, indent=2))

    # ── CSV ───────────────────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = [
        {"model": m, **{k: model_summaries[m].get(k) for k in flat_keys}}
        for m in models
    ]
    with open(run_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + flat_keys)
        writer.writeheader()
        writer.writerows(rows)

    # ── TXT ───────────────────────────────────────────────────────────────────
    model_w = max(len(m) for m in models) + 2

    # format each value
    def fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if v < 1.0 else f"{v:.2f}"
        return str(v)

    # split into groups of ≤6 metrics to stay under ~120 chars wide
    chunk = 6
    txt_lines: list[str] = [
        f"Comparison — {exp_name}",
        f"Run        : {run_dir.name}",
        f"Models     : {', '.join(models)}",
        "",
    ]
    for start in range(0, len(flat_keys), chunk):
        keys_chunk = flat_keys[start:start + chunk]
        col_w = max(max(len(k) for k in keys_chunk), 9)
        header = f"{'Model':<{model_w}}" + "  ".join(f"{k:>{col_w}}" for k in keys_chunk)
        txt_lines += [header, "─" * len(header)]
        for m in models:
            vals = [fmt(model_summaries[m].get(k)) for k in keys_chunk]
            txt_lines.append(f"{m:<{model_w}}" + "  ".join(f"{v:>{col_w}}" for v in vals))
        txt_lines.append("")

    (run_dir / "comparison.txt").write_text("\n".join(txt_lines))
    print(f"Comparison  → {run_dir}/comparison.*")
