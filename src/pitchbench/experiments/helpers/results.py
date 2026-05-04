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
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pitchbench.config as config
from pitchbench.experiments.helpers import cost as cost_tracker

RESULTS_ROOT = config.RESULTS_DIR
DATA_ROOT    = config.DATA_DIR


def _safe_stem(name: str) -> str:
    """Filesystem-safe version of model_name (OpenRouter slugs contain slashes)."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_") or "model"


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
    """Return base metadata: timestamp, git commit, CLI argv, and any extra kwargs."""
    import sys
    meta: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "argv":      sys.argv[:],
    }
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
        if dirty:
            print(
                "\n  *** WARNING: working tree has uncommitted changes — "
                "results will NOT be exactly reproducible from git_commit alone ***\n"
            )
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
    # Each experiment run accumulates LLM cost from a clean slate so per-model
    # totals reflect just this run, not anything billed earlier in the process.
    cost_tracker.reset()
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


# ── LLM usage / cost block ────────────────────────────────────────────────────

def _format_usage_block(usage: dict[str, Any]) -> list[str]:
    """Render an LLM USAGE block; returns [] when nothing was billed (local models)."""
    if not usage or not usage.get("calls"):
        return []
    lines = ["", "LLM USAGE", "=" * 50,
             f"  {'calls':<22} {usage.get('calls', 0):>12,}",
             f"  {'prompt tokens':<22} {usage.get('prompt_tokens', 0):>12,}",
             f"  {'completion tokens':<22} {usage.get('completion_tokens', 0):>12,}",
             f"  {'total tokens':<22} {usage.get('total_tokens', 0):>12,}"]
    cached = usage.get("cached_tokens", 0)
    if cached:
        lines.append(f"  {'cached prompt tokens':<22} {cached:>12,}")
    lines.append(f"  {'cost (USD)':<22} {'$' + format(usage.get('cost_usd', 0.0), '.6f'):>12}")
    return lines


# ── Cross-experiment aggregation ──────────────────────────────────────────────

def extract_format_accuracies(
    run_dir: Path,
    model_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Read per-format accuracies (and stimulus count) back from a run directory.

    Returns ``{model: {"n": int, "formats": {fmt: acc}}}``. ``n`` is taken from
    ``summary.total`` in ``results_<model>.json`` (falls back to record count,
    then ``None``). Models with no ``format_accuracy_<model>.csv`` file are
    skipped silently.
    """
    out: dict[str, dict[str, Any]] = {}
    for m in model_names:
        stem    = _safe_stem(m)
        fa_path = run_dir / f"format_accuracy_{stem}.csv"
        if not fa_path.exists():
            continue

        formats: dict[str, float] = {}
        n: int | None = None
        with open(fa_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fmt     = row["Format"].strip().lower()
                acc_str = row["Accuracy"].strip().rstrip("%")
                try:
                    formats[fmt] = float(acc_str) / 100.0
                except ValueError:
                    continue
                if n is None and row.get("n_samples", "").strip():
                    try:
                        n = int(row["n_samples"])
                    except ValueError:
                        pass
        if not formats:
            continue

        json_path = run_dir / f"results_{stem}.json"
        if n is None and json_path.exists():
            try:
                payload = json.loads(json_path.read_text())
                summary = payload.get("summary", {})
                n = summary.get("total")
                if n is None:
                    results = payload.get("results")
                    if isinstance(results, list):
                        n = len(results)
            except (OSError, json.JSONDecodeError):
                pass

        out[m] = {"n": n, "formats": formats}
    return out


def write_aggregate_format_accuracies(
    output_path: Path,
    runs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Write a single CSV aggregating per-experiment format accuracies + per-model means.

    Args:
        output_path: target CSV path
        runs:        ``{exp_name: {model: {"n": int, "formats": {fmt: acc}}}}``

    Layout (long form, columns: experiment, model, n_samples, format, accuracy):
        pitchbench_a1_pitch_id, audio_flamingo_next_think, 57,   midi,   0.198
        pitchbench_a1_pitch_id, audio_flamingo_next_think, 57,   abc,    0.068
        ...
        __MEAN__,               audio_flamingo_next_think, 1850, midi,   0.123
        __MEAN__,               audio_flamingo_next_think, 1850, abc,    0.087

    The ``__MEAN__`` rows give the per-model, per-format mean across experiments;
    ``n_samples`` on a ``__MEAN__`` row is the total stimuli summed across the
    experiments that contributed.
    """
    rows: list[dict[str, Any]] = []
    means: dict[tuple[str, str], list[float]] = {}
    totals: dict[str, int] = {}

    for exp_name, by_model in sorted(runs.items()):
        for model, info in by_model.items():
            n       = info.get("n")
            formats = info.get("formats", {})
            for fmt, acc in formats.items():
                rows.append({
                    "experiment": exp_name,
                    "model":      model,
                    "n_samples":  n if n is not None else "",
                    "format":     fmt,
                    "accuracy":   round(acc, 4),
                })
                means.setdefault((model, fmt), []).append(acc)
            if isinstance(n, int):
                totals[model] = totals.get(model, 0) + n

    for (model, fmt), accs in sorted(means.items()):
        if not accs:
            continue
        rows.append({
            "experiment": "__MEAN__",
            "model":      model,
            "n_samples":  totals.get(model, ""),
            "format":     fmt,
            "accuracy":   round(sum(accs) / len(accs), 4),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "model", "n_samples", "format", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ── Session-level cost summary (across multiple experiments) ──────────────────

def write_session_cost_summary(
    out_dir: Path,
    per_experiment: dict[str, dict[str, dict[str, Any]]],
) -> Path | None:
    """Write a cross-experiment LLM cost rollup.

    Args:
        out_dir:        directory to create and write into.
        per_experiment: ``{exp_name: {model_name: usage_dict}}`` where usage_dict
                        is whatever ``cost.all_totals()`` returned at the end of
                        each experiment.

    Returns the path of the .txt file, or ``None`` if no cost was recorded
    anywhere (in which case nothing is written).
    """
    grand_calls  = 0
    grand_tokens = 0
    grand_cost   = 0.0
    per_exp_total: list[tuple[str, int, int, float]] = []
    for exp_name in sorted(per_experiment):
        e_calls = e_toks = 0
        e_cost  = 0.0
        for u in per_experiment[exp_name].values():
            e_calls += u.get("calls", 0)
            e_toks  += u.get("total_tokens", 0)
            e_cost  += u.get("cost_usd", 0.0)
        per_exp_total.append((exp_name, e_calls, e_toks, e_cost))
        grand_calls  += e_calls
        grand_tokens += e_toks
        grand_cost   += e_cost

    if grand_calls == 0:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {
        "timestamp":      ts,
        "experiments":    per_experiment,
        "total_calls":    grand_calls,
        "total_tokens":   grand_tokens,
        "total_cost_usd": round(grand_cost, 6),
    }
    (out_dir / "cost_summary.json").write_text(json.dumps(payload, indent=2))

    # ── TXT ───────────────────────────────────────────────────────────────────
    name_w = max(34, max(len(e) for e, *_ in per_exp_total) + 2)
    lines = [
        "PitchBench session cost summary",
        f"Timestamp : {ts}",
        "",
        f"{'Experiment':<{name_w}}{'calls':>10}{'tokens':>14}{'cost (USD)':>16}",
        "─" * (name_w + 10 + 14 + 16),
    ]
    for exp_name, c, t, cost_usd in per_exp_total:
        lines.append(
            f"{exp_name:<{name_w}}{c:>10,}{t:>14,}{('$' + format(cost_usd, '.6f')):>16}"
        )
    lines += [
        "─" * (name_w + 10 + 14 + 16),
        f"{'TOTAL':<{name_w}}{grand_calls:>10,}{grand_tokens:>14,}"
        f"{('$' + format(grand_cost, '.6f')):>16}",
        "",
    ]
    txt_path = out_dir / "cost_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    return txt_path


# ── Per-model results ─────────────────────────────────────────────────────────

def save_format_accuracy_csv(
    run_dir: Path,
    model_name: str,
    per_format: dict[str, float],
    n_samples: int | None = None,
) -> None:
    """Write format_accuracy_<model>.csv — one row per prompt format.

    Columns: Format, n_samples, Accuracy  (e.g. "MIDI", 57, "19.8%")
    """
    path = run_dir / f"format_accuracy_{_safe_stem(model_name)}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Format", "n_samples", "Accuracy"])
        for fmt, acc in per_format.items():
            writer.writerow([fmt.upper(), n_samples if n_samples is not None else "", f"{acc:.1%}"])


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

    stem = f"results_{_safe_stem(model_name)}"

    # ── Marginal-accuracy summary (added automatically; included in JSON) ─────
    marginals = summarise_marginals(records, formats=formats, extra_metrics=extra_metrics)
    summary   = {**summary, "marginals": marginals}

    # ── LLM usage / cost (OpenRouter only — local servers record nothing) ─────
    usage    = cost_tracker.get(model_name)
    metadata = {**metadata, "usage": usage}
    summary  = {**summary, "cost_usd": usage["cost_usd"], "total_tokens": usage["total_tokens"]}

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {"metadata": metadata, "summary": summary, "results": records}
    (run_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))

    # ── TXT ───────────────────────────────────────────────────────────────────
    lines: list[str] = [
        f"Experiment : {exp_name}",
        f"Model      : {model_name}",
    ]
    for k, v in metadata.items():
        if k == "usage":
            continue                                  # rendered in its own block below
        lines.append(f"{k:10} : {v}")
    lines += ["", "SUMMARY", "=" * 50]
    if summary_lines:
        lines.extend(summary_lines)
    else:
        for k, v in summary.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                lines.append(f"  {k:<30} {v}")
    lines.extend(_format_marginals_block(marginals, formats=formats, extra_metrics=extra_metrics))
    usage_lines = _format_usage_block(usage)
    lines.extend(usage_lines)
    (run_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    # Echo the LLM USAGE block to stdout so the cost is visible at the end of
    # each model run (and gets teed into run_log.txt automatically).
    for ln in usage_lines:
        print(ln)

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
        save_format_accuracy_csv(run_dir, model_name, per_format, n_samples=summary.get("total"))

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

    # ── LLM cost rollup across models (OpenRouter only) ───────────────────────
    usage_per_model = {m: cost_tracker.get(m) for m in models}
    total_cost_usd  = round(sum(u["cost_usd"]    for u in usage_per_model.values()), 6)
    total_tokens    = sum(u["total_tokens"]      for u in usage_per_model.values())
    total_calls     = sum(u["calls"]             for u in usage_per_model.values())

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
        "usage":          {
            "per_model":      usage_per_model,
            "total_cost_usd": total_cost_usd,
            "total_tokens":   total_tokens,
            "total_calls":    total_calls,
        },
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

    # Render the LLM-cost rollup only when at least one model was billed —
    # local-only runs (everything zero) skip the block entirely.
    if total_calls:
        txt_lines += ["LLM USAGE", "─" * 50,
                      f"{'Model':<{model_w}}{'calls':>10}{'tokens':>14}{'cost (USD)':>16}"]
        for m in models:
            u = usage_per_model[m]
            txt_lines.append(
                f"{m:<{model_w}}{u['calls']:>10,}{u['total_tokens']:>14,}"
                f"{('$' + format(u['cost_usd'], '.6f')):>16}"
            )
        txt_lines.append(
            f"{'TOTAL':<{model_w}}{total_calls:>10,}{total_tokens:>14,}"
            f"{('$' + format(total_cost_usd, '.6f')):>16}"
        )
        txt_lines.append("")

    (run_dir / "comparison.txt").write_text("\n".join(txt_lines))
    print(f"Comparison  → {run_dir}/comparison.*")
