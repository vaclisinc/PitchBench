"""
Experiment runner — entry point for the ``pitchbench`` CLI.

Usage::

    # Generate audio + dataset (data/generated/<exp>/data.parquet + WAV files)
    pitchbench generate a1
    pitchbench generate a               # every experiment in category a
    pitchbench generate all

    # Evaluate one model on pre-generated data
    pitchbench evaluate a1 --model openrouter/google/gemini-2.5-flash
    pitchbench evaluate a  --model openrouter/google/gemini-2.5-flash
    pitchbench evaluate all --model openrouter/google/gemini-2.5-flash

    # Higher-level analysis (ANALYSIS mode, preset-driven)
    pitchbench analyze --preset q1 --model openrouter/google/gemini-2.5-flash
    pitchbench analyze a1 --preset q1 --model openrouter/google/gemini-2.5-flash

    # List experiments / presets
    pitchbench --list
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pitchbench.config as config
from pitchbench.model import cost as cost_tracker
from pitchbench.model.query import model_slug
from pitchbench.experiments.helpers.results import (
    aggregate_run_accuracies,
    aggregate_session_metrics,
    combine_accuracies_csvs,
    write_model_summary_csv,
    write_session_cost_summary,
)



# ── Regex / discovery ────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"^pitchbench_([a-z]+\d+[a-z]*)_(.+)$")
_ID_RE   = re.compile(r"^[a-z]+\d+[a-z]*$")
_CAT_RE  = re.compile(r"^[a-z]$")


def _scripts_dir() -> Path:
    return Path(__file__).parent / "scripts"


def discover() -> list[str]:
    """Return all experiment module names, sorted by their letter+digit ID."""
    names = [p.stem for p in _scripts_dir().glob("pitchbench_*.py")]
    return sorted(names, key=_sort_key)


def _sort_key(name: str) -> tuple:
    m = _NAME_RE.match(name)
    if not m:
        return (chr(127), 0, name)
    ident = m.group(1)
    cat   = ident[0]
    try:
        num = int(re.search(r"\d+", ident).group())
    except (AttributeError, ValueError):
        num = 0
    return (cat, num, name)


def _id_to_name(exp_id: str) -> str | None:
    exp_id = exp_id.lower()
    for n in discover():
        m = _NAME_RE.match(n)
        if m and m.group(1) == exp_id:
            return n
    return None


def _category_to_names(cat: str) -> list[str]:
    cat = cat.lower()
    return [n for n in discover() if (m := _NAME_RE.match(n)) and m.group(1).startswith(cat)]


def _resolve_experiments(positionals: list[str]) -> list[str]:
    """Resolve a list of positional tokens to full experiment module names.

    Accepts: experiment IDs (``a1``), category letters (``a``), full module
    names, ``"all"``, or the standalone ``"paper"`` selector for Table 1.
    Raises ``SystemExit`` on unknown tokens.
    """
    if not positionals or positionals == ["all"]:
        return discover()
    if positionals == ["paper"]:
        names = [_id_to_name(exp_id) for exp_id in config.PAPER_EXPERIMENT_IDS]
        if any(name is None for name in names):
            sys.exit("A required paper experiment is missing from the package.")
        return names

    known = discover()
    out: list[str] = []
    for token in positionals:
        token = token.lower()
        if token == "all":
            out.extend(discover())
        elif _CAT_RE.match(token):
            names = _category_to_names(token)
            if not names:
                sys.exit(f"No experiments found for category '{token}'.")
            out.extend(names)
        elif _ID_RE.match(token):
            name = _id_to_name(token)
            if name is None:
                ids = [_NAME_RE.match(n).group(1) for n in known if _NAME_RE.match(n)]
                sys.exit(f"Unknown experiment ID '{token}'. Available: {ids}")
            out.append(name)
        elif token in known:
            out.append(token)
        else:
            sys.exit(f"Unknown experiment '{token}'.")

    # Preserve order, deduplicate.
    seen: set[str] = set()
    deduped: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


# ── Analysis presets ─────────────────────────────────────────────────────────

def _analysis_presets() -> dict[str, Any]:
    try:
        ac = importlib.import_module("pitchbench.configs.analysis_config")
    except Exception:
        return {}
    presets = getattr(ac, "PRESETS", None)
    return presets if isinstance(presets, dict) else {}


def _apply_analysis_preset(preset_name: str) -> list[str] | None:
    preset = _analysis_presets().get(preset_name)
    if not isinstance(preset, dict):
        return None
    notation_formats = preset.get("notation_formats")
    if notation_formats is not None:
        config.pitchbench_general_NOTATION_FORMATS = notation_formats
    experiments = preset.get("experiments", {})
    if not isinstance(experiments, dict):
        return []
    for exp_overrides in experiments.values():
        if not isinstance(exp_overrides, dict):
            continue
        for attr, val in exp_overrides.items():
            setattr(config, attr, val)
    return list(experiments.keys())


# ── Per-category dispatch ────────────────────────────────────────────────────

def _generate_one(
    name:        str,
    *,
    sample_n:    int | None = None,
    sample_seed: int = 42,
) -> int:
    """Import the experiment module and call the right generate_cat_*_data.

    Dispatch is by isinstance(spec, ...) so experiments whose name starts with
    a different letter than their spec type (e.g. d7a/d7c use CatASpec) route
    correctly.  CatESpec is an alias for CatASpec so there is no separate check.
    """
    mod = importlib.import_module(f"pitchbench.experiments.scripts.{name}")
    try:
        spec = mod.SPEC
    except AttributeError:
        print(f"  [SKIP] {name}: no SPEC attribute (bespoke experiment not yet supported in generate)")
        return 0

    from pitchbench.experiments.helpers.cat_a import CatASpec, generate_cat_a_data
    from pitchbench.experiments.helpers.cat_b import CatBSpec, generate_cat_b_data
    from pitchbench.experiments.helpers.cat_c import CatCSpec, generate_cat_c_data
    from pitchbench.experiments.helpers.cat_d import CatDSpec, generate_cat_d_data
    from pitchbench.experiments.helpers.cat_f import CatFSpec, generate_cat_f_data

    if isinstance(spec, CatASpec):  # covers CatESpec (same class)
        return generate_cat_a_data(spec, sample_n=sample_n, sample_seed=sample_seed)
    if isinstance(spec, CatBSpec):
        return generate_cat_b_data(spec, sample_n=sample_n, sample_seed=sample_seed)
    if isinstance(spec, CatCSpec):
        return generate_cat_c_data(spec, sample_n=sample_n, sample_seed=sample_seed)
    if isinstance(spec, CatDSpec):
        return generate_cat_d_data(spec, sample_n=sample_n, sample_seed=sample_seed)
    if isinstance(spec, CatFSpec):
        return generate_cat_f_data(spec, sample_n=sample_n, sample_seed=sample_seed)
    print(f"  [SKIP] {name}: unrecognised spec type {type(spec).__name__}")
    return 0


def _evaluate_one(
    name:        str,
    model_name:  str,
    run_dir:     Path,
    sample_info: dict[str, Any] | None = None,
    model_label: str | None = None,
    filter_conditions: bool = False,
) -> dict | None:
    """Import the experiment module and call the right evaluate_cat_*_from_parquet.

    Dispatch is by isinstance(spec, ...) so experiments whose name starts with
    a different letter than their spec type (e.g. d7a/d7c use CatASpec) route
    correctly.  CatESpec is an alias for CatASpec so there is no separate check.
    """
    mod = importlib.import_module(f"pitchbench.experiments.scripts.{name}")
    try:
        spec = mod.SPEC
    except AttributeError:
        print(f"  [SKIP] {name}: no SPEC attribute (bespoke experiment not yet supported in evaluate/analyze)")
        return None

    from pitchbench.experiments.helpers.cat_a import CatASpec, evaluate_cat_a_from_parquet
    from pitchbench.experiments.helpers.cat_b import CatBSpec, evaluate_cat_b_from_parquet
    from pitchbench.experiments.helpers.cat_c import CatCSpec, evaluate_cat_c_from_parquet
    from pitchbench.experiments.helpers.cat_d import CatDSpec, evaluate_cat_d_from_parquet
    from pitchbench.experiments.helpers.cat_f import CatFSpec, evaluate_cat_f_from_parquet

    print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}\n")

    exp_run_dir = run_dir / name
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(spec, CatASpec):  # covers CatESpec (same class)
            return evaluate_cat_a_from_parquet(spec, model_name, exp_run_dir, sample_info, model_label=model_label, filter_conditions=filter_conditions)  # type: ignore[return-value]
        if isinstance(spec, CatBSpec):
            return evaluate_cat_b_from_parquet(spec, model_name, exp_run_dir, sample_info, model_label=model_label, filter_conditions=filter_conditions)  # type: ignore[return-value]
        if isinstance(spec, CatCSpec):
            return evaluate_cat_c_from_parquet(spec, model_name, exp_run_dir, sample_info, model_label=model_label, filter_conditions=filter_conditions)  # type: ignore[return-value]
        if isinstance(spec, CatDSpec):
            return evaluate_cat_d_from_parquet(spec, model_name, exp_run_dir, sample_info, model_label=model_label, filter_conditions=filter_conditions)  # type: ignore[return-value]
        if isinstance(spec, CatFSpec):
            return evaluate_cat_f_from_parquet(spec, model_name, exp_run_dir, sample_info, model_label=model_label, filter_conditions=filter_conditions)  # type: ignore[return-value]
        print(f"  [SKIP] {name}: unrecognised spec type {type(spec).__name__}")
        return None
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        return None


# ── Subcommand implementations ────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    """``pitchbench generate`` — produce audio + data.parquet for each experiment."""
    names = _resolve_experiments(args.experiments or [])
    if not names:
        sys.exit("No experiments matched. Use --list to see available experiments.")

    # Route engine output to data/generated/ instead of data/audio/.
    config.AUDIO_DIR = config.GENERATED_DIR

    print(f"Generating {len(names)} experiment(s) → {config.GENERATED_DIR}")
    if args.sample_n is not None:
        print(f"Sampling: {args.sample_n} stimuli/experiment (seed={args.sample_seed})")
    total_new = 0
    for n in names:
        print(f"\n{'─' * 60}\n{n}")
        try:
            total_new += _generate_one(n, sample_n=args.sample_n, sample_seed=args.sample_seed)
        except Exception as exc:
            print(f"  [ERROR] {n}: {exc}")

    print(f"\nDone. {total_new} new rows written across {len(names)} experiment(s).")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """``pitchbench evaluate`` — run one model on pre-generated data."""
    names = _resolve_experiments(args.experiments or [])
    if not names:
        sys.exit("No experiments matched.")

    if getattr(args, "dataset_dir", None):
        from pitchbench.baselines.evaluation import _prepare_official_dataset
        import pyarrow.parquet as pq
        if not args.dataset_revision:
            sys.exit("--dataset-dir requires --dataset-revision")
        counts = {
            name: {"experiment_id": _NAME_RE.match(name).group(1),
                   "sampled": pq.read_metadata(args.dataset_dir / name / "test-00000-of-00001.parquet").num_rows}
            for name in names
        }
        dataset_config = {
            "input_dataset": {
                "repository": "vaclis/PitchBench", "revision": args.dataset_revision,
                "local_dir": str(args.dataset_dir.resolve()), "split": "test",
                "parquet_filename": "test-00000-of-00001.parquet",
                "embedded_audio_field": "audio", "verify_embedded_audio": True,
                "write_shard_sha256": True, "preserve_audio_names": True,
            },
            "benchmark": {"expected_condition_count": sum(c["sampled"] for c in counts.values())},
        }
        _prepare_official_dataset(dataset_config, config._PROJECT_ROOT, counts)
        if getattr(args, "reference_answers", None):
            import gzip
            import json
            import pandas as pd
            for name in names:
                paths = list((args.reference_answers / name).glob("*/results_*.json*"))
                if len(paths) != 1:
                    raise ValueError(f"Expected one reference answer file for {name}")
                path = paths[0]
                with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
                    order = [Path(r["wav"]).name for r in json.load(handle)["results"]]
                parquet = config.GENERATED_DIR / name / "_questions.parquet"
                frame = pd.read_parquet(parquet)
                identities = frame["audio_path"].map(lambda p: Path(p).name)
                if len(set(order)) != len(order) or set(order) != set(identities) or len(frame) != len(order):
                    raise ValueError(f"Reference and official stimulus identities differ: {name}")
                frame.index = identities
                frame.loc[order].reset_index(drop=True).to_parquet(parquet, index=False)

    model_name = args.model
    model_label = args.name or model_slug(model_name)

    # Build the run directory: results/evaluation/<model_label>/<run_datetime>/
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        config._PROJECT_ROOT / "results" / "evaluation" / model_label / ts
        if not args.run_name
        else config._PROJECT_ROOT / "results" / "evaluation" / model_label / args.run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Point engine + config at the generated data.
    config.AUDIO_DIR   = config.GENERATED_DIR
    config.RESULTS_DIR = run_dir

    sample_info: dict[str, Any] = {"sample_n": args.sample_n, "sample_seed": args.sample_seed}
    if args.experiments == ["paper"] or getattr(args, "paper_formats", False) or getattr(args, "dataset_dir", None):
        sample_info["pitch_formats"] = ("midi", "spn", "hz")

    print(f"Evaluating {len(names)} experiment(s) with model: {model_name}")
    if args.name:
        print(f"Model label  : {model_label}")
    if args.sample_n is not None:
        print(f"Sampling: {args.sample_n} stimuli/experiment (seed={args.sample_seed})")
    print(f"Results → {run_dir}\n")

    all_runs:      dict[str, dict[str, Any]] = {}
    per_exp_costs: dict[str, dict[str, Any]] = {}

    for n in names:
        cost_tracker.reset()
        result = _evaluate_one(n, model_name, run_dir, sample_info, model_label=model_label)
        per_exp_costs[n] = cost_tracker.all_totals()
        if result is not None:
            all_runs[n] = result

    # ── Post-run aggregation ─────────────────────────────────────────────────
    if args.experiments == ["paper"] and set(all_runs) != set(names):
        missing = sorted(set(names) - set(all_runs))
        sys.exit("Paper evaluation is incomplete; refusing to report a partial overall. "
                 "Missing: " + ", ".join(missing))
    overall_dir = run_dir / "overall"
    overall_dir.mkdir(exist_ok=True)
    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_path = aggregate_session_metrics(
        run_dir, overall_dir / f"all_metrics_aggregate_{ts2}.csv"
    )
    if metrics_path:
        print(f"All scalar metrics → {metrics_path}")

    _run_a1_plots(run_dir, overall_dir)
    _run_eval_analysis(run_dir, overall_dir)

    eval_root   = run_dir.parent.parent          # results/evaluation/
    summary_dir = eval_root / "summary"
    _run_cross_model_analysis(eval_root, summary_dir)

    results_run = combine_accuracies_csvs(run_dir, overall_dir / "aggregated_results.csv")
    if results_run:
        print(f"aggregated_results.csv → {results_run}")

    cost_path = write_session_cost_summary(run_dir / "_sessions" / ts, per_exp_costs)
    if cost_path:
        print(f"\nSession cost summary → {cost_path}")
        print(cost_path.read_text())


def _inject_analysis_config() -> None:
    """Override config pitchbench_* variables with analysis_config values."""
    import pitchbench.configs.analysis_config as _ac
    for _n in dir(_ac):
        if _n.startswith("pitchbench_"):
            setattr(config, _n, getattr(_ac, _n))
    if hasattr(_ac, "SAMPLING"):
        config.EXPERIMENT_DEFAULTS = _ac.SAMPLING  # type: ignore[assignment]


def _run_a1_plots(run_dir: Path, overall_dir: Path) -> None:
    """Run analyze_a1 line plots and heatmaps if A1 results are present."""
    try:
        from pitchbench.analysis.a1 import (  # type: ignore[import]
            FORMATS, extract_a1_data, compute_stats, compute_l1_scores,
            plot_a1_results, plot_a1_heatmap, save_summary_csv, save_l1_csv,
        )
    except ImportError as exc:
        print(f"  [SKIP] a1 plots (missing dependency): {exc}")
        return

    data = extract_a1_data(str(run_dir))
    if not any(data[fmt] for fmt in FORMATS):
        return

    # Write a1-specific outputs into the a1 experiment subdirectory.
    a1_dirs = sorted(run_dir.glob("pitchbench_a1_*/"))
    a1_dir = a1_dirs[0] if a1_dirs else overall_dir

    print("\nA1 analysis plots:")
    stats    = compute_stats(data)
    l1_scores = compute_l1_scores(stats)
    save_summary_csv(stats,    str(a1_dir / "a1_summary.csv"))
    save_l1_csv(l1_scores, stats, str(a1_dir / "a1_l1.csv"))
    plot_a1_results(
        stats, data, l1_scores,
        output_file=str(a1_dir / "a1_line_plots.png"),
    )
    plot_a1_heatmap(data, output_file=str(a1_dir / "a1_heatmaps.png"))


def _run_eval_analysis(run_dir: Path, overall_dir: Path) -> None:
    """Aggregate accuracies, write summary, and produce per-model plots."""
    agg_path = aggregate_run_accuracies(run_dir, overall_dir / "aggregated_accuracies.csv")
    if not agg_path:
        print("  No accuracies CSVs found — skipping analysis")
        return
    print(f"\nAggregated accuracies → {agg_path}")

    summary_path = write_model_summary_csv(agg_path, overall_dir / "summary.csv")
    print(f"Summary               → {summary_path}")

    try:
        from pitchbench.analysis.overview import (  # type: ignore[import]
            plot_accuracy_by_instrument,
            plot_accuracy_by_notation,
            plot_accuracy_by_note,
        )
    except ImportError:
        print("  matplotlib/numpy not available — skipping plots")
        return

    _, plot_inst = plot_accuracy_by_instrument(agg_path, output_dir=overall_dir)
    print(f"By instrument         → {plot_inst}")
    _, plot_nota = plot_accuracy_by_notation(agg_path, output_dir=overall_dir)
    print(f"By notation           → {plot_nota}")
    _, plot_note = plot_accuracy_by_note(agg_path, output_dir=overall_dir)
    print(f"By note               → {plot_note}")


def _run_cross_model_analysis(eval_root: Path, summary_dir: Path) -> None:
    """Combine the latest run from every model under eval_root and re-run plots."""
    try:
        from pitchbench.analysis.overview import (  # type: ignore[import]
            aggregate_all_models,
            plot_accuracy_by_instrument,
            plot_accuracy_by_notation,
            plot_accuracy_by_note,
        )
    except ImportError:
        print("  matplotlib/numpy not available — skipping cross-model summary")
        return

    agg_path = aggregate_all_models(eval_root, summary_dir)
    if agg_path is None:
        return

    print(f"\nCross-model summary   → {agg_path}")
    _, plot_inst = plot_accuracy_by_instrument(agg_path, output_dir=summary_dir)
    print(f"By instrument (all)   → {plot_inst}")
    _, plot_nota = plot_accuracy_by_notation(agg_path, output_dir=summary_dir)
    print(f"By notation  (all)    → {plot_nota}")
    _, plot_note = plot_accuracy_by_note(agg_path, output_dir=summary_dir)
    print(f"By note      (all)    → {plot_note}")


def _run_analysis_tables(overall_dir: Path, analysis_results_dir: Path) -> None:
    """Write ablation CSV and line plots into overall_dir after an analysis run."""
    try:
        from pitchbench.analysis.ablation import (
            write_ablation_csv, plot_ablation_lines, plot_format_lines,
        )
    except ImportError as exc:
        print(f"  [SKIP] ablation outputs (missing dependency): {exc}")
        return
    
    print("\nAblation summary CSV:")
    combined_csv = overall_dir / "aggregated_accuracies.csv"
    if not combined_csv.exists():
        print("  [SKIP] ablation CSV: aggregated_accuracies.csv not found in output dir")
    else:
        write_ablation_csv(out_dir=overall_dir, combined_csv=combined_csv)

    print("Ablation line plots:")
    try:
        plot_ablation_lines(analysis_dir=analysis_results_dir, out_dir=overall_dir)
    except SystemExit:
        print("  [SKIP] ablation lines: no models found")
    print("Format line plots:")
    try:
        plot_format_lines(analysis_dir=analysis_results_dir, out_dir=overall_dir)
    except SystemExit:
        print("  [SKIP] format lines: no models found")


def cmd_analyze(args: argparse.Namespace) -> None:
    """``pitchbench analyze`` — analysis mode (analysis_config), preset-driven evaluation."""
    preset_name = args.preset

    _inject_analysis_config()
    # Route engine output and dataset I/O to data/analysis/ (separate from the
    # full benchmark data in data/generated/).
    config.AUDIO_DIR     = config.ANALYSIS_DIR
    config.GENERATED_DIR = config.ANALYSIS_DIR

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug        = model_slug(args.model)
    model_label = args.name or slug
    run_dir = (
        config._PROJECT_ROOT / "results" / "analysis" / model_label / ts
        if not args.run_name
        else config._PROJECT_ROOT / "results" / "analysis" / model_label / args.run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR = run_dir

    # Apply preset overrides (config attribute patching).
    known_names = discover()
    if args.experiments:
        # User specified explicit experiments; ignore preset experiment list.
        names = _resolve_experiments(args.experiments)
    else:
        preset_exp_names = _apply_analysis_preset(preset_name)
        if preset_exp_names is None:
            sys.exit(f"Unknown analysis preset '{preset_name}'. "
                     f"Available: {sorted(_analysis_presets().keys())}")
        unknown = [n for n in preset_exp_names if n not in known_names]
        if unknown:
            sys.exit(f"Preset '{preset_name}' references unknown experiment(s): {unknown}")
        names = preset_exp_names

    model_name  = args.model
    sample_info: dict[str, Any] | None = (
        {"sample_n": args.sample_n, "sample_seed": args.sample_seed}
        if args.sample_n is not None else None
    )

    print(f"Analysis mode → preset={preset_name}  model={model_name}")
    if args.name:
        print(f"Model label   : {model_label}")
    print(f"Experiments   : {', '.join(names)}")
    if sample_info:
        print(f"Sampling      : {args.sample_n} stimuli/experiment (seed={args.sample_seed})")
    print(f"Results       → {run_dir}\n")

    # Auto-generate analysis stimuli into data/analysis/ for any missing conditions.
    print(f"Generating analysis stimuli → {config.ANALYSIS_DIR}")
    for n in names:
        try:
            new_rows = _generate_one(n)
            if new_rows:
                print(f"  {n}: {new_rows} new rows written")
        except Exception as exc:
            print(f"  [WARN] {n}: generate failed — {exc}")
    print()

    all_runs:      dict[str, dict[str, Any]] = {}
    per_exp_costs: dict[str, dict[str, Any]] = {}

    for n in names:
        cost_tracker.reset()
        result = _evaluate_one(n, model_name, run_dir, sample_info, model_label=model_label, filter_conditions=True)
        per_exp_costs[n] = cost_tracker.all_totals()
        if result is not None:
            all_runs[n] = result

    overall_dir = run_dir / "overall"
    overall_dir.mkdir(exist_ok=True)
    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_path = aggregate_session_metrics(
        run_dir, overall_dir / f"all_metrics_aggregate_{ts2}.csv"
    )
    analysis_results_dir = run_dir.parent.parent
    _run_a1_plots(run_dir, overall_dir)
    _run_eval_analysis(run_dir, overall_dir)
    if metrics_path:
        print(f"All scalar metrics → {metrics_path}")
    _run_analysis_tables(overall_dir, analysis_results_dir)

    cost_path = write_session_cost_summary(run_dir / "_sessions" / ts, per_exp_costs)
    if cost_path:
        print(f"\nSession cost summary → {cost_path}")
        print(cost_path.read_text())


# ── --list ───────────────────────────────────────────────────────────────────

def _cmd_list() -> None:
    for n in discover():
        m = _NAME_RE.match(n)
        ident = f"[{m.group(1)}]" if m else "    "
        print(f"  {ident:>6}  {n}")
    preset_names = sorted(_analysis_presets().keys())
    if preset_names:
        print("\nAnalysis presets:")
        for p in preset_names:
            print(f"  [preset]  {p}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PitchBench CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available experiments and presets, then exit.",
    )

    sub = parser.add_subparsers(dest="command")

    # ── generate ─────────────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Generate audio + dataset files")
    gen.add_argument(
        "experiments", nargs="*",
        help="Experiment ID(s) (e.g. a1 b3), category letter(s) (e.g. a b), "
             "full module name(s), or 'all' (default when omitted).",
    )
    gen.add_argument("--sample-n",    type=int, default=None, dest="sample_n",
                     metavar="N", help="Generate only N stimuli per experiment (stratified).")
    gen.add_argument("--sample-seed", type=int, default=42,   dest="sample_seed",
                     metavar="SEED")

    # ── evaluate ─────────────────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="Run one model on pre-generated data")
    ev.add_argument("experiments", nargs="*")
    ev.add_argument("--model", default=config.DEFAULT_LOCAL_URL, metavar="MODEL",
                    help="openrouter/<id> or local server URL (default: http://localhost:8001)")
    ev.add_argument("--name", default=None, metavar="NAME",
                    help="Friendly name for the model (used for directory and CSV labels).")
    ev.add_argument("--run-name", default=None, dest="run_name", metavar="NAME",
                    help="Override the run directory name (default: timestamp).")
    ev.add_argument("--sample-n", type=int, default=None, dest="sample_n", metavar="N",
                    help="Evaluate only N stimuli per experiment (stratified).")
    ev.add_argument("--sample-seed", type=int, default=42, dest="sample_seed", metavar="SEED")
    ev.add_argument("--dataset-dir", type=Path, help="Official HF local-dir with embedded audio and revision metadata.")
    ev.add_argument("--reference-answers", type=Path, help="Saved model answer root; verify and match stimulus order before querying.")
    ev.add_argument("--dataset-revision", help="Required immutable revision for --dataset-dir.")
    ev.add_argument("--paper-formats", action="store_true", help="Use MIDI/SPN/Hz only for selected experiments.")
    ev.add_argument("--parallel-experiments", type=int, default=1, dest="parallel",
                    metavar="N", help="Number of experiments to evaluate in parallel.")

    # ── analyze ──────────────────────────────────────────────────────────────
    an = sub.add_parser("analyze", help="Analysis mode (preset-driven evaluation)")
    an.add_argument("experiments", nargs="*",
                    help="Override the preset's experiment list.")
    an.add_argument("--model", default=config.DEFAULT_LOCAL_URL, metavar="MODEL",
                    help="openrouter/<id> or local server URL (default: http://localhost:8001)")
    an.add_argument("--name", default=None, metavar="NAME",
                    help="Friendly name for the model (used for directory and CSV labels).")
    an.add_argument("--preset", default="q1", metavar="PRESET",
                    help="Analysis preset name (default: q1).")
    an.add_argument("--run-name", default=None, dest="run_name", metavar="NAME")
    an.add_argument("--sample-n", type=int, default=None, dest="sample_n", metavar="N",
                    help="Evaluate only N stimuli per experiment (stratified).")
    an.add_argument("--sample-seed", type=int, default=42, dest="sample_seed", metavar="SEED")

    args = parser.parse_args()

    if args.list:
        _cmd_list()
        return

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
