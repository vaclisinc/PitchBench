"""
Experiment runner — entry point for the ``pitchbench`` CLI.

Usage::

    pitchbench a1                               # shorthand for --id a1
    pitchbench pitchbench_a1_single_pitch_id           # run by full module name
    pitchbench --id a1                          # run by category+digit ID
    pitchbench a1 --preview                     # generate stimuli, skip queries
    pitchbench --id a1 --download               # generate audio files for a1
    pitchbench --download                       # generate audio for ALL experiments
    pitchbench a                                # run every experiment in category 'a'
    pitchbench a --download                     # generate audio for every 'a' experiment
    pitchbench q1                               # run analysis preset by name
    pitchbench all                              # run every experiment
    pitchbench --list                           # list available experiments

    # Random sub-sample (stratified, deterministic):
    pitchbench --id a1 --sample-n 20 --sample-seed 42 --preview
    pitchbench --id a1 --sample-n 20

Extra flags (e.g. --models, --seed, --sample-n) are forwarded to the
experiment module.
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
from pitchbench.experiments.helpers import cost as cost_tracker
from pitchbench.experiments.helpers.results import (
    aggregate_session_metrics, write_aggregate_format_accuracies,
    write_session_cost_summary,
)


def _prompt_for_models() -> list[str]:
    """Ask the user which model to run; default to config.DEFAULT_MODEL on Enter.

    Returns a list with one model slug (matching the --models nargs='+' shape).
    Falls back to [DEFAULT_MODEL] without prompting when stdin isn't a TTY.
    """
    default = config.DEFAULT_MODEL
    if not sys.stdin.isatty():
        return [default]
    try:
        choice = input(f"Model? [{default}]: ").strip()
    except EOFError:
        choice = ""
    return [choice] if choice else [default]

# pitchbench_<id>_<desc>.py — capture the id (lowercase letter + digit(s))
_NAME_RE = re.compile(r"^pitchbench_([a-z]+\d+[a-z]*)_(.+)$")
# Bare experiment-id shorthand, e.g. "a1", "b3" — accepted as a positional arg.
_ID_RE = re.compile(r"^[a-z]+\d+$")
# Bare category prefix, e.g. "a", "b" — runs every experiment in that category.
_CAT_RE = re.compile(r"^[a-z]$")


def _scripts_dir() -> Path:
    return Path(__file__).parent / "scripts"


def discover() -> list[str]:
    """Return all v2 experiment module names, sorted by their letter+digit ID."""
    names = [p.stem for p in _scripts_dir().glob("pitchbench_*.py")]
    return sorted(names, key=_sort_key)


def _sort_key(name: str) -> tuple:
    m = _NAME_RE.match(name)
    if not m:
        return (chr(127), 0, name)               # unknown → end of list
    ident = m.group(1)
    cat   = ident[0]
    try:
        num = int(ident[1:])
    except ValueError:
        num = 0
    return (cat, num, name)


def _id_to_name(exp_id: str) -> str | None:
    """Resolve a letter+digit id (e.g. ``"a1"``) to its full module name."""
    exp_id = exp_id.lower()
    for n in discover():
        m = _NAME_RE.match(n)
        if m and m.group(1) == exp_id:
            return n
    return None


def _category_to_names(cat: str) -> list[str]:
    """Return all experiment module names whose ID starts with ``cat``."""
    cat = cat.lower()
    out: list[str] = []
    for n in discover():
        m = _NAME_RE.match(n)
        if m and m.group(1).startswith(cat):
            out.append(n)
    return out


def _analysis_presets() -> dict[str, Any]:
    """Best-effort load of analysis preset definitions."""
    try:
        analysis_config = importlib.import_module("pitchbench.analysis_config")
    except Exception:
        return {}
    presets = getattr(analysis_config, "PRESETS", None)
    return presets if isinstance(presets, dict) else {}


def _apply_analysis_preset(preset_name: str) -> list[str] | None:
    """Apply config overrides for an analysis preset; return experiment names."""
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


def _run_experiment_worker(
    args: tuple[str, list[str]],
) -> tuple[str, dict | None, dict]:
    """Worker for ProcessPoolExecutor: runs one experiment in an isolated process."""
    name, extra = args
    from pitchbench.experiments.helpers import cost as _cost  # process-local copy
    _cost.reset()
    try:
        result = run_experiment(name, extra)
    except Exception as exc:
        print(f"\n[ERROR] Experiment {name} failed with error:\n{exc}\n", flush=True)
        return name, None, _cost.all_totals()
    return name, result, _cost.all_totals()


def run_experiment(name: str, extra_argv: list[str]) -> dict[str, dict[str, Any]] | None:
    """Run one experiment; return its per-model format accuracies (or None for preview)."""
    print(f"\n{'#' * 60}")
    print(f"# {name}")
    print(f"{'#' * 60}\n")
    mod = importlib.import_module(f"pitchbench.experiments.scripts.{name}")
    if "--preview" in extra_argv:
        mod.preview()
        return None
    return mod.run()


def _launch_servers(launch_args: list[str]) -> None:
    """Dispatch `pitchbench launch ...` to the model server launcher."""
    from pitchbench.model import serve_all

    old_argv = sys.argv
    try:
        sys.argv = [f"{old_argv[0]} launch", *launch_args]
        serve_all.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", nargs="?", default=None,
        help="Experiment name (e.g. pitchbench_a1_single_pitch_id) or 'all'",
    )
    parser.add_argument(
        "--id", dest="exp_id", type=str, default=None, metavar="ID",
        help="Experiment letter+digit ID (e.g. a1, b3, e2).",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Generate stimuli, skip model queries",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Generate (download) every audio file for the experiment(s) without "
             "querying any model. With no experiment specified, generates audio for "
             "every experiment.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available experiments and exit",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Model(s) to query (forwarded to the experiment module)",
    )
    parser.add_argument(
        "--sample-n", type=int, default=None, metavar="N",
        help="Draw N stimuli per experiment (stratified; forwarded to the experiment module)",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42, metavar="SEED",
        help="RNG seed for --sample-n (default: 42; forwarded to the experiment module)",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, metavar="NAME",
        help="Override the results subdirectory name (e.g. 'pilot_01'). "
             "Results land in results/<NAME>/. Also settable via PITCHBENCH_RUN env var.",
    )
    parser.add_argument(
        "--parallel-experiments", type=int, default=1, metavar="N",
        dest="LOCAL_CONCURRENCY",
        help="Number of experiments to run in parallel (default: 1 = sequential). "
             "Each experiment runs in its own process so cost tracking stays isolated.",
    )

    args, unknown = parser.parse_known_args()

    # Dedicated model-launch path:
    #   pitchbench launch                          -> launch all servers
    #   pitchbench launch audio_flamingo_next_*    -> launch one server
    #   pitchbench launch --only ... --port ...    -> forwarded to serve_all
    # Handle this before model-prompt logic so it never asks "Model?".
    if args.experiment == "launch":
        launch_args: list[str] = []
        if args.list:
            launch_args.append("--list")
        if unknown:
            if unknown[0].startswith("-"):
                launch_args.extend(unknown)
            else:
                launch_args.extend(["--only", unknown[0]])
                launch_args.extend(unknown[1:])
        _launch_servers(launch_args)
        return

    if args.run_name:
        config.RESULTS_DIR = config._PROJECT_ROOT / "results" / args.run_name

    if args.list:
        for n in discover():
            m = _NAME_RE.match(n)
            ident = f"[{m.group(1)}]" if m else "    "
            print(f"  {ident:>6}  {n}")
        preset_names = sorted(_analysis_presets().keys())
        if preset_names:
            print("\nAnalysis presets:")
            for p in preset_names:
                print(f"  [preset]  {p}")
        return

    # --download is a stimulus-only mode (no model queries) that also defaults to
    # "all experiments" when no specific experiment is given. Internally it routes
    # through each module's preview() — same as --preview — since preview() is the
    # standard "generate stimuli, skip queries" entry point in every script.
    stimulus_only = args.preview or args.download

    # If the user didn't pass --models, prompt with DEFAULT_MODEL (Enter accepts it).
    # Stimulus-only modes skip the prompt since no model is queried in that path.
    if args.models:
        models = args.models
    elif stimulus_only or args.list:
        models = []
    else:
        models = _prompt_for_models()

    models_extra  = (["--models"] + models) if models else []
    sample_extra  = (
        ["--sample-n", str(args.sample_n)] if args.sample_n is not None else []
    ) + (
        ["--sample-seed", str(args.sample_seed)] if args.sample_n is not None else []
    )
    extra = (["--preview"] if stimulus_only else []) + models_extra + sample_extra + unknown

    # Resolve which experiment to run. Three accepted forms for the positional:
    #   "a1"                       → letter+digit shorthand (resolved via _id_to_name)
    #   "pitchbench_a1_single_pitch_id"   → full module name
    #   "all"                      → every experiment
    exp_id = args.exp_id
    if exp_id is None and args.experiment and _ID_RE.match(args.experiment.lower()):
        exp_id = args.experiment.lower()

    known_names: list[str] = discover()
    preset_names: list[str] = sorted(_analysis_presets().keys())

    # Category prefix: "a", "b", … runs every experiment in that category.
    category: str | None = None
    if args.experiment and _CAT_RE.match(args.experiment.lower()):
        category = args.experiment.lower()
    elif args.exp_id and _CAT_RE.match(args.exp_id.lower()):
        category = args.exp_id.lower()

    preset_name: str | None = None
    if args.experiment and args.experiment in preset_names:
        preset_name = args.experiment
    elif exp_id is not None and exp_id in preset_names:
        preset_name = exp_id

    if preset_name is not None:
        preset_exp_names = _apply_analysis_preset(preset_name)
        if preset_exp_names is None:
            parser.error(f"Unknown analysis preset {preset_name!r}.")
        unknown_in_preset = [n for n in preset_exp_names if n not in known_names]
        if unknown_in_preset:
            parser.error(
                f"Analysis preset {preset_name!r} references unknown experiment(s): "
                f"{unknown_in_preset}"
            )
        name = preset_exp_names
    elif category is not None:
        cat_names = _category_to_names(category)
        if not cat_names:
            parser.error(f"No experiments found for category {category!r}.")
        name = cat_names
    elif exp_id is not None:
        name = _id_to_name(exp_id)
        if name is None:
            ids = [_NAME_RE.match(n).group(1) for n in known_names if _NAME_RE.match(n)]
            if preset_names:
                parser.error(
                    f"Unknown experiment ID {exp_id!r}. Available IDs: {ids}. "
                    f"Available presets: {preset_names}"
                )
            parser.error(f"Unknown experiment ID {exp_id!r}. Available: {ids}")
    elif args.experiment and (args.experiment == "all" or args.experiment in known_names):
        name = args.experiment
    elif args.download:
        # Bare `--download` means "download every experiment".
        # If argparse swallowed an unknown-flag value into the `experiment`
        # positional, that token won't match any module name — fall through
        # to "all" rather than reject the request.
        name = "all"
    elif args.experiment:
        # Fallback: pass through the unrecognised name; importlib will surface
        # a clear ModuleNotFoundError so the user sees what was attempted.
        name = args.experiment
    else:
        parser.error("provide an experiment name or --id <letter><digit> (or use --list)")

    if isinstance(name, list) or name == "all":
        names_to_run = discover() if name == "all" else name
        all_runs:      dict[str, dict[str, dict[str, Any]]]   = {}
        per_exp_costs: dict[str, dict[str, dict[str, Any]]]   = {}
        concurrency = max(1, args.LOCAL_CONCURRENCY)
        if concurrency > 1:
            print(f"Running {len(names_to_run)} experiments with LOCAL_CONCURRENCY={concurrency}")
            worker_args = [(n, extra) for n in names_to_run]
            with ProcessPoolExecutor(max_workers=concurrency) as pool:
                for exp_name, result, costs in pool.map(_run_experiment_worker, worker_args):
                    per_exp_costs[exp_name] = costs
                    if result:
                        all_runs[exp_name] = result
        else:
            for n in names_to_run:
                # Reset before each experiment so a crash before make_run_dir doesn't
                # mis-attribute the previous experiment's totals to this one.
                cost_tracker.reset()
                try:
                    result = run_experiment(n, extra)
                except Exception as exc:
                    print(f"\n[ERROR] Experiment {n} failed with error:\n{exc}\n")
                    per_exp_costs[n] = cost_tracker.all_totals()
                    continue
                per_exp_costs[n] = cost_tracker.all_totals()
                if result:
                    all_runs[n] = result

        if all_runs and not stimulus_only:
            ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = config.RESULTS_DIR / f"format_accuracy_aggregate_{ts}.csv"
            write_aggregate_format_accuracies(output_path, all_runs)
            print(f"\nAggregate format accuracies → {output_path}")

        if not stimulus_only:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Full scalar-metric aggregate (every experiment in the session)
            metrics_path = aggregate_session_metrics(
                config.RESULTS_DIR,
                config.RESULTS_DIR / f"all_metrics_aggregate_{ts}.csv",
            )
            if metrics_path is not None:
                print(f"All scalar metrics → {metrics_path}")
            # Per-category focused aggregates (one CSV per non-trivial category)
            ran_categories = sorted({
                m.group(1) for m in (_NAME_RE.match(n) for n in (names_to_run or []))
                if m
            })
            for cat in ran_categories:
                cat_path = aggregate_session_metrics(
                    config.RESULTS_DIR,
                    config.RESULTS_DIR / f"metrics_{cat}_aggregate_{ts}.csv",
                    only_categories={cat},
                )
                if cat_path is not None:
                    print(f"Category-{cat} metrics → {cat_path}")

        if not stimulus_only:
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_d = config.RESULTS_DIR / "_sessions" / ts
            txt_path  = write_session_cost_summary(session_d, per_exp_costs)
            if txt_path is not None:
                print(f"\nSession cost summary → {txt_path}")
                print(txt_path.read_text())
    else:
        run_experiment(name, extra)


if __name__ == "__main__":
    main()
