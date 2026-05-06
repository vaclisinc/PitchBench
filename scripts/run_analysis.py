"""
Run a named analysis preset or user-config overrides.

Usage
-----
# Run all 10 Q1 ablation experiments (MIDI-only, sine source):
    python scripts/run_analysis.py q1

# With a specific model:
    python scripts/run_analysis.py q1 --models audio_flamingo_next_instruct

# Preview only (generate stimuli, skip model queries):
    python scripts/run_analysis.py q1 --preview

# Apply user_config.py overrides on top of a preset:
    python scripts/run_analysis.py q1 --user-config

# Run user_config.py alone (no preset), using the experiments list inside it:
    python scripts/run_analysis.py --user-config

How it works
------------
Experiment scripts capture config values at *module import time*
(e.g. ``LOUDNESS_DB = config.pitchbench_a2_LOUDNESS_DB``).  To make
runtime overrides visible, this script:
  1. Patches the relevant ``pitchbench.config`` attributes.
  2. Pops the experiment's cached module from ``sys.modules`` so the next
     ``importlib.import_module`` does a fresh import that re-captures
     the patched values.
  3. Repeats for each experiment in the preset.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Force analysis config selection before pitchbench.config initializes.
os.environ["PITCHBENCH_MODE"] = "ANALYSIS"

# Config must be imported before analysis_config / user_config so that
# BENCHMARK_* constants are available to those modules at import time.
import pitchbench.config as config
from pitchbench.analysis_config import PRESETS
from pitchbench.experiments.run import run_experiment


def _apply_overrides(overrides: dict) -> None:
    for attr, val in overrides.items():
        setattr(config, attr, val)


def _disable_sampling(exp_names: list[str]) -> None:
    """Set per_stratum=None for each experiment so all conditions are run.

    The benchmark EXPERIMENT_DEFAULTS normally applies a per-stratum cap even
    without --sample-n.  Analysis runs want the full condition grid.
    """
    for name in exp_names:
        entry = config.EXPERIMENT_DEFAULTS.get(name)
        if entry is not None:
            entry = dict(entry)
            entry["per_stratum"] = None
            config.EXPERIMENT_DEFAULTS[name] = entry


def _fresh_run(exp_name: str, extra: list[str]) -> None:
    """Pop cached module, run experiment with fresh import, then clean up."""
    key = f"pitchbench.experiments.scripts.{exp_name}"
    sys.modules.pop(key, None)
    run_experiment(exp_name, extra)
    sys.modules.pop(key, None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "preset", nargs="?", default=None, choices=list(PRESETS),
        help="Named analysis preset (e.g. q1).  Omit when using --user-config alone.",
    )
    parser.add_argument(
        "--user-config", action="store_true",
        help="Load src/pitchbench/user_config.py and apply its USER_OVERRIDES "
             "on top of the preset (or use it as the sole config when no preset "
             "is given).",
    )
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--preview", action="store_true",
                        help="Generate stimuli only, skip model queries.")
    parser.add_argument(
        "--run-name", default=None, metavar="NAME",
        help="Override the results subdirectory name.  Defaults to the preset "
             "name, or 'user_config' when only --user-config is used.",
    )

    args, extra_argv = parser.parse_known_args()

    if args.preset is None and not args.user_config:
        parser.error("Provide a preset name (e.g. q1) or --user-config (or both).")

    # ── Resolve the run name ──────────────────────────────────────────────
    run_name = args.run_name or (args.preset if args.preset else "user_config")
    config.RESULTS_DIR = config._PROJECT_ROOT / "results" / "analysis" /run_name

    # ── Apply preset ──────────────────────────────────────────────────────
    experiments: list[str]
    if args.preset is not None:
        preset = PRESETS[args.preset]
        config.pitchbench_general_NOTATION_FORMATS = preset["notation_formats"]
        for exp_overrides in preset["experiments"].values():
            _apply_overrides(exp_overrides)
        experiments = list(preset["experiments"].keys())
    else:
        experiments = []

    # ── Apply user_config on top (wins on overlapping keys) ───────────────
    if args.user_config:
        from pitchbench.user_config import (  # noqa: PLC0415
            EXPERIMENTS as USER_EXPERIMENTS,
            NOTATION_FORMATS as USER_FORMATS,
            USER_OVERRIDES,
        )
        config.pitchbench_general_NOTATION_FORMATS = USER_FORMATS
        _apply_overrides(USER_OVERRIDES)
        if not experiments:
            experiments = USER_EXPERIMENTS

    # ── Disable default per-stratum sampling for analysis experiments ─────
    # Benchmark EXPERIMENT_DEFAULTS applies a per-stratum cap even without
    # --sample-n.  Analysis presets want the full condition grid.
    _disable_sampling(experiments)

    # ── Build forwarded args ──────────────────────────────────────────────
    extra: list[str] = list(extra_argv)
    if args.models:
        extra = ["--models"] + args.models + extra
    if args.preview:
        extra = ["--preview"] + extra

    # ── Run each experiment ───────────────────────────────────────────────
    for exp_name in experiments:
        _fresh_run(exp_name, extra)


if __name__ == "__main__":
    main()
