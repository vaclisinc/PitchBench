"""
Experiment runner — entry point for the ``pitchbench`` CLI.

Usage::

    pitchbench a1                               # shorthand for --id a1
    pitchbench pitchbench_a1_pitch_id           # run by full module name
    pitchbench --id a1                          # run by category+digit ID
    pitchbench a1 --preview                     # generate stimuli, skip queries
    pitchbench --id a1 --download               # generate audio files for a1
    pitchbench --download                       # generate audio for ALL experiments
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
from pathlib import Path

import pitchbench.config as config


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
_NAME_RE = re.compile(r"^pitchbench_([a-z]+\d+)_(.+)$")
# Bare experiment-id shorthand, e.g. "a1", "b3" — accepted as a positional arg.
_ID_RE = re.compile(r"^[a-z]+\d+$")


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


def run_experiment(name: str, extra_argv: list[str]) -> None:
    print(f"\n{'#' * 60}")
    print(f"# {name}")
    print(f"{'#' * 60}\n")
    mod = importlib.import_module(f"pitchbench.experiments.scripts.{name}")
    if "--preview" in extra_argv:
        mod.preview()
    else:
        mod.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", nargs="?", default=None,
        help="Experiment name (e.g. pitchbench_a1_pitch_id) or 'all'",
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

    args, unknown = parser.parse_known_args()

    if args.list:
        for n in discover():
            m = _NAME_RE.match(n)
            ident = f"[{m.group(1)}]" if m else "    "
            print(f"  {ident:>6}  {n}")
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
    #   "pitchbench_a1_pitch_id"   → full module name
    #   "all"                      → every experiment
    exp_id = args.exp_id
    if exp_id is None and args.experiment and _ID_RE.match(args.experiment.lower()):
        exp_id = args.experiment.lower()

    known_names: list[str] = discover()

    if exp_id is not None:
        name = _id_to_name(exp_id)
        if name is None:
            ids = [_NAME_RE.match(n).group(1) for n in known_names if _NAME_RE.match(n)]
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

    if name == "all":
        for n in discover():
            run_experiment(n, extra)
    else:
        run_experiment(name, extra)


if __name__ == "__main__":
    main()
