"""
Experiment runner — entry point for the ``pitchbench`` CLI.

Usage::

    pitchbench pitchbench_a1_pitch_id           # run by full module name
    pitchbench --id a1                          # run by category+digit ID
    pitchbench pitchbench_a1_pitch_id --preview # generate stimuli, skip queries
    pitchbench all                              # run every experiment
    pitchbench --list                           # list available experiments

Extra flags (e.g. --models, --seed) are forwarded to the experiment module.
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path

import pitchbench.config as config

# pitchbench_<id>_<desc>.py — capture the id (lowercase letter + digit(s))
_NAME_RE = re.compile(r"^pitchbench_([a-z]+\d+)_(.+)$")


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
        "--list", action="store_true",
        help="List all available experiments and exit",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Model(s) to query (forwarded to the experiment module)",
    )

    args, unknown = parser.parse_known_args()

    if args.list:
        for n in discover():
            m = _NAME_RE.match(n)
            ident = f"[{m.group(1)}]" if m else "    "
            print(f"  {ident:>6}  {n}")
        return

    models_extra = (["--models"] + args.models) if args.models else []
    extra        = (["--preview"] if args.preview else []) + models_extra + unknown

    if args.exp_id is not None:
        name = _id_to_name(args.exp_id)
        if name is None:
            ids = [_NAME_RE.match(n).group(1) for n in discover() if _NAME_RE.match(n)]
            parser.error(f"Unknown experiment ID {args.exp_id!r}. Available: {ids}")
    elif args.experiment:
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
