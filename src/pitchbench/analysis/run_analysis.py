"""
Batch analysis runner: run a named preset across one or more models.

Usage::

    # One model
    python -m pitchbench.analysis.run_analysis q1 \\
        --models openrouter/google/gemini-2.5-flash

    # Multiple models in sequence
    python -m pitchbench.analysis.run_analysis q1 \\
        --models openrouter/google/gemini-2.5-flash \\
                 openrouter/openai/gpt-4o-audio-preview

    # Override run directory name
    python -m pitchbench.analysis.run_analysis q1 \\
        --models openrouter/google/gemini-2.5-flash \\
        --run-name my_run

    # Limit stimuli per experiment
    python -m pitchbench.analysis.run_analysis q1 \\
        --models openrouter/google/gemini-2.5-flash \\
        --sample-n 20 --sample-seed 0

This is a thin wrapper around ``pitchbench analyze`` that handles batching
across multiple models.  For a single model you can also run::

    pitchbench analyze --preset q1 --model <model>
"""

from __future__ import annotations

import argparse
import sys

from pitchbench.experiments.run import _analysis_presets, cmd_analyze


def main() -> None:
    known_presets = list(_analysis_presets().keys())

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "preset",
        choices=known_presets if known_presets else None,
        metavar="PRESET",
        help=f"Analysis preset to run. Available: {known_presets or '(none defined)'}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="One or more model identifiers (openrouter/<slug> or http://localhost:PORT).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Override the results subdirectory name (default: timestamp).",
    )
    parser.add_argument(
        "--name",
        default=None,
        metavar="LABEL",
        help="Human-readable model label used in result paths (default: derived from model URL).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        metavar="N",
        help="Cap total stimuli per experiment (stratified).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        metavar="SEED",
    )

    args = parser.parse_args()

    for model in args.models:
        namespace = argparse.Namespace(
            preset=args.preset,
            model=model,
            name=args.name,
            run_name=args.run_name,
            experiments=[],
            sample_n=args.sample_n,
            sample_seed=args.sample_seed,
        )
        cmd_analyze(namespace)


if __name__ == "__main__":
    main()
