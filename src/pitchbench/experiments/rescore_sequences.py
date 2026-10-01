"""Compatibility entrypoint for the canonical, complete paper replay.

Use ``python -m pitchbench.analysis.replay --help`` for current arguments.
The old three-task replacement of a legacy overall table has been retired.
"""

import sys

from pitchbench.analysis.replay import main as replay_main


def main() -> None:
    print(
        "rescore_sequences now delegates to pitchbench.analysis.replay. "
        "Supply --dataset-dir for the fixed official dataset; "
        "add --baseline-input for all eight models.",
        file=sys.stderr,
    )
    replay_main()


if __name__ == "__main__":
    main()
