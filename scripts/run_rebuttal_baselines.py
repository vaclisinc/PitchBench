#!/usr/bin/env python3
"""Compatibility CLI for preparing and running the rebuttal baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitchbench.baselines.evaluation import main

if __name__ == "__main__":
    main()
