import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rebuttal_baselines.py"
NAMESPACE = runpy.run_path(str(SCRIPT))


def test_official_row_aliases_preserve_scorer_condition() -> None:
    condition = NAMESPACE["_condition_from_official_row"](
        "d4",
        {
            "audio": {"bytes": b"RIFF", "path": "x.wav"},
            "prompt": "prompt",
            "gt_seq": ["down", "up"],
            "source": "sine",
            "start_midi": 60,
        },
    )
    assert condition == {
        "source": "sine",
        "source_type": "waveform",
        "start_midi": 60,
        "gt_seq": ["down", "up"],
    }


def test_official_b5_timestamps_are_flattened_in_order() -> None:
    timestamps = NAMESPACE["_timing_ground_truth"](
        "b5",
        {
            "duration_ms": 500,
            "onsets_ms": [100, 1000],
        },
    )
    assert timestamps == [0.1, 0.6, 1.0, 1.5]
