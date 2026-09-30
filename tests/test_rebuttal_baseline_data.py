from pitchbench.baselines import evaluation

NAMESPACE = vars(evaluation)


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


def test_queue_preflight_requires_matching_pushed_snapshot(tmp_path, monkeypatch):
    import json
    import pytest

    monkeypatch.setenv("VACLAB_RUN_DIR", str(tmp_path))
    def git(*args):
        return {
            ("status", "--porcelain"): "",
            ("branch", "--show-current"): "",
            ("rev-parse", "HEAD"): "abc123",
            ("branch", "-r", "--contains", "abc123"): "origin/rebuttal-dsp-nn-baselines",
        }[args]
    monkeypatch.setattr(evaluation, "_git", git)
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"snapshot": True, "git_commit": "abc123"}))
    assert evaluation._formal_preflight()["commit"] == "abc123"
    path.write_text(json.dumps({"snapshot": True, "git_commit": "wrong"}))
    with pytest.raises(RuntimeError, match="does not match"):
        evaluation._formal_preflight()
