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


def test_frozen_counts_do_not_regenerate_audio_conditions(tmp_path, monkeypatch):
    import pandas as pd
    import pytest

    shard = tmp_path / "test.parquet"
    pd.DataFrame({"item": [1, 2]}).to_parquet(shard)
    monkeypatch.setattr(evaluation, "_official_shard_path", lambda *args: shard)
    monkeypatch.setattr(evaluation, "_official_shard_revision", lambda *args: "fixed")
    monkeypatch.setenv("PITCHBENCH_ROOT", str(tmp_path))
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(tmp_path / "config.yaml"))
    env = {"PITCHBENCH_ROOT": str(tmp_path), "PITCHBENCH_BASELINE_CONFIG": str(tmp_path / "config.yaml")}
    config = {"benchmark": {"experiments": ["f2"], "expected_condition_count": 2},
              "input_dataset": {"revision": "fixed"}}
    counts = evaluation._expected_counts(config, env)
    assert counts["pitchbench_f2_melodic_line_tonal"]["sampled"] == 2
    config["benchmark"]["expected_condition_count"] = 3
    with pytest.raises(RuntimeError, match="Expected 3 conditions, got 2"):
        evaluation._expected_counts(config, env)
    config["input_dataset"]["revision"] = "different"
    with pytest.raises(RuntimeError, match="unexpected dataset revision"):
        evaluation._expected_counts(config, env)


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


def test_runtime_assets_survive_checkout_snapshot_changes(tmp_path, monkeypatch):
    assets = tmp_path / "shared-data"
    for name in ("preloaded", "soundfonts"):
        (assets / name).mkdir(parents=True)
    config = {"runtime": {"runtime_root": str(tmp_path / "run"),
                          "persistent_results_dir": str(tmp_path / "results"),
                          "asset_data_dir": str(assets)}}
    evaluation._prepare_runtime(config)
    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path / "different-checkout")
    root, _ = evaluation._prepare_runtime(config)
    assert (root / "data/preloaded").resolve() == assets / "preloaded"
