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
    env = {
        "PITCHBENCH_ROOT": str(tmp_path),
        "PITCHBENCH_BASELINE_CONFIG": str(tmp_path / "config.yaml"),
    }
    config = {
        "benchmark": {"experiments": ["f2"], "expected_condition_count": 2},
        "input_dataset": {"revision": "fixed"},
    }
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
            (
                "branch",
                "-r",
                "--contains",
                "abc123",
            ): "origin/rebuttal-dsp-nn-baselines",
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
    config = {
        "runtime": {
            "runtime_root": str(tmp_path / "run"),
            "persistent_results_dir": str(tmp_path / "results"),
            "asset_data_dir": str(assets),
        }
    }
    evaluation._prepare_runtime(config)
    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path / "different-checkout")
    root, _ = evaluation._prepare_runtime(config)
    assert (root / "data/preloaded").resolve() == assets / "preloaded"


def test_standalone_preflight_accepts_clean_local_and_detached_commits(
    tmp_path, monkeypatch
):
    """Public reproduction needs neither a remote nor a VacLab installation."""
    import subprocess
    import pytest

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "fixture",
    )
    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("VACLAB_RUN_DIR", raising=False)
    head = git("rev-parse", "HEAD")
    assert evaluation._formal_preflight()["commit"] == head
    git("checkout", "--detach", "-q", head)
    receipt = evaluation._formal_preflight()
    assert receipt == {"branch": "detached", "commit": head, "execution": "standalone"}
    (tmp_path / "uncommitted.txt").write_text("dirty")
    with pytest.raises(RuntimeError, match="clean PitchBench tree"):
        evaluation._formal_preflight()


def test_queue_preflight_does_not_fall_back_to_standalone(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("VACLAB_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(
        evaluation,
        "_git",
        lambda *args: {
            ("status", "--porcelain"): "",
            ("branch", "--show-current"): "",
            ("rev-parse", "HEAD"): "abc123",
        }[args],
    )
    with pytest.raises(RuntimeError, match="queue receipt"):
        evaluation._formal_preflight()


def test_environment_check_does_not_prepare_or_evaluate_data(monkeypatch, capsys):
    from argparse import Namespace
    from pathlib import Path

    monkeypatch.setattr(
        evaluation,
        "parse_args",
        lambda: Namespace(config=Path("unused.yaml"), check_environment=True),
    )
    monkeypatch.setattr(
        evaluation, "_load_config", lambda path: {"baselines": {"dsp": {}}}
    )
    monkeypatch.setattr(
        evaluation, "_validate_runtime_environment", lambda config: None
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Environment check must not prepare data or run inference")

    for name in ["_formal_preflight", "_prepare_runtime", "_run"]:
        monkeypatch.setattr(evaluation, name, forbidden)
    evaluation.main()
    assert "no inference" in capsys.readouterr().out


def test_environment_validation_checks_the_complete_requirements_file(
    tmp_path, monkeypatch
):
    import sys
    import pytest

    (tmp_path / "requirements.txt").write_text("# frozen\ntransitive-package==2.0\n")
    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path)
    config = {
        "runtime": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "requirements_file": "requirements.txt",
        }
    }
    monkeypatch.setattr(evaluation.importlib.metadata, "version", lambda name: "1.0")
    with pytest.raises(
        RuntimeError, match="transitive-package: expected 2.0, found 1.0"
    ):
        evaluation._validate_runtime_environment(config)
    monkeypatch.setattr(evaluation.importlib.metadata, "version", lambda name: "2.0")
    evaluation._validate_runtime_environment(config)
