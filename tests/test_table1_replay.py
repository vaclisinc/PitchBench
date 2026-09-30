"""Check the audit against independent ground truth and malformed evidence."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pitchbench.analysis import replay


@pytest.fixture
def audit_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "TASKS", {"D4": "Contour"})
    monkeypatch.setattr(replay, "MODELS", {"test_model": "Test"})
    name = "pitchbench_d4_contour_continuous"
    dataset = tmp_path / "dataset"
    folder = dataset / name
    folder.mkdir(parents=True)
    official = dict(
        source="sine",
        start_midi=60,
        end_midi=62,
        traj_name="up",
        interval_st=2,
        gt_seq=["up"],
        audio={"path": "official.wav"},
    )
    pd.DataFrame([official]).to_parquet(folder / "test-00000-of-00001.parquet")
    meta = dataset / ".cache/huggingface/download" / name
    meta.mkdir(parents=True)
    (meta / "test-00000-of-00001.parquet.metadata").write_text(replay.REVISION + "\n")
    result = tmp_path / "paper/evaluation/_test_model" / name / "run"
    result.mkdir(parents=True)
    # D4 legacy filenames differ from the official release: match conditions.
    saved = dict(
        source="sine",
        start_midi=60,
        end_midi=62,
        traj_name="up",
        interval_st=2,
        trajectory_gt="up",
        raw_response="up, down",
        trajectory_correct=1,
        wav="legacy.wav",
    )
    path = result / "results_test_model.json"
    path.write_text(json.dumps({"results": [saved]}))
    monkeypatch.setattr(
        replay.subprocess, "check_output", lambda *a, **k: "test-commit\n"
    )
    return tmp_path, dataset, path, saved


def test_replay_scores_answers_instead_of_trusting_saved_flags(audit_fixture):
    root, dataset, path, row = audit_fixture
    out = root / "out"
    replay.replay(root, dataset, out)
    metrics = pd.read_csv(out / "metrics.csv")
    assert metrics.iloc[0]["score"] == 0
    receipt = json.loads((out / "run.json").read_text())
    assert receipt["score_mismatches"] == 1
    assert receipt["missing_responses"] == []


@pytest.mark.parametrize(
    "corruption", ["duplicate", "ground_truth", "unknown_identity"]
)
def test_replay_rejects_invalid_raw_evidence(audit_fixture, corruption):
    root, dataset, path, row = audit_fixture
    rows = [row]
    if corruption == "duplicate":
        rows.append(row.copy())
    elif corruption == "ground_truth":
        row["trajectory_gt"] = "down"
    else:
        row["start_midi"] = 50
    path.write_text(json.dumps({"results": rows}))
    with pytest.raises(
        ValueError, match="Duplicate|Ground truth mismatch|Unknown result identity"
    ):
        replay.replay(root, dataset, root / "out")


def test_offline_replay_never_uses_optional_llm_parser(monkeypatch):
    from pitchbench.experiments.helpers import music

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setenv("PITCHBENCH_DOREMI_LLM", "1")
    monkeypatch.delenv("PITCHBENCH_OFFLINE_SCORING", raising=False)

    def remote_call(*args, **kwargs):
        raise AssertionError("Offline scoring must not call an API")

    monkeypatch.setattr(music, "_doremi_llm_call", remote_call)
    with music.offline_scoring():
        assert music.extract_solfege("__NO_DETECTION__") is None
        assert music.extract_all_solfege("__NO_DETECTION__") == []
        assert music.extract_solfege("fa#") == 6
    assert music._doremi_llm_enabled()


def test_table_replay_does_not_read_doremi_answers():
    row = dict(
        wav="test.wav", source_type="waveform", raw_midi="60", raw_spn="", raw_hz=""
    )
    # No raw_doremi or prompt_doremi exists: the paper replay must not need it.
    score = replay.score_record(
        "A1", "pitchbench_a1_single_pitch_id", row, dict(source="sine", midi=60)
    )
    assert score["any_correct"] == 1


def test_unrequested_format_makes_no_model_call(monkeypatch):
    from pitchbench.model import query

    called = []

    def respond(model, wav, prompt):
        called.append(prompt)
        return {"result": prompt}

    monkeypatch.setattr(query, "query_alm", respond)
    result = query.query_four_formats(
        "test", "audio.wav", "midi", "spn", "", "hz", verbose=False
    )
    assert called == ["midi", "spn", "hz"]
    assert result[2]["result"] == ""
