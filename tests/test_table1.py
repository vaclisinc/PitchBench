"""Protect the paper table against stale exports and incomplete evidence."""

import csv
import io
import importlib
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from pitchbench.analysis.table1 import MODELS, SOURCES, TASKS, load_scores, render

REPO = Path(__file__).resolve().parents[1]
C4_PERCENT = dict(zip(list(MODELS)[:6], [0, 1.5, 0, 0.5, 12, 6.5]))


@pytest.mark.parametrize("module", ["analyze", "overview"])
def test_c4_any_excludes_solfege_only_matches(tmp_path, module):
    aggregate = importlib.import_module(f"pitchbench.analysis.{module}")
    folder = tmp_path / "model/pitchbench_c4_chord_pitches/run"
    folder.mkdir(parents=True)
    (folder / "results_model.csv").write_text(
        "midi_correct,spn_correct,doremi_correct,hz_correct\n" "0,0,1,0\n0,1,0,0\n"
    )
    assert aggregate._compute_c4_any_from_results(tmp_path, "model") == (2, 0.5)


def test_final_scores_match_raw_replay_counts_and_sequence_scores():
    scores = load_scores(REPO)
    assert len(scores) == 28 * 8
    with (REPO / SOURCES["recomputed"]).open() as handle:
        for row in csv.DictReader(handle):
            expected = Decimal(row["score_sum"]) * 100 / int(row["n_samples"])
            assert scores[row["model"], row["task"]] == expected
    # Independently established accuracy counts, including the manuscript errors.
    cases = [
        ("openrouter_google_gemini_flash_latest", "D4", 0, 160),
        ("openrouter_google_gemini_flash_latest", "D5", 9, 120),
        ("openrouter_openai_gpt_4o_audio_preview", "B3", 0, 160),
        ("openrouter_openai_gpt_4o_audio_preview", "B4", 1, 120),
        ("dashscope_qwen3_5_omni_plus", "C1", 22, 228),
        ("audio_flamingo_next_instruct", "D7a", 18, 130),
        ("baseline/basic-pitch", "B3", 130, 160),
        ("baseline/basic-pitch", "E1", 219, 240),
    ]
    for model, task, correct, total in cases:
        assert scores[model, task] == Decimal(correct) * 100 / total
    for model, percent in C4_PERCENT.items():
        assert scores[model, "C4"] == Decimal(str(percent))


def test_means_use_all_28_unrounded_tasks():
    rows = list(csv.DictReader(io.StringIO(render(REPO)["table1.csv"])))
    assert [r["task"] for r in rows] == [*TASKS, "Mean"]
    for model in MODELS:
        expected = sum(Decimal(row[model]) for row in rows[:-1]) / 28
        assert Decimal(rows[-1][model]) == expected
    # Double rounding used to display 9.7 and 13.9 from rounded intermediate CSVs.
    from pitchbench.analysis.table1 import _display

    scores = load_scores(REPO)
    assert _display(scores["dashscope_qwen3_5_omni_plus", "C1"]) == "9.6"
    assert _display(scores["audio_flamingo_next_instruct", "D7a"]) == "13.8"
    assert scores["baseline/basic-pitch", "F1"] == Decimal("76.5") * 100 / 130
    assert _display(scores["baseline/basic-pitch", "F1"]) == "58.8"


def test_committed_exports_match_evidence():
    for name, content in render(REPO).items():
        assert (
            REPO / "paper/figures-and-tables" / name
        ).read_bytes() == content.encode()


@pytest.mark.parametrize(
    "corruption", ["duplicate", "missing", "unexpected", "nan", "range"]
)
def test_reject_incomplete_or_invalid_evidence(tmp_path, corruption):
    for relative in SOURCES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    target = tmp_path / SOURCES["recomputed"]
    with target.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "missing":
        rows.pop()
    elif corruption == "unexpected":
        rows[0]["model"] = "baseline/unknown"
    else:
        rows[0]["score"] = "NaN" if corruption == "nan" else "1.01"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Duplicate|Coverage mismatch|Invalid score"):
        load_scores(tmp_path)
