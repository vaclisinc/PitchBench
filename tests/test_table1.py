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
        "midi_correct,spn_correct,doremi_correct,hz_correct\n"
        "0,0,1,0\n0,1,0,0\n"
    )
    assert aggregate._compute_c4_any_from_results(tmp_path, "model") == (2, 0.5)


def test_final_scores_match_independently_saved_result_bundles():
    scores = load_scores(REPO)
    assert len(scores) == 28 * 8
    with (REPO / "results/d8-ordered-note-f1/scores_by_model_experiment.csv").open() as handle:
        for row in csv.DictReader(handle):
            task = row["experiment"].split("_")[1].upper()
            task = "D7a" if task == "D7A" else task
            expected = C4_PERCENT[row["model"]] if task == "C4" else float(row["score"]) * 100
            assert float(scores[row["model"], task]) == pytest.approx(expected)
    with (REPO / "results/d8-baselines-lcs/updated_table.csv").open() as handle:
        for row in csv.DictReader(handle):
            for model, field in (("baseline/dsp", "dsp_score_pct"), ("baseline/basic-pitch", "basic_pitch_score_pct")):
                assert float(scores[model, row["experiment"]]) == pytest.approx(float(row[field]))


def test_means_use_all_28_tasks_and_agree_with_published_evidence():
    rows = list(csv.DictReader(io.StringIO(render(REPO)["table1.csv"])))
    assert [r["task"] for r in rows] == [*TASKS, "Mean"]
    for model in MODELS:
        expected = sum(Decimal(row[model]) for row in rows[:-1]) / 28
        assert Decimal(rows[-1][model]) == expected
    with (REPO / "results/d8-ordered-note-f1/overall.csv").open() as handle:
        legacy_c4 = dict(zip(list(MODELS)[:6], [1.5, 2.5, 0, 1.5, 21, 8.5]))
        for row in csv.DictReader(handle):
            model = row["model"]
            corrected = float(row["mean_score"]) * 100 + (C4_PERCENT[model] - legacy_c4[model]) / 28
            assert float(rows[-1][model]) == pytest.approx(corrected)
    assert float(rows[-1]["baseline/dsp"]) == pytest.approx(69.92997772208298)
    assert float(rows[-1]["baseline/basic-pitch"]) == pytest.approx(73.13607630186577)


def test_committed_exports_match_evidence():
    for name, content in render(REPO).items():
        assert (REPO / "paper/figures-and-tables" / name).read_bytes() == content.encode()


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "unexpected", "nan", "range"])
def test_reject_incomplete_or_invalid_evidence(tmp_path, corruption):
    for relative in SOURCES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    target = tmp_path / SOURCES["baseline_d8"]
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
        rows[0]["any_note_f1"] = "NaN" if corruption == "nan" else "1.01"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Duplicate|Coverage mismatch|Invalid score"):
        load_scores(tmp_path)
