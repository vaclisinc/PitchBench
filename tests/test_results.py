"""Tests for results.py — aggregate_run_accuracies, write_model_summary_csv,
write_aggregate_format_accuracies, combine_accuracies_csvs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pitchbench.experiments.helpers.results import (
    aggregate_run_accuracies,
    combine_accuracies_csvs,
    write_aggregate_format_accuracies,
    write_model_summary_csv,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_accuracies_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n_samples", "value", "value_pct"])
        writer.writeheader()
        writer.writerows(rows)


_A1_ROWS = [
    {"metric": "total",         "n_samples": "10", "value": "",      "value_pct": ""},
    {"metric": "accuracy.midi", "n_samples": "10", "value": "0.500", "value_pct": "50.0%"},
    {"metric": "accuracy.spn",  "n_samples": "10", "value": "0.400", "value_pct": "40.0%"},
    {"metric": "accuracy.any",  "n_samples": "10", "value": "0.600", "value_pct": "60.0%"},
]

_B1_ROWS = [
    {"metric": "total",     "n_samples": "20", "value": "",      "value_pct": ""},
    {"metric": "accuracy",  "n_samples": "20", "value": "0.750", "value_pct": "75.0%"},
]


# ── aggregate_run_accuracies ──────────────────────────────────────────────────

class TestAggregateRunAccuracies:
    def test_combines_experiment_csvs(self, tmp_path):
        _write_accuracies_csv(
            tmp_path / "pitchbench_a1_single_pitch_id" / "accuracies_my_model.csv",
            _A1_ROWS,
        )
        _write_accuracies_csv(
            tmp_path / "pitchbench_b1_timing" / "accuracies_my_model.csv",
            _B1_ROWS,
        )
        out = tmp_path / "overall" / "aggregated_accuracies.csv"
        result = aggregate_run_accuracies(tmp_path, out)

        assert result == out
        assert out.exists()

        with out.open() as f:
            rows = list(csv.DictReader(f))

        experiments = {r["experiment"] for r in rows}
        assert "pitchbench_a1_single_pitch_id" in experiments
        assert "pitchbench_b1_timing" in experiments

        models = {r["model"] for r in rows}
        assert "my_model" in models

        # Every row has required columns
        for row in rows:
            assert "metric" in row
            assert "n_samples" in row
            assert "value" in row
            assert "value_pct" in row

    def test_returns_none_when_no_csvs(self, tmp_path):
        out = tmp_path / "overall" / "aggregated_accuracies.csv"
        result = aggregate_run_accuracies(tmp_path, out)
        assert result is None
        assert not out.exists()

    def test_ignores_files_outside_pitchbench_dirs(self, tmp_path):
        # File at top level (not in pitchbench_* dir) should be ignored
        _write_accuracies_csv(tmp_path / "accuracies_stray.csv", _A1_ROWS)
        out = tmp_path / "aggregated.csv"
        result = aggregate_run_accuracies(tmp_path, out)
        assert result is None

    def test_multiple_models_per_experiment(self, tmp_path):
        exp_dir = tmp_path / "pitchbench_a1_single_pitch_id"
        _write_accuracies_csv(exp_dir / "accuracies_model_a.csv", _A1_ROWS)
        _write_accuracies_csv(exp_dir / "accuracies_model_b.csv", _A1_ROWS)
        out = tmp_path / "agg.csv"
        aggregate_run_accuracies(tmp_path, out)

        with out.open() as f:
            rows = list(csv.DictReader(f))
        models = {r["model"] for r in rows}
        assert "model_a" in models
        assert "model_b" in models


# ── write_model_summary_csv ───────────────────────────────────────────────────

def _make_aggregated_csv(tmp_path, rows: list[dict]) -> Path:
    p = tmp_path / "aggregated_accuracies.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "experiment", "metric", "n_samples", "value", "value_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return p


class TestWriteModelSummaryCsv:
    def _agg_rows(self):
        return [
            {"model": "model_x", "experiment": "pitchbench_a1_x", "metric": "accuracy.any",
             "n_samples": "10", "value": "0.6", "value_pct": "60.0%"},
            {"model": "model_x", "experiment": "pitchbench_b1_y", "metric": "accuracy",
             "n_samples": "20", "value": "0.75", "value_pct": "75.0%"},
        ]

    def test_wide_columns_present(self, tmp_path):
        agg = _make_aggregated_csv(tmp_path, self._agg_rows())
        out = tmp_path / "summary.csv"
        write_model_summary_csv(agg, out)
        assert out.exists()

        with out.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        assert "model_name" in headers
        assert "mean_accuracy" in headers
        assert "pitchbench_a1_x_accuracy" in headers
        assert "pitchbench_a1_x_n" in headers
        assert "pitchbench_b1_y_accuracy" in headers
        assert len(rows) == 1

    def test_mean_accuracy_computed(self, tmp_path):
        agg = _make_aggregated_csv(tmp_path, self._agg_rows())
        out = tmp_path / "summary.csv"
        write_model_summary_csv(agg, out)

        with out.open() as f:
            row = list(csv.DictReader(f))[0]

        mean = float(row["mean_accuracy"])
        assert abs(mean - (0.6 + 0.75) / 2) < 1e-5

    def test_prefers_accuracy_any_over_accuracy(self, tmp_path):
        rows = [
            {"model": "m", "experiment": "exp_1", "metric": "accuracy",
             "n_samples": "5", "value": "0.1", "value_pct": "10%"},
            {"model": "m", "experiment": "exp_1", "metric": "accuracy.any",
             "n_samples": "5", "value": "0.9", "value_pct": "90%"},
        ]
        agg = _make_aggregated_csv(tmp_path, rows)
        out = tmp_path / "summary.csv"
        write_model_summary_csv(agg, out)

        with out.open() as f:
            row = list(csv.DictReader(f))[0]

        assert float(row["exp_1_accuracy"]) == pytest.approx(0.9)

    def test_multiple_models(self, tmp_path):
        rows = [
            {"model": "m1", "experiment": "exp_1", "metric": "accuracy.any",
             "n_samples": "10", "value": "0.5", "value_pct": "50%"},
            {"model": "m2", "experiment": "exp_1", "metric": "accuracy.any",
             "n_samples": "10", "value": "0.8", "value_pct": "80%"},
        ]
        agg = _make_aggregated_csv(tmp_path, rows)
        out = tmp_path / "summary.csv"
        write_model_summary_csv(agg, out)

        with out.open() as f:
            result_rows = list(csv.DictReader(f))

        assert len(result_rows) == 2
        model_names = {r["model_name"] for r in result_rows}
        assert {"m1", "m2"} == model_names


# ── write_aggregate_format_accuracies ─────────────────────────────────────────

class TestWriteAggregateFormatAccuracies:
    def _runs(self):
        return {
            "pitchbench_a1_x": {
                "model_alpha": {"n": 10, "formats": {"midi": 0.5, "spn": 0.4, "doremi": 0.3, "hz": 0.2}},
            },
            "pitchbench_a2_y": {
                "model_alpha": {"n": 20, "formats": {"midi": 0.8, "spn": 0.7, "doremi": 0.6, "hz": 0.5}},
            },
        }

    def test_writes_per_experiment_rows(self, tmp_path):
        out = tmp_path / "fa.csv"
        write_aggregate_format_accuracies(out, self._runs())

        with out.open() as f:
            rows = list(csv.DictReader(f))

        experiments = {r["experiment"] for r in rows if r["experiment"] != "__MEAN__"}
        assert "pitchbench_a1_x" in experiments
        assert "pitchbench_a2_y" in experiments

    def test_mean_row_present(self, tmp_path):
        out = tmp_path / "fa.csv"
        write_aggregate_format_accuracies(out, self._runs())

        with out.open() as f:
            rows = list(csv.DictReader(f))

        mean_rows = [r for r in rows if r["experiment"] == "__MEAN__"]
        assert len(mean_rows) > 0

    def test_columns_present(self, tmp_path):
        out = tmp_path / "fa.csv"
        write_aggregate_format_accuracies(out, self._runs())

        with out.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

        for col in ["experiment", "model", "n_samples", "format", "accuracy"]:
            assert col in headers


# ── combine_accuracies_csvs ───────────────────────────────────────────────────

class TestCombineAccuraciesCsvs:
    def test_combines_and_prepends_model(self, tmp_path):
        _write_accuracies_csv(tmp_path / "accuracies_model_a.csv", _A1_ROWS)
        _write_accuracies_csv(tmp_path / "accuracies_model_b.csv", _A1_ROWS)

        out = combine_accuracies_csvs(tmp_path, "combined.csv")
        assert out is not None

        with out.open() as f:
            rows = list(csv.DictReader(f))

        models = {r["model"] for r in rows}
        assert "model_a" in models
        assert "model_b" in models
        assert len(rows) == len(_A1_ROWS) * 2

    def test_returns_none_when_empty(self, tmp_path):
        result = combine_accuracies_csvs(tmp_path, "combined.csv")
        assert result is None
