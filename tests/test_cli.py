"""Tests for pitchbench.experiments.run — experiment resolution, sorting, slug."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from pitchbench.experiments.run import _resolve_experiments, _sort_key


# ── _sort_key ─────────────────────────────────────────────────────────────────

class TestSortKey:
    def test_canonical_name_sorts_by_cat_num(self):
        a1 = _sort_key("pitchbench_a1_single_pitch_id")
        a2 = _sort_key("pitchbench_a2_duration")
        b1 = _sort_key("pitchbench_b1_timing")
        assert a1 < a2 < b1

    def test_same_cat_ordered_by_number(self):
        assert _sort_key("pitchbench_a1_x") < _sort_key("pitchbench_a10_x")

    def test_unknown_name_sorts_last(self):
        known  = _sort_key("pitchbench_a1_single_pitch_id")
        unknown = _sort_key("not_a_pitchbench_name")
        assert known < unknown

    def test_variant_suffix_sorts_after_base(self):
        # a1a vs a1 — the letter suffix increments alphabetically
        assert _sort_key("pitchbench_a1_x") < _sort_key("pitchbench_a1a_x")


# ── _resolve_experiments ──────────────────────────────────────────────────────

_FAKE_EXPERIMENTS = [
    "pitchbench_a1_single_pitch_id",
    "pitchbench_a2_duration",
    "pitchbench_b1_timing",
    "pitchbench_e1_audio_effects",
]


@pytest.fixture()
def fake_discover(monkeypatch):
    monkeypatch.setattr("pitchbench.experiments.run.discover", lambda: _FAKE_EXPERIMENTS)


class TestResolveExperiments:
    def test_all_returns_everything(self, fake_discover):
        result = _resolve_experiments(["all"])
        assert result == _FAKE_EXPERIMENTS

    def test_empty_returns_everything(self, fake_discover):
        result = _resolve_experiments([])
        assert result == _FAKE_EXPERIMENTS

    def test_experiment_id(self, fake_discover):
        result = _resolve_experiments(["a1"])
        assert result == ["pitchbench_a1_single_pitch_id"]

    def test_category_letter(self, fake_discover):
        result = _resolve_experiments(["a"])
        assert set(result) == {
            "pitchbench_a1_single_pitch_id",
            "pitchbench_a2_duration",
        }

    def test_full_module_name(self, fake_discover):
        result = _resolve_experiments(["pitchbench_b1_timing"])
        assert result == ["pitchbench_b1_timing"]

    def test_multiple_ids_preserved_order(self, fake_discover):
        result = _resolve_experiments(["b1", "a1"])
        assert result == [
            "pitchbench_b1_timing",
            "pitchbench_a1_single_pitch_id",
        ]

    def test_deduplicates(self, fake_discover):
        result = _resolve_experiments(["a1", "a1"])
        assert result == ["pitchbench_a1_single_pitch_id"]

    def test_unknown_id_raises(self, fake_discover):
        with pytest.raises(SystemExit):
            _resolve_experiments(["z99"])

    def test_unknown_category_raises(self, fake_discover):
        with pytest.raises(SystemExit):
            _resolve_experiments(["z"])

    def test_case_insensitive(self, fake_discover):
        result = _resolve_experiments(["A1"])
        assert result == ["pitchbench_a1_single_pitch_id"]
