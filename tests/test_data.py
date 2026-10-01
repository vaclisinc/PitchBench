"""Tests for pitchbench.experiments.helpers.data — Parquet I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pitchbench.config as config
from pitchbench.experiments.helpers.data import append_dataset, dataset_path, read_dataset


# ── dataset_path ──────────────────────────────────────────────────────────────

class TestDatasetPath:
    def test_returns_parquet_under_generated_dir(self):
        p = dataset_path("pitchbench_a1_single_pitch_id")
        assert p == config.GENERATED_DIR / "pitchbench_a1_single_pitch_id" / "_questions.parquet"

    def test_extension_is_parquet(self):
        p = dataset_path("pitchbench_b1_timing")
        assert p.suffix == ".parquet"
        assert p.name == "_questions.parquet"


# ── append_dataset ────────────────────────────────────────────────────────────

class TestAppendDataset:
    def _rows(self, n: int = 3, prefix: str = "file") -> list[dict]:
        return [
            {
                "audio_path": f"data/generated/exp/{prefix}_{i}.wav",
                "gt_midi":    60 + i,
                "gt_spn":     "C4",
            }
            for i in range(n)
        ]

    def test_creates_parquet_on_first_write(self, tmp_path):
        p = tmp_path / "data.parquet"
        n = append_dataset(p, self._rows())
        assert n == 3
        assert p.exists()

    def test_returns_count_of_new_rows(self, tmp_path):
        p = tmp_path / "data.parquet"
        append_dataset(p, self._rows(3, "a"))
        n = append_dataset(p, self._rows(2, "b"))
        assert n == 2

    def test_skips_duplicate_audio_paths(self, tmp_path):
        p = tmp_path / "data.parquet"
        rows = self._rows(3)
        append_dataset(p, rows)
        n = append_dataset(p, rows)   # exact same rows → all duplicates
        assert n == 0

    def test_partial_overlap_only_appends_new(self, tmp_path):
        p = tmp_path / "data.parquet"
        append_dataset(p, self._rows(3, "a"))
        # Two new + one duplicate
        mixed = self._rows(1, "a") + self._rows(2, "b")
        n = append_dataset(p, mixed)
        assert n == 2

    def test_total_row_count_after_appends(self, tmp_path):
        import pandas as pd
        p = tmp_path / "data.parquet"
        append_dataset(p, self._rows(3, "a"))
        append_dataset(p, self._rows(2, "b"))
        df = pd.read_parquet(p)
        assert len(df) == 5

    def test_empty_rows_returns_zero(self, tmp_path):
        p = tmp_path / "data.parquet"
        assert append_dataset(p, []) == 0
        assert not p.exists()

    def test_d7b_dual_audio_dedup(self, tmp_path):
        p = tmp_path / "data.parquet"
        rows = [
            {"audio_1_path": "a1.wav", "audio_2_path": "a2.wav", "gt_midi": 60},
            {"audio_1_path": "b1.wav", "audio_2_path": "b2.wav", "gt_midi": 62},
        ]
        append_dataset(p, rows)
        n = append_dataset(p, rows)  # all duplicates
        assert n == 0

    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "data.parquet"
        append_dataset(p, self._rows(1))
        assert p.exists()


# ── read_dataset ──────────────────────────────────────────────────────────────

class TestReadDataset:
    def _write(self, path: Path, rows: list[dict]) -> None:
        import pandas as pd
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    def test_raises_file_not_found_with_hint(self, tmp_path):
        p = tmp_path / "missing" / "data.parquet"
        with pytest.raises(FileNotFoundError, match="pitchbench generate"):
            read_dataset(p)

    def test_returns_list_of_dicts(self, tmp_path):
        p = tmp_path / "data.parquet"
        self._write(p, [{"audio_path": "x.wav", "gt_midi": 60}])
        rows = read_dataset(p)
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)

    def test_values_are_preserved(self, tmp_path):
        p = tmp_path / "data.parquet"
        self._write(p, [{"audio_path": "x.wav", "gt_midi": 72}])
        rows = read_dataset(p)
        assert rows[0]["gt_midi"] == 72

    def test_condition_json_parsed(self, tmp_path):
        p = tmp_path / "data.parquet"
        cond = {"source": "piano", "duration_ms": 500}
        self._write(p, [{"audio_path": "x.wav", "condition_json": json.dumps(cond)}])
        rows = read_dataset(p)
        assert rows[0]["_condition"] == cond

    def test_invalid_condition_json_gives_empty_dict(self, tmp_path):
        p = tmp_path / "data.parquet"
        self._write(p, [{"audio_path": "x.wav", "condition_json": "{bad json"}])
        rows = read_dataset(p)
        assert rows[0]["_condition"] == {}

    def test_row_count_matches(self, tmp_path):
        p = tmp_path / "data.parquet"
        self._write(p, [{"audio_path": f"{i}.wav", "gt_midi": i} for i in range(5)])
        rows = read_dataset(p)
        assert len(rows) == 5


def test_paper_formats_drop_solfege_queries_without_mutating_dataset():
    from pitchbench.experiments.helpers.data import select_pitch_formats
    row = {'prompt_main': 'count', 'prompt_midi': 'midi', 'prompt_abc': 'spn',
           'prompt_solfege': 'doremi', 'prompt_freq': 'hz'}
    selected = select_pitch_formats([row], ('midi', 'spn', 'hz'))[0]
    assert selected == {'prompt_main': 'count', 'prompt_midi': 'midi',
                        'prompt_spn': 'spn', 'prompt_doremi': '', 'prompt_hz': 'hz'}
    assert row['prompt_solfege'] == 'doremi'
