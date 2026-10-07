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


def test_paper_selector_is_exactly_the_28_table_tasks():
    from pitchbench.analysis.table1 import TASKS
    from pitchbench.experiments.run import _resolve_experiments
    names = _resolve_experiments(['paper'])
    assert [name.split('_')[1] for name in names] == [task.lower() for task in TASKS]
    assert len(names) == 28


def test_paper_evaluation_rejects_missing_task_instead_of_partial_overall(tmp_path, monkeypatch):
    from argparse import Namespace
    import pitchbench.experiments.run as run
    monkeypatch.setattr(run.config, '_PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(run, '_evaluate_one', lambda name, *a, **k: None if '_b3_' in name else {})
    args = Namespace(experiments=['paper'], model='test', name=None,
                     run_name='test', sample_n=1, sample_seed=42)
    with pytest.raises(SystemExit, match='incomplete.*pitchbench_b3'):
        run.cmd_evaluate(args)


def test_official_audio_is_identity_checked_before_querying(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    import pandas as pd
    import pitchbench.experiments.run as run

    name = 'pitchbench_e6_slightly_off'
    dataset = tmp_path / 'dataset'
    shard = dataset / name / 'test-00000-of-00001.parquet'
    shard.parent.mkdir(parents=True)
    pd.DataFrame([
        {'audio': {'path': 'a.wav', 'bytes': b'original-a'}, 'source': 'sine', 'midi': 60,
         'prompt_midi': 'MIDI?', 'prompt_abc': 'SPN?', 'prompt_freq': 'Hz?', 'prompt_solfege': 'DoReMi?'},
        {'audio': {'path': 'b.wav', 'bytes': b'original-b'}, 'source': 'piano', 'midi': 61,
         'prompt_midi': 'MIDI?', 'prompt_abc': 'SPN?', 'prompt_freq': 'Hz?', 'prompt_solfege': 'DoReMi?'},
    ]).to_parquet(shard)
    metadata = dataset / '.cache/huggingface/download' / name / (shard.name + '.metadata')
    metadata.parent.mkdir(parents=True)
    metadata.write_text('fixed\n')
    reference = tmp_path / 'reference'
    answers = reference / name / 'run/results_test.json'
    answers.parent.mkdir(parents=True)
    answers.write_text(json.dumps({'results': [{'wav': 'b.wav'}, {'wav': 'a.wav'}]}))
    runtime = tmp_path / 'runtime'
    monkeypatch.setattr(run.config, '_PROJECT_ROOT', runtime)
    monkeypatch.setattr(run.config, 'GENERATED_DIR', runtime / 'data/generated')
    calls = []
    def evaluate(name, model, output, sample_info, **kwargs):
        frame = pd.read_parquet(runtime / 'data/generated' / name / '_questions.parquet')
        assert [__import__('pathlib').Path(p).name for p in frame.audio_path] == ['b.wav', 'a.wav']
        assert [__import__('pathlib').Path(p).read_bytes() for p in frame.audio_path] == [b'original-b', b'original-a']
        assert list(frame.prompt_doremi) == ['', '']
        assert sample_info['pitch_formats'] == ('midi', 'spn', 'hz')
        calls.append(name)
        return {}
    monkeypatch.setattr(run, '_evaluate_one', evaluate)
    monkeypatch.setattr(run, '_run_eval_analysis', lambda *a: None)
    monkeypatch.setattr(run, '_run_cross_model_analysis', lambda *a: None)
    monkeypatch.setattr(run, '_run_a1_plots', lambda *a: None)
    args = Namespace(experiments=['e6'], model='test', name=None, run_name='test',
                     sample_n=None, sample_seed=42, dataset_dir=dataset,
                     dataset_revision='fixed', reference_answers=reference)
    run.cmd_evaluate(args)
    assert calls == [name]
    answers.write_text(json.dumps({'results': [{'wav': 'a.wav'}, {'wav': 'wrong.wav'}]}))
    with pytest.raises(ValueError, match='identities differ'):
        run.cmd_evaluate(args)
    assert calls == [name]
    args.dataset_revision = 'wrong'
    with pytest.raises(RuntimeError, match='expected dataset revision'):
        run.cmd_evaluate(args)
