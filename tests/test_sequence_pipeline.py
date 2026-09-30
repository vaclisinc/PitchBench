"""Exercise real sequence scorers through both evaluation paths and overall CSVs."""

import csv
import importlib
import json
from dataclasses import replace

import pytest
import pitchbench.config as config

config.AUDIO_DIR = config.GENERATED_DIR

from pitchbench.experiments.helpers import cat_d, cat_f, data
from pitchbench.experiments.helpers.music import midi_to_freq
from pitchbench.experiments.helpers.results import aggregate_run_accuracies, write_model_summary_csv


@pytest.mark.parametrize("mode", ["direct", "parquet"])
def test_sequence_f1_reaches_package_overall(mode, tmp_path, monkeypatch):
    cond = dict(
        source="piano", sources=["piano", "violin"], target_pitches=[60, 62, 64],
        trial=0, n_notes=3, midi_sequence=[60, 62, 64], note_sequence=["C4", "D4", "E4"],
        doremi_sequence=["do", "re", "mi"], hz_sequence=[midi_to_freq(p) for p in [60, 62, 64]],
        n=2, x=1, tempo="slow", tone_ms=1000, total_ms=3000,
        inst_cfg="same", source_label="piano", all_n_notes=[3, 3],
        chorale_id="test", chorale_slug="test", voice_name="soprano", qpm=120,
        start_qL=0, end_qL=3,
    )
    prompts = {f: f for f in ("midi", "spn", "doremi", "hz")}
    conditions = [cond, {**cond, "source": "violin", "source_label": "violin"}]
    tasks = ("d8_sequence_pitches", "f1_melodic_line_atonal", "f2_melodic_line_tonal")
    for task in tasks:
        module = importlib.import_module(f"pitchbench.experiments.scripts.pitchbench_{task}")
        runner = cat_d if task.startswith("d8") else cat_f
        spec = replace(module.SPEC, build_conditions_fn=lambda: conditions,
                       wav_fn=lambda c: tmp_path / "audio.wav", prompts_fn=lambda c: prompts,
                       label_fn=lambda j: "test")
        monkeypatch.setattr(runner, "get_model_info", lambda name: {})
        monkeypatch.setattr(runner, "query_alm", lambda name, wav, prompt: {
            "result": "60 64" if prompt == "midi" else ""})
        monkeypatch.setattr(runner, "dispatch", lambda jobs, fn, **kw: [fn(j) for j in jobs])
        folder = tmp_path / spec.exp_name
        folder.mkdir()
        if mode == "direct":
            runner.run_one_model(spec, "test", conditions, folder, None, model_label="custom-label")
        else:
            def generation_unavailable():
                raise AssertionError("Stored-data evaluation must not regenerate conditions")
            spec = replace(spec, build_conditions_fn=generation_unavailable)
            monkeypatch.setattr(data, "read_dataset", lambda path: [{
                "audio_path": "audio.wav", "_condition": c,
                **{f"prompt_{f}": p for f, p in prompts.items()},
            } for c in conditions])
            monkeypatch.setattr(runner, "apply_default_sampling", lambda exp, rows, **kw: (rows, {}))
            evaluate = runner.evaluate_cat_d_from_parquet if runner is cat_d else runner.evaluate_cat_f_from_parquet
            evaluate(spec, "test", folder, model_label="custom-label")
        payload = json.loads((folder / "results_custom_label.json").read_text())
        assert payload["summary"]["accuracy"]["any"] == pytest.approx(0.8)
        assert payload["metadata"]["score_name"] == "ordered_note_f1_lcs"
        assert payload["results"][0]["any_note_f1"] == pytest.approx(0.8)
        exact = "any_sequence_correct" if runner is cat_d else "any_seq_correct"
        assert payload["results"][0][exact] == 0
        with (folder / "accuracies_custom_label.csv").open() as handle:
            metrics = list(csv.DictReader(handle))
        assert float(next(r for r in metrics if r["metric"] == "accuracy.any")["value"]) == 0.8
        assert any("any_note_f1" in r["metric"] for r in metrics)
    aggregate = aggregate_run_accuracies(tmp_path, tmp_path / "overall/aggregated_accuracies.csv")
    summary = write_model_summary_csv(aggregate, tmp_path / "overall/summary.csv")
    with summary.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["model_name"] == "custom_label"
    assert float(row["mean_accuracy"]) == 0.8
    for task in tasks:
        assert float(row[f"pitchbench_{task}_accuracy"]) == 0.8


def test_overall_includes_all_28_tasks_including_timing(tmp_path):
    from pitchbench.analysis.table1 import TASKS
    from pitchbench.experiments.helpers.cat_b import compute_summary
    from pitchbench.experiments.helpers.results import save_accuracies_csv

    for task in TASKS:
        folder = tmp_path / f"pitchbench_{task.lower()}_test"
        folder.mkdir()
        if task in {"B3", "B4", "B5"}:
            summary = compute_summary([{"correct": value} for value in [1, 0, 0, 0]], "timing", None)
        else:
            summary = {"total": 4, "accuracy": {"n": 4, "any": 0.5}}
        save_accuracies_csv(folder, "test", summary)
    aggregate = aggregate_run_accuracies(tmp_path, tmp_path / "aggregate.csv")
    result = write_model_summary_csv(aggregate, tmp_path / "summary.csv")
    with result.open() as handle:
        row = next(csv.DictReader(handle))
    values = [float(v) for k, v in row.items() if k.startswith("pitchbench_") and k.endswith("_accuracy")]
    assert len(values) == 28
    assert values.count(0.25) == 3
    assert float(row["mean_accuracy"]) == round((25 * 0.5 + 3 * 0.25) / 28, 6)
