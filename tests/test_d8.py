import ast

import pytest
import pitchbench.config as config

config.AUDIO_DIR = config.GENERATED_DIR

from pitchbench.experiments.helpers.cat_d import compute_summary
from pitchbench.experiments.helpers.music import midi_to_freq, midi_to_note
from pitchbench.experiments.scripts.pitchbench_d8_sequence_pitches import SPEC, record_for


def score(**responses):
    pitches = [60, 62, 64]
    return record_for(
        dict(source="sine", trial=0, n_notes=3, midi_sequence=pitches,
             note_sequence=[midi_to_note(p) for p in pitches],
             doremi_sequence=["do", "re", "mi"],
             hz_sequence=[midi_to_freq(p) for p in pitches]),
        "example.wav", {f: responses.get(f, "") for f in ("midi", "spn", "doremi", "hz")},
    )


@pytest.mark.parametrize("prediction,expected,exact", [
    ("60 62 64", 1, 1),
    ("60 64", 4 / 5, 0),
    ("60 62 64 65", 6 / 7, 0),
    ("60 60 62 64", 6 / 7, 0),
    ("64 62 60", 1 / 3, 0),
    ("60 63 64", 2 / 3, 0),
    ("", 0, 0),
])
def test_d8_full_sequence_scoring(prediction, expected, exact):
    record = score(midi=prediction)
    assert record["midi_note_f1"] == pytest.approx(expected)
    assert record["any_note_f1"] == pytest.approx(expected)
    assert record["midi_sequence_correct"] == exact
    assert record["any_sequence_correct"] == exact
    assert len(ast.literal_eval(record["midi_pred"])) == len(prediction.split())


def test_d8_format_matching_and_no_solfege_in_any():
    assert score(doremi="do re mi")["any_note_f1"] == 0
    assert score(doremi="do re mi")["doremi_note_f1"] == 1
    assert score(spn="C4 D4 E4")["spn_note_f1"] == 1
    record = score(spn="C4 D4 E4 F4")
    assert record["spn_note_f1"] == 6 / 7
    assert record["spn_sequence_correct"] == 0
    assert score(hz="261.63 293.66 329.63")["hz_note_f1"] == 1
    # 2 Hz is inside Category F's 1% tolerance but outside D8's 1 Hz.
    assert score(hz="263.63 295.66 331.63")["hz_note_f1"] == 0


def test_d8_summary_averages_per_stimulus_format_maximum():
    records = [score(midi="60 62 64"), score(spn="C4 E4")]
    summary = compute_summary(records, SPEC.headline_metrics, None, SPEC.metric_suffix)
    assert summary["accuracy"]["any"] == 0.9
    assert summary["accuracy"]["midi"] == 0.5
    assert summary["accuracy"]["spn"] == 0.4
    # Other D tasks keep binary accuracy by default.
    assert compute_summary([{"count_correct": 1}, {"count_correct": 0}],
                           ("count",), None)["accuracy"]["count"] == 0.5


@pytest.mark.parametrize("mode", ["direct", "parquet"])
def test_d8_evaluation_exports_lcs_headline_and_exact_diagnostic(mode, tmp_path, monkeypatch):
    import csv
    import json
    from dataclasses import replace
    from pitchbench.experiments.helpers import cat_d, data
    from pitchbench.experiments.helpers.plots import _detect_primary_score

    pitches = [60, 62, 64]
    cond = dict(source="sine", trial=0, n_notes=3, midi_sequence=pitches,
                note_sequence=["C4", "D4", "E4"], doremi_sequence=["do", "re", "mi"],
                hz_sequence=[midi_to_freq(p) for p in pitches])
    spec = replace(SPEC, build_conditions_fn=lambda: [cond], wav_fn=lambda c: tmp_path / "audio.wav")
    monkeypatch.setattr(cat_d, "get_model_info", lambda name: {})
    monkeypatch.setattr(cat_d, "query_alm", lambda name, wav, prompt: {
        "result": "60 64" if "MIDI" in prompt else ""})
    monkeypatch.setattr(cat_d, "dispatch", lambda jobs, fn, **kw: [fn(j) for j in jobs])
    if mode == "direct":
        cat_d.run_one_model(spec, "test", [cond], tmp_path, None)
    else:
        monkeypatch.setattr(data, "read_dataset", lambda path: [{
            "audio_path": "audio.wav", "_condition": cond,
            **{f"prompt_{f}": p for f, p in spec.prompts_fn(cond).items()},
        }])
        monkeypatch.setattr(cat_d, "apply_default_sampling", lambda exp, rows, **kw: (rows, {}))
        cat_d.evaluate_cat_d_from_parquet(spec, "test", tmp_path)
    payload = json.loads((tmp_path / "results_test.json").read_text())
    assert payload["summary"]["accuracy"]["any"] == 0.8
    assert payload["metadata"]["score_name"] == "ordered_note_f1_lcs"
    assert payload["results"][0]["any_sequence_correct"] == 0
    assert _detect_primary_score(payload["results"])[0] == "any_note_f1"
    with (tmp_path / "accuracies_test.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(next(r for r in rows if r["metric"] == "accuracy.any")["value"]) == 0.8
