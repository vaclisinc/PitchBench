import pitchbench.config as pitchbench_config

pitchbench_config.AUDIO_DIR = pitchbench_config.GENERATED_DIR

from pitchbench.experiments.helpers.cat_f import (
    ordered_note_f1,
    parse_midi_seq,
    score_polyphonic_record,
)


def test_parse_midi_seq_keeps_extra_notes() -> None:
    assert parse_midi_seq("60 62 64 65 67 69") == [60, 62, 64, 65, 67, 69]


def test_ordered_note_f1_partial_credit_example() -> None:
    score = ordered_note_f1(
        [60, 62, 64, 65, 67],
        [60, 62, 63, 65, 67, 69],
    )
    assert score["matches"] == 4
    assert score["precision"] == 4 / 6
    assert score["recall"] == 4 / 5
    assert score["f1"] == 8 / 11


def test_ordered_note_f1_penalizes_reversed_order() -> None:
    score = ordered_note_f1(
        [60, 62, 64, 65, 67],
        [67, 65, 64, 62, 60],
    )
    assert score["matches"] == 1
    assert score["f1"] == 0.2


def test_polyphonic_record_any_f1_excludes_doremi() -> None:
    record = score_polyphonic_record(
        {
            "target_pitches": [60, 62, 64],
            "source": "piano",
            "sources": ["piano"],
        },
        "example.wav",
        {
            "midi": "60 63 64 65",
            "spn": "C4 D4 E4",
            "doremi": "do re mi",
            "hz": "",
        },
    )
    assert record["midi_note_f1"] == 4 / 7
    assert record["spn_note_f1"] == 1.0
    assert record["any_note_f1"] == 1.0
    assert record["midi_pred_count"] == 4
    assert record["midi_count_error"] == 1
    assert record["midi_seq_correct"] == 0
