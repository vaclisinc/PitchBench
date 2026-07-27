from pathlib import Path

import numpy as np

from pitchbench.baselines import runtime
from pitchbench.baselines.runtime import Analysis, Event

CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "pb_rebuttal_baselines_001.yaml"
)
MUSCRIPTOR_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "pb_rebuttal_muscriptor_001.yaml"
)


def _path(tmp_path: Path, experiment: str) -> Path:
    directory = tmp_path / f"pitchbench_{experiment}_example"
    directory.mkdir()
    return directory / "sample.wav"


def _analysis(
    *,
    events: list[Event],
    frame_times: list[float] | None = None,
    monophonic: list[float] | None = None,
    polyphonic: list[tuple[float, ...]] | None = None,
) -> Analysis:
    return Analysis(
        duration=10.0,
        events=tuple(events),
        frame_times=np.asarray(frame_times or [], dtype=np.float32),
        monophonic_midi=np.asarray(monophonic or [], dtype=np.float32),
        polyphonic_pitches=tuple(polyphonic or []),
    )


def test_d2_uses_continuous_pitch_before_rounding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    result = _analysis(
        events=[
            Event(0.0, 1.0, 69.00, 1.0),
            Event(1.2, 2.2, 69.01, 1.0),
        ]
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/dsp",
        _path(tmp_path, "d2"),
        'Which tone is higher in pitch? Reply with ONLY "first" or "second".',
    )
    assert answer["result"] == "second"


def test_b2_reads_only_prompt_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    result = _analysis(
        events=[
            Event(1.0, 2.0, 60.0, 1.0),
            Event(4.0, 6.0, 67.0, 1.0),
        ]
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/dsp",
        _path(tmp_path, "b2"),
        "Identify the pitch that is sounding at exactly 0:05.00. "
        "What is the MIDI note number?",
    )
    assert answer["result"] == "67"


def test_c3_matches_transposed_chord_template(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    frames = [(62.0, 65.0, 69.0)] * 10
    result = _analysis(
        events=[],
        frame_times=list(np.linspace(0.0, 10.0, len(frames))),
        polyphonic=frames,
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/dsp",
        _path(tmp_path, "c3"),
        "This audio contains a chord. What is its harmonic quality?",
    )
    assert answer["result"] == "minor"


def test_muscriptor_chord_decoder_uses_its_configured_presence_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(MUSCRIPTOR_CONFIG))
    frames = [(60.0, 64.0, 67.0)] * 10
    result = _analysis(
        events=[],
        frame_times=list(np.linspace(0.0, 10.0, len(frames))),
        polyphonic=frames,
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/muscriptor",
        _path(tmp_path, "c1"),
        "How many distinct pitches are sounding simultaneously?",
    )
    assert answer["result"] == "3"


def test_d7a_anchors_measured_interval_to_prompt_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    result = _analysis(
        events=[
            Event(0.0, 1.0, 61.2, 1.0),
            Event(1.5, 2.5, 68.2, 1.0),
        ]
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/dsp",
        _path(tmp_path, "d7a"),
        "The FIRST tone is MIDI note 60. What is the MIDI note number "
        "of the SECOND tone?",
    )
    assert answer["result"] == "67"


def test_d1_counts_distinct_pitches_not_fragmented_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    result = _analysis(
        events=[
            Event(0.0, 0.4, 60.0, 1.0),
            Event(0.4, 1.0, 60.1, 1.0),
            Event(1.5, 2.5, 64.0, 1.0),
        ]
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/basic-pitch",
        _path(tmp_path, "d1"),
        "How many distinct musical pitches are played in this sequence?",
    )
    assert answer["result"] == "2"


def test_d8_uses_prompt_count_to_stabilize_frame_track(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    result = _analysis(
        events=[Event(0.0, 3.0, 72.0, 1.0)],
        frame_times=list(np.arange(9, dtype=float)),
        monophonic=[60.0, 60.0, 60.0, 64.0, 64.0, 67.0, 67.0, 67.0, 67.0],
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/basic-pitch",
        _path(tmp_path, "d8"),
        "You will hear 3 musical notes played one after another. "
        "Identify all 3 MIDI note numbers in order.",
    )
    assert answer["result"] == "60 64 67"


def test_e6_snaps_hz_response_to_equal_temperament(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    result = _analysis(events=[Event(0.0, 1.0, 60.4, 1.0)])
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/basic-pitch",
        _path(tmp_path, "e6"),
        "Identify the nearest in-tune pitch in Hertz.",
    )
    assert answer["result"] == "261.6256"


def test_missing_detection_is_returned_as_incorrect_sentinel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: _analysis(events=[]))
    answer = runtime.query_baseline(
        "baseline/dsp",
        _path(tmp_path, "a1"),
        "What is the MIDI note number?",
    )
    assert answer["result"] == "__NO_DETECTION__"


def test_d4_detects_slow_down_then_up_contour(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PITCHBENCH_BASELINE_CONFIG", str(CONFIG))
    descending = np.linspace(60.0, 59.0, 50)
    ascending = np.linspace(59.0, 60.0, 50)
    contour = np.concatenate([descending, ascending])
    result = _analysis(
        events=[Event(0.0, 10.0, 59.5, 1.0)],
        frame_times=list(np.linspace(0.0, 10.0, len(contour))),
        monophonic=list(contour),
    )
    monkeypatch.setattr(runtime, "_analysis", lambda *_args: result)
    answer = runtime.query_baseline(
        "baseline/dsp",
        _path(tmp_path, "d4"),
        "Describe whether the pitch moves up or down.",
    )
    assert answer["result"] == "down, up"
