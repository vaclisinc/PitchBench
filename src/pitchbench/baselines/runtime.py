"""Task-aware non-language baselines for the paper-locked PitchBench suite.

The acoustic front ends never receive condition dictionaries or ground-truth
fields. They consume only the waveform. The deterministic decoder receives the
public experiment ID (from the generated-audio directory) and the exact prompt,
which is the same task information exposed to evaluated audio-language models.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_PREFIX = "baseline/"
_CONFIG_ENV = "PITCHBENCH_BASELINE_CONFIG"
REPO_ROOT = Path(__file__).resolve().parents[3]
_config_lock = threading.Lock()
_config_cache: tuple[Path, int, dict[str, Any]] | None = None
_analysis_lock = threading.Lock()
_basic_pitch_model: Any | None = None
_muscriptor_model: Any | None = None


@dataclass(frozen=True)
class Event:
    onset: float
    offset: float
    midi: float
    confidence: float
    contour: tuple[float, ...] = ()
    instrument: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.offset - self.onset)


@dataclass(frozen=True)
class Analysis:
    duration: float
    events: tuple[Event, ...]
    frame_times: np.ndarray
    monophonic_midi: np.ndarray
    polyphonic_pitches: tuple[tuple[float, ...], ...]


_analysis_cache: dict[tuple[str, str], Analysis] = {}


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required baseline config field: {path}.{key}")
    return mapping[key]


def _load_config() -> tuple[dict[str, Any], Path, str]:
    raw_path = os.environ.get(_CONFIG_ENV)
    if not raw_path:
        raise RuntimeError(f"{_CONFIG_ENV} must point to the formal baseline YAML")
    path = Path(raw_path).resolve()
    stat = path.stat()
    global _config_cache
    with _config_lock:
        if _config_cache is None or _config_cache[:2] != (path, stat.st_mtime_ns):
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Baseline config must be a mapping: {path}")
            for section in (
                "experiment",
                "benchmark",
                "input_dataset",
                "runtime",
                "shared_decoder",
                "baselines",
                "outputs",
            ):
                _required(payload, section, "root")
            if (
                len(payload["benchmark"]["experiments"])
                != payload["benchmark"]["expected_experiment_count"]
            ):
                raise ValueError(
                    "Configured experiment list does not match expected_experiment_count"
                )
            _config_cache = (path, stat.st_mtime_ns, payload)
        config = _config_cache[2]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return config, path, digest


def baseline_concurrency(model_name: str) -> int:
    if not model_name.startswith(BASELINE_PREFIX):
        raise ValueError(f"Not a baseline model: {model_name}")
    config, _, _ = _load_config()
    value = int(_required(config["runtime"], "concurrency", "runtime"))
    if value < 1:
        raise ValueError("runtime.concurrency must be positive")
    return value


def baseline_model_info(model_name: str) -> dict[str, Any]:
    config, path, digest = _load_config()
    backend = model_name.removeprefix(BASELINE_PREFIX)
    if backend not in {"dsp", "basic-pitch", "muscriptor"}:
        raise ValueError(f"Unknown baseline backend: {model_name}")
    section_name = {
        "basic-pitch": "basic_pitch",
        "dsp": "dsp",
        "muscriptor": "muscriptor",
    }[backend]
    section = config["baselines"][section_name]
    if section["model_name"] != model_name:
        raise ValueError(
            f"Config model_name {section['model_name']!r} does not match {model_name!r}"
        )
    info: dict[str, Any] = {
        "model": model_name,
        "provider": "offline_baseline",
        "backend": backend,
        "experiment_id": config["experiment"]["experiment_id"],
        "config_path": str(path),
        "config_sha256": digest,
        "parameters": section,
    }
    if backend == "basic-pitch":
        try:
            from importlib.metadata import PackageNotFoundError, version

            info["installed_package_version"] = version(section["package"])
        except PackageNotFoundError:
            info["installed_package_version"] = "unavailable"
    elif backend == "muscriptor":
        source_dir = (REPO_ROOT / section["source_dir"]).resolve()
        info["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            text=True,
        ).strip()
    return info


def _experiment_id(audio_path: str | Path) -> str:
    for parent in [Path(audio_path).parent, *Path(audio_path).parents]:
        match = re.match(r"pitchbench_([a-z]+\d+[a-z]*)_", parent.name)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot infer PitchBench experiment ID from {audio_path}")


def _read_audio(path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sample_rate != target_sr:
        from scipy.signal import resample_poly

        divisor = math.gcd(sample_rate, target_sr)
        audio = resample_poly(audio, target_sr // divisor, sample_rate // divisor)
        sample_rate = target_sr
    return np.asarray(audio, dtype=np.float32), sample_rate


def _segments_from_mask(
    mask: np.ndarray,
    times: np.ndarray,
    *,
    minimum_seconds: float,
    frame_seconds: float,
) -> list[tuple[float, float, int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    segments: list[tuple[float, float, int, int]] = []
    for start, end in zip(starts, ends):
        onset = max(0.0, float(times[start] - frame_seconds / 2))
        offset = float(times[end - 1] + frame_seconds / 2)
        if offset - onset >= minimum_seconds:
            segments.append((onset, offset, int(start), int(end)))
    return segments


def _events_from_activity_and_pitch(
    segments: list[tuple[float, float, int, int]],
    *,
    duration: float,
    times: np.ndarray,
    midi_track: np.ndarray,
    shared: dict[str, Any],
) -> list[Event]:
    """Split active regions at stable pitch jumps, then join short dropouts."""
    from scipy.ndimage import median_filter

    split_threshold = float(shared["event_pitch_split_threshold_cents"]) / float(
        shared["cents_per_semitone"]
    )
    split_filter = int(shared["event_pitch_split_median_frames"])
    events: list[Event] = []
    for onset, offset, _, _ in segments:
        indices = np.flatnonzero(
            (times >= onset) & (times <= offset) & np.isfinite(midi_track)
        )
        if indices.size == 0:
            continue
        values = midi_track[indices]
        filtered = median_filter(values, size=split_filter, mode="nearest")
        boundaries = np.flatnonzero(np.abs(np.diff(filtered)) >= split_threshold) + 1
        for run in np.split(np.arange(indices.size), boundaries):
            if run.size == 0:
                continue
            run_indices = indices[run]
            run_onset = max(onset, float(times[run_indices[0]]))
            run_offset = min(duration, offset, float(times[run_indices[-1]]))
            if run_offset - run_onset < float(shared["minimum_event_seconds"]):
                continue
            run_values = midi_track[run_indices]
            events.append(
                Event(
                    onset=run_onset,
                    offset=run_offset,
                    midi=float(np.median(run_values)),
                    confidence=1.0,
                    contour=tuple(float(value) for value in run_values),
                )
            )

    merged: list[Event] = []
    maximum_gap = float(shared["event_merge_gap_seconds"])
    maximum_pitch_difference = float(shared["event_merge_pitch_cents"]) / float(
        shared["cents_per_semitone"]
    )
    for event in sorted(events, key=lambda item: (item.onset, item.offset)):
        if (
            merged
            and event.onset - merged[-1].offset <= maximum_gap
            and abs(event.midi - merged[-1].midi) <= maximum_pitch_difference
        ):
            previous = merged.pop()
            contour = previous.contour + event.contour
            merged.append(
                Event(
                    onset=previous.onset,
                    offset=event.offset,
                    midi=float(np.median(contour)),
                    confidence=min(previous.confidence, event.confidence),
                    contour=contour,
                )
            )
        else:
            merged.append(event)

    glitch_maximum = float(shared["event_glitch_max_seconds"])
    index = 1
    while index < len(merged) - 1:
        previous, glitch, following = merged[index - 1 : index + 2]
        if (
            glitch.duration <= glitch_maximum
            and following.onset - previous.offset <= maximum_gap + glitch.duration
            and abs(previous.midi - following.midi) <= maximum_pitch_difference
        ):
            contour = previous.contour + following.contour
            merged[index - 1 : index + 2] = [
                Event(
                    onset=previous.onset,
                    offset=following.offset,
                    midi=float(np.median(contour)),
                    confidence=min(previous.confidence, following.confidence),
                    contour=contour,
                )
            ]
            index = max(1, index - 1)
        else:
            index += 1
    return merged


def _dsp_analysis(audio_path: str | Path, need_polyphony: bool) -> Analysis:
    config, _, _ = _load_config()
    shared = config["shared_decoder"]
    dsp = config["baselines"]["dsp"]
    yin_cfg = dsp["yin"]
    activity_cfg = dsp["activity"]
    target_sr = int(dsp["sample_rate"])
    audio, sample_rate = _read_audio(audio_path, target_sr)
    duration = len(audio) / sample_rate

    import librosa
    from scipy.ndimage import binary_closing

    activity_hop = int(activity_cfg["hop_length"])
    rms = librosa.feature.rms(
        y=audio,
        frame_length=int(activity_cfg["frame_length"]),
        hop_length=activity_hop,
        center=True,
    )[0]
    rms_threshold = max(
        float(activity_cfg["absolute_rms_floor"]),
        float(activity_cfg["relative_rms_threshold"]) * float(np.max(rms, initial=0.0)),
    )
    active = rms >= rms_threshold
    closing_frames = max(
        1,
        round(float(activity_cfg["closing_seconds"]) * sample_rate / activity_hop),
    )
    active = binary_closing(active, structure=np.ones(closing_frames, dtype=bool))
    activity_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sample_rate, hop_length=activity_hop
    )

    yin_hop = int(yin_cfg["hop_length"])
    f0 = librosa.yin(
        audio,
        fmin=float(yin_cfg["fmin_hz"]),
        fmax=min(float(yin_cfg["fmax_hz"]), sample_rate * 0.49),
        sr=sample_rate,
        frame_length=int(yin_cfg["frame_length"]),
        hop_length=yin_hop,
        trough_threshold=float(yin_cfg["trough_threshold"]),
        center=bool(yin_cfg["center"]),
        pad_mode=str(yin_cfg["pad_mode"]),
    )
    yin_times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sample_rate, hop_length=yin_hop
    )
    active_at_yin = (
        np.interp(
            yin_times,
            activity_times,
            active.astype(float),
            left=0.0,
            right=0.0,
        )
        >= 0.5
    )
    midi_track = np.full(len(f0), np.nan, dtype=np.float32)
    valid = active_at_yin & np.isfinite(f0) & (f0 > 0)
    midi_track[valid] = librosa.hz_to_midi(f0[valid]).astype(np.float32)

    segments = _segments_from_mask(
        active,
        activity_times,
        minimum_seconds=float(shared["minimum_event_seconds"]),
        frame_seconds=int(activity_cfg["frame_length"]) / sample_rate,
    )
    events = _events_from_activity_and_pitch(
        segments,
        duration=duration,
        times=yin_times,
        midi_track=midi_track,
        shared=shared,
    )

    poly_times = np.array([], dtype=np.float32)
    poly_frames: tuple[tuple[float, ...], ...] = ()
    if need_polyphony:
        poly_times, poly_frames = _dsp_multipitch(audio, sample_rate, dsp["multipitch"])

    return Analysis(
        duration=duration,
        events=tuple(events),
        frame_times=poly_times if need_polyphony else yin_times,
        monophonic_midi=midi_track,
        polyphonic_pitches=poly_frames,
    )


def _dsp_multipitch(
    audio: np.ndarray,
    sample_rate: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
    import librosa
    from scipy.ndimage import median_filter

    bins_per_octave = int(settings["bins_per_octave"])
    hop_length = int(settings["hop_length"])
    fmin_midi = int(settings["fmin_midi"])
    n_bins = int(settings["n_octaves"]) * bins_per_octave
    magnitude = np.abs(
        librosa.cqt(
            audio,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=float(librosa.midi_to_hz(fmin_midi)),
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )
    config, _, _ = _load_config()
    shared = config["shared_decoder"]
    candidates = np.arange(
        int(shared["analysis_midi_min"]),
        int(shared["analysis_midi_max"]) + 1,
    )
    harmonics = list(settings["harmonics"])
    weights = list(settings["harmonic_weights"])
    if len(harmonics) != len(weights):
        raise ValueError(
            "multipitch harmonics and harmonic_weights must have equal length"
        )

    salience = np.zeros((len(candidates), magnitude.shape[1]), dtype=np.float32)
    fundamental = np.zeros_like(salience)
    for row, midi in enumerate(candidates):
        fundamental_index = round((midi - fmin_midi) * bins_per_octave / 12)
        if 0 <= fundamental_index < n_bins:
            fundamental[row] = magnitude[fundamental_index]
        for harmonic, weight in zip(harmonics, weights):
            harmonic_midi = midi + 12 * math.log2(float(harmonic))
            index = round((harmonic_midi - fmin_midi) * bins_per_octave / 12)
            if 0 <= index < n_bins:
                salience[row] += float(weight) * magnitude[index]
    salience = median_filter(
        salience,
        size=(1, int(settings["activation_median_filter_frames"])),
        mode="nearest",
    )
    frames: list[tuple[float, ...]] = []
    relative_threshold = float(settings["salience_relative_threshold"])
    fundamental_threshold = float(settings["fundamental_relative_threshold"])
    maximum_polyphony = int(settings["maximum_polyphony"])
    minimum_separation = float(settings["minimum_pitch_separation_semitones"])
    for frame_index in range(salience.shape[1]):
        column = salience[:, frame_index]
        fund_column = fundamental[:, frame_index]
        maximum = float(np.max(column, initial=0.0))
        fundamental_maximum = float(np.max(fund_column, initial=0.0))
        if maximum <= 0 or fundamental_maximum <= 0:
            frames.append(())
            continue
        ranked = np.argsort(column)[::-1]
        selected: list[float] = []
        for index in ranked:
            if column[index] < relative_threshold * maximum:
                break
            if fund_column[index] < fundamental_threshold * fundamental_maximum:
                continue
            midi = float(candidates[index])
            if any(abs(midi - prior) < minimum_separation for prior in selected):
                continue
            selected.append(midi)
            if len(selected) >= maximum_polyphony:
                break
        frames.append(tuple(sorted(selected)))
    times = librosa.frames_to_time(
        np.arange(len(frames)), sr=sample_rate, hop_length=hop_length
    ).astype(np.float32)
    return times, tuple(frames)


def _basic_pitch_analysis(audio_path: str | Path, need_polyphony: bool) -> Analysis:
    del need_polyphony
    config, _, _ = _load_config()
    settings = config["baselines"]["basic_pitch"]
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    global _basic_pitch_model
    with _analysis_lock:
        if _basic_pitch_model is None:
            model_path = Path(ICASSP_2022_MODEL_PATH)
            if model_path.name != settings["model_filename"]:
                raise RuntimeError(
                    f"Expected Basic Pitch {settings['model_filename']}, got {model_path}"
                )
            _basic_pitch_model = Model(model_path)
        model = _basic_pitch_model

    output, _, note_events = predict(
        audio_path,
        model,
        onset_threshold=float(settings["onset_threshold"]),
        frame_threshold=float(settings["frame_threshold"]),
        minimum_note_length=float(settings["minimum_note_length_ms"]),
        minimum_frequency=settings["minimum_frequency_hz"],
        maximum_frequency=settings["maximum_frequency_hz"],
        multiple_pitch_bends=bool(settings["multiple_pitch_bends"]),
        melodia_trick=bool(settings["melodia_trick"]),
    )
    contour_bins = float(settings["contour_bins_per_semitone"])
    events: list[Event] = []
    duration = 0.0
    for onset, offset, midi, confidence, bends in note_events:
        bend_values = tuple(
            float(midi) + float(bend) / contour_bins for bend in (bends or [])
        )
        representative = float(np.median(bend_values)) if bend_values else float(midi)
        events.append(
            Event(
                onset=float(onset),
                offset=float(offset),
                midi=representative,
                confidence=float(confidence),
                contour=bend_values,
            )
        )
        duration = max(duration, float(offset))

    frame_count = int(output["contour"].shape[0])
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

    times = np.arange(frame_count, dtype=np.float32) * FFT_HOP / AUDIO_SAMPLE_RATE
    if frame_count:
        duration = max(duration, float(times[-1]))
    mono = np.full(frame_count, np.nan, dtype=np.float32)
    poly_frames: list[tuple[float, ...]] = []
    for frame_index, time_s in enumerate(times):
        active = [event for event in events if event.onset <= time_s <= event.offset]
        active.sort(key=lambda event: event.midi)
        pitches: list[float] = []
        for event in active:
            if event.contour:
                relative = (time_s - event.onset) / max(event.duration, 1e-9)
                contour_index = min(
                    len(event.contour) - 1,
                    max(0, round(relative * (len(event.contour) - 1))),
                )
                pitches.append(event.contour[contour_index])
            else:
                pitches.append(event.midi)
        poly_frames.append(tuple(pitches))
        if pitches:
            mono[frame_index] = pitches[int(np.argmax([e.confidence for e in active]))]
    return Analysis(
        duration=duration,
        events=tuple(sorted(events, key=lambda event: (event.onset, event.midi))),
        frame_times=times,
        monophonic_midi=mono,
        polyphonic_pitches=tuple(poly_frames),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _muscriptor_analysis(audio_path: str | Path, need_polyphony: bool) -> Analysis:
    del need_polyphony
    config, _, _ = _load_config()
    settings = config["baselines"]["muscriptor"]
    source_dir = (REPO_ROOT / settings["source_dir"]).resolve()
    checkpoint = (REPO_ROOT / settings["checkpoint"]).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"MuScriptor source directory is missing: {source_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MuScriptor checkpoint is missing: {checkpoint}")

    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from muscriptor import TranscriptionModel
    from muscriptor.events import NoteEndEvent, NoteStartEvent

    global _muscriptor_model
    with _analysis_lock:
        if _muscriptor_model is None:
            actual_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=source_dir,
                text=True,
            ).strip()
            if actual_revision != str(settings["source_revision"]):
                raise RuntimeError(
                    f"MuScriptor source revision mismatch: expected "
                    f"{settings['source_revision']}, found {actual_revision}"
                )
            actual_checkpoint_sha = _sha256(checkpoint)
            if actual_checkpoint_sha != str(settings["checkpoint_sha256"]):
                raise RuntimeError(
                    f"MuScriptor checkpoint SHA256 mismatch: expected "
                    f"{settings['checkpoint_sha256']}, found {actual_checkpoint_sha}"
                )
            _muscriptor_model = TranscriptionModel.load_model(
                checkpoint,
                device=str(settings["device"]),
            )
        model = _muscriptor_model

    events: list[Event] = []
    for item in model.transcribe(
        audio_path,
        use_sampling=bool(settings["use_sampling"]),
        temperature=float(settings["temperature"]),
        cfg_coef=float(settings["cfg_coef"]),
        instruments=settings["instruments"],
        batch_size=int(settings["batch_size"]),
        no_eos_is_ok=bool(settings["no_eos_is_ok"]),
        beam_size=int(settings["beam_size"]),
    ):
        if isinstance(item, NoteStartEvent):
            continue
        if not isinstance(item, NoteEndEvent):
            continue
        start = item.start_event
        onset = max(0.0, float(start.start_time))
        offset = max(onset, float(item.end_time))
        if offset <= onset:
            continue
        events.append(
            Event(
                onset=onset,
                offset=offset,
                midi=float(start.pitch),
                confidence=1.0,
                instrument=str(start.instrument),
            )
        )

    import soundfile as sf

    audio_info = sf.info(str(audio_path))
    duration = float(audio_info.frames) / float(audio_info.samplerate)
    frame_hop = float(settings["frame_hop_seconds"])
    if frame_hop <= 0:
        raise ValueError("baselines.muscriptor.frame_hop_seconds must be positive")
    frame_count = max(1, math.ceil(duration / frame_hop) + 1)
    times = np.arange(frame_count, dtype=np.float32) * frame_hop
    mono = np.full(frame_count, np.nan, dtype=np.float32)
    mono_score = np.full(frame_count, -np.inf, dtype=np.float32)
    poly_sets: list[set[float]] = [set() for _ in range(frame_count)]
    for event in events:
        start_index = max(0, math.ceil(event.onset / frame_hop))
        end_index = min(frame_count, math.floor(event.offset / frame_hop) + 1)
        if end_index <= start_index:
            continue
        for index in range(start_index, end_index):
            poly_sets[index].add(event.midi)
        score = event.duration
        indices = np.arange(start_index, end_index)
        replace = score > mono_score[indices]
        selected = indices[replace]
        mono[selected] = event.midi
        mono_score[selected] = score

    return Analysis(
        duration=duration,
        events=tuple(
            sorted(events, key=lambda event: (event.onset, event.midi, event.offset))
        ),
        frame_times=times,
        monophonic_midi=mono,
        polyphonic_pitches=tuple(tuple(sorted(pitches)) for pitches in poly_sets),
    )


def _analysis(
    model_name: str, audio_path: str | Path, need_polyphony: bool
) -> Analysis:
    key = (model_name, str(Path(audio_path).resolve()))
    with _analysis_lock:
        cached = _analysis_cache.get(key)
    if cached is not None and (not need_polyphony or cached.polyphonic_pitches):
        return cached
    backend = model_name.removeprefix(BASELINE_PREFIX)
    if backend == "dsp":
        result = _dsp_analysis(audio_path, need_polyphony)
    elif backend == "basic-pitch":
        result = _basic_pitch_analysis(audio_path, need_polyphony)
    elif backend == "muscriptor":
        result = _muscriptor_analysis(audio_path, need_polyphony)
    else:
        raise ValueError(f"Unknown baseline backend: {model_name}")
    with _analysis_lock:
        _analysis_cache[key] = result
    return result


def _format_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    minutes, remainder = divmod(centiseconds, 6000)
    whole_seconds, hundredths = divmod(remainder, 100)
    return f"{minutes}:{whole_seconds:02d}.{hundredths:02d}"


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_SOLFEGE = (
    "do",
    "do#",
    "re",
    "re#",
    "mi",
    "fa",
    "fa#",
    "sol",
    "sol#",
    "la",
    "la#",
    "si",
)


def _pitch_format(prompt: str) -> str:
    lower = prompt.lower()
    if "solf" in lower or "fixed-do" in lower:
        return "doremi"
    if "scientific pitch notation" in lower or "note name and octave" in lower:
        return "spn"
    if "hertz" in lower or re.search(r"\bhz\b", lower):
        return "hz"
    return "midi"


def _format_pitch(midi: float, prompt: str) -> str:
    output_format = _pitch_format(prompt)
    nearest = round(midi)
    if output_format == "midi":
        return str(nearest)
    if output_format == "spn":
        return f"{_NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}"
    if output_format == "doremi":
        return _SOLFEGE[nearest % 12]
    return f"{440.0 * 2 ** ((midi - 69.0) / 12.0):.4f}"


def _format_pitch_list(pitches: list[float], prompt: str) -> str:
    return " ".join(_format_pitch(pitch, prompt) for pitch in pitches)


def _dominant_event(analysis: Analysis) -> Event | None:
    if not analysis.events:
        return None
    return max(
        analysis.events,
        key=lambda event: (
            event.duration * max(event.confidence, 1e-6),
            event.confidence,
        ),
    )


def _events_in_order(analysis: Analysis) -> list[Event]:
    return sorted(
        analysis.events, key=lambda event: (event.onset, event.offset, event.midi)
    )


def _pitch_at_time(analysis: Analysis, timestamp: float) -> float | None:
    active = [
        event for event in analysis.events if event.onset <= timestamp <= event.offset
    ]
    if active:
        return max(active, key=lambda event: event.confidence).midi
    if analysis.frame_times.size and analysis.monophonic_midi.size:
        index = int(np.argmin(np.abs(analysis.frame_times - timestamp)))
        value = float(analysis.monophonic_midi[index])
        return value if np.isfinite(value) else None
    return None


def _two_region_pitches(analysis: Analysis) -> tuple[float, float] | None:
    if not analysis.frame_times.size or not analysis.monophonic_midi.size:
        return None
    config, _, _ = _load_config()
    shared = config["shared_decoder"]
    first_fractions = [float(value) for value in shared["d2_first_region_fractions"]]
    second_fractions = [float(value) for value in shared["d2_second_region_fractions"]]
    if not (
        len(first_fractions) == len(second_fractions) == 2
        and 0
        <= first_fractions[0]
        < first_fractions[1]
        < second_fractions[0]
        < second_fractions[1]
        <= 1
    ):
        raise ValueError(
            "D2 region fractions must be two ordered, non-overlapping intervals"
        )
    first_start, first_end = (
        fraction * analysis.duration for fraction in first_fractions
    )
    second_start, second_end = (
        fraction * analysis.duration for fraction in second_fractions
    )
    first = analysis.monophonic_midi[
        (analysis.frame_times >= first_start) & (analysis.frame_times <= first_end)
    ]
    second = analysis.monophonic_midi[
        (analysis.frame_times >= second_start) & (analysis.frame_times <= second_end)
    ]
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if not first.size or not second.size:
        return None
    return float(np.median(first)), float(np.median(second))


def _prompt_timestamp(prompt: str) -> float:
    match = re.search(r"exactly\s+(\d+):(\d+(?:\.\d+)?)", prompt, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse queried timestamp from prompt: {prompt}")
    return 60 * int(match.group(1)) + float(match.group(2))


def _simultaneous_pitches(analysis: Analysis, model_name: str) -> list[float]:
    config, _, _ = _load_config()
    shared = config["shared_decoder"]
    if analysis.polyphonic_pitches and analysis.frame_times.size:
        start = float(shared["steady_state_start_fraction"]) * analysis.duration
        end = float(shared["steady_state_end_fraction"]) * analysis.duration
        selected_frames = [
            frame
            for time_s, frame in zip(analysis.frame_times, analysis.polyphonic_pitches)
            if start <= time_s <= end
        ]
        counts: dict[int, int] = {}
        for frame in selected_frames:
            for pitch in {round(value) for value in frame}:
                counts[pitch] = counts.get(pitch, 0) + 1
        if selected_frames:
            backend = model_name.removeprefix(BASELINE_PREFIX)
            if backend == "muscriptor":
                fraction = float(
                    config["baselines"]["muscriptor"]["chord_presence_fraction"]
                )
            else:
                fraction = float(
                    config["baselines"]["dsp"]["multipitch"]["chord_presence_fraction"]
                )
            required = max(1, math.ceil(fraction * len(selected_frames)))
            pitches = sorted(
                float(pitch) for pitch, count in counts.items() if count >= required
            )
            if pitches:
                return pitches
    midpoint = analysis.duration / 2
    active = [
        event.midi
        for event in analysis.events
        if event.onset <= midpoint <= event.offset
    ]
    return sorted(active)


def _closest_chord_quality(pitches: list[float]) -> str | None:
    if not pitches:
        return None
    config, _, _ = _load_config()
    templates = config["shared_decoder"]["chord_templates"]
    tie_order = config["shared_decoder"]["chord_template_tie_order"]
    observed = {round(pitch) % 12 for pitch in pitches}
    best: tuple[int, int, int, str] | None = None
    for quality_index, quality in enumerate(tie_order):
        intervals = templates[quality]
        for root in range(12):
            candidate = {(root + int(interval)) % 12 for interval in intervals}
            symmetric_difference = len(observed ^ candidate)
            missing = len(candidate - observed)
            key = (symmetric_difference, missing, quality_index, quality)
            if best is None or key < best:
                best = key
    return best[-1] if best else None


def _contour_directions(
    values: np.ndarray, expected_max: int | None = None
) -> list[str]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return []
    config, _, _ = _load_config()
    from scipy.ndimage import median_filter

    filtered = median_filter(
        finite,
        size=int(config["shared_decoder"]["contour_median_filter_frames"]),
        mode="nearest",
    )
    edge_count = max(1, round(0.15 * len(filtered)))
    middle_start = max(0, round(0.40 * len(filtered)))
    middle_end = min(len(filtered), round(0.60 * len(filtered)))
    start = float(np.median(filtered[:edge_count]))
    middle = float(np.median(filtered[middle_start:middle_end]))
    end = float(np.median(filtered[-edge_count:]))
    threshold = float(config["shared_decoder"]["pitch_change_threshold_cents"]) / float(
        config["shared_decoder"]["cents_per_semitone"]
    )
    if middle > start + threshold and middle > end + threshold:
        directions = ["up", "down"]
    elif middle < start - threshold and middle < end - threshold:
        directions = ["down", "up"]
    elif end > start + threshold:
        directions = ["up"]
    elif end < start - threshold:
        directions = ["down"]
    else:
        directions = []
    if expected_max is not None:
        directions = directions[:expected_max]
    return directions


def _parse_int_from_prompt(pattern: str, prompt: str) -> int | None:
    match = re.search(pattern, prompt, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _target_voice(prompt: str, n_voices: int) -> int:
    lower = prompt.lower()
    if "soprano" in lower or "highest melodic line" in lower:
        return 0
    if "bass voice" in lower or "lowest melodic line" in lower:
        return n_voices - 1
    if "alto voice" in lower or "second-from-top" in lower:
        return 1
    if "tenor voice" in lower or "third-from-top" in lower:
        return 2
    match = re.search(r"(\w+)-from-top", lower)
    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    if match and match.group(1) in ordinals:
        return min(n_voices - 1, ordinals[match.group(1)] - 1)
    return 0


def _rank_track(analysis: Analysis, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    n_voices = (
        _parse_int_from_prompt(r"hear\s+(\d+)\s+simultaneous melodic lines", prompt)
        or 4
    )
    target_rank = _target_voice(prompt, n_voices)
    times: list[float] = []
    pitches: list[float] = []
    for time_s, frame in zip(analysis.frame_times, analysis.polyphonic_pitches):
        ordered = sorted(frame, reverse=True)
        if len(ordered) <= target_rank:
            continue
        times.append(float(time_s))
        pitches.append(float(ordered[target_rank]))
    return np.asarray(times), np.asarray(pitches)


def _track_events_fixed_count(
    times: np.ndarray,
    pitches: np.ndarray,
    expected_count: int | None,
) -> list[float]:
    if pitches.size == 0:
        return []
    rounded = np.rint(pitches).astype(int)
    runs: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(rounded) + 1):
        if index == len(rounded) or rounded[index] != rounded[start]:
            runs.append(
                {
                    "start": start,
                    "end": index,
                    "pitch": float(np.median(pitches[start:index])),
                }
            )
            start = index
    if expected_count is not None:
        while len(runs) > expected_count:
            merge_index = min(
                range(len(runs) - 1),
                key=lambda index: (
                    min(
                        runs[index]["end"] - runs[index]["start"],
                        runs[index + 1]["end"] - runs[index + 1]["start"],
                    ),
                    abs(runs[index]["pitch"] - runs[index + 1]["pitch"]),
                ),
            )
            left, right = runs[merge_index], runs[merge_index + 1]
            merged_slice = pitches[left["start"] : right["end"]]
            runs[merge_index : merge_index + 2] = [
                {
                    "start": left["start"],
                    "end": right["end"],
                    "pitch": float(np.median(merged_slice)),
                }
            ]
        while len(runs) < expected_count:
            split_index = max(
                range(len(runs)),
                key=lambda index: runs[index]["end"] - runs[index]["start"],
            )
            run = runs[split_index]
            if run["end"] - run["start"] < 2:
                break
            middle = (run["start"] + run["end"]) // 2
            runs[split_index : split_index + 1] = [
                {
                    "start": run["start"],
                    "end": middle,
                    "pitch": float(np.median(pitches[run["start"] : middle])),
                },
                {
                    "start": middle,
                    "end": run["end"],
                    "pitch": float(np.median(pitches[middle : run["end"]])),
                },
            ]
    return [run["pitch"] for run in runs]


def _sequence_pitches(
    analysis: Analysis,
    expected_count: int | None,
) -> list[float]:
    """Convert the monophonic frame track into ordered, stable pitch runs."""
    if analysis.frame_times.size and analysis.monophonic_midi.size:
        finite = np.isfinite(analysis.monophonic_midi)
        if np.any(finite):
            return _track_events_fixed_count(
                analysis.frame_times[finite],
                analysis.monophonic_midi[finite],
                expected_count,
            )
    pitches = [event.midi for event in _events_in_order(analysis)]
    if expected_count is not None:
        pitches = pitches[:expected_count]
    return pitches


def _event_pitches_without_adjacent_duplicates(analysis: Analysis) -> list[float]:
    """Keep decoded event order while collapsing same-note fragmentation."""
    pitches: list[float] = []
    for event in _events_in_order(analysis):
        if pitches and round(event.midi) == round(pitches[-1]):
            continue
        pitches.append(event.midi)
    return pitches


def _missing() -> str:
    config, _, _ = _load_config()
    return str(config["runtime"]["missing_prediction_text"])


def _decode(model_name: str, audio_path: str | Path, prompt: str) -> str:
    experiment = _experiment_id(audio_path)
    polyphonic = experiment in {"c1", "c2", "c3", "c4", "f1", "f2"}
    analysis = _analysis(model_name, audio_path, polyphonic)
    events = _events_in_order(analysis)

    if experiment == "b2":
        pitch = _pitch_at_time(analysis, _prompt_timestamp(prompt))
        return _format_pitch(pitch, prompt) if pitch is not None else _missing()

    if experiment == "b3":
        if not events:
            return _missing()
        return f"{_format_timestamp(events[0].onset)}, {_format_timestamp(events[-1].offset)}"

    if experiment == "b4":
        target = _parse_int_from_prompt(r"\(MIDI\s+(\d+)", prompt)
        if target is None or not events:
            return _missing()
        match = min(events, key=lambda event: abs(event.midi - target))
        return f"{_format_timestamp(match.onset)}, {_format_timestamp(match.offset)}"

    if experiment == "b5":
        if not events:
            return _missing()
        timestamps = [
            timestamp
            for event in events
            for timestamp in (
                _format_timestamp(event.onset),
                _format_timestamp(event.offset),
            )
        ]
        return ", ".join(timestamps)

    if experiment in {"c1", "c2", "c3", "c4"}:
        pitches = _simultaneous_pitches(analysis, model_name)
        if experiment == "c1":
            return (
                str(len({round(pitch) for pitch in pitches})) if pitches else _missing()
            )
        if experiment == "c2":
            return (
                str(round(max(pitches) - min(pitches)))
                if len(pitches) >= 2
                else _missing()
            )
        if experiment == "c3":
            return _closest_chord_quality(pitches) or _missing()
        return _format_pitch_list(sorted(pitches), prompt) if pitches else _missing()

    if experiment == "d1":
        distinct = {round(event.midi) for event in events}
        return str(len(distinct)) if distinct else _missing()

    if experiment == "d2":
        pair = _two_region_pitches(analysis)
        if pair is None and len(events) >= 2:
            pair = (events[0].midi, events[1].midi)
        if pair is None:
            return _missing()
        return "first" if pair[0] > pair[1] else "second"

    if experiment == "d3":
        pitches = _event_pitches_without_adjacent_duplicates(analysis)
        if len(pitches) < 2:
            return _missing()
        directions = [
            "up" if second > first else "down"
            for first, second in itertools.pairwise(pitches)
        ]
        return ", ".join(directions)

    if experiment == "d4":
        directions = _contour_directions(analysis.monophonic_midi, expected_max=2)
        return ", ".join(directions) if directions else _missing()

    if experiment == "d5":
        expected = _parse_int_from_prompt(r"^(\d+)\s+tones", prompt)
        pitches = _sequence_pitches(analysis, expected)
        if not pitches:
            return _missing()
        order = sorted(range(len(pitches)), key=lambda index: pitches[index])
        return " ".join(str(index + 1) for index in order)

    if experiment == "d6":
        pitches = _sequence_pitches(analysis, expected_count=2)
        if len(pitches) < 2:
            return _missing()
        return str(round(pitches[1] - pitches[0]))

    if experiment == "d7a":
        pitches = _sequence_pitches(analysis, expected_count=2)
        if len(pitches) < 2:
            return _missing()
        interval = pitches[1] - pitches[0]
        output_format = _pitch_format(prompt)
        if output_format == "midi":
            reference = _parse_int_from_prompt(
                r"FIRST tone is MIDI note\s+(\d+)", prompt
            )
            return (
                str(round(reference + interval))
                if reference is not None
                else _missing()
            )
        if output_format == "hz":
            match = re.search(r"FIRST tone is\s+(\d+(?:\.\d+)?)\s+Hz", prompt)
            if not match:
                return _missing()
            reference_midi = 69 + 12 * math.log2(float(match.group(1)) / 440.0)
            return _format_pitch(reference_midi + interval, prompt)
        note_match = re.search(
            r"FIRST tone is\s+'?([A-Ga-g](?:#|b)?-?\d+|'?(?:do|re|mi|fa|sol|la|si)(?:#|b)?)",
            prompt,
        )
        if not note_match:
            return _missing()
        token = note_match.group(1).strip("'").lower()
        if output_format == "spn":
            pitch_class = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[
                token[0]
            ]
            accidental = 1 if "#" in token else (-1 if "b" in token[1:] else 0)
            octave = int(re.search(r"-?\d+", token).group())
            reference_midi = 12 * (octave + 1) + pitch_class + accidental
        else:
            reverse_solfege = {name: index for index, name in enumerate(_SOLFEGE)}
            reference_midi = float(reverse_solfege[token])
        return _format_pitch(reference_midi + interval, prompt)

    if experiment == "d8":
        expected = _parse_int_from_prompt(r"hear\s+(\d+)\s+musical notes", prompt)
        pitches = _sequence_pitches(analysis, expected)
        return _format_pitch_list(pitches, prompt) if pitches else _missing()

    if experiment in {"f1", "f2"}:
        expected = _parse_int_from_prompt(
            r"(?:exactly|plays exactly)\s+(\d+)\s+notes", prompt
        )
        times, pitches = _rank_track(analysis, prompt)
        track = _track_events_fixed_count(times, pitches, expected)
        return _format_pitch_list(track, prompt) if track else _missing()

    event = _dominant_event(analysis)
    if experiment == "e6" and event is not None:
        return _format_pitch(float(round(event.midi)), prompt)
    return _format_pitch(event.midi, prompt) if event is not None else _missing()


def query_baseline(
    model_name: str,
    audio_path: str | Path,
    prompt: str,
) -> dict[str, Any]:
    if not model_name.startswith(BASELINE_PREFIX):
        raise ValueError(f"Not a baseline model: {model_name}")
    result = _decode(model_name, audio_path, prompt)
    return {
        "result": result,
        "raw_response": {
            "offline_baseline": True,
            "answer": result,
            "experiment_id": _experiment_id(audio_path),
        },
        "top_tokens": None,
        "embedding": None,
        "usage": None,
    }


def dump_cache_manifest() -> str:
    """Return a compact diagnostic manifest for receipts and tests."""
    with _analysis_lock:
        entries = [
            {
                "model": model,
                "audio_path": path,
                "event_count": len(analysis.events),
            }
            for (model, path), analysis in _analysis_cache.items()
        ]
    entries.sort(key=lambda item: (item["model"], item["audio_path"]))
    return json.dumps(entries, sort_keys=True)
