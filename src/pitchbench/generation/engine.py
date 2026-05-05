"""
Central audio engine for PitchBench.

All experiments generate their stimuli through this module.  Generated files
are stored under config.AUDIO_DIR (data/audio/) with descriptive names and are
never overwritten — if a file already exists it is returned immediately.

Filename convention:  {notes}_{source}_{descriptor}.wav
  notes      single MIDI   : "C4", "Cs4" (C#4), "As4" (A#4)
             sequence      : "C4-E4-G4"  (hyphen-separated)
             chord         : "C4+E4+G4"  (plus-separated)
             glide         : "C4~A4"     (tilde)
             arbitrary Hz  : "440hz"     (rounded to nearest integer)
  source     waveform or instrument name: "sine", "piano", etc.
  descriptor encodes extra parameters:  "2000ms", "seq_750ms_300msgap",
             "chord_1500ms", "in_silence_at45000ms_for2000ms_of60000ms",
             "glide_3000ms_linear", "vol-12db", "fx_reverb_l"

Waveform sources:  always available (numpy only)
Instrument sources: require FluidSynth (pyfluidsynth + SF2 soundfont).
                    Missing FluidSynth → ValueError with a clear install hint.
"""

from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

import pitchbench.config as config

# ── Module-level constants ────────────────────────────────────────────────────

SR          = config.SAMPLE_RATE
AUDIO_DIR   = config.AUDIO_DIR
_FADE_S     = 0.01   # 10 ms fade-in/out on every synthesised segment

_current_exp: str = ""


def set_exp(exp_name: str) -> None:
    """Set the active experiment name; audio files go in AUDIO_DIR/{exp_name}/."""
    global _current_exp
    _current_exp = exp_name


def _audio_dir() -> Path:
    if _current_exp:
        d = AUDIO_DIR / _current_exp
        d.mkdir(parents=True, exist_ok=True)
        return d
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_DIR

# ── Note slug helpers ─────────────────────────────────────────────────────────

_NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]


def _note_slug(midi: int) -> str:
    return _NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def _hz_slug(freq_hz: float) -> str:
    return f"{freq_hz:.1f}hz".replace(".", "p")


def _notes_slug(midis: Sequence[int], sep: str) -> str:
    return sep.join(_note_slug(m) for m in midis)


# ── Low-level WAV I/O ─────────────────────────────────────────────────────────

def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ── Waveform synthesis primitives ─────────────────────────────────────────────

def _envelope(n: int, fade_s: float = _FADE_S) -> np.ndarray:
    fade = min(int(SR * fade_s), n // 4)
    env = np.ones(n, dtype=np.float32)
    if fade > 0:
        env[:fade]  = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
    return env


def _synth_waveform(freq_hz: float, duration_s: float, waveform: str) -> np.ndarray:
    n = int(SR * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False)
    if waveform == "sine":
        sig = np.sin(2 * np.pi * freq_hz * t)
    elif waveform == "sawtooth":
        sig = 2 * ((freq_hz * t) % 1.0) - 1
    elif waveform == "square":
        sig = np.sign(np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    elif waveform == "triangle":
        phase = (freq_hz * t) % 1.0
        sig = 2 * np.abs(2 * phase - 1) - 1
    else:
        raise ValueError(f"Unknown waveform: {waveform!r}")
    audio = (sig * _envelope(n)).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak
    return audio


def _synth_glide(
    start_hz: float,
    end_hz: float,
    duration_s: float,
    waveform: str,
    shape: str = "linear",
) -> np.ndarray:
    """
    Synthesise a continuously-varying pitch via phase accumulation.
    shape:  "linear"       start → end
            "arch"         start → end → start (peak halfway)
            "valley"       start → end → start (trough halfway)
            "log"          logarithmic (perceptually linear in pitch)
    """
    n      = int(SR * duration_s)
    t_norm = np.linspace(0.0, 1.0, n, endpoint=False)

    if shape == "linear":
        freq_t = start_hz + (end_hz - start_hz) * t_norm

    elif shape in ("arch", "valley"):
        # go to end and back, parabolic shape
        peak_val = end_hz if shape == "arch" else (2 * start_hz - end_hz)
        freq_t   = start_hz + (peak_val - start_hz) * 4 * t_norm * (1 - t_norm)

    elif shape == "log":
        log_s = np.log2(start_hz)
        log_e = np.log2(max(end_hz, 1.0))
        freq_t = 2 ** (log_s + (log_e - log_s) * t_norm)

    else:
        raise ValueError(f"Unknown glide shape: {shape!r}")

    # Phase accumulation prevents discontinuities
    phase = np.zeros(n, dtype=np.float64)
    dt = 1.0 / SR
    for i in range(1, n):
        phase[i] = phase[i - 1] + 2 * np.pi * float(freq_t[i - 1]) * dt
    phase = phase.astype(np.float32)

    if waveform == "sine":
        sig = np.sin(phase)
    elif waveform == "sawtooth":
        sig = 2 * ((phase / (2 * np.pi)) % 1.0) - 1
    elif waveform == "square":
        sig = np.sign(np.sin(phase)).astype(np.float32)
    elif waveform == "triangle":
        p = (phase / (2 * np.pi)) % 1.0
        sig = 2 * np.abs(2 * p - 1) - 1
    else:
        raise ValueError(f"Unknown waveform: {waveform!r}")

    audio = (sig * _envelope(n)).astype(np.float32)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak
    return audio


# ── FluidSynth instrument rendering ──────────────────────────────────────────

def _render_instrument(
    midi: int,
    instrument: str,
    duration_s: float,
) -> np.ndarray:
    """Render one MIDI note via FluidSynth.  Raises if fluidsynth is missing."""
    try:
        import fluidsynth
    except ImportError:
        raise ValueError(
            f"FluidSynth not available — cannot render instrument {instrument!r}.\n"
            "Install with:  pip install pyfluidsynth\n"
            "and ensure libfluidsynth is present on your system."
        )

    sf2 = config.SF2_PATH
    program = config.GM_PROGRAMS_V1.get(instrument)
    if program is None:
        raise ValueError(f"Unknown instrument {instrument!r}. "
                         f"Available: {list(config.GM_PROGRAMS_V1)}")

    fs = fluidsynth.Synth(samplerate=float(SR))
    sfid = fs.sfload(sf2)
    fs.program_select(0, sfid, 0, program)

    note_on_dur  = max(min(duration_s * 0.85, duration_s - 0.1), duration_s * 0.5)
    release_dur  = duration_s - note_on_dur
    n_body  = int(note_on_dur  * SR)
    n_tail  = int(release_dur  * SR)
    n_total = int(duration_s   * SR)

    fs.noteon(0, midi, 100)
    body = np.array(fs.get_samples(n_body),  dtype=np.float32).reshape(-1, 2).mean(axis=1)
    fs.noteoff(0, midi)
    tail = np.array(fs.get_samples(n_tail),  dtype=np.float32).reshape(-1, 2).mean(axis=1)
    fs.delete()

    audio = np.concatenate([body, tail])[:n_total]
    peak  = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9
    # short fade-out
    fade_n = int(0.05 * SR)
    if fade_n < len(audio):
        audio[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    return audio.astype(np.float32)


def _render_instrument_detuned(
    midi: int,
    instrument: str,
    duration_s: float,
    detune_cents: float,
) -> np.ndarray:
    """Render a MIDI note via FluidSynth with a pitch-bend offset in cents.

    GM pitch bend range is ±200 cents (±2 semitones).  The maximum residual
    when snapping an arbitrary Hz value to the nearest MIDI note is 50 cents,
    well within range.
    """
    try:
        import fluidsynth
    except ImportError:
        raise ValueError(
            f"FluidSynth not available — cannot render instrument {instrument!r}.\n"
            "Install with:  pip install pyfluidsynth\n"
            "and ensure libfluidsynth is present on your system."
        )

    sf2 = config.SF2_PATH
    program = config.GM_PROGRAMS_V1.get(instrument)
    if program is None:
        raise ValueError(f"Unknown instrument {instrument!r}. "
                         f"Available: {list(config.GM_PROGRAMS_V1)}")

    _BEND_RANGE_CENTS = 200.0
    bend_value = int(round(8192 + detune_cents / _BEND_RANGE_CENTS * 8192))
    bend_value = max(0, min(16383, bend_value))

    fs = fluidsynth.Synth(samplerate=float(SR))
    sfid = fs.sfload(sf2)
    fs.program_select(0, sfid, 0, program)
    fs.pitch_bend(0, bend_value)

    note_on_dur  = max(min(duration_s * 0.85, duration_s - 0.1), duration_s * 0.5)
    release_dur  = duration_s - note_on_dur
    n_body  = int(note_on_dur  * SR)
    n_tail  = int(release_dur  * SR)
    n_total = int(duration_s   * SR)

    fs.noteon(0, midi, 100)
    body = np.array(fs.get_samples(n_body),  dtype=np.float32).reshape(-1, 2).mean(axis=1)
    fs.noteoff(0, midi)
    tail = np.array(fs.get_samples(n_tail),  dtype=np.float32).reshape(-1, 2).mean(axis=1)
    fs.delete()

    audio = np.concatenate([body, tail])[:n_total]
    peak  = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9
    fade_n = int(0.05 * SR)
    if fade_n < len(audio):
        audio[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    return audio.astype(np.float32)


# ── Source routing ────────────────────────────────────────────────────────────

def _is_waveform(source: str) -> bool:
    return source in config.WAVEFORMS


def _render_single(midi: int, duration_s: float, source: str) -> np.ndarray:
    if _is_waveform(source):
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        return _synth_waveform(freq, duration_s, source)
    return _render_instrument(midi, source, duration_s)


def _render_hz(freq_hz: float, duration_s: float, source: str) -> np.ndarray:
    if _is_waveform(source):
        return _synth_waveform(freq_hz, duration_s, source)
    # Instrument: snap to nearest MIDI note, apply residual as pitch bend
    midi = round(69 + 12 * math.log2(freq_hz / 440.0))
    midi_freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    residual_cents = 1200.0 * math.log2(freq_hz / midi_freq)
    return _render_instrument_detuned(midi, source, duration_s, residual_cents)


# ── Public API ────────────────────────────────────────────────────────────────

def tone(
    midi: int,
    source: str,
    duration_ms: int,
) -> Path:
    """Single sustained tone.  Returns path to cached WAV."""
    name = f"{_note_slug(midi)}_{source}_{duration_ms}ms.wav"
    path = _audio_dir() / name
    if path.exists():
        return path
    _write_wav(path, _render_single(midi, duration_ms / 1000, source))
    return path


def tone_hz(
    freq_hz: float,
    source: str,
    duration_ms: int,
) -> Path:
    """Single tone at an arbitrary frequency."""
    name = f"{_hz_slug(freq_hz)}_{source}_{duration_ms}ms.wav"
    path = _audio_dir() / name
    if path.exists():
        return path
    _write_wav(path, _render_hz(freq_hz, duration_ms / 1000, source))
    return path


def tone_with_vibrato(
    midi: int,
    source: str,
    duration_ms: int,
    rate_hz: float,
    depth_cents: float,
) -> Path:
    """Single sustained tone with sinusoidal vibrato (waveforms only).

    Vibrato is implemented via phase accumulation over a time-varying frequency
    f(t) = f0 * 2^( (depth/1200) * sin(2π·rate·t) ), preventing DC drift even
    with large depths. ``rate_hz=0`` or ``depth_cents=0`` gives a flat tone.
    """
    if not _is_waveform(source):
        raise ValueError("Vibrato synthesis only supported for waveform sources")
    slug = _note_slug(midi)
    name = f"{slug}_{source}_{duration_ms}ms_vib{rate_hz:g}hz{depth_cents:g}c.wav"
    path = _audio_dir() / name
    if path.exists():
        return path

    duration_s = duration_ms / 1000
    n          = int(SR * duration_s)
    t          = np.linspace(0.0, duration_s, n, endpoint=False)
    f0         = 440.0 * 2 ** ((midi - 69) / 12)

    if rate_hz <= 0 or depth_cents <= 0:
        freq_t = np.full(n, f0, dtype=np.float64)
    else:
        # cents → semitone → exponent
        freq_t = f0 * (2.0 ** ((depth_cents / 1200.0) * np.sin(2 * np.pi * rate_hz * t)))

    phase  = np.zeros(n, dtype=np.float64)
    dt     = 1.0 / SR
    for i in range(1, n):
        phase[i] = phase[i - 1] + 2 * np.pi * float(freq_t[i - 1]) * dt
    phase = phase.astype(np.float32)

    if source == "sine":
        sig = np.sin(phase)
    elif source == "sawtooth":
        sig = 2 * ((phase / (2 * np.pi)) % 1.0) - 1
    elif source == "square":
        sig = np.sign(np.sin(phase)).astype(np.float32)
    elif source == "triangle":
        p   = (phase / (2 * np.pi)) % 1.0
        sig = 2 * np.abs(2 * p - 1) - 1
    else:
        raise ValueError(f"Unknown waveform: {source!r}")

    audio = (sig * _envelope(n)).astype(np.float32)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak
    _write_wav(path, audio)
    return path


def sequence(
    midis: Sequence[int],
    source: str,
    tone_ms: int,
    gap_ms: int,
) -> Path:
    """Notes played sequentially with a silence gap between each."""
    notes = _notes_slug(midis, "-")
    name  = f"{notes}_{source}_seq_{tone_ms}ms_{gap_ms}msgap.wav"
    path  = _audio_dir() / name
    if path.exists():
        return path

    tone_s = tone_ms / 1000
    gap    = np.zeros(int(SR * gap_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, m in enumerate(midis):
        parts.append(_render_single(m, tone_s, source))
        if i < len(midis) - 1:
            parts.append(gap)
    audio = np.concatenate(parts)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak
    _write_wav(path, audio)
    return path


def sequence_hz(
    freqs_hz: Sequence[float],
    source: str,
    tone_ms: int,
    gap_ms: int,
) -> Path:
    """Sequential tones at arbitrary Hz."""
    slugs = "-".join(_hz_slug(f) for f in freqs_hz)
    name  = f"{slugs}_{source}_seq_{tone_ms}ms_{gap_ms}msgap.wav"
    path  = _audio_dir() / name
    if path.exists():
        return path

    tone_s = tone_ms / 1000
    gap    = np.zeros(int(SR * gap_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, f in enumerate(freqs_hz):
        parts.append(_render_hz(f, tone_s, source))
        if i < len(freqs_hz) - 1:
            parts.append(gap)
    audio = np.concatenate(parts)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak
    _write_wav(path, audio)
    return path


def chord(
    midis: Sequence[int],
    source: str | Sequence[str],
    duration_ms: int,
) -> Path:
    """Notes played simultaneously (mixed).

    ``source`` is either a single string (every note rendered with the same
    timbre) or a sequence of strings the same length as ``midis``, in which
    case each note gets its own source and the results are mixed together.
    Filename encodes the per-note sources, so ``[piano, violin, flute]`` →
    ``..._piano-violin-flute_chord_...``.
    """
    sorted_midis = sorted(midis)
    notes        = _notes_slug(sorted_midis, "+")

    # Normalise source argument
    if isinstance(source, str):
        sources_list = [source] * len(sorted_midis)
        src_slug     = source
    else:
        if len(source) != len(midis):
            raise ValueError(
                f"chord(): source list length {len(source)} ≠ midis length {len(midis)}"
            )
        # Re-zip to align with sorted_midis ordering
        pairs        = sorted(zip(midis, source), key=lambda p: p[0])
        sources_list = [s for _, s in pairs]
        src_slug     = "-".join(sources_list)

    name = f"{notes}_{src_slug}_chord_{duration_ms}ms.wav"
    path = _audio_dir() / name
    if path.exists():
        return path

    dur_s = duration_ms / 1000
    tones = [_render_single(m, dur_s, s) for m, s in zip(sorted_midis, sources_list)]
    maxlen = max(len(t) for t in tones)
    mixed  = np.zeros(maxlen, dtype=np.float32)
    for t in tones:
        mixed[:len(t)] += t / len(tones)
    peak = float(np.max(np.abs(mixed)))
    if peak > 1e-10:
        mixed *= 0.9 / peak
    _write_wav(path, mixed)
    return path


def clip_with_notes(
    notes: Sequence[tuple[int, int, int]],
    source: str,
    total_dur_ms: int,
    *,
    name_hint: str = "",
) -> tuple[Path, list[tuple[float, float, int]]]:
    """Build a fixed-length silent clip with a sequence of non-overlapping notes.

    ``notes`` is an iterable of ``(midi, onset_ms, dur_ms)`` triples. The
    returned WAV is cached under a deterministic filename derived from the
    note sequence + total length + ``name_hint``, and the second return value
    is the list of ``(onset_s, offset_s, midi)`` ground-truth times for use
    in scoring.

    Notes that fall outside ``[0, total_dur_ms]`` are clipped to the boundary
    rather than discarded so the caller can detect bad inputs in stimulus
    generation rather than silently lose them.
    """
    notes = list(notes)
    if not notes:
        raise ValueError("clip_with_notes requires at least one note")
    notes_sorted = sorted(notes, key=lambda x: x[1])

    # Filename — short hash-style summary for reproducibility
    parts = "-".join(f"{_note_slug(m)}@{on}+{d}" for m, on, d in notes_sorted)
    suffix = f"_{name_hint}" if name_hint else ""
    name   = f"clip_{parts}_{source}_total{total_dur_ms}ms{suffix}.wav"
    path   = _audio_dir() / name

    # Compute ground-truth onset/offset (in seconds) for the caller; this is
    # cheap to recompute even when we hit the cache.
    gt: list[tuple[float, float, int]] = []
    for m, on_ms, dur_ms in notes_sorted:
        on_s  = on_ms / 1000
        off_s = (on_ms + dur_ms) / 1000
        gt.append((on_s, off_s, m))

    if path.exists():
        return path, gt

    SR_local = SR
    total_n  = int(SR_local * total_dur_ms / 1000)
    audio    = np.zeros(total_n, dtype=np.float32)
    for m, on_ms, dur_ms in notes_sorted:
        if dur_ms <= 0:
            continue
        tone_arr = _render_single(m, dur_ms / 1000, source)
        start_n  = max(0, int(SR_local * on_ms / 1000))
        end_n    = min(start_n + len(tone_arr), total_n)
        if start_n >= total_n:
            continue
        audio[start_n:end_n] += tone_arr[:end_n - start_n]

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio = audio / peak * 0.9
    _write_wav(path, audio)
    return path, gt


def polyphonic_mix(
    lines: Sequence[tuple[Sequence[tuple[int, int, int]], str]],
    total_dur_ms: int,
    name_hint: str = "",
) -> Path:
    """Render and mix multiple independent melodic lines into one clip.

    Each element of ``lines`` is a ``(notes, source)`` pair where
    ``notes`` is a sequence of ``(midi, onset_ms, dur_ms)`` triples and
    ``source`` is a waveform name or GM instrument slug.  Each line is
    normalised independently before mixing so no single part dominates.

    Returns path to cached WAV.
    """
    key = repr((
        [([(m, o, d) for m, o, d in notes], src) for notes, src in lines],
        total_dur_ms,
        name_hint,
    ))
    fp   = hashlib.sha256(key.encode()).hexdigest()[:14]
    n    = len(lines)
    sfx  = f"_{name_hint}" if name_hint else ""
    path = _audio_dir() / f"poly{n}_{fp}{sfx}_total{total_dur_ms}ms.wav"
    if path.exists():
        return path

    total_n = int(SR * total_dur_ms / 1000)
    mixed   = np.zeros(total_n, dtype=np.float32)
    for notes, source in lines:
        line_buf = np.zeros(total_n, dtype=np.float32)
        for midi, onset_ms, dur_ms in notes:
            if dur_ms <= 0:
                continue
            seg   = _render_single(midi, dur_ms / 1000, source)
            start = max(0, int(SR * onset_ms / 1000))
            end   = min(start + len(seg), total_n)
            if start >= total_n:
                continue
            line_buf[start:end] += seg[:end - start]
        peak = float(np.max(np.abs(line_buf)))
        if peak > 1e-10:
            line_buf = line_buf / peak * (0.9 / n)
        mixed += line_buf

    peak = float(np.max(np.abs(mixed)))
    if peak > 1e-10:
        mixed = (mixed / peak * 0.9).astype(np.float32)
    _write_wav(path, mixed)
    return path


def tone_in_silence(
    midi: int,
    source: str,
    tone_start_ms: int,
    tone_dur_ms: int,
    total_dur_ms: int,
) -> Path:
    """A single tone embedded at a specific position in a silent clip."""
    slug = _note_slug(midi)
    name = (
        f"{slug}_{source}_in_silence"
        f"_at{tone_start_ms}ms_for{tone_dur_ms}ms_of{total_dur_ms}ms.wav"
    )
    path = _audio_dir() / name
    if path.exists():
        return path

    total_n = int(SR * total_dur_ms / 1000)
    audio   = np.zeros(total_n, dtype=np.float32)
    tone_arr = _render_single(midi, tone_dur_ms / 1000, source)
    start_n  = int(SR * tone_start_ms / 1000)
    end_n    = min(start_n + len(tone_arr), total_n)
    audio[start_n:end_n] = tone_arr[:end_n - start_n]
    _write_wav(path, audio)
    return path


def glide(
    start_midi: int,
    end_midi: int,
    source: str,
    duration_ms: int,
    shape: str = "linear",
) -> Path:
    """Continuously varying pitch from start_midi to end_midi."""
    if not _is_waveform(source):
        raise ValueError("Glide synthesis only supported for waveform sources")
    name = (
        f"{_note_slug(start_midi)}~{_note_slug(end_midi)}"
        f"_{source}_glide_{duration_ms}ms_{shape}.wav"
    )
    path = _audio_dir() / name
    if path.exists():
        return path
    start_hz = 440.0 * 2 ** ((start_midi - 69) / 12)
    end_hz   = 440.0 * 2 ** ((end_midi   - 69) / 12)
    _write_wav(path, _synth_glide(start_hz, end_hz, duration_ms / 1000, source, shape))
    return path


def tone_at_volume(
    midi: int,
    source: str,
    duration_ms: int,
    loudness_db: float,
) -> Path:
    """Tone at a specific amplitude level relative to 0 dBFS."""
    slug = _note_slug(midi)
    db_s = f"{loudness_db:+.0f}db".replace("+", "p").replace("-", "m")
    name = f"{slug}_{source}_{duration_ms}ms_vol{db_s}.wav"
    path = _audio_dir() / name
    if path.exists():
        return path
    amp   = 0.9 * (10 ** (loudness_db / 20))
    audio = _render_single(midi, duration_ms / 1000, source)
    # re-scale to requested amplitude
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio = audio / peak * amp
    _write_wav(path, audio)
    return path


def tone_with_effect(
    midi: int,
    source: str,
    duration_ms: int,
    effect_name: str,
    effect_params: dict,
    noise_seed: int = 0,
) -> Path:
    """Tone with an audio effect applied.

    Supported effect types (``effect_params['type']``):
        ``noise``        — additive Gaussian noise at requested SNR (dB)
        ``reverb``       — comb-filter reverb (delay_s, decay)
        ``clip``         — soft clipping at fractional threshold
        ``harmonic``     — extra harmonic at ratio · f0
        ``eq_hi``        — single-pole high-shelf, +12 dB above ``cutoff_hz``
        ``eq_lo``        — single-pole low-shelf,  +12 dB below ``cutoff_hz``
        ``compression``  — soft-knee compressor (ratio, threshold_db)
    """
    slug = _note_slug(midi)
    name = f"{slug}_{source}_{duration_ms}ms_fx_{effect_name}.wav"
    path = _audio_dir() / name
    if path.exists():
        return path
    audio = _render_single(midi, duration_ms / 1000, source)
    freq  = 440.0 * 2 ** ((midi - 69) / 12)
    audio = _apply_effect(audio, freq, effect_params, noise_seed)
    _write_wav(path, audio)
    return path


def glide_chain(
    midis: Sequence[int],
    source: str,
    ms_per_st: int,
    shape: str = "linear",
) -> Path:
    """Continuous chained glide visiting each MIDI in turn (no silence gaps).

    Each segment connects ``midis[i] → midis[i+1]`` with duration
    ``|midis[i+1] - midis[i]| * ms_per_st`` ms. Uses the same phase-accumulated
    glide as :func:`glide`, then concatenates segments end-to-end. Waveform
    sources only — FluidSynth pitch-bend chains are unreliable.
    """
    if not _is_waveform(source):
        raise ValueError("glide_chain only supported for waveform sources")
    if len(midis) < 2:
        raise ValueError("glide_chain requires at least 2 pitches")

    notes = "~".join(_note_slug(m) for m in midis)
    name  = f"{notes}_{source}_glidechain_{ms_per_st}msst_{shape}.wav"
    path  = _audio_dir() / name
    if path.exists():
        return path

    parts: list[np.ndarray] = []
    for a, b in zip(midis[:-1], midis[1:]):
        seg_ms = max(1, abs(b - a)) * ms_per_st
        seg_s  = seg_ms / 1000
        a_hz   = 440.0 * 2 ** ((a - 69) / 12)
        b_hz   = 440.0 * 2 ** ((b - 69) / 12)
        parts.append(_synth_glide(a_hz, b_hz, seg_s, source, shape))

    audio = np.concatenate(parts).astype(np.float32)
    peak  = float(np.max(np.abs(audio)))
    if peak > 1e-10:
        audio *= 0.9 / peak
    _write_wav(path, audio)
    return path


def _resample_audio(audio: np.ndarray, duration_factor: float) -> np.ndarray:
    """Change duration and pitch by linear interpolation (simulates tape speed change).

    duration_factor is the output/input length ratio:
        > 1 → longer output (slower playback), pitch falls by 12·log2(duration_factor) semitones.
        < 1 → shorter output (faster playback), pitch rises.
    """
    n_in  = len(audio)
    n_out = max(1, int(round(n_in * duration_factor)))
    if n_in == n_out:
        return audio.copy()
    x_in  = np.linspace(0.0, 1.0, n_in,  endpoint=False)
    x_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out   = np.interp(x_out, x_in, audio.astype(np.float64)).astype(np.float32)
    peak  = float(np.max(np.abs(out)))
    return (out / peak * 0.9).astype(np.float32) if peak > 1e-8 else out


def _time_stretch_pv(
    audio: np.ndarray,
    duration_factor: float,
    n_fft: int = 1024,
    hop_in: int = 256,
) -> np.ndarray:
    """Phase-vocoder time stretch: change duration without changing pitch.

    duration_factor is the output/input length ratio:
        > 1 → longer output (slower tempo), pitch unchanged.
        < 1 → shorter output (faster tempo), pitch unchanged.
    Pure numpy — no librosa dependency.
    """
    n       = len(audio)
    hop_out = max(1, int(round(hop_in * duration_factor)))
    n_fft_h = n_fft // 2 + 1
    window  = np.hanning(n_fft).astype(np.float64)
    omega   = 2.0 * np.pi * np.arange(n_fft_h) / n_fft

    n_frames = max(1, (n + hop_in - 1) // hop_in)
    out_len  = n_frames * hop_out + n_fft
    output   = np.zeros(out_len, dtype=np.float64)
    norm_buf = np.zeros(out_len, dtype=np.float64)

    phase_acc:  np.ndarray       = np.zeros(n_fft_h, dtype=np.float64)
    prev_phase: np.ndarray | None = None

    for i in range(n_frames):
        start = i * hop_in
        frame = np.zeros(n_fft, dtype=np.float64)
        end   = min(start + n_fft, n)
        frame[:end - start] = audio[start:end].astype(np.float64)

        spec  = np.fft.rfft(frame * window)
        mag   = np.abs(spec)
        phase = np.angle(spec)

        if prev_phase is None:
            phase_acc = phase.copy()
        else:
            delta = phase - prev_phase - omega * hop_in
            delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
            phase_acc += (omega + delta / hop_in) * hop_out

        prev_phase = phase.copy()

        frame_out = np.real(np.fft.irfft(mag * np.exp(1j * phase_acc), n=n_fft))
        out_start = i * hop_out
        output[out_start:out_start + n_fft] += frame_out * window
        norm_buf[out_start:out_start + n_fft] += window ** 2

    norm_buf = np.maximum(norm_buf, 1e-8)
    output  /= norm_buf
    expected = max(1, int(round(n * duration_factor)))
    out      = output[:expected].astype(np.float32)
    peak     = float(np.max(np.abs(out)))
    return (out / peak * 0.9).astype(np.float32) if peak > 1e-8 else out


def tone_time_modified(
    midi: int,
    source: str,
    duration_ms: int,
    mode: str,
    factor: float,
) -> Path:
    """Tone with time modification applied.

    ``factor`` is the output/input duration ratio (e.g. 2.0 = twice as long):
        ``'clean'``    — no modification (factor ignored)
        ``'resample'`` — change duration + pitch (longer → lower; shorter → higher)
        ``'stretch'``  — phase-vocoder: change duration, pitch unchanged

    Pitch shift from resampling: ``semitones = round(-12 * log2(factor))``
        factor=2.0 → −12 st (octave down); factor=0.5 → +12 st (octave up)
    """
    slug     = _note_slug(midi)
    factor_s = f"{factor:.2f}x".replace(".", "p")
    name     = f"{slug}_{source}_{duration_ms}ms_{mode}_{factor_s}.wav"
    path     = _audio_dir() / name
    if path.exists():
        return path

    audio = _render_single(midi, duration_ms / 1000, source)

    if mode == "clean" or factor == 1.0:
        out = audio.copy()
    elif mode == "resample":
        out = _resample_audio(audio, factor)
    elif mode == "stretch":
        out = _time_stretch_pv(audio, factor)
    else:
        raise ValueError(f"Unknown time-modification mode: {mode!r}")

    peak = float(np.max(np.abs(out)))
    if peak > 1e-10:
        out = (out / peak * 0.9).astype(np.float32)
    _write_wav(path, out)
    return path


def tone_with_background(
    midi: int,
    source: str,
    duration_ms: int,
    background_name: str,
    snr_db: float,
    seed: int = 0,
) -> Path:
    """Tone mixed with an additive background at a precise SNR.

    Synthetic backgrounds (white/pink/brown noise, cafe babble, traffic, rain)
    are generated deterministically per call; if a wav at
    ``stimuli/backgrounds/<name>.wav`` exists it is used and looped to length.
    """
    slug = _note_slug(midi)
    name = f"{slug}_{source}_{duration_ms}ms_bg_{background_name}_snr{snr_db:+g}db.wav"
    path = _audio_dir() / name
    if path.exists():
        return path

    duration_s = duration_ms / 1000
    n          = int(SR * duration_s)
    audio      = _render_single(midi, duration_s, source)[:n]
    if len(audio) < n:                              # FluidSynth may end early
        audio = np.pad(audio, (0, n - len(audio)))
    background = _synth_background(background_name, n, seed)

    sig_p   = float(np.mean(audio ** 2)) + 1e-12
    bg_p    = float(np.mean(background ** 2)) + 1e-12
    target_bg_p = sig_p / (10 ** (snr_db / 10))
    bg_scale    = np.sqrt(target_bg_p / bg_p)
    mixed   = audio + (background * bg_scale).astype(np.float32)

    peak = float(np.max(np.abs(mixed)))
    if peak > 1e-10:
        mixed = (mixed / peak * 0.9).astype(np.float32)
    _write_wav(path, mixed)
    return path


# ── Background-noise synthesis ────────────────────────────────────────────────

_BG_EXTS = (".wav", ".mp3", ".flac", ".ogg")


def _load_background_clip(name: str) -> np.ndarray | None:
    """Load ``data/downloaded/background/<name>.<ext>`` as mono float32 @ SR.

    Returns ``None`` if no matching file exists.  Stereo is downmixed; sample
    rate is converted by linear interpolation when it doesn't match SR.
    """
    bg_dir = config.DATA_DIR / "downloaded" / "background"
    for ext in _BG_EXTS:
        bg_path = bg_dir / f"{name}{ext}"
        if bg_path.exists():
            break
    else:
        return None

    import soundfile as sf
    data, sr = sf.read(str(bg_path), always_2d=False, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        # linear-interp resample (good enough for non-tonal background beds)
        n_out = int(round(len(data) * SR / sr))
        x_in  = np.linspace(0.0, 1.0, len(data), endpoint=False)
        x_out = np.linspace(0.0, 1.0, n_out,    endpoint=False)
        data  = np.interp(x_out, x_in, data).astype(np.float32)
    return data.astype(np.float32, copy=False)


def _synth_background(name: str, n: int, seed: int) -> np.ndarray:
    """Return ``n`` samples of the named background, deterministic given seed.

    Lookup order:
      1. ``data/downloaded/background/<name>.{wav,mp3,flac,ogg}`` — real-world
         recordings, truncated to the first ``n`` samples (no looping). Stereo
         is downmixed to mono and the clip is resampled to SR if needed.
      2. Synthetic ``white_noise`` — Gaussian, deterministic given seed.

    Anything else raises ValueError.
    """
    clip = _load_background_clip(name)
    if clip is not None:
        if len(clip) < n:
            raise ValueError(
                f"Background {name!r}: clip is {len(clip)/SR:.2f}s, "
                f"shorter than requested {n/SR:.2f}s — pick a longer file"
            )
        return clip[:n]

    rng = np.random.default_rng(seed)

    if name == "white_noise":
        return rng.normal(0.0, 1.0, n).astype(np.float32)

    raise ValueError(
        f"Unknown background {name!r}. Available: 'white_noise' or any file at "
        f"data/downloaded/background/<name>.{{wav,mp3,flac,ogg}}"
    )


# ── Effect implementations (mirrors exp_7 logic) ─────────────────────────────

def _apply_effect(
    audio: np.ndarray,
    freq_hz: float,
    params: dict,
    noise_seed: int,
) -> np.ndarray:
    eff = params.get("type")

    if eff is None:  # clean
        return audio.copy()

    if eff == "noise":
        sig_power   = float(np.mean(audio ** 2))
        noise_power = sig_power / (10 ** (params["snr_db"] / 10))
        rng   = np.random.default_rng(noise_seed)
        noise = rng.normal(0, np.sqrt(max(noise_power, 1e-12)), len(audio)).astype(np.float32)
        out   = audio + noise
        peak  = float(np.max(np.abs(out)))
        return (out / peak * 0.9).astype(np.float32) if peak > 1e-8 else out

    if eff == "reverb":
        delay_n = int(params["delay_s"] * SR)
        decay   = params["decay"]
        out = audio.copy()
        for i in range(delay_n, len(out)):
            out[i] += decay * out[i - delay_n]
        peak = float(np.max(np.abs(out)))
        return (out / peak * 0.9).astype(np.float32) if peak > 1e-8 else out

    if eff == "clip":
        peak     = float(np.max(np.abs(audio)))
        clip_val = params["threshold"] * peak
        out      = np.clip(audio, -clip_val, clip_val)
        return (out / clip_val * 0.9).astype(np.float32) if clip_val > 1e-8 else out

    if eff == "harmonic":
        n    = len(audio)
        t    = np.linspace(0.0, n / SR, n, endpoint=False)
        env  = _envelope(n)
        harmonic = (
            params["level"] * 0.9
            * np.sin(2 * np.pi * freq_hz * params["ratio"] * t)
            * env
        ).astype(np.float32)
        out  = audio + harmonic
        peak = float(np.max(np.abs(out)))
        return (out / peak * 0.9).astype(np.float32) if peak > 1e-8 else out

    if eff in ("eq_hi", "eq_lo"):
        # Single-pole shelving filter with ±gain_db gain above (eq_hi) /
        # below (eq_lo) the cutoff. Implementation: split signal into low
        # (1-pole LP) and high (residual) bands, weight, recombine.
        cutoff_hz = float(params.get("cutoff_hz", 1000.0))
        gain_db   = float(params.get("gain_db", 12.0))
        rc        = 1.0 / (2 * np.pi * cutoff_hz)
        a         = 1.0 / (1.0 + rc * SR)
        low       = np.zeros_like(audio, dtype=np.float64)
        prev      = 0.0
        for i, v in enumerate(audio):
            prev = a * float(v) + (1.0 - a) * prev
            low[i] = prev
        high = audio - low.astype(np.float32)
        gain = 10 ** (gain_db / 20.0)
        if eff == "eq_hi":
            out = (low.astype(np.float32) + high * gain).astype(np.float32)
        else:                                       # eq_lo
            out = ((low * gain).astype(np.float32) + high).astype(np.float32)
        peak = float(np.max(np.abs(out)))
        return (out / peak * 0.9).astype(np.float32) if peak > 1e-8 else out

    if eff == "compression":
        # Soft-knee downward compressor on the linear envelope.
        ratio        = float(params.get("ratio", 4.0))
        threshold_db = float(params.get("threshold_db", -20.0))
        thr_lin      = 10 ** (threshold_db / 20.0)
        eps          = 1e-9
        abs_a        = np.maximum(np.abs(audio), eps)
        # gain per sample (in linear amplitude domain)
        over         = np.maximum(abs_a / thr_lin, 1.0)
        gain         = over ** (1.0 / ratio - 1.0)
        out          = (audio * gain).astype(np.float32)
        peak = float(np.max(np.abs(out)))
        return (out / peak * 0.9).astype(np.float32) if peak > 1e-8 else out

    if eff == "saturation":
        # Plugin-style harmonic saturation via pedalboard's tanh waveshaper,
        # wrapped in 4× oversampling (resample_poly applies an anti-alias FIR)
        # so harmonics above Nyquist do not fold back as inharmonic junk.
        # Output is RMS-matched to the dry signal so loudness does not leak
        # into the experiment as drive_db increases.
        from pedalboard import Distortion, Pedalboard
        from scipy.signal import resample_poly

        drive_db = float(params.get("drive_db", 12.0))
        OS       = 4
        up       = resample_poly(audio.astype(np.float32), OS, 1).astype(np.float32)
        shaped   = Pedalboard([Distortion(drive_db=drive_db)])(up, sample_rate=SR * OS)
        out      = resample_poly(shaped, 1, OS).astype(np.float32)

        rms_in  = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-9
        rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) + 1e-9
        out     = (out * (rms_in / rms_out)).astype(np.float32)
        peak    = float(np.max(np.abs(out)))
        if peak > 0.99:
            out = (out * (0.99 / peak)).astype(np.float32)
        return out

    return audio.copy()
