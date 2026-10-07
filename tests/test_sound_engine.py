import math
import os

import numpy as np
import pytest

import pitchbench.config as config

config.AUDIO_DIR = config.GENERATED_DIR

import pitchbench.sound.engine as engine  # noqa: E402


def _f0_cents(audio: np.ndarray, ref_hz: float) -> float:
    n = engine.SR // 2
    seg = audio[engine.SR // 4: engine.SR // 4 + n] * np.hanning(n)
    nfft = 1 << 20
    spec = np.abs(np.fft.rfft(seg, nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / engine.SR)
    hps = spec.copy()
    for k in (2, 3):
        hps[: len(spec) // k] *= spec[::k][: len(spec) // k]
    band = (freqs > 50) & (freqs < 2000)
    hz = freqs[band][np.argmax(hps[band])]
    return 1200 * math.log2(hz / ref_hz)


def _peak_cents(audio: np.ndarray, ref_hz: float) -> float:
    """Sub-cent location of the spectral peak nearest ``ref_hz``."""
    seg = audio[engine.SR // 10: engine.SR] * np.hanning(engine.SR - engine.SR // 10)
    nfft = 1 << 21
    spec = np.abs(np.fft.rfft(seg, nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / engine.SR)
    band = (freqs > ref_hz * 0.97) & (freqs < ref_hz * 1.03)
    i = int(np.argmax(np.where(band, spec, 0)))
    y0, y1, y2 = np.log(spec[i - 1: i + 2])
    peak = i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return 1200 * math.log2(peak * engine.SR / nfft / ref_hz)


def _real_soundfont() -> bool:
    return os.path.isfile(config.SF2_PATH) and os.path.getsize(config.SF2_PATH) > 1_000_000


needs_fluidsynth = pytest.mark.skipif(not _real_soundfont(), reason="no real SoundFont available")


@needs_fluidsynth
@pytest.mark.parametrize("cents", [-22.0, 0.0, 22.0])
def test_instrument_detune_renders_at_target_pitch(cents):
    pytest.importorskip("fluidsynth")
    audio = engine._render_instrument_detuned(69, "piano", 1.5, cents)
    # A raw (unsigned) pitch-bend value once made every note ~+2 semitones sharp.
    assert abs(_f0_cents(audio, 440.0) - cents) < 10


@needs_fluidsynth
@pytest.mark.parametrize("cents", [1.0, 2.0, -1.0])
def test_instrument_detune_resolves_single_cents(cents):
    # FluidSynth quantises pitch to whole cents, so a 1-cent pitch bend is a no-op.
    pytest.importorskip("fluidsynth")
    base = engine._render_instrument_detuned(69, "piano", 1.5, 0.0)
    shifted = engine._render_instrument_detuned(69, "piano", 1.5, cents)
    measured = _peak_cents(shifted, 440.0) - _peak_cents(base, 440.0)
    assert abs(measured - cents) < 0.2


@needs_fluidsynth
def test_instrument_detune_keeps_duration():
    pytest.importorskip("fluidsynth")
    audio = engine._render_instrument_detuned(60, "violin", 1.0, -40.0)
    assert len(audio) == engine.SR
