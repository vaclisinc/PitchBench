import math
import os

import numpy as np
import pytest

import pitchbench.config as config

config.AUDIO_DIR = config.GENERATED_DIR

import pitchbench.sound.engine as engine  # noqa: E402


@pytest.mark.parametrize("cents,expected", [
    (0.0, 0),
    (22.0, 901),
    (-22.0, -901),
    (100.0, 4096),
    (-200.0, -8192),
    (200.0, 8191),
    (500.0, 8191),
    (-500.0, -8192),
])
def test_pitch_bend_value_is_signed(cents, expected):
    # pyfluidsynth's Synth.pitch_bend adds the 8192 centre offset itself.
    assert engine._pitch_bend_value(cents) == expected


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


def _real_soundfont() -> bool:
    return os.path.isfile(config.SF2_PATH) and os.path.getsize(config.SF2_PATH) > 1_000_000


@pytest.mark.skipif(not _real_soundfont(), reason="no real SoundFont available")
@pytest.mark.parametrize("cents", [-22.0, 0.0, 22.0])
def test_instrument_detune_renders_at_target_pitch(cents):
    pytest.importorskip("fluidsynth")
    midi = 69
    audio = engine._render_instrument_detuned(midi, "piano", 1.5, cents)
    measured = _f0_cents(audio, 440.0)
    # Before the fix every note came out ~+2 semitones sharp.
    assert abs(measured - cents) < 10
