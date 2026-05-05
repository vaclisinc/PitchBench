# e4 Harmonic Saturation & e5 Time Stretch

**Date:** 2026-05-04
**Status:** planned

## Goal
Add two new audio-effects experiments to the e-series: e4 tests whether ALMs can identify pitch through harmonic saturation (tanh soft clipping), and e5 tests whether models distinguish pitch-preserving time stretching from pitch-altering resampling — motivated by music-producer feedback that these are fundamentally different operations.

## Context
- Follows the same structure as `pitchbench_e2_audio_effects.py` (IV = effect/condition, DV = 4-format pitch accuracy)
- Engine's `_apply_effect` already handles reverb/clip/eq/harmonic; saturation is a new type
- e5 requires new engine primitives (phase vocoder + resample) not yet in `engine.py`
- Restricted pitch set for e5: MIDI 36–84 (so ±12 semitone shifts stay in audible range)
- All sources (waveforms + GM instruments); base duration 3000 ms for e5
- EVAL=True branch in config locks values for paper runs

## Phase 1: e4 harmonic saturation

### Action items
- [ ] Add `"saturation"` effect type to `engine._apply_effect` — tanh waveshaping: `out = tanh(drive·x) / tanh(drive)`, normalised to 0.9 peak
- [ ] Add `BENCHMARK_E4_SATURATIONS` dict to `config.py` with 4 levels: `clean`, `sat_light` (drive=2), `sat_medium` (drive=5), `sat_heavy` (drive=20)
- [ ] Add e4 sampling entry to `SAMPLING_CONFIG`: `{"per_stratum": 10, "strata": ("saturation_level", "midi")}`
- [ ] Add `pitchbench_e4_PITCHES`, `pitchbench_e4_TONE_MS`, `pitchbench_e4_SATURATIONS`, `pitchbench_e4_SOURCES` in both EVAL branches of `config.py`
- [ ] Create `src/pitchbench/experiments/scripts/pitchbench_e4_harmonic_saturation.py` following e2 structure: `build_conditions`, `generate_stimuli`, `run_one_model`, `preview`, `run`
- [ ] Verify `pitchbench --id e4 --preview` generates audio without error

## Phase 2: e5 time stretch vs resample

### Action items
- [ ] Add `_resample_audio(audio, speed_factor)` to engine: linear interpolation, `n_out = n_in / speed_factor`
- [ ] Add `_time_stretch_pv(audio, speed_factor)` to engine: numpy phase vocoder (STFT → phase accumulation → ISTFT), `n_out = n_in / speed_factor`, pitch unchanged
- [ ] Add `tone_time_modified(midi, source, duration_ms, mode, speed_factor) -> Path` to engine public API
- [ ] Add `BENCHMARK_PITCHES_E5 = [36, 43, 48, 54, 58, 60, 64, 67, 69, 72, 77, 84]` to `config.py`
- [ ] Add `BENCHMARK_E5_CONDITIONS` list with 5 entries: `clean` (factor=1.0), `resample_0.5x` (factor=0.5), `resample_2x` (factor=2.0), `stretch_0.5x` (factor=0.5), `stretch_2x` (factor=2.0)
- [ ] Add e5 sampling entry to `SAMPLING_CONFIG`: `{"per_stratum": 10, "strata": ("mode", "midi")}`
- [ ] Add `pitchbench_e5_*` config vars in both EVAL branches
- [ ] Create `src/pitchbench/experiments/scripts/pitchbench_e5_time_stretch.py`: `midi_gt` = `midi + round(12 * log2(factor))` for resample, `midi` for stretch/clean; include `original_midi` in each record as extra metadata
- [ ] Verify `pitchbench --id e5 --preview` generates 5 condition × sources × pitches stimuli

## Out of scope
- Auto-tune (e6): more complex, needs pitch detection + pitch quantisation — future experiment
- Librosa/rubberband dependency: keep numpy-only phase vocoder for reproducibility without optional deps
