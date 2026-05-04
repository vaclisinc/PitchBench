# PitchBench

Benchmark suite for evaluating pitch and acoustic perception in audio language models (ALMs). Probes pitch identification, temporal localisation, chord recognition, melodic contour, and more — reporting per-format accuracy (MIDI, SPN, doremi, Hz) to expose where verbal decoding fails.

## Install

```bash
uv sync --all-extras     # recommended (includes FluidSynth + model-server stack)
uv sync                  # core only
uv sync --extra generation   # adds pretty-midi, pyfluidsynth
uv sync --extra model        # adds torch, transformers, fastapi
```

Requires Python ≥ 3.12. Instrument sources require FluidSynth and a GM soundfont (default: `/usr/share/sounds/sf2/FluidR3_GM.sf2`, override with `PITCHBENCH_SF2`).

## Running experiments

```bash
pitchbench --list                          # list all experiments
pitchbench --id a1                         # run by short ID
pitchbench pitchbench_a1_pitch_id          # run by full module name
pitchbench --id a1 --preview               # generate stimuli only, skip queries
pitchbench all                             # run every experiment
```

**Local model servers** — start each server first, then pass `--models`:

```bash
python -m pitchbench.model.api             # music_flamingo on :8000
python -m pitchbench.model.api_fl_next     # audio_flamingo_next_instruct on :8001

pitchbench --id a1 --models audio_flamingo_next_instruct
pitchbench all --models music_flamingo audio_flamingo_next_instruct
```

Model URLs are set in `config.MODEL_URLS` and overridable via env vars (`MF_URL`, `AF_NEXT_INST_URL`, …).

**OpenRouter** — set `OPENROUTER_KEY` in `.env`, then use the `openrouter/` prefix:

```bash
pitchbench --id a1 --models openrouter/google/gemini-2.5-flash
pitchbench all --models openrouter/google/gemini-2.5-pro
```

Audio-capable slugs are whitelisted in `config.OPENROUTER_AUDIO_MODELS`. Mix local and cloud models freely — a `comparison.*` file is written automatically when more than one model runs.

## Experiment categories

| ID | Topic |
|----|-------|
| `a1–a5` | Single-pitch identification |
| `b1–b5` | Onsets, offsets, time-localised pitch |
| `c1–c4` | Chords / dyads / simultaneous pitches |
| `d1–d7` | Sequences, contour, intervals |
| `e1–e3` | Loudness, audio effects, background noise |
| `f1–f3` | Embedding probes (PCA, kNN oracle, token logits) |
| `z1`    | Real-recording datasets (NSynth) |

### Category A — Single-pitch identification

| ID | Name | What it tests |
|----|------|---------------|
| a1 | `pitch_id` | Baseline: identify one sustained note (MIDI 36–84) across all sources and formats |
| a2 | `pitch_with_reference` | Anchored identification — a labelled reference is played first; tests relative-pitch use |
| a3 | `pitch_by_duration` | Sweeps 8 durations (50 ms–6 s); finds the model's temporal integration window |
| a4 | `pitch_with_vibrato` | FM vibrato at 0–10 Hz / 0–200 cents; does the model track centre pitch or oscillation? |
| a5 | `pitch_slightly_off` | Detuned tones (up to 40 % of a semitone); tests pitch quantisation behaviour |

### Category B — Onsets, offsets, time-localised pitch

| ID | Name | What it tests |
|----|------|---------------|
| b1 | `pitch_in_silence` | Hidden note in a 60 s silent clip; with and without a timing hint |
| b2 | `onset_offset_single` | Detect when a single note starts/ends (no pitch asked) |
| b3 | `onset_offset_specific` | Onset/offset of a named target among distractors |
| b4 | `pitch_at_time` | Which pitch is playing at a queried timestamp in a 3–5 note sequence |
| b5 | `onset_offset_each` | Full timing transcription of all notes in a 3–8 note sequence |

### Category C — Chords / dyads / simultaneous pitches

| ID | Name | What it tests |
|----|------|---------------|
| c1 | `dyad_interval` | Name the interval (semitone count) between two simultaneous tones |
| c2 | `chord_pitch_count` | Count distinct pitches in a chord (triads, 7ths, random sets) |
| c3 | `chord_pitch_id` | List every note in a chord (dyads, triads, 7th chords) |
| c4 | `chord_quality` | Name chord quality and/or root+quality from a sounding chord |

### Category D — Sequences, contour, intervals

| ID | Name | What it tests |
|----|------|---------------|
| d1 | `seq_pitch_count` | Count distinct pitches in a sequential passage (1–10 pitches) |
| d2 | `pitch_difference` | Binary higher/lower judgment; delta 1 cent–1200 cents |
| d3 | `interval_id_seq` | Name the interval between two sequential notes (ascending & descending) |
| d4 | `contour_discrete` | Output up/down tokens for each transition in a step-wise melody |
| d5 | `contour_continuous` | Classify a gliding pitch as flat / up / down / arch / valley |
| d6 | `pitch_ranking` | Rank 3–5 sequential tones from lowest to highest |
| d7 | `seq_pitch_id` | Transcribe all pitches in a 3–10 note sequence in order |

### Category E — Loudness, audio effects, background noise

| ID | Name | What it tests |
|----|------|---------------|
| e1 | `loudness` | Pitch accuracy vs. amplitude (−30 to 0 dBFS) |
| e2 | `audio_effects` | Pitch under white noise (4 SNR levels), reverb, and hard clipping |
| e3 | `background_effects` | Pitch over real-world backgrounds (crowd, rain, bells, street) at 4 SNRs |

### Category F — Embedding-space probes

| ID | Name | What it tests |
|----|------|---------------|
| f1 | `embedding_geometry` | PCA, linear probe, kNN, and cosine-similarity on encoder embeddings |
| f2 | `token_logits` | Vocabulary probability mass on pitch tokens at generation step 1 |
| f3 | `knn_oracle` | Compare 1-NN embedding accuracy vs. verbal output accuracy |

### Category Z — Real-recording datasets

| ID | Name | What it tests |
|----|------|---------------|
| z1 | `pitch_id_nsynth` | Pitch identification on NSynth real-instrument samples (ecological validity) |

## Stimulus engine

Audio is generated through `pitchbench.generation.engine` and cached deterministically in `data/audio/<exp_name>/`.

- **Waveforms** (numpy, always available): `sine`, `sawtooth`, `square`, `triangle`
- **GM instruments** (FluidSynth): `piano`, `violin`, `flute`, `trumpet`, … (see `config.GM_PROGRAMS_V1`)
- **Backgrounds** (e3): `white_noise` is synthesised; real recordings live in `data/downloaded/background/<name>.mp3` and are truncated to the clip length — never looped

## Output

Each run writes to `results/<exp_name>/run_<NNN>_<YYYYMMDD_HHMMSS>/`:

| File | Contents |
|------|----------|
| `results_<model>.json` | Full record: metadata, git commit, prompts, per-item responses |
| `results_<model>.txt`  | Human-readable summary |
| `results_<model>.csv`  | One row per stimulus |
| `run_log.txt`          | Full stdout tee |
| `comparison.*`         | Cross-model summary (when ≥ 2 models ran) |

Runs are never overwritten. Every JSON embeds the git commit hash and dirty flag for reproducibility.

## Repository layout

```
src/pitchbench/
  config.py               # all constants, model URLs, audio params
  generation/engine.py    # central audio engine
  experiments/
    run.py                # pitchbench CLI
    helpers/              # api, music, plots, results
    scripts/              # one file per experiment
  model/                  # FastAPI servers per model
tests/                    # pytest (mocks model API, no artefacts left)
data/                     # generated stimuli + downloaded assets (gitignored)
results/                  # experiment outputs (committed, never overwritten)
```

Runtime directories resolve relative to the working directory; set `PITCHBENCH_ROOT` to pin them elsewhere.

## Tests

```bash
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
