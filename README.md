# PitchBench

Benchmark suite for evaluating pitch and acoustic perception in audio language models (ALMs). Probes pitch identification, temporal localisation, chord recognition, melodic contour, and more — reporting per-format accuracy (MIDI, SPN, doremi, Hz) to expose where verbal decoding fails.

## Install

```bash
uv sync --all-extras     # recommended (includes FluidSynth bindings + model-server stack)
uv sync                  # core only
uv sync --extra generation   # adds pretty-midi, pyfluidsynth
uv sync --extra model        # adds torch, transformers, fastapi
```

Requires Python ≥ 3.12.

## Setup

### 1. FluidSynth (optional — for GM instrument sources)

Install the system library for your platform, then `uv sync --extra generation`:

| Platform | Command |
|----------|---------|
| Linux / WSL | `sudo apt install fluidsynth` |
| macOS | `brew install fluid-synth` |
| Windows | `choco install fluidsynth` or download from [fluidsynth.org](https://www.fluidsynth.org) |

Without FluidSynth, GM instruments (piano, violin, …) are skipped and only waveform sources (sine, sawtooth, square, triangle) are used. A GM soundfont is also required (default: `/usr/share/sounds/sf2/FluidR3_GM.sf2`, override with `PITCHBENCH_SF2`).

### 2. Model backend (pick one)

**Option A — OpenRouter (default, no GPU needed)**

Add your key to `.env` (or PitchBench will prompt for it on first run):
```
OPENROUTER_KEY=sk-or-...
```

If you run `pitchbench --id <X>` without `--models`, you'll be prompted to choose a model with `openrouter/google/gemini-3.1-flash-lite-preview` as the default — just press Enter to accept.

**Option B — Local model servers**

Requires `uv sync --extra model` (installs torch, transformers, fastapi). Start each server before running experiments:

```bash
python -m pitchbench.model.api_music_fl       # music_flamingo on :8000
python -m pitchbench.model.api_audio_fl_next  # audio_flamingo_next_instruct on :8001
```

Model URLs are set in `config.MODEL_URLS` and overridable via env vars (`MF_URL`, `AF_NEXT_INST_URL`, …).

## Running experiments

```bash
pitchbench --list                          # list all experiments
pitchbench --id a1 --preview               # generate stimuli only, no model queries
pitchbench --id a1 --models openrouter/google/gemini-2.5-flash   # OpenRouter
pitchbench --id a1 --models audio_flamingo_next_instruct          # local server
pitchbench all --models openrouter/google/gemini-2.5-flash        # run everything
```

You can mix local and cloud models in one run — a `comparison.*` file is written automatically when more than one model runs:

```bash
pitchbench --id a1 --models music_flamingo openrouter/google/gemini-2.5-flash
```

Any OpenRouter slug can be passed via `--models openrouter/<slug>`. If the model can't accept audio, OpenRouter's error message is surfaced verbatim — no local whitelist to maintain.

## Sampling

Two ways to size a run: the **paper defaults** baked into `config.EXPERIMENT_DEFAULTS` (used for the standard benchmark), and `--sample-n N` (used for quick testing). Both draw deterministically and stratified, so the same seed always yields the same subset.

### Standard benchmark (no flag) — config-driven, per-stratum

With no `--sample-n`, each experiment reads its entry in `config.EXPERIMENT_DEFAULTS`:

```python
"pitchbench_a3_single_pitch_by_duration": {"per_stratum": 5, "strata": ("midi", "duration_ms")},
```

- `per_stratum` is **per stratum cell**. Total drawn = `per_stratum × num_distinct_strata_keys`, computed from the actual condition list.
- `per_stratum: None` → run the full grid (no sub-sampling). Used for `a1`, `d5`, `f2`, and the embedding probes (`g1`/`g2`/`g3`, analysis track, which never sub-sample).

This is the mode used to produce paper results — tune sample sizes by editing `config.py`, not by passing flags.

```bash
# Run the standard benchmark
pitchbench --id a3 --models openrouter/google/gemini-2.5-flash
pitchbench all       --models openrouter/google/gemini-2.5-flash
```

### Quick testing — `--sample-n N` (total cap)

`--sample-n N` **overrides the config** and is treated as a **total cap** across all strata, not per cell. The strata fields still come from `EXPERIMENT_DEFAULTS[...]["strata"]`, so coverage stays balanced — each cell gets `floor(N / k)`, remainder distributed in sorted-key order.

| Flag | Default | Effect |
|------|---------|--------|
| `--sample-n N` | (config `per_stratum`) | Override: draw exactly N stimuli **total**, stratified by the config's `strata` |
| `--sample-seed S` | `42` | RNG seed for the stratified draw |

```bash
# Quick sanity pass: 10 stimuli from a1, reproducible
pitchbench --id a1 --sample-n 10 --preview
pitchbench --id a1 --sample-n 10 --models openrouter/google/gemini-2.5-flash

# Same fixed budget across every experiment (cheap pipeline check)
pitchbench all --sample-n 50 --models openrouter/google/gemini-2.5-flash
```

Caveat: if `N < num_strata_cells`, some cells get a zero quota; if any cell has fewer items than its quota, `stratified_sample` raises `ValueError` — pick a larger `N` or a coarser strata in the config.

### Reproducibility

Sampling parameters (`sample_n`, `sample_seed`, `total_available`, `stratified_by`) are embedded in every result JSON. The `.txt` summary echoes them:

```
  Sampling     : 10 of 1159 (stratified by '(midi, duration_ms)', seed=42)
```

## Experiment categories

| ID | Topic |
|----|-------|
| `a1–a5` | Single-pitch identification |
| `b1–b5` | Onsets, offsets, time-localised pitch |
| `c1–c4` | Chords / dyads / simultaneous pitches |
| `d1–d7` | Sequences, contour, intervals |
| `e1–e3` | Loudness, audio effects, background noise |
| `f1–f2` | Melodic-line and voice identification in polyphony |
| `g1–g3` | Embedding probes (PCA, kNN oracle, token logits) — analysis track, not in CLI |

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

### Category F — Melodic-line and voice identification in polyphony

| ID | Name | What it tests |
|----|------|---------------|
| f1 | `melodic_line_id` | Transcribe one designated line from 2–4 simultaneous synthetic voices; sweeps n, register rank, tempo (slow/medium/fast), and instrument config (similar / mixed) |
| f2 | `chorale_voice_id` | Same task on real Bach chorales (music21 corpus): transcribe the soprano, alto, tenor, or bass from the longest non-crossing segment; requires `music21` |

### Category G — Embedding-space probes (analysis track, not in CLI)

These run via `python -m pitchbench.experiments.eval.<name>` and require a model
server that exposes `/embed` and/or `/generate_with_probs` (currently
`music_flamingo`, `audio_flamingo_next_*`). OpenRouter and Kimi cleanly skip.

| ID | Name | What it tests |
|----|------|---------------|
| g1 | `embedding_geometry` | PCA, linear probe, kNN, and cosine-similarity on encoder embeddings |
| g2 | `token_logits` | Vocabulary probability mass on pitch tokens at generation step 1 |
| g3 | `knn_oracle` | Compare 1-NN embedding accuracy vs. verbal output accuracy |

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
