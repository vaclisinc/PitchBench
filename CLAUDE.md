# Project: PitchBench

Benchmark for testing what audio/acoustic signals ALMs (Audio Language Models) do
and don't understand. Results feed a paper on how to train or fine-tune models
based on those gaps.

All experiments live under `src/pitchbench/experiments/scripts/` and must be
entirely reproducible — this is for an academic paper.

## Layout (src/ package)

```
LICENSE
pyproject.toml             # build system, deps, [project.scripts] pitchbench=…
README.md
src/
  pitchbench/
    __init__.py            # exposes __version__
    config.py              # ALL hardcoded constants (paths, URLs, model IDs, audio params)
    generation/
      engine.py            # central audio engine — every stimulus goes through it
    experiments/
      run.py               # CLI entry point (pitchbench …)
      helpers/             # api.py, music.py, plots.py, results.py
      scripts/             # one file per experiment: pitchbench_<id>_<desc>.py
    model/                 # model API server(s): api_music_fl.py, api_audio_fl_next.py, …
tests/                     # pytest unit tests (mock the model API)
data/                      # generated stimuli + downloaded assets (gitignored)
  audio/                   # engine cache, per experiment
  downloaded/background/   # real-world background recordings (mp3) used by e3
results/                   # experiment outputs — committed, timestamped, never overwritten
stimuli/                   # optional pre-generated reference stimuli
```

Runtime directories (`data/`, `results/`, `stimuli/`, `_datasets/`) resolve
relative to the current working directory by default. Override the base via the
`PITCHBENCH_ROOT` env var when needed.

## Setup

```bash
# Editable install with everything for local development
pip install -e ".[all]"

# Or a leaner install (core deps only — no FluidSynth, no model-server stack)
pip install -e .

# Optional groups
pip install -e ".[generation]"   # adds pretty-midi, pyfluidsynth
pip install -e ".[model]"        # adds torch, transformers, fastapi, …
```

A model server must be reachable before running an experiment that queries one.
Each model has its own URL slot in `config.MODEL_URLS`; override via env vars
(`MF_URL`, `AF_NEXT_INST_URL`, …).

## Running experiments

The package installs a `pitchbench` CLI:

```bash
pitchbench --list                                # list available experiments
pitchbench pitchbench_a1_pitch_id                # run by full module name
pitchbench --id a1                               # run by category+digit ID
pitchbench pitchbench_a1_pitch_id --preview      # generate stimuli, skip queries
pitchbench all                                   # run every experiment
pitchbench --id a1 --models audio_flamingo_next_instruct
```

Experiment IDs are `<category><digit>` where category encodes the topic group:

| Category | Topic                                          |
|----------|------------------------------------------------|
| `a`      | Single-pitch identification                    |
| `b`      | Onsets, offsets, time-localised pitch          |
| `c`      | Chords / dyads / simultaneous pitches          |
| `d`      | Sequences, contour, intervals                  |
| `e`      | Loudness, audio effects, background noise      |
| `f`      | Polyphony / multi-instrument identification    |
| `z`      | Real-recording datasets (e.g. NSynth)          |

A separate analysis track lives under `experiments/eval/` (NOT discovered by
the `pitchbench` CLI — run with `python -m pitchbench.experiments.eval.<name>`):

| ID  | Script                                | Topic                                                |
|-----|---------------------------------------|------------------------------------------------------|
| `g1`| `pitchbench_g1_embedding_geometry.py` | PCA / linear probe / kNN on encoder embeddings       |
| `g2`| `pitchbench_g2_token_logits.py`       | Step-1 token probability mass on pitch tokens        |
| `g3`| `pitchbench_g3_knn_oracle.py`         | kNN oracle on embeddings vs. verbal output           |

These need `mode="probs"` / `mode="embed"` from the model server, so they only
work for local servers that expose `/generate_with_probs` and `/embed`
(currently: music_flamingo, audio_flamingo_next_*). OpenRouter and Kimi
cleanly raise `NotImplementedError` and are skipped.

## Stimulus generation

Every experiment renders its audio through `pitchbench.generation.engine`. The
engine writes to `data/audio/<exp_name>/` with deterministic filenames; if a
file already exists it is reused. Sources:

- **Waveforms** (always available): `sine`, `sawtooth`, `square`, `triangle`
- **GM instruments** (require FluidSynth + soundfont): `piano`, `violin`,
  `flute`, `trumpet`, … see `config.GM_PROGRAMS_V1`
- **Backgrounds** (used by `e3_background_effects`): `white_noise` (synthetic)
  plus any file at `data/downloaded/background/<name>.{wav,mp3,flac,ogg}`
  (currently `church-bells`, `crowd-noise`, `rain`, `street-noise`). Real
  backgrounds are truncated to the fragment length — never looped.

The SF2 soundfont path is `config.SF2_PATH` (defaults to
`/usr/share/sounds/sf2/FluidR3_GM.sf2`, override via `PITCHBENCH_SF2`).

## Results

Every run writes three files to `results/<exp_name>/<run_NNN>_<YYYYMMDD_HHMMSS>/`:

| File | Contents |
|------|----------|
| `results_<model>.json` | Full record: metadata, git commit, summary, per-item raw + parsed responses |
| `results_<model>.txt`  | Human-readable summary |
| `results_<model>.csv`  | One row per stimulus, for analysis |

Plus `run_log.txt` (full stdout tee) and a `comparison.{json,csv,txt}` when more
than one model ran. Results are **never overwritten** — each run gets its own
timestamped directory, and the git commit hash + dirty flag is embedded in
every JSON for exact reproducibility.

## Writing a new experiment

1. Create `src/pitchbench/experiments/scripts/pitchbench_<id>_<desc>.py`
   (e.g. `pitchbench_a6_pitch_with_chorus.py`).
2. Implement two entry points: `run()` and `preview()`.
3. Import via the package namespace:
   ```python
   import pitchbench.config as config
   import pitchbench.generation.engine as engine
   from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
   from pitchbench.experiments.helpers.results import (
       get_run_metadata, make_run_dir, save_comparison, save_results,
   )
   ```
4. Build metadata with `get_run_metadata(...)` and pass it to `save_results()`.
5. Use `engine.set_exp(EXP_NAME)` before generating audio so files land in the
   right cache subdirectory.

## Tests

```bash
pytest tests/                 # all
pytest tests/test_engine.py   # just the engine
```

Tests mock the model API, redirect `config.AUDIO_DIR` to a temp dir, and assert
on the saved JSON / CSV.

## Rules (enforced for paper reproducibility)

- Stimuli are generated deterministically from fixed parameters — no randomness
  without a fixed seed.
- Every result file must include: model info, git commit, all prompts used, all
  config values relevant to the run.
- Never edit results after the fact — re-run with the same config instead.
- `data/` may be regenerated; `results/` must be preserved.
