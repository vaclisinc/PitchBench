# PitchBench

Benchmark suite for evaluating pitch and acoustic perception in audio language
models (ALMs).  PitchBench probes a model's ability to identify, count, locate,
and compare pitches under controlled stimulus conditions, then reports
per-format accuracy (MIDI, SPN, doremi, Hz) so you can see where the model's
internal pitch representation fails the verbal decode.

## Install

```bash
# Editable install with everything (recommended for development)
pip install -e ".[all]"

# Core only — no FluidSynth, no model-server stack
pip install -e .

# À-la-carte
pip install -e ".[generation]"   # adds pretty-midi, pyfluidsynth
pip install -e ".[model]"        # adds torch, transformers, fastapi, …
```

Requires Python ≥ 3.12.  Stimulus generation for instrument sources additionally
requires the system FluidSynth library and a General-MIDI soundfont
(`/usr/share/sounds/sf2/FluidR3_GM.sf2` by default — override with the
`PITCHBENCH_SF2` env var).

## Quick start

```bash
# 1. Start a model server (one process per model)
python -m pitchbench.model.api                    # music_flamingo on :8000
# or:
python -m pitchbench.model.serve_all              # every configured model

# 2. List experiments
pitchbench --list

# 3. Run one experiment against the default models
pitchbench --id a1                                # by category+digit ID
pitchbench pitchbench_a1_pitch_id                 # by full module name

# 4. Preview only (generate stimuli, skip model queries)
pitchbench --id a1 --preview

# 5. Run everything
pitchbench all
```

## Experiment categories

Experiments are named `pitchbench_<id>_<desc>.py` where `<id>` is a category
letter plus a digit:

| ID prefix | Topic                                          |
|-----------|------------------------------------------------|
| `a1–a5`   | Single-pitch identification                    |
| `b1–b5`   | Onsets, offsets, time-localised pitch          |
| `c1–c4`   | Chords / dyads / simultaneous pitches          |
| `d1–d7`   | Sequences, contour, intervals                  |
| `e1–e3`   | Loudness, audio effects, background noise      |
| `f1–f3`   | Embedding probes (PCA, kNN oracle, token logits)|
| `z1`      | Real-recording datasets (NSynth)               |

Every experiment exposes a `run()` and `preview()` entry point and is invoked
through the `pitchbench` CLI.

## Stimulus engine

All audio is generated through `pitchbench.generation.engine`, which writes
deterministic WAVs to `data/audio/<exp_name>/`.

- **Waveforms** (numpy only): `sine`, `sawtooth`, `square`, `triangle`
- **GM instruments** (FluidSynth): piano, violin, flute, trumpet, … (see
  `pitchbench.config.GM_PROGRAMS_V1`)
- **Backgrounds** (used by `e3`): `white_noise` is synthesised; any file at
  `data/downloaded/background/<name>.{wav,mp3,flac,ogg}` is loaded, downmixed
  to mono, resampled to 16 kHz if needed, and **truncated** to the fragment
  length — never looped.  Currently bundled: `church-bells`, `crowd-noise`,
  `rain`, `street-noise`.

## Output

Each run lands in `results/<exp_name>/run_<NNN>_<YYYYMMDD_HHMMSS>/`:

| File | Contents |
|------|----------|
| `results_<model>.json` | Full record: metadata, git commit, summary, per-item responses |
| `results_<model>.txt`  | Human-readable summary |
| `results_<model>.csv`  | One row per stimulus |
| `run_log.txt`          | Full stdout tee of the run |
| `comparison.{json,csv,txt}` | Cross-model summary (when ≥ 2 models ran) |

Runs are **never overwritten**, and every JSON record embeds the current git
commit hash and dirty flag for exact reproducibility.

## Repository layout

```
src/pitchbench/
  config.py
  generation/engine.py
  experiments/
    run.py              # `pitchbench` CLI
    helpers/            # api / music / plots / results
    scripts/            # one experiment per file
  model/                # FastAPI servers per model
tests/                  # pytest unit tests
data/                   # generated stimuli + downloaded assets (gitignored)
results/                # experiment outputs (committed, never overwritten)
```

Runtime directories (`data/`, `results/`, `stimuli/`, `_datasets/`) resolve
relative to the working directory.  Set `PITCHBENCH_ROOT=/path/to/project` to
pin them somewhere else.

## Tests

```bash
pytest tests/
```

Tests mock the model API and redirect the audio cache to a temp dir, so they
run offline and leave no artefacts.

## License

MIT — see [LICENSE](LICENSE).
