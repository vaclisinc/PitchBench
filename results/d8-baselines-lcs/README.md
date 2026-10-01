# D8 DSP and Basic Pitch reproduction

Historical audit evidence. For the final paper scores and current replay command,
use [the complete Table 1 bundle](../table1-recomputed/README.md). To reproduce
this historical bundle specifically, check out the commit recorded in its run
receipt; its old command is not the current paper evaluation workflow.

Reran the original two rebuttal baselines on all **171 official D8 stimuli** (342 model/stimulus records), using dataset revision `f6c672608057cb877bfaacaeeff9bfb49eef7778`, seed 42, Python 3.11.15 and the pinned acoustic parameters.

| Baseline | Original displayed exact match | Rerun exact match | Ordered Note F1 (LCS) |
|---|---:|---:|---:|
| DSP (YIN + harmonic CQT adapter) | 80.1% | 80.1% | 94.6% |
| Basic Pitch 0.4.0 (ONNX, CPU) | 98.8% | 98.8% | 99.6% |

D8 uses the monophonic DSP path; the shared DSP baseline also includes harmonic CQT for polyphonic tasks. Ordered Note F1 is `2 * LCS / (n_gt + n_pred)`. Matching is exact MIDI / equivalent SPN pitch / ±1 Hz. ANY takes the maximum of MIDI, SPN and Hz for each stimulus, then the macro mean; solfège is excluded. Extra predictions are retained. Exact match and positional matches remain diagnostics.

The rerun reproduces both original exact-match values at the precision available in the historical table. All stimulus identities and target sequences match the six saved LALM evaluation sets. For each baseline, an independently executed preflight clip produced identical raw answers in all four formats when repeated in the full run. This verifies the tested repeats; the missing original raw baseline responses cannot be compared byte-for-byte.

`results_dsp.json` and `results_basic_pitch.json` preserve every raw answer, prompt, target, per-item score and evaluation metadata. `metrics.csv` and `item_scores.csv` contain unrounded numeric evidence. `config.yaml` is the resolved run configuration; `receipt.json` records the dataset hash, code commit and installed package versions; `model.json` records the packaged ONNX weight hash. `run.json` is the queue provenance, including the initial startup failure and successful retry. `validation.json` records the reproduction checks.

`updated_table.csv` replaces D8 in the original DSP/Basic Pitch comparison. Its approximate 28-task means are **69.9%** and **73.1%**. Only D8 was rerun; the other 27 values come from the historical table rounded to 0.1 percentage points. `overall.json` gives the approximate means and the conservative ±0.0482-percentage-point rounding bound. Do not treat these as recomputed full-precision results for the other tasks.

## Recompute scores without inference

From the project checkout with its Python dependencies available:

```bash
PYTHONPATH=src python -m pitchbench.experiments.rescore_sequences \
  --baseline-input results/d8-baselines-lcs --output-dir /tmp/d8-baseline-rescore
```

This independently reconstructs all scores from the saved raw answers and rejects disagreements with stored scores. The two CSV outputs must match this bundle byte-for-byte.

## Repeat inference

Use Python 3.11.15 and the exact versions recorded in `receipt.json`, with Basic Pitch 0.4.0's packaged `nmp.onnx` and ONNX Runtime 1.19.2. The environment intentionally uses the CPU/ONNX backend. `setuptools==75.8.0` supplies the `pkg_resources` dependency required by resampy 0.4.2.

Fetch the pinned D8 shard with `hf download pitchbench-authors/PitchBench --type dataset --revision f6c672608057cb877bfaacaeeff9bfb49eef7778 --include 'pitchbench_d8*/*' --local-dir YOUR_DATASET_DIR`. Copy `config.yaml` into a new run directory and update only its local input/output/environment paths. Run the existing `scripts/run_rebuttal_baselines.py CONFIG --phase all` entrypoint through the workspace queue. The runner validates package versions, shard revision, 171-row coverage and the committed code snapshot before completing its receipt. All audio, temporary caches, plots and logs remain under shared data or the working run, outside this compact Git bundle.
