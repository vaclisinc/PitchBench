# Paper results

[**Table 1**](table1.md) is the canonical camera-ready comparison: 28 tasks ×
eight models. [CSV](table1.csv) retains full precision in percentage units;
[LaTeX](table1.tex) and Markdown display one decimal place. All exports use one
[verified score file](../../results/table1-recomputed/metrics.csv).

## Build or check the table

From the repository root, using Python 3.10 or later (standard library only):

```bash
PYTHONPATH=src python -m pitchbench.analysis.table1
PYTHONPATH=src python -m pitchbench.analysis.table1 --check
```

No inference or dataset download is needed to rebuild the table. The builder
checks all 224 cells, sample counts, metrics, means and source hashes, rejecting
missing, duplicate or invalid evidence. Do not edit the generated exports.
LaTeX requires `booktabs`, `multirow`, and `graphicx`.

## Scoring

Pitch-format scores use **MIDI, SPN and Hz only**. DoReMi is neither queried by
the paper baseline runner nor read by the table audit. For exact-accuracy tasks,
ANY is the per-item OR of those three formats. D8/F1/F2 use Ordered Note F1,
`2 × LCS / (N_gt + N_pred)`: take the maximum of the three format scores for each
item, then average across items. Extra predicted notes remain in the denominator.
D8 retains ±1 Hz matching; F1/F2 retain ±1%. Other tasks use their existing task
accuracy definitions. See [experiment definitions](../../EXPERIMENTS.md).

Mean weights all 28 task scores equally. Numeric scores remain unrounded until
display. D7 in the paper is D7a (concatenated reference); split-reference variants
and Y1 are outside Table 1. GPT-4o B1 has 158 saved answers for 160 stimuli: its
score uses those 158 answers, with the missing identities recorded in the audit.

## Verify the evidence

The [complete audit bundle](../../results/table1-recomputed/README.md) contains
full-precision scores, counts, compact baseline raw answers, source hashes,
reproduction provenance. All ALM
cells are replayed from saved raw answers against the official ground truth;
both baselines are rerun on all 5,802 fixed stimuli.

## Download the fixed dataset

Keep the dataset and run outputs outside the source checkout. Download with
`--local-dir` so the revision metadata is retained alongside the Parquet files:

```bash
export PITCHBENCH_DATASET_DIR="$(pwd)/../pitchbench-dataset"
uvx --from huggingface-hub hf download vaclis/PitchBench \
  --repo-type dataset --revision f6c672608057cb877bfaacaeeff9bfb49eef7778 \
  --include "pitchbench_*/test-00000-of-00001.parquet" \
  --local-dir "$PITCHBENCH_DATASET_DIR"
```

Preserve `.cache/huggingface/download` inside this directory when copying the
dataset; it records the downloaded revision and is checked before scoring.

## Replay saved answers

Using the regular package environment (Python 3.12 or later):

```bash
PYTHONPATH=src python -m pitchbench.analysis.replay \
  --dataset-dir "$PITCHBENCH_DATASET_DIR" \
  --baseline-input results/table1-recomputed/baselines \
  --output-dir /tmp/pitchbench-table1-check
```

This is the single paper replay implementation. It matches stimulus identities,
checks saved ground truth, reparses only the three paper formats, and rejects
unexpected duplicates or missing samples. It makes no model calls and disables
optional external LLM parsing. Compare `metrics.csv` numerically (tolerance
`2e-15` on the 0–1 scale for sequence F1).

## Reproduce baseline inference

DSP and Basic Pitch inference use a **separate Python 3.11.15 / NumPy 1.26.4
environment** to reproduce the recorded run. The regular package environment
uses Python ≥3.12 / NumPy ≥2. The baseline environment executes this source
checkout via `PYTHONPATH=src`; it does not install the regular PitchBench wheel.

From the repository root, create an empty environment with `uv`:

```bash
export PITCHBENCH_BASELINE_ENV="$(pwd)/../pitchbench-baseline-env"
uv venv --no-project --python 3.11.15 "$PITCHBENCH_BASELINE_ENV"
uv pip install --python "$PITCHBENCH_BASELINE_ENV/bin/python" --no-deps \
  -r configs/baselines-requirements.txt
PYTHONPATH=src "$PITCHBENCH_BASELINE_ENV/bin/python" \
  -m pitchbench.baselines.evaluation configs/paper_baselines.yaml --check-environment
```

[The requirements file](../../configs/baselines-requirements.txt) pins the
complete runtime package set from the successful reproduction. `--no-deps` is
intentional: Basic Pitch's package metadata also requests unused inference
backends; the paper run uses only ONNX Runtime. Installing the entire pinned
file supplies the dependencies used by that backend and by the benchmark.
The environment check verifies the pinned versions and loads the packaged
ONNX model without querying or evaluating audio. Its SHA-256 must be
`2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`.

Copy [the reusable recipe](../../configs/paper_baselines.yaml) outside the
checkout, e.g. `/ABSOLUTE/PATH/TO/RUN/config.yaml`. Set these absolute paths:

| Config field | Value |
| --- | --- |
| `input_dataset.local_dir` | The downloaded dataset directory above |
| `runtime.environment_dir` | The baseline environment above (provenance) |
| `runtime.runtime_root` | `/ABSOLUTE/PATH/TO/RUN/runtime` |
| `runtime.persistent_results_dir` | `/ABSOLUTE/PATH/TO/RUN/artifacts/evaluation` |

Keep the dataset revision, acoustic parameters and decoder settings unchanged.
From a clean checkout, run:

```bash
PYTHONPATH=src "$PITCHBENCH_BASELINE_ENV/bin/python" \
  -m pitchbench.baselines.evaluation /ABSOLUTE/PATH/TO/RUN/config.yaml --phase all
```

A clean branch, tag or detached commit works outside VacLab; a remote upstream
and a queue receipt are not required. The receipt records the actual commit.
Inside VacLab, follow the workspace queue policy: run a small preflight, commit
and push, then enqueue. Queue executions still validate the matching receipt
and pushed commit.

## Generate the paper figures

From the source checkout:

```bash
PYTHONPATH=src python -m pitchbench.analysis.publication
```

This regenerates Figures 1–4 in this directory with the manuscript layout.
Each figure has a vector PDF and a 600-dpi PNG preview; editable SVG exports
are also produced. Use `--output-dir DIR` for a separate destination or
`--figures 1 2 3` to select figures. The generation manifest records the
source hashes and numerical coverage. Figure 1's fonts are bundled with the
package, including their open-font license.

- Figure 1: the three-level, 28-task pyramid.
- Figure 2: A1 predicted pitch distributions for six ALMs in MIDI/SPN/Hz;
  the companion CSVs account for every answer and identify exclusions.
- Figure 3: sample-weighted per-pitch accuracy, MIDI/SPN, C3–C5.
- Figure 4: the manuscript's four-format diagnostic accuracy comparison
  (MIDI/SPN/DoReMi/Hz). It retains this diagnostic's exact-match scores and
  sample-weighted aggregation. ANY is shown separately with its own denominator;
  it is not the three-format, 28-task Table 1 overall. The complete 30 source
  cells are in `accuracies_by_notation.csv`.

`table2.csv` reports the separate analysis sample and its ablation conditions;
its denominators differ from Table 1. Its numeric inputs are under
`paper/analysis/`. The long-form `accuracies_by_model_experiment.csv` is
regenerated by the Table 1 builder from the same full-precision scores.
