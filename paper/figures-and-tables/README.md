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
reproduction provenance and a comparison against the previous table. All ALM
cells are replayed from saved raw answers against the official ground truth;
both baselines are rerun on all 5,802 fixed stimuli.

To replay all raw answers without model calls, download the official
`pitchbench-authors/PitchBench` dataset at revision
`f6c672608057cb877bfaacaeeff9bfb49eef7778`, then run with the package dependencies:

```bash
PYTHONPATH=src python -m pitchbench.analysis.replay \
  --dataset-dir /PATH/TO/FIXED/DATASET \
  --baseline-input results/table1-recomputed/baselines \
  --output-dir /tmp/pitchbench-table1-check
```

The audit matches stimulus identities, checks saved ground truth, reparses only
the three paper formats, and fails on unexpected duplicates or missing samples.
Optional external LLM parsing is disabled. Compare `metrics.csv` numerically
(tolerance `2e-15` on the 0–1 scale for sequence F1).

For fresh baseline inference, use [the reusable recipe](../../configs/paper_baselines.yaml)
and the frozen config/environment in the audit bundle. DSP uses YIN and harmonic
CQT; Basic Pitch 0.4.0 uses its packaged ONNX model. Adapters receive audio, task
ID and the public prompt. Set the recipe's absolute dataset, environment and run
paths, then use the canonical entrypoint:

```bash
PYTHONPATH=src python -m pitchbench.baselines.evaluation /PATH/TO/RUN/config.yaml --phase all
```

In VacLab, run a tiny preflight, commit/push, then submit through the workspace
queue. Keep outputs under the run directory and set `runtime.asset_data_dir` to
the canonical checkout's `data` directory. The old D8-only result bundles and
legacy aggregate CSVs remain historical evidence; they are not inputs to Table 1.
