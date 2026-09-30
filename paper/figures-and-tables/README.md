# Paper results

[**Table 1**](table1.md) is the canonical comparison: 28 tasks, six audio language
models, DSP, and Basic Pitch. [CSV](table1.csv) preserves source precision in
percentage units; [LaTeX](table1.tex) is ready to include in the manuscript
(requires `booktabs`, `multirow`, and `graphicx`). Markdown and LaTeX display one
decimal place. Do not edit these exports independently.

## Rebuild and verify

From the repository root, with Python 3.10 or later (standard library only):

```bash
PYTHONPATH=src python -m pitchbench.analysis.table1
PYTHONPATH=src python -m pitchbench.analysis.table1 --check
```

The builder rejects missing, duplicate, unexpected, non-finite, or out-of-range
scores. `--check` verifies all four committed exports, including source hashes.
Use `--output-dir DIR` to preview elsewhere. Table generation requires no model
calls, audio downloads, or access to the original experiment environments.

## Metrics and evidence

All scores are percentages. Mean is the equally weighted mean of 28 tasks,
computed before display rounding. D7 in the manuscript means D7a (concatenated
reference audio); Y1 and the split-reference variants are outside Table 1.

D8, F1 and F2 use Ordered Note F1, `2 × LCS / (N_gt + N_pred)`, retaining extra
predictions. ANY takes the maximum MIDI/SPN/Hz score per stimulus, then averages
over stimuli. Solfège is excluded from ANY for these tasks. D8 retains ±1 Hz
matching; F1/F2 retain ±1%. Other rows retain the published task accuracy metric;
see [experiment definitions](../../EXPERIMENTS.md).

| Cells | Evidence used by the builder |
| --- | --- |
| ALMs, D8/F1/F2 | [Sequence metrics](../../results/d8-ordered-note-f1/metrics.csv), `any_note_f1` |
| ALMs, C4 | Saved per-item MIDI/SPN/Hz correctness flags in `paper/evaluation/`; paths and counts in [source mapping](table1.sources.json) |
| ALMs, other tasks | [Original task aggregates](accuracies_by_model_experiment.csv), `accuracy` |
| DSP / Basic Pitch, D8 | [Baseline metrics](../../results/d8-baselines-lcs/metrics.csv), `any_note_f1` |
| DSP / Basic Pitch, other tasks | [Preserved baseline scores](../../results/d8-baselines-lcs/historical_table.csv), displayed percentages |

[table1.sources.json](table1.sources.json) records exact source hashes, model IDs,
task IDs, cell selection rules, aggregation, and display rounding. The original
ALM aggregate file is an immutable evidence input; its old D8/F1/F2 values are
replaced by the sequence metrics when building Table 1. Its C4 values are also
replaced: the old analysis fallback incorrectly counted solfège-only matches,
although the paper and the C4 evaluator define ANY using MIDI/SPN/Hz. The evidence bundles in
`results/` preserve raw answers, per-item scores, and provenance for auditing;
their intermediate tables are not separate paper tables.

**Precision limit:** the 27 non-D8 baseline scores are available only rounded to
0.1 percentage points. Baseline means therefore remain approximate (69.9% DSP,
73.1% Basic Pitch), with a conservative rounding bound below ±0.0483 percentage
points. The builder preserves the available ALM aggregate precision; it does
not infer additional precision from the displayed values.

## Manuscript cross-check

An audit against `PitchBench-paper` Table 1 at commit `f5fe30e` found both
manuscript discrepancies and the legacy C4 aggregation bug above. The table
below reports the saved per-item scores, with C4 using the existing three-format
ANY rule. It does not introduce new model calls or change the task scorers.

| Model / task | Manuscript (%) | Correct / total | Verified score (%) |
| --- | ---: | ---: | ---: |
| GPT-4o audio / B3 | 0.8 | 0 / 160 | 0.0 |
| GPT-4o audio / B4 | 0.0 | 1 / 120 | 0.8 |
| Gemini Flash / D3 | 0.0 | 2 / 180 | 1.1 |
| Gemini Flash / D4 | 35.4 | 0 / 160 | 0.0 |
| Gemini Flash / D5 | 0.0 | 9 / 120 | 7.5 |
| AF-next-instruct / C4 | 0.0 | 0 / 200 | 0.0 |
| Gemini Pro / C4 | 1.5 | 3 / 200 | 1.5 |
| Gemini Flash / C4 | 0.0 | 0 / 200 | 0.0 |
| GPT-4o audio / C4 | 0.0 | 1 / 200 | 0.5 |
| Qwen plus / C4 | 15.6 | 24 / 200 | 12.0 |
| Qwen flash / C4 | 12.8 | 13 / 200 | 6.5 |

B3/B4 use saved `correct` flags; D3/D4/D5 use `sequence_correct`,
`trajectory_correct`, and `answer_correct`, respectively. Their files are under
`paper/evaluation/_<model>/pitchbench_<task>_*/run_*/results_<model>.json`.
The first table-consolidation commit `35c44ae` retained the faulty legacy C4
aggregates; current exports correct them and recompute all means. Thus the
initial ten differing cells were not ten proven manuscript errors: two C4 cells
already matched the three-format protocol, leaving eight manuscript task-cell
discrepancies against the saved evidence. No manuscript files were edited by
this audit. Git history alone does not establish how the incorrect cells arose.

### Independent confirmation of Gemini Flash D4/D5

Both saved runs were rechecked directly from `raw_response`, without trusting
the stored correctness flags. Calling each task's `record_for` reproduced every
flag: **D4 = 0/160**, **D5 = 9/120**. Both scorer files are byte-identical to the
initial-submission commit `471221b`; this discrepancy is not caused by the
rebuttal's sequence-metric changes.

- [D4 raw results](../evaluation/_openrouter_google_gemini_flash_latest/pitchbench_d4_contour_continuous/run_001_20260506_193621/results_openrouter_google_gemini_flash_latest.json):
  all 160 responses contain 62 alternating direction tokens (152 start with
  `up`, eight with `down`). Ground truths, independently reconstructed from
  `traj_name`, contain only one or two tokens. Direct normalized sequence
  comparison also gives zero exact matches.
- [D5 raw results](../evaluation/_openrouter_google_gemini_flash_latest/pitchbench_d5_sequence_ranking_by_pitch/run_001_20260506_193644/results_openrouter_google_gemini_flash_latest.json):
  independently sorting each item's `presented_hz` reproduces every ground-truth
  ranking. Nine raw responses match those rankings verbatim, at zero-based row
  indices `0, 6, 13, 19, 22, 27, 37, 53, 80`. For example, frequencies
  `[233.0819, 220.0, 246.9417]` imply `2 1 3`, exactly the first raw response.
- The same model's D7a result is **46/130 = 35.4%**, matching the value appearing
  in the manuscript's D4 cell. This suggests a table transcription error, but
  does not prove its cause. No alternative Gemini Flash D4/D5 run was found in
  the local project results or shared `outputs/pitchbench` directory. These
  conclusions concern the saved May 6 evaluation, not a new inference run.

## Recompute sequence scores from saved answers

With the project's scoring dependencies available:

```bash
PYTHONPATH=src python -m pitchbench.experiments.rescore_sequences \
  --output-dir /tmp/pitchbench-alm-rescore
PYTHONPATH=src python -m pitchbench.experiments.rescore_sequences \
  --baseline-input results/d8-baselines-lcs \
  --output-dir /tmp/pitchbench-baseline-rescore
```

These reconstruct the sequence evidence without inference. See the
[ALM evidence](../../results/d8-ordered-note-f1/README.md) and
[baseline evidence](../../results/d8-baselines-lcs/README.md) for provenance and
validation. Floating-point aggregate serialization can differ at machine precision
between Python versions; compare numerical aggregates with an absolute tolerance
of `2e-15` on the 0–1 scale. The unified sequence scorer replaces the former F-only rescore
script and intermediate reports, which remain available in Git history.

## Baseline implementation

DSP uses YIN for monophonic continuous F0 (30–2,500 Hz, 256-ms window, 16-ms hop)
and a harmonic CQT for simultaneous pitches (288 bins, 36 bins/octave over eight
octaves, 16-ms hop). Basic Pitch 0.4.0 uses its packaged ONNX model for note events
and pitch-bend contours. The deterministic task adapters receive only audio,
the task ID and the public prompt, never hidden synthesis conditions or targets.

The reusable [baseline recipe](../../configs/paper_baselines.yaml) contains the
fixed acoustic and decoder settings. Copy it into a run directory, replace the
absolute dataset/environment/output placeholders, and assign a run ID. Keep all
runtime and result paths under that run directory or shared storage. For queue
snapshots, also set `runtime.asset_data_dir` to the canonical checkout's `data`
directory so background/soundfont links survive snapshot cleanup. Use the pinned
environment recorded in the [baseline receipt](../../results/d8-baselines-lcs/receipt.json);
Basic Pitch inference uses Python 3.11.15 and ONNX Runtime 1.19.2.

```bash
PYTHONPATH=src python -m pitchbench.baselines.evaluation /ABSOLUTE/PATH/TO/RUN/config.yaml --phase all
```

In VacLab, submit inference through the workspace queue after a tiny preflight.
`scripts/run_rebuttal_baselines.py` remains a compatibility wrapper for old run
receipts. The frozen D8 reproduction config stays with its result bundle; the
old numbered experiment configs are available through their original Git commits.

Other figure CSVs and images in this directory are retained evidence from the
submission. They are not regenerated by the Table 1 builder.
