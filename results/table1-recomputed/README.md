# Table 1 evidence

This bundle supplies 216 available scores and eight explicit invalidations in
the 28-task, eight-model comparison. The table builder reads `metrics.csv` and
computes the equal-weight mean only for models with all 28 scores.

The [E6/D2 correction bundle](../e6-d2-corrected/README.md) records new ALM and
baseline inference, old/new scores, model availability and actual API costs.
Qwen was deferred by the owner; the original GPT-4o endpoint is unavailable,
and the historical Flash alias now resolves to a different model. Their
E6/D2 cells and overall means are shown as — (blank in CSV). The old answer
files remain historical records and are excluded by `unavailable.json`.

## Protocol

- Dataset: `vaclis/PitchBench@6aebaf876b7d7d0c95d28cacdb808f4dc75f0829`,
  containing 5,802 fixed stimuli.
- Pitch formats: MIDI, SPN, and Hz. ANY is the per-stimulus maximum across
  these formats; solfège does not contribute.
- D8/F1/F2: Ordered Note F1, `2 × LCS / (N_gt + N_pred)`, averaged over
  stimuli. Extra predictions remain in the denominator. D8 uses ±1 Hz
  matching; F1/F2 use ±1%. Other tasks use accuracy.
- Means retain full precision. Percentages display one decimal using
  ROUND_HALF_UP; displayed ties share the same rank.
- Available ALM scores are reparsed from 33,642 saved responses against official
  ground truth. Both baselines cover all 5,802 stimuli: E6/D2 were rerun on
  corrected audio; the other 26 task answers and provenance are unchanged.

## Contents

Recorded project paths are relative to the repository. Shared data, model and
run paths use `${VACLAB_ROOT}` as a portable placeholder; substitute your own
workspace root when reading archived commands. These receipts describe past
executions; use `configs/paper_baselines.yaml` for a fresh run. Raw/source/original
hashes identify the external original bytes, while committed file hashes identify
the sanitized evidence shipped here. Git commit IDs in execution receipts record
the revisions used before the public history rewrite.

| File | Purpose |
| --- | --- |
| `metrics.csv` | Per-model/task score, sum, observed count, and expected count |
| `overall.csv` | Equal-weight mean across 28 tasks; blank for incomplete models |
| `unavailable.json` | Explicit model/task invalidations and reasons |
| `run.json` | Replay provenance, input hashes, ground-truth checks, coverage |
| `baselines/` | Compressed baseline answers and stimulus metadata |
| `config.yaml` | Historical full-dataset baseline configuration; corrected two-task config is in `../e6-d2-corrected/` |
| `baseline-receipt.json`, `baseline-run.json` | Original inference provenance for the unchanged tasks |
| `model.json` | Basic Pitch model identity |
| `package-verification.json` | Source/wheel tests, answer preservation, and environment checks |
| `pipeline-verification.json` | Baseline task and overall agreement |
| `score_mismatches.csv` | Saved-score consistency checks where scores are present |

Compressed ALM answers are under `paper/evaluation/`. Each answer file preserves
its source hash and inference metadata. Derived benchmark scores are stored only
in this bundle and the generated Table 1 exports.

## Coverage

The replay verifies 45,246 usable responses and all 224 cell slots (216 scored,
eight explicitly unavailable), with zero score mismatches. The DSP overall
is 73.2236292106%; Basic Pitch is 75.2324890337%.

GPT-4o B1 has 158 saved responses out of 160 stimuli. Its score uses those 158
responses. The missing identities are:

- `C2_cello_in_silence_at22000ms_for5000ms_of60000ms.wav`
- `C4_piano_in_silence_at53000ms_for5000ms_of60000ms.wav`

No missing response is fabricated or silently scored as zero. The table caption
and source manifest disclose this denominator.

## Reproduce

Follow the [reproduction guide](../../paper/figures-and-tables/README.md) for
the fixed dataset, package installation, and separate baseline environment.
From the source checkout:

```bash
PYTHONPATH=src python -m pitchbench.analysis.replay \
  --dataset-dir /PATH/TO/FIXED/DATASET \
  --baseline-input results/table1-recomputed/baselines \
  --output-dir /tmp/pitchbench-table1-check
PYTHONPATH=src python -m pitchbench.analysis.table1 --check
```

The replay makes no model calls. Its `metrics.csv` must match this bundle
numerically, with tolerance `2e-15` on the 0–1 scale. It checks every stimulus
identity and saved ground-truth field; unexpected missing or duplicate records
are rejected. The table builder needs only the standard library.

Fresh baseline inference uses `configs/paper_baselines.yaml` and the pinned
ONNX environment in `configs/baselines-requirements.txt`. Change only local
input/environment/output paths when reproducing these results.
