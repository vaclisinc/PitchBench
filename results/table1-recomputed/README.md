# Table 1 evidence

This bundle supplies all 224 scores in the paper comparison: 28 tasks, six
audio language models, DSP, and Basic Pitch. The table builder reads
`metrics.csv` and computes the equal-weight mean of the 28 task scores.

## Protocol

- Dataset: `pitchbench-authors/PitchBench@f6c672608057cb877bfaacaeeff9bfb49eef7778`,
  containing 5,802 fixed stimuli.
- Pitch formats: MIDI, SPN, and Hz. ANY is the per-stimulus maximum across
  these formats; solfège does not contribute.
- D8/F1/F2: Ordered Note F1, `2 × LCS / (N_gt + N_pred)`, averaged over
  stimuli. Extra predictions remain in the denominator. D8 uses ±1 Hz
  matching; F1/F2 use ±1%. Other tasks use accuracy.
- Means retain full precision. Percentages display one decimal using
  ROUND_HALF_UP; displayed ties share the same rank.
- ALM scores are reparsed from 34,810 saved responses against official ground
  truth. Each baseline covers all 5,802 official audio files.

## Contents

| File | Purpose |
| --- | --- |
| `metrics.csv` | Per-model/task score, sum, observed count, and expected count |
| `overall.csv` | Equal-weight mean across 28 tasks |
| `run.json` | Replay provenance, input hashes, ground-truth checks, coverage |
| `baselines/` | Compressed baseline answers and stimulus metadata |
| `config.yaml` | Resolved baseline inference configuration |
| `baseline-receipt.json`, `baseline-run.json` | Inference environment and run provenance |
| `model.json` | Basic Pitch model identity |
| `package-verification.json` | Source/wheel tests, answer preservation, and environment checks |
| `pipeline-verification.json` | Baseline task and overall agreement |
| `score_mismatches.csv` | Saved-score consistency checks where scores are present |

Compressed ALM answers are under `paper/evaluation/`. Each answer file preserves
its source hash and inference metadata. Derived benchmark scores are stored only
in this bundle and the generated Table 1 exports.

## Coverage

The replay verifies 46,414 responses and all 224 model/task cells. The DSP overall
is 69.9180285613%; Basic Pitch is 73.1234522372%.

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
