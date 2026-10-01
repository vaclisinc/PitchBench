# Complete Table 1 verification

This bundle is the sole numeric input to the camera-ready Table 1. It combines
raw-answer replay for the six ALMs with a complete fixed-dataset reproduction
of DSP and Basic Pitch. The final table is generated from `metrics.csv`; no cell
is patched from a rounded intermediate table.

## Protocol

- 28 tasks, eight models, 224 cells. The official dataset contains 5,802 stimuli
  at `pitchbench-authors/PitchBench@f6c672608057cb877bfaacaeeff9bfb49eef7778`.
- ALMs: reparse the 34,810 saved answers against official ground truth. No new
  ALM inference. All stimulus identities must match; unexpected missing or
  duplicate records are rejected.
- Baselines: rerun each model on all 5,802 official audio files with the frozen
  acoustic/decoder configuration and the Python 3.11.15 baseline environment.
  Basic Pitch 0.4.0 uses the same packaged ONNX weights as the earlier D8 run.
- Only MIDI/SPN/Hz are queried and scored for pitch-format tasks. DoReMi is not
  read by the table replay. Non-pitch tasks retain their original task scorer.
- D8/F1/F2: per-item maximum Ordered Note F1 across the three formats, then mean
  over items. LCS matching retains extra predictions. D8 uses ±1 Hz, F1/F2 ±1%.
  Other tasks use accuracy. Overall weights each of the 28 task means equally.
- Retain full numeric precision until final display. Display uses one decimal,
  ROUND_HALF_UP; tied displayed scores receive the same rank.

`baselines/` contains compressed result records with raw answers and ground
truth fields. Repeated prompts and unused DoReMi fields are omitted; public
prompts are in the fixed dataset. Each file records the SHA-256 of its original
full JSON. Large logs, audio and full runtime outputs remain in shared storage.
`run.json` hashes every input to the replay; baseline inference provenance,
frozen config, environment and model hash are recorded separately.

## Verification result

All 224 cells and 46,414 saved responses were replayed using the installed wheel
in Python 3.12.7: zero saved-score mismatches. The two baselines each cover all
5,802 stimuli. Their task CSVs and 28-task overall summaries agree with the
independent replay to `2e-15` on the 0–1 scale; see
[pipeline-verification.json](pipeline-verification.json).

DSP's 28 displayed scores all reproduce the previous table. Basic Pitch has
three display changes:

| Task | Previous (%) | Verified unrounded (%) | Display (%) | Explanation |
| --- | ---: | ---: | ---: | --- |
| B3 | 81.2 | 130/160 × 100 = 81.25 | 81.3 | Use ROUND_HALF_UP consistently |
| E1 | 91.2 | 219/240 × 100 = 91.25 | 91.3 | Use ROUND_HALF_UP consistently |
| F1 | 58.9 | 76.5/130 × 100 = 58.846153846… | 58.8 | Remove intermediate four-decimal rounding |

An independent LCS calculation directly on Basic Pitch F1's raw MIDI answers
also gives a score sum of 76.5 over 130 items. The old cat-F summary rounded the
mean to `0.5885`, which then displayed as `58.9`; the full-precision mean displays
as `58.8`. MIDI supplies the maximum three-format F1 for every item in this cell.

The exact overall scores are **69.9180285613% DSP** and **73.1234522372% Basic
Pitch**, still displayed as 69.9 and 73.1. Both D8 reruns reproduce all 171 earlier
raw MIDI/SPN/Hz answers and their per-format/ANY F1 values exactly. Original
historical raw answers for the other 27 baseline tasks were unavailable, so
those comparisons establish agreement with the rounded table, not historical
per-item identity.

[table_differences.csv](table_differences.csv) lists all displayed differences
against both the pre-audit repository table (`846c310`) and the manuscript
(`f5fe30e`), including means. The manuscript comparison changes 13 task cells
and three means; the pre-audit repository comparison changes five task cells.

## Known data gap

GPT-4o B1 has 158 saved responses for 160 official stimuli. Its score uses the
158 available responses, preserving the original observed denominator. No
missing response is fabricated or silently scored as zero. Missing identities:

- `C2_cello_in_silence_at22000ms_for5000ms_of60000ms.wav`
- `C4_piano_in_silence_at53000ms_for5000ms_of60000ms.wav`

The legacy GPT C4 aggregate also reported 400 samples despite 200 unique raw
records; the verified table uses 200.

## Verified manuscript discrepancies

The comparison reference is the manuscript table at `PitchBench-paper@f5fe30e`.
Manuscript files were not edited by this audit. The following accuracy cells
were independently reparsed from raw responses and checked against official GT:

| Model / task | Manuscript (%) | Correct / total | Verified (%) |
| --- | ---: | ---: | ---: |
| GPT-4o / B3 | 0.8 | 0 / 160 | 0.0 |
| GPT-4o / B4 | 0.0 | 1 / 120 | 0.8 |
| Gemini Flash / D3 | 0.0 | 2 / 180 | 1.1 |
| Gemini Flash / D4 | 35.4 | 0 / 160 | 0.0 |
| Gemini Flash / D5 | 0.0 | 9 / 120 | 7.5 |
| GPT-4o / C4 | 0.0 | 1 / 200 | 0.5 |
| Qwen plus / C4 | 15.6 | 24 / 200 | 12.0 |
| Qwen flash / C4 | 12.8 | 13 / 200 | 6.5 |
| Qwen plus / C1 | 9.7 | 22 / 228 | 9.6 |
| AF-next / D7a | 13.9 | 18 / 130 | 13.8 |

The final two differences came from rounding the intermediate task mean before
rounding the displayed percentage. The earlier C4 aggregate incorrectly counted
solfège-only matches; the proper ANY rule uses three formats. AF-next C4=0.0 and
Gemini Pro C4=1.5 already agreed with the manuscript and the correct protocol.

For Gemini Flash D4, every saved response contains 62 alternating direction
tokens, while each ground truth contains one or two. D5 has nine verbatim correct
rankings; sorting each saved `presented_hz` also reproduces the stored ranking.
Both original task scorers match the initial-submission commit `471221b`.
Gemini Flash D7a happens to equal 46/130=35.4%, suggesting a transcription error
for D4, but the available history does not establish the cause.

## Package verification

The public `pitchbench evaluate paper` selector uses exactly the 28 paper tasks
and MIDI/SPN/Hz prompts. Missing tasks prevent a partial paper overall from being
reported. Stored-data evaluation does not regenerate synthesis conditions or
exclude instruments based on local synthesizer availability.

D8/F1/F2 use F1 through both direct and Parquet evaluation, task CSVs and overall.
B3/B4/B5 retain their task-level timing rollup. Numeric summaries preserve full
precision. Regression tests cover these paths, custom model labels, omitted
formats and the prohibition on remote parsing during replay.

The installed wheel was tested in a separate Python 3.12 environment using the
project lockfile. See `package-verification.json` for the exact scope and result.
The baseline environment remains separately pinned to preserve reproducibility.
These checks cover this branch and its locally built wheel; this audit does not
publish a new PyPI release.

## Reproduce

From the repository root, with the package dependencies installed and the fixed
Hugging Face dataset revision available locally:

```bash
PYTHONPATH=src python -m pitchbench.analysis.replay \
  --dataset-dir /PATH/TO/FIXED/DATASET \
  --baseline-input results/table1-recomputed/baselines \
  --output-dir /tmp/pitchbench-table1-check
PYTHONPATH=src python -m pitchbench.analysis.table1 --check
```

The replay makes no model calls. Compare the replayed `metrics.csv` numerically
with this bundle (tolerance `2e-15` on the 0–1 scale); `score_mismatches.csv` must
contain only its header. The two missing GPT-4o B1 answers remain explicitly
listed in the receipt. The table builder itself needs only the standard library.

For fresh baseline inference, use `configs/paper_baselines.yaml` and this bundle's
frozen `config.yaml`. Change machine-specific input/environment/output paths;
keep the dataset revision, acoustic settings, decoder settings and pinned
package versions. Run the existing entrypoint:

```bash
PYTHONPATH=src python -m pitchbench.baselines.evaluation /PATH/TO/RUN/config.yaml --phase all
```

VacLab runs must use the workspace queue after a small real-data preflight.
The successful reproduction is attempt 2 in `baseline-run.json`; attempt 1 was
cancelled after identifying an optional solfège parser fallback. Attempt 2
explicitly disables remote parsing and excludes DoReMi queries. Incomplete
attempt-1 answers are not used in this bundle.
