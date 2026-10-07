# Corrected E6/D2 rerun — 2026-10-07

Corrected official audio at `vaclis/PitchBench@6aebaf876b7d7d0c95d28cacdb808f4dc75f0829` replaces E6/D2 evidence for AF-Next Instruct, Gemini 3.1 Pro, DSP and Basic Pitch. Each rerun covers all 160 E6 and 132 D2 stimuli. The other 26 tasks (208 score rows and 215 evidence files) are unchanged.

## Scores

All numbers are percentages; arrows compare the historical invalid-audio run with the corrected run. The historical overall means are retained here only for comparison.

| Model | E6 old → new | D2 old → new | Mean old → new | Δ mean (pp) | Rank, same four models |
| --- | ---: | ---: | ---: | ---: | ---: |
| AF-next-instruct | 23.75 → 48.13 | 50.00 → 51.52 | 17.71 → 18.63 | 0.92 | 4 → 4 |
| Gemini 3.1 Pro | 12.50 → 13.13 | 63.64 → 71.21 | 19.52 → 19.82 | 0.29 | 3 → 3 |
| Gemini 3 Flash | 13.75 → — | 50.76 → — | 15.62 → — | — | — |
| GPT-4o audio | 11.25 → — | 50.00 → — | 10.94 → — | — | — |
| Qwen 3.5 plus | 22.50 → — | 65.15 → — | 51.39 → — | — | — |
| Qwen 3.5 flash | 30.00 → — | 57.58 → — | 37.86 → — | — | — |
| DSP | 23.75 → 98.13 | 80.30 → 98.48 | 69.92 → 73.22 | 3.31 | 2 → 2 |
| Basic Pitch | 19.38 → 73.13 | 76.52 → 81.82 | 73.12 → 75.23 | 2.11 | 1 → 1 |

Rank compares the same four completed models on all 28 tasks. `comparison.csv` also records historical ranks among all eight models. Incomplete models have no current mean or overall rank; missing cells are never filled with zeros or old scores.

## Unavailable models

- Qwen 3.5 plus/flash: owner delegated reruns to a collaborator. Old E6/D2 answers remain historical files, explicitly excluded by `../table1-recomputed/unavailable.json`.
- GPT-4o audio: the original `openai/gpt-4o-audio-preview` ID returned HTTP 400 (invalid model ID).
- Gemini Flash: the handoff ID without `~` is invalid. The actual historical `~google/gemini-flash-latest` alias now resolves to `google/gemini-3.8-flash`. One availability probe confirmed this; its answer is excluded from the Gemini 3 Flash column. No replacement model was evaluated.

These models’ E6/D2 cells and 28-task means display —. To restore a model, rerun both corrected tasks using its verified original endpoint/version, replace its saved answers, remove its two explicit invalidations, replay, and rebuild/check Table 1. A replacement model needs a separate consistently evaluated comparison.

## Protocol and checks

- Official Parquet WAV bytes are extracted without synthesis. E6 changes 121 instrument files and D2 changes 101; non-audio fields and waveform bytes are unchanged.
- Saved answers have the same stimulus identities as the official shards but a different order. The loader checks identities and reorders official rows to the shared historical answer order before ALM queries.
- E6 queries MIDI, SPN and Hz only; D2 queries one binary prompt. Original prompts and task scorers are retained. All 612 Gemini Pro calls produced nonempty completions.
- The new loader rejects mismatched dataset revisions and answer identities before API calls. Real FluidR3_GM rendering tests cover pitch accuracy, one-cent distinctions and duration.
- DSP and Basic Pitch reuse the Python 3.11.15 / NumPy 1.26.4 pinned baseline environment and identical decoder settings. DSP E6 instrument accuracy improves from 0/121 to 119/121; Basic Pitch from 0/121 to 86/121.
- AF-Next uses `nvidia/audio-flamingo-next-hf@5634886e2615c2f587dcf8b93c6edfe9907930ca`, bfloat16, deterministic generation, 128-token maximum, on one L40S. `af-environment.json` records its serving packages. The historical answer metadata did not pin a checkpoint revision.
- Full replay verifies 45,246 usable responses with zero score mismatches. Eight explicitly unavailable model/task cells are excluded.

## Cost and provenance

Formal Gemini Pro inference: **US$1.091526** (612 calls). Including four preflight calls, one latency probe and one successful Flash alias probe: **US$1.10109425**. Invalid-model requests returned no usage charge. Local GPU and CPU jobs incurred no API charges.

`inference-sources.json` maps every replacement to its raw output hash, committed evidence hash and provider usage. `gemini-pro-session-cost.json` is the authoritative saved usage record: the existing result saver looked up usage by display label and wrote zero in the per-result metadata, so that metadata is preserved as produced rather than hand-edited. Queue receipts record actual commits, commands, resources and attempts; the baseline first attempt failed before inference because environment symlinks were not ignored, then succeeded after the ignore fix. Original 26-task baseline provenance remains in `../table1-recomputed/`; the corrected two-task config and receipt are here.

The detailed active-run files and logs remain under `outputs/pitchbench/e6-d2-corrected*`. Reproduce the official-data ALM route with the command in `paper/figures-and-tables/README.md`; the reusable baseline recipe is `configs/paper_baselines.yaml`.
