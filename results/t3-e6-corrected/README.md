# Table 3 E6 corrected-audio rerun — 2026-10-29

Reruns the E6 (slight detuning) row of the paper's unified analysis table on
the 252-stimulus analysis grid (preset `q1`: MIDI 29/38/47/56/65/74/83 × six
sources × six detune levels, 5 s), regenerated with the fixed sound engine.
The original analysis answers in `paper/analysis/` used instrument audio that
ignored the detune and are unchanged here.

| Model | E6 ANY (%) | Correct / N | Empty responses |
| --- | ---: | ---: | ---: |
| AF-Next-Instruct | 15.48 | 39 / 252 | 0 |
| Gemini 3.1 Pro | 9.92 | 25 / 252 | 19 of 756 |
| Gemini 3 Flash | 5.16 | 13 / 252 | 0 |

ANY is the per-stimulus OR of MIDI, SPN and Hz; only those three formats were
queried (`--paper-formats`). Gemini 3 Flash uses `google/gemini-3-flash-preview`
(see `../e6-d2-corrected/README.md`). GPT-4o audio is unavailable. Qwen E6 in the
paper comes from the collaborator's reruns, whose answers are not in this repo.

## Protocol and checks

- The regenerated grid has the same 252 (MIDI, source, detune) identities as
  the historical analysis answers.
- Instrument rendering needs libfluidsynth (here 3.5.6 from a conda env via
  `LD_LIBRARY_PATH`) and `PITCHBENCH_SF2=data/soundfonts/FluidR3_GM.sf2`;
  without them only the sine rows are generated. A spectral check confirms
  each detune is applied: per source, the residual offset from the target
  frequency is constant across detune levels.
- Gemini 3.1 Pro returned 19 empty completions (9 SPN, 7 MIDI, 3 Hz); the
  existing scorer counts them as wrong. Finish reasons were not saved.
- Table 3 means (29 rows) recompute the other 28 rows from `paper/analysis/`
  and reproduce the displayed means of every model before this change.

## Cost

Gemini 3.1 Pro US$1.400222 (756 calls), Gemini 3 Flash US$0.125691 (756 calls),
plus two small Flash preflights (under US$0.01). AF-Next ran locally on one GPU.
Raw outputs and logs remain under `outputs/pitchbench/t3-e6-corrected-*`.
