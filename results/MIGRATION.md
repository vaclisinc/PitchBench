# PitchBench v1 → v2 experiment rename

PitchBench v2 introduced lowercase letter-prefix experiment IDs
(`<category><digit>`) and renamed every script from `exp_<N>_<desc>.py` to
`pitchbench_<id>_<desc>.py`. Old `results/exp_<old>/...` directories are kept
unchanged for traceability of past papers; new runs land in
`results/pitchbench_<id>_<desc>/run_<NNN>_<TS>/`. Use the table below to map
an old result directory to the script that produced it.

| v1 file                              | v2 file                                           |
| ------------------------------------ | ------------------------------------------------- |
| `exp_1_pitch.py`                     | `pitchbench_a1_pitch_id.py`                       |
| `exp_12_anchored_pitch.py`           | `pitchbench_a2_pitch_with_reference.py`           |
| `exp_10_pitch_duration.py`           | `pitchbench_a3_pitch_by_duration.py`              |
| `exp_2_pitch_in_silence.py`          | `pitchbench_b1_pitch_in_silence.py`               |
| `exp_9_simultaneous_pitches.py`      | `pitchbench_c3_chord_pitch_id.py`                 |
| `exp_5_pitch_count.py` (split)       | `pitchbench_c2_chord_pitch_count.py` (sim. half)  |
|                                      | `pitchbench_d1_seq_pitch_count.py` (seq. half)    |
| `exp_4_relative_pitch.py` (split)    | `pitchbench_d2_pitch_difference.py` (binary)      |
|                                      | `pitchbench_d6_pitch_ranking.py` (rank)           |
| `exp_11_pitch_trajectory.py`         | `pitchbench_d5_contour_continuous.py`             |
| `exp_8_multi_pitch.py`               | `pitchbench_d7_seq_pitch_id.py`                   |
| `exp_6_loudness.py`                  | `pitchbench_e1_loudness.py`                       |
| `exp_7_effects.py`                   | `pitchbench_e2_audio_effects.py`                  |
| `exp_13_embedding_geometry.py`       | `pitchbench_f1_embedding_geometry.py`             |
| `exp_14_token_logits.py`             | `pitchbench_f2_token_logits.py`                   |
| `exp_15_knn_oracle.py`               | `pitchbench_f3_knn_oracle.py`                     |
| `exp_3_nsynth.py`                    | `pitchbench_z1_pitch_id_nsynth.py`                |

## What's new in v2

- **`a4` `a5` `b2` `b3` `b4` `b5` `c1` `c4` `d3` `d4` `e3`** — eleven new
  experiments. See the per-script docstring for details.
- Every pitch-ID experiment asks **four** formats now: MIDI, SPN, Doremi, Hz.
- Every result `.txt` includes a marginal-accuracy table per varying IV.
- `query_alm` (in `helpers/api.py`) is the central ALM dispatcher and supports
  both local servers (Audio Flamingo, Qwen3-Omni) and OpenRouter
  (`openrouter/google/gemini-2.5-flash` etc.) through a single interface.
