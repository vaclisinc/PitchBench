# D8, F1 and F2 Ordered Note F1

This bundle records the sequence-only rescore. Its non-sequence rows and overall
means retain the legacy C4 aggregation, which included solfège. Use the
[canonical Table 1](../../paper/figures-and-tables/table1.md) for the final
comparison with C4 corrected to MIDI/SPN/Hz; the sequence metrics here are unchanged.

Rescored saved responses; no new model calls. Main scores use order-preserving one-to-one LCS matches: `2M/(N_gt+N_pred)`. ANY is the maximum of MIDI, SPN and Hz for each stimulus, then the macro mean. Solfège is excluded from ANY.

D8 retains its ±1 Hz matching tolerance; F1/F2 retain ±1%. Full predicted sequences are scored, including extra notes. Strict exact match is auxiliary. Legacy D8 accuracy truncated extra predictions; it is reported separately for comparison.

Scores below are percentages. Overall means replace D8/F1/F2 in the original 28-task table; the other tasks are unchanged. Raw evaluation inputs remain unchanged.

| Model | D8 legacy | D8 strict | D8 LCS | F1 LCS | F2 LCS | Overall |
|---|---:|---:|---:|---:|---:|---:|
| audio_flamingo_next_instruct | 0.0 | 0.0 | 14.1 | 26.3 | 26.8 | 17.8 |
| dashscope_qwen3_5_omni_flash | 2.9 | 2.9 | 48.0 | 30.5 | 28.7 | 37.9 |
| dashscope_qwen3_5_omni_plus | 55.0 | 55.0 | 86.5 | 35.5 | 46.4 | 51.7 |
| openrouter_google_gemini_3_1_pro_preview | 0.0 | 0.0 | 18.2 | 16.4 | 14.2 | 19.6 |
| openrouter_google_gemini_flash_latest | 0.0 | 0.0 | 14.3 | 30.6 | 31.9 | 15.6 |
| openrouter_openai_gpt_4o_audio_preview | 0.0 | 0.0 | 10.0 | 26.9 | 35.7 | 11.0 |

`metrics.csv` and `item_scores.csv` contain unrounded aggregate and per-stimulus evidence. `scores_by_model_experiment.csv` contains the updated 28-task table. `run.json` records the code commit, source hashes and reproduction command.
