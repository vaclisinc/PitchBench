# Response on Category F scoring

This records the earlier F-only rescore. D8 now uses Ordered Note F1 too; see the [combined D8/F1/F2 results and updated overall means](../../results/d8-ordered-note-f1/README.md).

Thank you for raising this point. We agree that full-sequence exact match is too strict for the Category F transcription tasks because one missing or extra note can make an otherwise partially correct sequence score zero.

We therefore rescored all saved Category F model responses using **Ordered Note F1**. For each response, we find the longest common subsequence of ground-truth and predicted pitches. If the alignment contains $M$ matched notes, the score is $2M/(N_{\mathrm{gt}}+N_{\mathrm{pred}})$. This gives partial credit while penalizing wrong, missing, extra, duplicated, and out-of-order notes.

Following the aggregation used in our main table, we compute the score separately for MIDI, scientific pitch notation, and Hz, take the maximum of the three scores for each stimulus, and then macro-average over stimuli. Solfège is excluded because it does not encode octave. No models were queried again; the results below were computed from the original saved responses.

| Model | f1 Atonal | f2 Tonal | Updated mean |
|---|---:|---:|---:|
| AF-next-instruct | 26.3 | 26.8 | 17.3 |
| Gemini 3.1 Pro | 16.4 | 14.2 | 18.9 |
| Gemini 3 Flash | 30.6 | 31.9 | 15.1 |
| GPT-4o audio | 26.9 | 35.7 | 10.6 |
| Qwen-3.5 omni plus | **35.5** | **46.4** | 50.6 |
| Qwen-3.5 omni flash | 30.5 | 28.7 | 36.3 |

We will replace the Category F exact-match values in the paper with these Ordered Note F1 scores, relabel the table values from accuracy to score where appropriate, and clarify the scoring and cross-format aggregation in the evaluation section.
