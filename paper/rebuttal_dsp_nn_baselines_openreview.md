# Response on DSP and neural pitch baselines

Thank you for this suggestion. We agree that specialized pitch systems are an important reference for interpreting the audio-language-model results. We have now evaluated two frozen, non-language-model baselines on all 28 PitchBench experiments (5,802 examples per baseline):

1. **DSP:** a deterministic pipeline using YIN for continuous monophonic \(F_0\), an RMS/onset envelope for note boundaries, and harmonic-CQT peak tracking for simultaneous pitches. This baseline has no learned parameters.
2. **Neural:** the frozen [Spotify Basic Pitch](https://arxiv.org/abs/2203.09893) v0.4.0 checkpoint, which predicts polyphonic note events and pitch-bend contours. It was not fine-tuned on PitchBench.

Both systems produce a shared intermediate representation containing continuous pitch, nearest MIDI pitch, onset, offset, and confidence. A deterministic task adapter then converts these estimates into the answer requested by each experiment. The adapter receives the public task definition and question (e.g., a queried timestamp or labelled reference pitch), but never receives the per-example condition record, synthesis metadata, or ground-truth answer.

## How the baselines cover all 28 tasks

| Experiments | Operation on estimated pitch/events |
|---|---|
| A1--A3 | Return the dominant pitch and convert it to the requested notation. |
| B1 | Ignore silent regions and return the dominant active pitch. |
| B2 | Return the pitch active at the timestamp in the question. |
| B3--B5 | Return the detected onset/offset of one note, a named note, or all notes, respectively. |
| C1--C4 | Count simultaneous pitches, compute the outer interval, match a chord template, or return the pitch set. |
| D1 | Count sequential note events. |
| D2 | Compare the continuous median \(F_0\) of the two tones before semitone rounding. |
| D3--D4 | Return adjacent pitch-change signs or smoothed rising/falling contour segments. |
| D5 | Rank successive tones using continuous pitch estimates. |
| D6 | Return the rounded signed semitone difference between successive tones. |
| D7a | Measure the interval in the audio and apply it to the labelled reference pitch in the question. |
| D8 | Return pitches ordered by onset time. |
| E1--E4 | Apply the same pitch decoder to the acoustically transformed signal. |
| E5 | Return the centre of the vibrato contour. |
| E6 | Return the nearest equal-tempered pitch to the continuous estimate. |
| F1--F2 | Track simultaneous pitches over time, select the requested register/voice, and return its ordered notes. |

Thus, tasks such as D2 (sub-semitone higher/lower comparison) and D7a (reasoning relative to a reference pitch) are evaluated through explicit deterministic operations on continuous acoustic estimates rather than being reduced to note transcription.

## Results

The table reports the unchanged PitchBench headline metric for each experiment (%). F1 and F2 use ordered-note F1.

| Category | DSP | Basic Pitch |
|---|---|---|
| A | A1 95.8, A2 97.0, A3 82.9 | A1 98.7, A2 99.0, A3 82.4 |
| B | B1 98.1, B2 94.0, B3 78.8, B4 65.8, B5 54.2 | B1 99.4, B2 99.3, B3 81.2, B4 51.7, B5 22.5 |
| C | C1 28.1, C2 30.2, C3 58.9, C4 34.0 | C1 47.8, C2 62.1, C3 77.6, C4 61.0 |
| D | D1 75.0, D2 80.3, D3 61.7, D4 73.8, D5 65.0, D6 83.0, D7a 92.3, D8 64.3 | D1 20.7, D2 76.5, D3 40.0, D4 73.8, D5 53.3, D6 48.7, D7a 55.4, D8 12.3 |
| E | E1 94.6, E2 36.7, E3 97.5, E4 96.9, E5 85.6, E6 23.8 | E1 91.2, E2 88.8, E3 96.7, E4 99.5, E5 68.8, E6 19.4 |
| F | F1 20.6, F2 14.7 | F1 58.9, F2 27.3 |
| **Macro average over 28 experiments** | **67.3** | **64.8** |

For comparison, the best of the six audio-language models in our submitted table has a 47.7% macro average (54.6% when weighted by examples), while DSP and Basic Pitch reach 67.3% and 64.8% macro average (71.5% and 71.7% weighted), respectively.

D2 directly tests frequency differences below one semitone. On the 1--25 cent subset, DSP obtains 61.7% and Basic Pitch 53.3%; on the 50--1,200 cent subset, both obtain 95.8%. This shows that using a continuous estimate enables above-chance comparison, but very small differences remain difficult, especially for Basic Pitch's pitch-bend representation.

All thresholds are fixed in one YAML configuration. Basic Pitch uses its documented default note-decoding thresholds; missing detections are counted as incorrect; and both baselines use the same official test data and unchanged PitchBench scorers. Overall, the specialized systems substantially improve atomic pitch identification, but they still struggle with very fine frequency differences, multi-event organization, and polyphonic voice tracking. We will add these baselines and their full per-experiment results to the revision.
