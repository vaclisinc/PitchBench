# Response on DSP and neural pitch baselines

Thank you for this suggestion. We agree that specialized pitch systems are important references for interpreting the audio-language-model results. We therefore evaluated three frozen, non-language-model baselines on all 28 PitchBench experiments (5,802 examples per baseline):

1. **DSP:** a deterministic pipeline using YIN for continuous monophonic \(F_0\), an RMS/onset envelope for note boundaries, and harmonic-CQT peak tracking for simultaneous pitches. It has no learned parameters.
2. **Basic Pitch:** the frozen [Spotify Basic Pitch](https://arxiv.org/abs/2203.09893) v0.4.0 checkpoint, which predicts polyphonic note events and pitch-bend contours.
3. **MuScriptor:** the frozen [MuScriptor-medium](https://arxiv.org/abs/2607.08168) 307M multi-instrument audio-to-MIDI checkpoint with greedy decoding.

Neither neural model was fine-tuned on PitchBench. All three systems use the same deterministic task adapter and unchanged PitchBench scorers. The adapter receives only the public task definition and question (for example, a queried timestamp or labelled reference pitch), never the per-example condition record, synthesis metadata, or ground-truth answer.

## How the baselines answer all 28 tasks

| Experiments | Operation on estimated pitch/events |
|---|---|
| A1--A3 | Return the dominant pitch and convert it to the requested notation. |
| B1 | Ignore silent regions and return the dominant active pitch. |
| B2 | Return the pitch active at the queried timestamp. |
| B3--B5 | Return the detected onset/offset of one note, a named note, or all notes. |
| C1--C4 | Count simultaneous pitches, compute the outer interval, match a chord template, or return the pitch set. |
| D1 | Count sequential note events. |
| D2 | Compare the median pitch of the two tones before semitone rounding when a continuous estimate is available. |
| D3--D4 | Return adjacent pitch-change signs or smoothed rising/falling contour segments. |
| D5 | Rank successive tones by estimated pitch. |
| D6 | Return the rounded signed semitone difference between successive tones. |
| D7a | Measure the interval in the audio and apply it to the labelled reference pitch in the question. |
| D8 | Return pitches ordered by onset time. |
| E1--E4 | Apply the same pitch decoder to the acoustically transformed signal. |
| E5 | Return the centre of the vibrato contour. |
| E6 | Return the nearest equal-tempered pitch to the continuous estimate. |
| F1--F2 | Track simultaneous pitches over time, select the requested register/voice, and return its ordered notes. |

MuScriptor emits integer MIDI events without pitch bends, so it cannot represent sub-semitone differences in D2. We include this limitation rather than adding a DSP \(F_0\) estimate to the neural model.

## Results

The table reports the unchanged PitchBench headline metric for each experiment (%). F1 and F2 use ordered-note F1.

| Category | DSP | Basic Pitch | MuScriptor |
|---|---|---|---|
| A | A1 95.8, A2 97.0, A3 82.9 | A1 98.7, A2 99.0, A3 82.4 | A1 15.2, A2 10.0, A3 6.2 |
| B | B1 98.1, B2 94.0, B3 78.8, B4 65.8, B5 54.2 | B1 99.4, B2 99.3, B3 81.2, B4 51.7, B5 22.5 | B1 64.4, B2 48.0, B3 19.4, B4 18.3, B5 0.8 |
| C | C1 28.1, C2 30.2, C3 58.9, C4 34.0 | C1 47.8, C2 62.1, C3 77.6, C4 61.0 | C1 4.4, C2 0.4, C3 5.2, C4 2.0 |
| D | D1 75.0, D2 80.3, D3 61.7, D4 73.8, D5 65.0, D6 83.0, D7a 92.3, D8 64.3 | D1 20.7, D2 76.5, D3 40.0, D4 73.8, D5 53.3, D6 48.7, D7a 55.4, D8 12.3 | D1 8.6, D2 14.4, D3 11.1, D4 20.0, D5 0.8, D6 6.4, D7a 7.7, D8 0.6 |
| E | E1 94.6, E2 36.7, E3 97.5, E4 96.9, E5 85.6, E6 23.8 | E1 91.2, E2 88.8, E3 96.7, E4 99.5, E5 68.8, E6 19.4 | E1 11.2, E2 9.2, E3 15.8, E4 4.7, E5 3.1, E6 0.0 |
| F | F1 20.6, F2 14.7 | F1 58.9, F2 27.3 | F1 66.1, F2 29.9 |
| **Macro average over 28 experiments** | **67.3** | **64.8** | **14.4** |

The corresponding example-weighted averages are 71.5% (DSP), 71.7% (Basic Pitch), and 13.6% (MuScriptor). For reference, the best of the six audio-language models in our submitted table has a 47.7% macro average (54.6% example-weighted).

MuScriptor helps answer whether Basic Pitch's F scores reflect a broken run. On F1, MuScriptor improves ordered-note F1 from 58.9% to 66.1%. On F2, the improvement is concentrated in the soprano (77.3% vs. 66.5%): alto is 31.3% vs. 31.4%, tenor is 9.4% for both, and bass is 1.6% vs. 1.9%. Thus, a stronger transcription front end improves the upper line, while the simple register-based adapter remains unable to reliably separate inner and lower voices.

D2 directly tests differences below one semitone. On the 1--25-cent subset, DSP obtains 61.7% and Basic Pitch 53.3%; on the 50--1,200-cent subset, both obtain 95.8%. MuScriptor obtains 10.0% and 18.1%, respectively, because its discrete MIDI output cannot preserve these frequency differences and it often produces extraneous events on the isolated synthetic stimuli.

All thresholds and decoding choices were fixed before the formal runs in complete YAML configurations. Missing detections were counted as incorrect, and all baselines used the same official data, sampling seed, prompts, and scorers. The results show that specialized DSP and note-transcription systems provide strong references, but no single baseline covers every kind of pitch competence: continuous DSP is best for fine frequency comparison, Basic Pitch is robust on isolated notes and transformations, and MuScriptor is most informative for upper-voice transcription while showing a substantial domain mismatch on short synthetic stimuli.
