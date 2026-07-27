# Response on DSP and neural pitch baselines

Thank you for this suggestion. We agree that specialized pitch estimators provide an important reference point for interpreting the audio-language-model results. We will add two frozen, non-language-model baselines and evaluate them on the same 28 experiments with the same PitchBench scoring:

1. **DSP baseline.** A deterministic signal-processing pipeline uses YIN for continuous monophonic \(F_0\), an energy/onset envelope for note boundaries, and harmonic spectral peak tracking for simultaneous pitches. The YIN component follows the same implementation we used in our recent acoustic pitch diagnostics. This baseline has no learned parameters.
2. **Neural baseline.** We use the frozen [Spotify Basic Pitch](https://arxiv.org/abs/2203.09893) checkpoint, an instrument-agnostic model that predicts polyphonic note events, onsets, offsets, and pitch-bend contours. We do not fine-tune it on PitchBench.

Both front ends are converted to the same intermediate event representation: continuous pitch, nearest MIDI pitch, onset, offset, and confidence. A deterministic task adapter then converts those estimated events into the answer requested by each experiment. The adapter receives the public experiment ID (the task definition) and the same question shown to the evaluated model, such as a queried timestamp, a labelled reference pitch, the number of tones, or the requested voice. It never receives the per-example condition record, ground-truth answer, or synthesis metadata.

The 28 experiments are handled as follows:

| Experiments | Deterministic operation on the estimated pitch/events |
|---|---|
| A1--A3 | Return the dominant active pitch; convert it to MIDI, note name, solfège, or Hz as requested. |
| B1 | Ignore silent frames and return the dominant active pitch. |
| B2 | Return the pitch active at the timestamp stated in the question. |
| B3 | Return the first onset and final offset of the detected note. |
| B4 | Return the onset and offset of the detected event matching the note named in the question. |
| B5 | Return all detected onset/offset pairs in chronological order. |
| C1 | Count the distinct pitches active in the steady-state chord region. |
| C2 | Return the semitone distance between the lowest and highest simultaneous pitch. |
| C3 | Match the detected pitch-class set to the closest chord-quality template. |
| C4 | Return the set of simultaneous pitches. |
| D1 | Count the detected sequential note events. |
| D2 | Compare the continuous median \(F_0\) of the two tones before semitone rounding. This directly evaluates differences from 1 cent to one octave. |
| D3 | Return the sign of each adjacent pitch change. |
| D4 | Smooth the continuous pitch contour and return its sequence of rising/falling segments. |
| D5 | Sort the continuous pitch estimates of the successive tones and return their position indices. |
| D6 | Return the rounded signed semitone difference between the two successive tones. |
| D7a | Estimate the measured interval between the two tones and apply it to the labelled reference pitch given in the question. |
| D8 | Return detected pitches ordered by onset time. |
| E1--E4 | Apply the same dominant-pitch decoder to the acoustically transformed signal. |
| E5 | Return the median/centre of the vibrato contour. |
| E6 | Return the nearest equal-tempered pitch to the continuous estimate. |
| F1--F2 | Track simultaneous pitches through time, select the requested non-crossing register/voice, and return that track's ordered notes. |

For Basic Pitch, D2 and D5 use its pitch-bend contour rather than only its integer MIDI note events; thus the baseline is not artificially forced to semitone resolution, although its native contour resolution may still limit very small differences. The DSP baseline uses YIN's continuous \(F_0\) estimate for these comparisons. For polyphonic tasks, the DSP baseline uses harmonic spectral peaks, whereas Basic Pitch uses its native multipitch output.

All thresholds and task rules are fixed in one YAML configuration rather than fitted per example. We use Basic Pitch's documented default note-decoding settings, pin and report all software versions, save raw predictions, count missing detections as incorrect rather than skipping them, and apply the unchanged PitchBench scorers. These additions show how much of each task can be solved by a conventional signal-processing system or a specialized transcription network, and make the gap between specialized pitch extraction and open-ended pitch reasoning by general-purpose audio-language models clearer.
