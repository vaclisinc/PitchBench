# Response on DSP and neural pitch baselines

Thank you for this suggestion. We agree that specialized pitch systems are important references for interpreting the audio-language-model results. We therefore evaluated three frozen, non-language-model baselines on all 28 PitchBench experiments (5,802 examples per baseline):

1. **DSP:** a deterministic monophonic-\(F_0\) and multipitch pipeline with no learned parameters.
2. **Basic Pitch:** the frozen [Spotify Basic Pitch](https://arxiv.org/abs/2203.09893) v0.4.0 ONNX checkpoint.
3. **MuScriptor:** the frozen [MuScriptor-medium](https://arxiv.org/abs/2607.08168) 307M audio-to-MIDI checkpoint with greedy decoding. We set its instrument condition to `acoustic_piano`; this conditions generation and masks all other instrument tokens. The same fixed piano prior is applied to every PitchBench timbre.

Neither neural model was fine-tuned on PitchBench. The task adapter receives only the audio, public task ID, and exact question (e.g., a queried timestamp, requested number of tones, or labelled reference pitch), never the hidden condition record, synthesis parameters, or ground-truth answer.

## Exact front ends

**DSP.** Audio is resampled to 16 kHz. For monophonic tasks we use YIN, a classical time-domain fundamental-frequency estimator that compares the signal with delayed copies using a cumulative normalized difference; it is not a neural model. We use 30--2,500 Hz, a 4,096-sample window (256 ms), a 256-sample hop (16 ms), and an RMS activity mask. For simultaneous-pitch tasks only, we use a 288-bin CQT: 36 bins/octave over 8 octaves from A0 (27.5 Hz) to approximately 6.91 kHz, with the same 16-ms hop. Thus the CQT has 3 bins/semitone, or approximately 33.3 cents/bin. Integer MIDI candidates are scored by the first four harmonics with weights \(1, 1/2, 1/3, 1/4\).

**Basic Pitch.** We use onset threshold 0.5, frame threshold 0.3, minimum note length 127.7 ms, pitch bends enabled, and the Melodia option enabled. Its pitch contour has 3 bins/semitone (approximately 33.3 cents/bin). We retain the predicted note events and pitch-bend contour; we do not reduce its output to integer MIDI before tasks that require a continuous comparison.

**MuScriptor.** We use the medium checkpoint (307M parameters), one `acoustic_piano` instrument condition, greedy decoding, and integer MIDI events. MuScriptor has no pitch-bend output, so it cannot preserve sub-semitone differences.

## How note transcription is converted into task answers

The conversion follows the semantics of each question rather than treating every experiment as note extraction:

| Experiments | Deterministic operation on estimated pitch/events |
|---|---|
| A1--A3, B1--B2 | Return the dominant active pitch, or the pitch active at the queried timestamp, in the requested notation. |
| B3--B5 | Return detected onset/offset times for one note, a named note, or all notes. |
| C1--C4 | From pitches active in the stable middle region, count distinct simultaneous pitches, compute the outer interval, match a chord template, or return the pitch set. |
| D1 | Count **distinct musical pitches**, not raw note events. This avoids counting repeated fragments of one decoded pitch as new pitches. |
| D2 | Compare the median continuous pitch of the first and second tones before semitone rounding. |
| D3--D4 | Return adjacent note-event pitch-change signs or smoothed rising/falling contour segments. Adjacent fragments with the same rounded pitch are collapsed. |
| D5--D6 | Recover the successive stable tones, then rank them or return their signed rounded semitone difference. |
| D7a | Measure the interval in the audio and add it to the reference pitch explicitly given in the question. |
| D8 | Return stable pitches in onset order. |
| E1--E5 | Apply the same pitch decoder to the transformed signal; vibrato is represented by its contour centre. |
| E6 | Select the dominant note event, add its median pitch bend to obtain a continuous MIDI estimate, then round to the nearest equal-tempered pitch. No detuning metadata is used. |
| F1--F2 | At each time, rank simultaneous pitches by register, select the requested line/voice, and return its ordered notes. |

For C1, a pitch must occur in at least 30% of frames in the middle 20--80% of the active region. When a sequence question explicitly states the requested number of tones, that public number is used to merge or split stable detected segments. Missing detections are counted as incorrect.

## Results

The table reports the unchanged PitchBench headline metric for each experiment (%). F1 and F2 use ordered-note F1.

| Category | DSP | Basic Pitch | MuScriptor (piano-conditioned) |
|---|---|---|---|
| A | A1 95.8, A2 97.0, A3 82.9 | A1 98.7, A2 99.0, A3 82.4 | A1 7.7, A2 8.0, A3 4.3 |
| B | B1 98.1, B2 94.0, B3 78.8, B4 65.8, B5 54.2 | B1 99.4, B2 99.3, B3 81.2, B4 51.7, B5 22.5 | B1 69.4, B2 60.0, B3 6.9, B4 22.5, B5 0.0 |
| C | C1 28.1, C2 30.2, C3 58.9, C4 34.0 | C1 47.8, C2 62.1, C3 77.6, C4 61.0 | C1 1.3, C2 1.3, C3 1.0, C4 0.5 |
| D | D1 76.4, D2 80.3, D3 61.7, D4 73.8, D5 88.3, D6 96.2, D7a 98.5, D8 80.1 | D1 42.9, D2 76.5, D3 56.7, D4 73.8, D5 67.5, D6 97.5, D7a 100.0, D8 98.8 | D1 7.1, D2 8.3, D3 17.2, D4 17.5, D5 5.0, D6 11.0, D7a 11.5, D8 0.6 |
| E | E1 94.6, E2 36.7, E3 97.5, E4 96.9, E5 85.6, E6 23.8 | E1 91.2, E2 88.8, E3 96.7, E4 99.5, E5 68.8, E6 19.4 | E1 9.2, E2 4.2, E3 6.7, E4 4.7, E5 1.9, E6 0.6 |
| F | F1 20.6, F2 14.7 | F1 58.9, F2 27.3 | F1 67.8, F2 30.9 |
| **Macro average over 28 experiments** | **69.4** | **73.1** | **13.8** |

The corresponding example-weighted averages are 73.2% (DSP), 78.5% (Basic Pitch), and 11.7% (MuScriptor). For reference, the best of the six audio-language models in our submitted table has a 47.7% macro average (54.6% example-weighted).

D2 directly tests frequency differences smaller than one semitone. On the 1--25-cent subset, DSP obtains 61.7%, Basic Pitch 53.3%, and MuScriptor 5.0%; on the 50--1,200-cent subset, they obtain 95.8%, 95.8%, and 11.1%, respectively. DSP and Basic Pitch can compare continuous \(F_0\)/pitch-bend estimates, whereas MuScriptor emits only integer MIDI and often produces no event or extra events for these short synthetic tones.

MuScriptor also helps verify that Basic Pitch's Category-F result is not a broken run. On F1, MuScriptor improves ordered-note F1 from 58.9% to 67.8%. On F2, MuScriptor versus Basic Pitch obtains 78.8% vs. 66.5% for soprano, 31.0% vs. 31.4% for alto, 11.2% vs. 9.4% for tenor, and 2.5% vs. 1.9% for bass. Thus, the transcription front end improves the upper line, while the simple register-based adapter remains weak for inner and lower voices.

All thresholds and task conversions were fixed in complete YAML configurations before their formal runs. All baselines used the same official data revision, sampling seed, questions, and scorers. The results show why both DSP and neural transcription references are useful: continuous DSP is strongest for fine frequency comparison, Basic Pitch is robust across most isolated-note and sequence transformations, and piano-conditioned MuScriptor is informative for upper-voice transcription but has a severe domain mismatch on short synthetic and non-piano stimuli.
