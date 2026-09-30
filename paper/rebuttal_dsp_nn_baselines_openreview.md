# Response on DSP and neural pitch baselines

On top of the discussion above, our goal is not to outperform specialized DSP/neural pitch systems, but to make the LALM community aware that pitch should be treated as a basic capability of LALMs. At the same time, the fact that standard DSP/neural systems solve many of these tasks shows that the relevant information is recoverable from audio—and therefore that LALMs should, in principle, also be able to acquire it.

We evaluated three frozen baselines on all 28 PitchBench experiments (5,802 examples each):

- **DSP:** YIN for continuous monophonic \(F_0\) (30--2,500 Hz, 256-ms window, 16-ms hop) and a harmonic CQT for simultaneous pitches. The CQT has exactly **288 bins: 36 bins/octave over 8 octaves**, or **3 bins/semitone (33.3 cents/bin)**, with a 16-ms hop.
- **Neural pitch:** [Basic Pitch](https://arxiv.org/abs/2203.09893) v0.4.0 for note events and pitch-bend contours (3 bins/semitone), and [MuScriptor-medium](https://arxiv.org/abs/2607.08168) with greedy decoding and `acoustic_piano` conditioning for MIDI transcription.
- For tasks beyond direct note extraction, a fixed deterministic adapter converts these outputs into the requested answer: e.g., distinct-pitch counts, continuous higher/lower comparisons, nearest equal-tempered pitch, interval relative to the reference stated in the question, or an ordered pitch/voice sequence. It never receives hidden synthesis parameters or ground truth.

This historical table reports the metrics used for the baseline runs (%); F1 and F2 use ordered-note F1. D8 below still reports the original exact-match score. D8 now uses Ordered Note F1; the [updated LALM results](../results/d8-ordered-note-f1/README.md) must not be compared directly with this table's D8 row until the baseline responses are rescored.

| Experiment | DSP | Basic Pitch | MuScriptor (piano-conditioned) |
|---|---|---|---|
| A1 | 95.8 | 98.7 | 7.7 |
| A2 | 97.0 | 99.0 | 8.0 |
| A3 | 82.9 | 82.4 | 4.3 |
| B1 | 98.1 | 99.4 | 69.4 |
| B2 | 94.0 | 99.3 | 60.0 |
| B3 | 78.8 | 81.2 | 6.9 |
| B4 | 65.8 | 51.7 | 22.5 |
| B5 | 54.2 | 22.5 | 0.0 |
| C1 | 28.1 | 47.8 | 1.3 |
| C2 | 30.2 | 62.1 | 1.3 |
| C3 | 58.9 | 77.6 | 1.0 |
| C4 | 34.0 | 61.0 | 0.5 |
| D1 | 76.4 | 42.9 | 7.1 |
| D2 | 80.3 | 76.5 | 8.3 |
| D3 | 61.7 | 56.7 | 17.2 |
| D4 | 73.8 | 73.8 | 17.5 |
| D5 | 88.3 | 67.5 | 5.0 |
| D6 | 96.2 | 97.5 | 11.0 |
| D7a | 98.5 | 100.0 | 11.5 |
| D8 | 80.1 | 98.8 | 0.6 |
| E1 | 94.6 | 91.2 | 9.2 |
| E2 | 36.7 | 88.8 | 4.2 |
| E3 | 97.5 | 96.7 | 6.7 |
| E4 | 96.9 | 99.5 | 4.7 |
| E5 | 85.6 | 68.8 | 1.9 |
| E6 | 23.8 | 19.4 | 0.6 |
| F1 | 20.6 | 58.9 | 67.8 |
| F2 | 14.7 | 27.3 | 30.9 |
| **Macro average over 28 experiments** | **69.4** | **73.1** | **13.8** |
