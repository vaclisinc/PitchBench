# Response on DSP and neural pitch baselines

On top of the discussion above, our goal is not to outperform specialized DSP/neural pitch systems, but to make the LALM community aware that pitch should be treated as a basic capability of LALMs. At the same time, the fact that standard DSP/neural systems solve many of these tasks shows that the relevant information is recoverable from audio—and therefore that LALMs should, in principle, also be able to acquire it.

We evaluated three frozen baselines on all 28 PitchBench experiments (5,802 examples each):

- **DSP:** YIN for continuous monophonic \(F_0\) (30--2,500 Hz, 256-ms window, 16-ms hop) and a harmonic CQT for simultaneous pitches. The CQT has exactly **288 bins: 36 bins/octave over 8 octaves**, or **3 bins/semitone (33.3 cents/bin)**, with a 16-ms hop.
- **Neural pitch:** [Basic Pitch](https://arxiv.org/abs/2203.09893) v0.4.0 for note events and pitch-bend contours (3 bins/semitone), and [MuScriptor-medium](https://arxiv.org/abs/2607.08168) with greedy decoding and `acoustic_piano` conditioning for MIDI transcription.
- For tasks beyond direct note extraction, a fixed deterministic adapter converts these outputs into the requested answer: e.g., distinct-pitch counts, continuous higher/lower comparisons, nearest equal-tempered pitch, interval relative to the reference stated in the question, or an ordered pitch/voice sequence. It never receives hidden synthesis parameters or ground truth.

The table reports the unchanged PitchBench headline metric for each experiment (%); F1 and F2 use ordered-note F1.

| Category | DSP | Basic Pitch | MuScriptor (piano-conditioned) |
|---|---|---|---|
| A | A1 95.8, A2 97.0, A3 82.9 | A1 98.7, A2 99.0, A3 82.4 | A1 7.7, A2 8.0, A3 4.3 |
| B | B1 98.1, B2 94.0, B3 78.8, B4 65.8, B5 54.2 | B1 99.4, B2 99.3, B3 81.2, B4 51.7, B5 22.5 | B1 69.4, B2 60.0, B3 6.9, B4 22.5, B5 0.0 |
| C | C1 28.1, C2 30.2, C3 58.9, C4 34.0 | C1 47.8, C2 62.1, C3 77.6, C4 61.0 | C1 1.3, C2 1.3, C3 1.0, C4 0.5 |
| D | D1 76.4, D2 80.3, D3 61.7, D4 73.8, D5 88.3, D6 96.2, D7a 98.5, D8 80.1 | D1 42.9, D2 76.5, D3 56.7, D4 73.8, D5 67.5, D6 97.5, D7a 100.0, D8 98.8 | D1 7.1, D2 8.3, D3 17.2, D4 17.5, D5 5.0, D6 11.0, D7a 11.5, D8 0.6 |
| E | E1 94.6, E2 36.7, E3 97.5, E4 96.9, E5 85.6, E6 23.8 | E1 91.2, E2 88.8, E3 96.7, E4 99.5, E5 68.8, E6 19.4 | E1 9.2, E2 4.2, E3 6.7, E4 4.7, E5 1.9, E6 0.6 |
| F | F1 20.6, F2 14.7 | F1 58.9, F2 27.3 | F1 67.8, F2 30.9 |
| **Macro average over 28 experiments** | **69.4** | **73.1** | **13.8** |
