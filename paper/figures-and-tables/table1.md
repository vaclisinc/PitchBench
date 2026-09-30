# Table 1 — PitchBench results

Scores are percentages. D8, F1 and F2 use Ordered Note F1; other tasks use accuracy. Mean weights each of the 28 tasks equally. D7a is the concatenated reference task (D7 in the paper); Y1 is excluded. DSP and Basic Pitch means are approximate: their 27 non-D8 inputs were preserved only to 0.1 percentage points (mean rounding bound: ±0.0483 percentage points).

Best displayed score per row is bold. All formats are generated from the same evidence; see [sources and reproduction](README.md).

| Task | AF-next-instruct | Gemini 3.1 Pro | Gemini 3 Flash | GPT-4o audio | Qwen-3.5 omni plus | Qwen-3.5 omni flash | DSP | Basic Pitch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 Pitch ID | 35.9 | 14.9 | 6.0 | 6.1 | 91.6 | 75.1 | 95.8 | **98.7** |
| A2 Loudness | 43.0 | 22.5 | 23.0 | 14.5 | 90.5 | 71.5 | 97.0 | **99.0** |
| A3 Duration | 29.5 | 20.0 | 21.0 | 13.8 | 74.8 | 61.0 | **82.9** | 82.4 |
| B1 Silence | 5.6 | 25.6 | 21.3 | 14.6 | 90.0 | 75.6 | 98.1 | **99.4** |
| B2 At Time | 10.0 | 17.3 | 16.7 | 10.7 | 77.3 | 56.7 | 94.0 | **99.3** |
| B3 Time Pitch | 0.0 | 0.0 | 2.5 | 0.0 | 13.1 | 9.4 | 78.8 | **81.2** |
| B4 Time Spec. | 0.0 | 0.0 | 1.7 | 0.8 | 20.0 | 2.5 | **65.8** | 51.7 |
| B5 Time Multi. | 0.0 | 0.0 | 0.0 | 0.0 | 30.0 | 2.5 | **54.2** | 22.5 |
| C1 Count | 13.6 | 46.5 | 39.9 | 3.1 | 9.7 | 20.2 | 28.1 | **47.8** |
| C2 Interval | 7.8 | 6.9 | 9.1 | 5.2 | 9.1 | 11.2 | 30.2 | **62.1** |
| C3 Quality | 9.9 | 13.0 | 10.4 | 10.9 | 13.0 | 13.0 | 58.9 | **77.6** |
| C4 Chord P. | 1.5 | 2.5 | 0.0 | 1.5 | 21.0 | 8.5 | 34.0 | **61.0** |
| D1 Seq. Count | 26.4 | 51.4 | 25.0 | 20.0 | **82.1** | 78.6 | 76.4 | 42.9 |
| D2 High/Low | 50.0 | 63.6 | 50.8 | 50.0 | 65.2 | 57.6 | **80.3** | 76.5 |
| D3 Contour D. | 0.0 | 15.0 | 1.1 | 0.0 | 15.6 | 12.8 | **61.7** | 56.7 |
| D4 Contour C. | 0.0 | 43.8 | 0.0 | 3.1 | 51.9 | 17.5 | **73.8** | **73.8** |
| D5 Rank | 4.2 | 5.0 | 7.5 | 1.7 | 20.0 | 2.5 | **88.3** | 67.5 |
| D6 Seq. Int. | 3.8 | 19.5 | 5.1 | 3.0 | 15.7 | 5.5 | 96.2 | **97.5** |
| D7a Ref. Pitch | 13.9 | 58.5 | 35.4 | 14.6 | 96.2 | 76.9 | 98.5 | **100.0** |
| D8 Seq. Pitch | 14.1 | 18.2 | 14.3 | 10.0 | 86.5 | 48.0 | 94.6 | **99.6** |
| E1 Effects | 35.0 | 21.3 | 19.6 | 12.9 | 86.7 | 55.8 | **94.6** | 91.2 |
| E2 Background | 12.1 | 4.2 | 14.2 | 12.1 | 42.9 | 22.9 | 36.7 | **88.8** |
| E3 Saturation | 45.0 | 14.2 | 24.2 | 10.8 | 86.7 | 69.2 | **97.5** | 96.7 |
| E4 Stretch | 34.9 | 18.2 | 9.4 | 10.4 | 88.5 | 79.2 | 96.9 | **99.5** |
| E5 Vibrato | 24.4 | 2.5 | 3.1 | 3.8 | 65.6 | 39.4 | **85.6** | 68.8 |
| E6 Off Pitch | 23.8 | 12.5 | 13.8 | 11.3 | 22.5 | **30.0** | 23.8 | 19.4 |
| F1 Atonal | 26.3 | 16.4 | 30.6 | 26.9 | 35.5 | 30.5 | 20.6 | **58.9** |
| F2 Tonal | 26.8 | 14.2 | 31.9 | 35.7 | **46.4** | 28.7 | 14.7 | 27.3 |
| Mean | 17.8 | 19.6 | 15.6 | 11.0 | 51.7 | 37.9 | 69.9 | **73.1** |
