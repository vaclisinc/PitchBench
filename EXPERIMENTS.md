# PitchBench Dataset Experiment Breakdown

For each experiment, the key independent variables and their benchmark values are described in sequence, along with a note on how conditions are stratified when sub-sampling is applied. Sources are drawn from a catalogue of 19 timbres: four synthetic waveforms (sine, sawtooth, square, and triangle) and 15 General MIDI instruments (piano, electric keyboard, guitar, flute, trumpet, trombone, clarinet, oboe, violin, cello, organ, bass, synthesizer lead, synthesizer pad, and voice). When models are prompted to output a pitch, four notation formats are queried per stimulus: MIDI integer, Scientific Pitch Notation (SPN, e.g., C4), fixed-do solfège (do = C, re = D, and so on), and frequency in Hertz. The conditions (i.e. parameter settings) for each experiment were stratified according to the variables salient to that specific experiment. Each experiment produced a total of 120-240 audio fragments, with the exception of A1.

---

## Level 1: Atomic Pitch Perception

At the lowest level, models identify a single pitch in audio clips containing a single tone for their full duration. These tasks isolate core frequency identification across variations in amplitude and note duration.

### A1: Single Pitch Identification

Each stimulus is a single sustained tone rendered by one source, either a synthetic waveform or a General MIDI instrument, and the model is asked to report the pitch in all four notation formats. The pitch set sweeps all 61 MIDI values from 29 to 89, the note range leading to audible and identifiable sound fragments for all sources. Exact-match scoring is applied for MIDI, SPN, and solfège, while a ±1 % tolerance is used for the Hz format. Duration is fixed at five seconds. Each condition corresponds to a unique (midi, source) pair.

### A2: Pitch Identification under Loudness Variation

The single-tone identification task is repeated at five amplitude levels, ranging from −30 dBFS (near inaudible) to +6 dBFS (slightly above unity gain), to test whether pitch identification degrades at amplitude extremes. Conditions are stratified by (midi, loudness_db), with four conditions per stratum and a total of 10 MIDI values distributed across the range.

### A3: Pitch Identification under Duration Variation

The single-tone task is repeated at seven duration levels spanning three orders of magnitude, from 50 ms (a transient attack) to 60 seconds (a sustained drone), in order to test whether very short or very long tones impair pitch extraction across all 19 timbres. Conditions are stratified by (midi, duration_ms), with three conditions per stratum and a total of 10 MIDI values distributed across the range.

---

## Level 2: Contextual Pitch Perception

At the intermediate level, pitch must be identified within a temporal, structural, or acoustic context.

### B: Temporal Localization

Experiments B1 through B5 test whether models can correctly locate and identify pitches when a tone or sequence is embedded in a longer clip, or when a specific onset or offset time must be reported.

#### B1: Pitch of a Hidden Tone in Silence

A single five-second tone is embedded at one of eight positions within a 60-second silent clip. The prompt states that exactly one note is present but does not specify when it occurs, requiring the model to identify the pitch without prior knowledge of its onset time. Eight positions uniformly sample the full clip duration: 2,000, 7,000, 14,000, 22,000, 27,000, 41,000, 47,000, and 53,000 ms. Conditions are stratified by (midi, pos_ms), with two conditions per stratum and a total of 10 MIDI values distributed across the range.

#### B2: Pitch at a Queried Timestamp

A sequence of five or ten non-overlapping notes is arranged within a 60-second clip, and the model is asked to report the pitch sounding at a single queried timestamp that falls at the midpoint of one designated target note. This task requires simultaneous temporal localization and pitch identification. Conditions are stratified by (n_notes, target_idx), with ten conditions per stratum and a total of 10 MIDI values distributed across the range.

#### B3: Onset and Offset of a Single Tone

The model must report the onset and offset times, in MM:SS.cc format, of a single sustained tone embedded in a 60-second silent clip. The tone appears at one of eight positions and may be either one second or five seconds in duration. A predicted timestamp is scored correct if it falls within 250 ms of the ground truth. One condition is generated per (pos_ms, duration_ms, midi) stratum.

#### B4: Onset and Offset of a Named Note Among Distractors

A sequence of notes is arranged in a 60-second clip, and the prompt names one target note and asks for its onset and offset. Five or seven distractor pitches occupy the remaining positions. The target may appear first, in the middle, or last in the sequence, testing whether serial position affects localization accuracy. Scoring uses the same ±250 ms tolerance as B3. Ten conditions per (target_pos, n_distractors, duration_ms) stratum are generated.

#### B5: All Note Onsets and Offsets

The model must report the onset and offset of every note in a sequence of three, five, or eight notes, listed in chronological order as comma-separated MM:SS.cc timestamps. Both regular (evenly spaced) and irregular (randomly spaced) rhythms are tested. A stimulus is scored correct only if the model returns the right number of timestamps and each falls within 250 ms of its ground truth. Ten conditions per (rhythm, n_notes, duration_ms) stratum are generated.

---

### C: Simultaneous Pitches

Experiments C1 through C4 present multiple tones sounding simultaneously: as chords and dyads: and ask the model to count pitches, label intervals, classify harmonic quality, or enumerate all constituent pitches.

#### C1: Chord Pitch Count

Simultaneously sounding tones drawn from standard chord types or a random pitch set are presented, and the model must count how many distinct pitches it hears. Standard chord types span seven qualities (major, minor, diminished, augmented, dominant seventh, major seventh, and minor seventh) plus two suspended chords (sus2 and sus4) and one half-diminished chord, yielding three- or four-note chords; random sets extend coverage to one through six simultaneous pitches. Both same-instrument and mixed-instrument chords are tested. One condition is generated per (n, chord_quality, root_midi, same_instrument) stratum.

#### C2: Dyad Interval Identification

Two simultaneous tones are presented and the model must report the interval between them as an integer number of semitones, covering the full range from a minor second (1 semitone) to an octave (12 semitones). Same-instrument and mixed-instrument renderings are crossed with ten root pitches and two note durations. Scoring requires an exact integer match. One condition is generated per (interval_st, same_instrument, root_midi) stratum.

#### C3: Chord Quality Identification

A chord is presented and the model must classify its harmonic quality from a closed set of ten labels: major, minor, diminished, augmented, dominant seventh, major seventh, minor seventh, half-diminished, sus2, and sus4. Both single-timbre and mixed-timbre renderings are tested. One condition is generated per (chord_quality_gt, same_instrument, root_midi) stratum.

#### C4: Simultaneous Pitch Enumeration

All pitches of a chord must be listed in all four notation formats. Scoring is set-exact: the predicted set must match the ground-truth set exactly, regardless of order. Chord types cover 13 dyad interval classes (unison through octave), the four triads, and three seventh-chord types (dominant, major, and minor), rooted at each of the ten benchmark pitches. This is the most demanding simultaneous-pitch task, requiring full enumeration rather than counting or labelling. Duration is fixed at five seconds, and one condition is generated per (chord_type, n_notes, root_midi) stratum.

---

### D: Sequential Pitch Tasks

Experiments D1 through D8 involve notes played one after another, asking the model to count, compare, trace the contour of, rank, measure intervals between, or enumerate the pitches in a temporal sequence.

#### D1: Sequential Pitch Count

A sequence of one to ten distinct pitches is played one after another with no pitch repeated, and the model must count how many distinct pitches it heard. Both regular (constant 300 ms inter-note gap) and irregular (randomly varied gaps) rhythms are used to test whether temporal regularity aids counting. Three seeded trials per cell ensure robustness across random pitch orderings. Five conditions are generated per (n, rhythm, duration_ms) stratum.

#### D2: Binary Higher/Lower Judgment

Two tones separated by a brief silence are presented and the model must report which---the first or the second---is higher in pitch. The pitch difference spans 11 levels from 1 cent to 1,200 cents (one octave), crossed with four inter-tone silences. Presentation order is randomized, placing chance performance at 50%. One condition is generated per (delta_cents, order, duration_ms, base_name) stratum.

#### D3: Discrete Melodic Contour

A sequence of notes is played step-wise and the model must output the directional shape as a comma-separated list of `up`/`down` tokens, one per transition. Four sequence lengths (2 to 7 transitions) are crossed with five step sizes (1 to 11 semitones) and three note durations. Scoring requires an exact match of the full token sequence. Three conditions are generated per (n_transitions, step_size_st, note_duration_ms) stratum.

#### D4: Continuous Pitch Trajectory

A single pitch glides continuously through one of four trajectories (rising, falling, rise-then-fall, or fall-then-rise) and the model must describe the shape using the same `up`/`down` vocabulary as D3, with synonym normalization applied before scoring. Glides are linear or arch/valley sweeps of 1, 4, 7, or 12 semitones. Duration is fixed at five seconds. One condition is generated per (traj_name, interval_st, start_midi) stratum.

#### D5: Pitch Ranking

Three to seven tones are presented in a random order and the model must rank them from lowest to highest, outputting their position indices. Pitch differences between adjacent ranks span five levels from 25 to 400 cents, and both regular and irregular rhythms are included. One condition is generated per (n_notes, delta_cents, rhythm, base_name) stratum.

#### D6: Sequential Dyad Interval

Two notes are played in succession and the model must report the signed semitone interval as an integer, with positive values indicating an ascending interval and negative values a descending one. All 12 positive intervals are tested in both directions, crossed with four inter-note silences, two note durations, and ten root pitches. An off-by-one diagnostic is recorded alongside the headline exact-match metric. One condition is generated per (signed_st, base_midi) stratum.

#### D7a: Pitch with Linguistic Reference, Concatenated Audio

A reference tone and a target tone are concatenated into one audio file, separated by a 500 ms gap. The prompt reveals the reference pitch in the queried notation format and asks the model to name the target. 13 signed intervals from −12 to +12 semitones are tested with 5 reference notes. Two conditions are generated per (ref_midi, interval) stratum.

#### D7b: Pitch with Linguistic Reference, Split Audio

Conditions are identical to D7a, but the reference and target tones are delivered as two separate audio inputs, accommodating models with multi-audio API support. Comparing D7a and D7b on the same model isolates how much of any anchoring benefit is attributable to the linguistic reference versus the additional demand of segmenting a concatenated signal. Models that do not support multi-audio calls are automatically skipped after a run-time probe. D7a is the experiment used in the benchmark to compare fairly across models.

#### D8: Sequential Pitch Identification

All pitches in a sequence of three, five, or ten notes must be listed in order in all four notation formats. Three conditions are generated per (n_notes, source) stratum.

Headline scoring uses Ordered Note F1: $2M/(N_{gt}+N_{pred})$, where $M$ is the LCS match count. Full predictions are retained, including extra notes. MIDI matches exactly, SPN matches equivalent pitches, and Hz retains the original ±1 Hz tolerance. ANY takes the maximum of MIDI/SPN/Hz per stimulus, then averages over stimuli; solfège is diagnostic only. Strict full-sequence exact match and positional matches remain auxiliary diagnostics. See the [saved-response rescore and updated model table](results/d8-ordered-note-f1/README.md).

---

### E: Acoustic Variations

Experiments E1 through E6 pair the single-tone pitch identification task with signal-level manipulations: including audio effects, background noise, harmonic distortion, vibrato, and intonation error: to test the robustness of pitch perception across degraded or perturbed stimuli.

#### E1: Pitch under Audio Effects

A single sustained tone is processed with one of six DSP effects and the model must still identify the pitch. The effects cover two filter types (high-pass and low-pass), hard quantization (4-bit bitcrushing), heavy saturation (30 dB soft-clip drive), long convolution reverb (room size 0.9), and heavy chorus (rate 1.2 Hz, depth 0.9, mix 0.6). Each output is RMS-normalized to the dry signal so that loudness cannot serve as an effect-strength cue. Two conditions are generated per (effect_type, midi, source_type) stratum.

#### E2: Pitch under Background Noise

A sustained tone is mixed with a background sound at one of four signal-to-noise ratios and the model must identify the pitch while ignoring the background. Backgrounds include white noise, artificial competing tones, and four real-world recordings: church bells, crowd noise, rain, and street noise: mixed at SNRs of +10, 0, −6, and −12 dB. One condition is generated per (background, snr_db, midi) stratum.

#### E3: Pitch under Harmonic Saturation

A single tone is processed with a tanh soft-clipper at three drive levels. Because soft-clipping preserves the fundamental frequency while enriching the overtone spectrum, the correct pitch answer is unchanged. Light (6 dB), medium (15 dB), and heavy (30 dB) drive levels span the range from mild tape-like warmth to near-hard-clipping, testing whether a richer harmonic spectrum disrupts pitch extraction. Two conditions are generated per (saturation_level, midi, source_type) stratum.

#### E4: Pitch under Time Stretching

The duration of a three-second tone is modified either by resampling (which changes both duration and pitch, analogous to speed-change playback) or by phase-vocoder time stretching (which changes duration while preserving pitch). The model must identify the pitch as it actually sounds in the processed audio. Resampling at 0.5× halves duration and raises pitch by 12 semitones; resampling at 2× doubles duration and lowers pitch by 12 semitones; stretching at either factor leaves pitch unchanged. The ground-truth MIDI is always the perceived, post-processing pitch. Two conditions are generated per (condition, midi, source_type) stratum.

#### E5: Pitch under Vibrato

A tone with sinusoidal frequency modulation is presented and the model must identify the nominal centre pitch, ignoring the oscillation. Rate and depth are crossed spanning musically common (3 Hz, 25 cents) to perceptually extreme (10 Hz, 200 cents) values. Two conditions are generated per (vibrato_rate_hz, vibrato_depth_cents, midi) stratum.

#### E6: Pitch Slightly Out of Tune

A tone is detuned by a small amount within the perceptual basin of attraction of its nominal pitch: up to 45% of the half-semitone bandwidth toward each neighbour: so that the correct answer is always the original MIDI note. The model is asked to identify the nearest in-tune pitch. Four conditions are generated per (midi, detune_hz) stratum.

---

## Level 3: Melodic Pitch Perception

At the highest level, PitchBench tests whether models can identify pitches within a melodic line in polyphonic settings where multiple voices sound simultaneously, building onto all capabilities established at level 1 (absolute pitch) and 2 (pitch within chord and sequences). 

### F1: Melodic Line in Synthetic Polyphony

Two or three synthetic melodic voices play simultaneously with no rests: each voice always sustains a note and consecutive notes flow directly into one another without gaps: and the model must transcribe one designated voice identified by its register rank (highest, second-from-top, and so on down to lowest). The target voice always has exactly ten notes; distractor voices have one to twenty notes with randomly varying per-note durations. Both same-instrument and mixed-instrument configurations are tested at two tempos: slow (one second per note) and medium (500 ms per note). Two seeded trials are run per cell. One condition is generated per (n, source_label, tempo, x) stratum.

### F2: Voice Identification in Bach Chorales

Excerpts from four-part Bach chorales (soprano-alto-tenor-bass) drawn from the music21 corpus are rendered as audio, and the model must transcribe one designated voice: soprano, alto, tenor, or bass: identified by name. For each (chorale, voice) pair, the longest contiguous segment in which the target voice maintains its register rank without crossing any other voice is extracted (minimum four notes, maximum 30 seconds), ensuring that the register cue in the prompt remains valid throughout. Both same-instrument (all four voices on one timbre) and mixed-instrument configurations (each voice on a distinct timbre) are tested across five chorales: BWV 66.6, 4.8, 7.7, 26.6, and 57.8. Eight conditions are generated per (chorale_slug, x) stratum.
