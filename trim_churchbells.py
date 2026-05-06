#!/usr/bin/env python3
"""Trim churchbells.wav to keep only seconds 4-11."""

import soundfile as sf

# Load the audio file
audio_path = "data/downloaded/background/street-noise.wav"
y, sr = sf.read(audio_path)

# Calculate sample indices for seconds 4-11
start_sample = int(2 * sr)
end_sample = int(9 * sr)

# Trim the audio
y_trimmed = y[start_sample:end_sample]

# Save the trimmed audio (overwrite the original)
sf.write(audio_path, y_trimmed, sr)

print(f"Trimmed church-bells.wav to seconds 4-11")
print(f"Original duration: {len(y) / sr:.2f}s, New duration: {len(y_trimmed) / sr:.2f}s")
