"""Diagnose whether OpenRouter actually forwards audio to audio-capable models.

For each model:
  1. Test 1 — sentinel: send WAV + ask for NO_AUDIO_RECEIVED if it didn't arrive.
                Also inspect usage.prompt_tokens_details.audio_tokens.
  2. Test 2 — pitch ID: send the same WAV with a real pitch-ID prompt, then
                send a *different* pitch with the same prompt. If the answer
                doesn't change with the audio, the model is guessing.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(Path(".env"))
API_KEY = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_KEY not in .env")

MODELS = [
    "openai/gpt-audio",
    "openai/gpt-4o-audio-preview",
    "google/gemini-3.1-pro-preview",
]

# Two distinct pitches — if the model is actually listening, answers should differ.
WAV_A3 = Path("data/audio/pitchbench_a1_single_pitch_id/A3_sine_5000ms.wav")  # gt = la / A3 / 220 Hz
WAV_C2 = Path("data/audio/pitchbench_a1_single_pitch_id/C2_sine_5000ms.wav")  # gt = do / C2 / 65 Hz

SENTINEL_PROMPT = (
    "What do you hear? If you did not receive any audio, "
    "reply with exactly: NO_AUDIO_RECEIVED."
)
PITCH_PROMPT = (
    "Identify the pitch of the single note in this recording. "
    "Reply with only the note name in scientific pitch notation (e.g. C4, F#3)."
)


def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def call(model: str, wav: Path, prompt: str) -> dict:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": _b64(wav), "format": "wav"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "usage": {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vaclis-CNMAT/PitchBench",
        "X-Title": "PitchBench-debug",
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=body, headers=headers, timeout=120,
    )
    return {"status": resp.status_code, "json": resp.json()}


def summarise(out: dict) -> str:
    if out["status"] != 200:
        return f"HTTP {out['status']}: {json.dumps(out['json'])[:300]}"
    j = out["json"]
    msg = ((j.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    if isinstance(msg, list):
        msg = " ".join(p.get("text", "") for p in msg if isinstance(p, dict))
    usage = j.get("usage") or {}
    audio_tok = (usage.get("prompt_tokens_details") or {}).get("audio_tokens", "—")
    pt = usage.get("prompt_tokens", "?")
    ct = usage.get("completion_tokens", "?")
    cost = usage.get("cost", "?")
    return (
        f"  reply: {msg!r}\n"
        f"  usage: prompt={pt} (audio={audio_tok})  completion={ct}  cost=${cost}"
    )


for model in MODELS:
    print(f"\n{'=' * 72}\nMODEL: {model}\n{'=' * 72}")

    print(f"\n[1] Sentinel test on A3_sine ({WAV_A3.name}):")
    print(summarise(call(model, WAV_A3, SENTINEL_PROMPT)))

    print(f"\n[2] Pitch-ID on A3_sine (truth = A3):")
    print(summarise(call(model, WAV_A3, PITCH_PROMPT)))

    print(f"\n[3] Pitch-ID on C2_sine (truth = C2):")
    print(summarise(call(model, WAV_C2, PITCH_PROMPT)))
