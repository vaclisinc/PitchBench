"""
Smoke-test for DashScope rate limits.

Fires N requests in parallel and prints the status for each.
Usage:
    python scripts/test_dashscope_rpm.py                        # defaults
    python scripts/test_dashscope_rpm.py --model qwen3.5-omni-plus --n 70 --workers 20
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


def _make_tiny_wav() -> bytes:
    """Generate a 100 ms 440 Hz sine-wave WAV in memory."""
    import math
    rate = 16000
    duration = 0.1
    n = int(rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(32767 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _make_request(idx: int, audio_b64: str, model_id: str, api_key: str) -> dict:
    body = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": f"data:;base64,{audio_b64}", "format": "wav"},
                },
                {"type": "text", "text": "Reply with one word: yes"},
            ],
        }],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t0 = time.time()
    try:
        with requests.post(
            f"{DASHSCOPE_BASE_URL}/chat/completions",
            json=body, headers=headers, timeout=30, stream=True,
        ) as resp:
            elapsed = round(time.time() - t0, 3)
            body_preview = ""
            if not resp.ok:
                body_preview = resp.text[:200]
            return {
                "idx": idx,
                "status": resp.status_code,
                "elapsed_s": elapsed,
                "retry_after": resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-reset-requests"),
                "body_preview": body_preview,
            }
    except Exception as exc:
        return {"idx": idx, "status": -1, "elapsed_s": round(time.time() - t0, 3),
                "retry_after": None, "body_preview": str(exc)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DashScope RPM limit")
    parser.add_argument("--model", default="qwen3.5-omni-plus",
                        help="DashScope model ID (without dashscope/ prefix)")
    parser.add_argument("--n", type=int, default=70,
                        help="Total number of requests to fire")
    parser.add_argument("--workers", type=int, default=20,
                        help="Thread pool size (fire this many concurrently)")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY not set")

    wav_bytes = _make_tiny_wav()
    audio_b64 = _b64(wav_bytes)

    print(f"Firing {args.n} requests to '{args.model}' with {args.workers} workers …")
    print(f"  WAV size: {len(wav_bytes)} bytes (100 ms sine)")
    print()

    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_make_request, i, audio_b64, args.model, api_key): i
            for i in range(args.n)
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            tag = "OK " if r["status"] == 200 else f"ERR {r['status']}"
            ra  = f" [Retry-After={r['retry_after']}]" if r["retry_after"] else ""
            bp  = f" — {r['body_preview']}" if r["body_preview"] else ""
            print(f"  [{r['idx']:03d}] {tag}  {r['elapsed_s']:.3f}s{ra}{bp}")

    t_total = round(time.time() - t_start, 1)
    ok    = sum(1 for r in results if r["status"] == 200)
    r429  = sum(1 for r in results if r["status"] == 429)
    other = len(results) - ok - r429

    print()
    print(f"Done in {t_total}s — {ok} OK, {r429} rate-limited (429), {other} other errors")
    if r429:
        print(f"  → confirmed rate limit hit after ~{ok} concurrent requests")
    else:
        print(f"  → no rate limit hit with {args.n} parallel requests")


if __name__ == "__main__":
    main()
