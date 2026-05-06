"""
Central ALM (Audio Language Model) facade for PitchBench.

Single public entry point: ``query_alm`` — routes the call to either
  • a local FastAPI model server (registered in ``config.MODEL_URLS``),
  • OpenRouter chat-completions API (model_name starts with ``openrouter/``), or
  • Alibaba DashScope OpenAI-compatible chat-completions
    (model_name starts with ``dashscope/``, e.g. ``dashscope/qwen3-omni-flash``).

Every call returns a structured dict capturing the cleaned text result, the
provider's raw JSON response, and a ``model_params`` block holding everything
needed to reproduce the call (model name, endpoint, max tokens, temperature,
audio sha256, prompt). This is the contract every experiment relies on for
academic-paper reproducibility.

Convenience wrappers:
  - ``query_four_formats``: issue the same audio through MIDI / SPN / Doremi / Hz
    prompts; returns four result dicts.
  - ``query_three_formats``: legacy 3-prompt variant (MIDI / SPN / Doremi) used
    by the older pitch-ID scripts; new experiments should use the 4-format one.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import io
import json
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any, Literal

import requests

import pitchbench.config as config

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)               # .env wins over a stale/empty shell var
except ImportError:                          # python-dotenv not installed; fall back to env
    pass

import pitchbench.config as config
from pitchbench.experiments.helpers import cost as cost_tracker

OPENROUTER_PREFIX = "openrouter/"
DASHSCOPE_PREFIX  = "dashscope/"
MANUAL_MODEL_NAME = "manual"
_OPENROUTER_GPT4O_MIN_WAV_MS = 100


# ── small utilities ───────────────────────────────────────────────────────────

def _audio_sha256(audio_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _audio_b64(audio_path: str | Path) -> str:
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _audio_bytes(audio_path: str | Path) -> bytes:
    with open(audio_path, "rb") as f:
        return f.read()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_openrouter_gpt4o_model(model_name: str) -> bool:
    if not model_name.startswith(OPENROUTER_PREFIX):
        return False
    model_id = model_name[len(OPENROUTER_PREFIX):].lower()
    return model_id.startswith("openai/gpt-4o") or model_id.startswith("openai/gpt-audio")


def _pad_wav_bytes_with_silence(wav_bytes: bytes, min_duration_ms: int) -> tuple[bytes, int]:
    """Pad a WAV clip with trailing silence up to ``min_duration_ms``.

    Returns the upload bytes and the number of milliseconds of silence added.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
        frame_rate = params.framerate
        sampwidth = params.sampwidth
        n_channels = params.nchannels
        n_frames = params.nframes

    if frame_rate <= 0 or sampwidth <= 0 or n_channels <= 0:
        raise ValueError("Invalid WAV parameters for GPT-4o padding")

    min_frames = (min_duration_ms * frame_rate + 999) // 1000
    if n_frames >= min_frames:
        return wav_bytes, 0

    missing_frames = min_frames - n_frames
    silence = b"\x00" * (missing_frames * sampwidth * n_channels)

    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
        dst.writeframes(silence)

    padding_ms = int(round(missing_frames * 1000.0 / frame_rate))
    return out.getvalue(), padding_ms


def _openrouter_upload_audio(audio_path: str | Path, model_name: str) -> dict[str, Any]:
    """Return OpenRouter upload payload info for one audio file.

    GPT-4o-family OpenRouter models reject very short WAVs, so those uploads are
    padded in-memory with trailing silence only. The on-disk file is unchanged.
    """
    raw_bytes = _audio_bytes(audio_path)
    upload_bytes = raw_bytes
    padding_ms = 0
    min_duration_ms: int | None = None

    if _is_openrouter_gpt4o_model(model_name):
        min_duration_ms = _OPENROUTER_GPT4O_MIN_WAV_MS
        upload_bytes, padding_ms = _pad_wav_bytes_with_silence(raw_bytes, min_duration_ms)

    return {
        "b64": base64.b64encode(upload_bytes).decode("ascii"),
        "uploaded_sha256": _sha256_bytes(upload_bytes),
        "padding_ms": padding_ms,
        "min_duration_ms": min_duration_ms,
    }


def model_slug(info: dict | str) -> str:
    """Build a filesystem-safe model identifier.

    Accepts either a /health response dict or a model_name string.
    OpenRouter slugs become e.g. ``openrouter_google_gemini_2_5_flash``.
    """
    if isinstance(info, str):
        return re.sub(r"[^a-zA-Z0-9]+", "_", info).strip("_")
    raw        = info.get("model", "")
    short      = raw.split("/")[-1].removesuffix("-hf") if raw else ""
    checkpoint = info.get("checkpoint", "")
    parts      = [p for p in [short, checkpoint] if p]
    slug       = "_".join(parts) if parts else "unknown_model"
    return re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_")


# ── model info / health ───────────────────────────────────────────────────────

def get_model_info(model_name: str) -> dict:
    """Return /health for local models, or a synthetic info dict for OpenRouter / manual."""
    if model_name == MANUAL_MODEL_NAME:
        return {
            "model":    MANUAL_MODEL_NAME,
            "provider": "manual",
        }
    if model_name.startswith(OPENROUTER_PREFIX):
        return {
            "model":               model_name,
            "checkpoint":          "openrouter",
            "provider":            "openrouter",
            "openrouter_model_id": model_name[len(OPENROUTER_PREFIX):],
        }
    if model_name.startswith(DASHSCOPE_PREFIX):
        return {
            "model":              model_name,
            "checkpoint":         "dashscope",
            "provider":           "dashscope",
            "dashscope_model_id": model_name[len(DASHSCOPE_PREFIX):],
        }
    url = config.MODEL_URLS.get(model_name)
    if url is None or url == "openrouter":
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {list(config.MODEL_URLS)}"
        )
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"\nModel '{model_name}' not reachable at {url}\n"
            f"Start its server, or set the URL via env var (see config.MODEL_URLS).\n"
            "Or skip model queries:  python experiments/run.py <exp> --preview"
        )


# ── local server handlers ─────────────────────────────────────────────────────

# Retry transient failures (5xx, connection drops, read timeouts) — shared GPU
# servers OOM occasionally under concurrent load. Mirrors the OpenRouter path.
_LOCAL_RETRY_STATUSES = (500, 502, 503, 504)
_LOCAL_RETRY_ATTEMPTS = 3


def _post_local_with_retry(
    url: str, audio_path: str, *, data: dict | None, timeout_s: float,
    accept_404_405: bool = False,
) -> requests.Response:
    """POST a multipart audio upload to a local model server with backoff retries.

    Retries on connection errors, read timeouts, and 5xx responses. Surfaces the
    server's response body (FastAPI puts the real exception in JSON ``detail``)
    when the final attempt still fails, so logs aren't reduced to ``HTTP 500``.
    """
    last_err: str | None = None
    for attempt in range(_LOCAL_RETRY_ATTEMPTS):
        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    url,
                    data=data or {},
                    files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                    timeout=timeout_s,
                )
            if accept_404_405 and resp.status_code in (404, 405):
                return resp
            if resp.status_code in _LOCAL_RETRY_STATUSES:
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if attempt < _LOCAL_RETRY_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Local server {url} failed after retries: {last_err}")
            if not resp.ok:
                raise RuntimeError(
                    f"Local server {url} HTTP {resp.status_code}: {resp.text[:500]}"
                )
            return resp
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            last_err = str(e)
            if attempt < _LOCAL_RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Local server {url} failed after retries: {last_err}")
    # Unreachable — every branch above either returns or raises.
    raise RuntimeError(f"Local server {url} failed after retries: {last_err}")


def _local_text(model_name: str, audio_path: str, prompt: str,
                max_new_tokens: int, timeout_s: float) -> dict:
    url = config.MODEL_URLS[model_name]
    resp = _post_local_with_retry(
        f"{url}/analyze/upload", audio_path,
        data={"prompt": prompt, "max_new_tokens": max_new_tokens},
        timeout_s=timeout_s,
    )
    raw = resp.json()
    return {"result": raw.get("result", ""), "raw_response": raw,
            "top_tokens": None, "embedding": None, "usage": None}


def _local_probs(model_name: str, audio_path: str, prompt: str,
                 max_new_tokens: int, top_k: int, timeout_s: float) -> dict:
    url = config.MODEL_URLS[model_name]
    resp = _post_local_with_retry(
        f"{url}/generate_with_probs", audio_path,
        data={"prompt": prompt, "max_new_tokens": max_new_tokens, "top_k": top_k},
        timeout_s=timeout_s, accept_404_405=True,
    )
    if resp.status_code in (404, 405):
        raise NotImplementedError(
            f"{model_name} does not expose /generate_with_probs (HTTP {resp.status_code})"
        )
    raw = resp.json()
    return {"result": raw.get("result", ""), "raw_response": raw,
            "top_tokens": raw.get("top_tokens"), "embedding": None, "usage": None}


def _local_embed(model_name: str, audio_path: str, timeout_s: float) -> dict:
    url = config.MODEL_URLS[model_name]
    resp = _post_local_with_retry(
        f"{url}/embed", audio_path, data=None,
        timeout_s=timeout_s, accept_404_405=True,
    )
    if resp.status_code in (404, 405):
        raise NotImplementedError(
            f"{model_name} does not expose /embed (HTTP {resp.status_code})"
        )
    raw = resp.json()
    return {"result": None, "raw_response": raw,
            "top_tokens": None, "embedding": raw.get("embedding"), "usage": None}


# ── Manual handler ────────────────────────────────────────────────────────────

def _manual_text(audio_path: str, prompt: str) -> dict:
    """Prompt the human at the CLI to provide a response for this stimulus.

    Stand-in for an ALM call — the wav path and prompt are printed and the
    user's typed answer is returned as ``result``. Requires an interactive TTY.
    """
    if not sys.stdin.isatty():
        raise SystemExit(
            f"Model {MANUAL_MODEL_NAME!r} requires an interactive terminal "
            "(stdin is not a TTY)."
        )
    print()
    print(f"  [manual]  {Path(audio_path).name}")
    print(f"      audio :  {audio_path}")
    print(f"      prompt:  {prompt}")
    try:
        answer = input("      answer:  ").strip()
    except EOFError:
        answer = ""
    return {
        "result":       answer,
        "raw_response": {"manual": True, "answer": answer, "prompt": prompt},
        "top_tokens":   None,
        "embedding":    None,
        "usage":        None,
    }


# ── OpenRouter handler ────────────────────────────────────────────────────────

def _resolve_openrouter_key() -> str:
    """Return the OpenRouter API key, prompting the user once if missing.

    On a TTY, asks the user to paste a key (hidden via getpass) and offers to
    save it to ./.env so they aren't prompted again. On a non-TTY (CI, nohup),
    falls back to a hard SystemExit so silent runs don't hang.
    """
    api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    if not sys.stdin.isatty():
        raise SystemExit(
            "OPENROUTER_KEY not set. Add it to .env or export it before calling OpenRouter."
        )

    print("OPENROUTER_KEY not found in environment or .env.")
    print("Get one at https://openrouter.ai/keys")
    try:
        api_key = getpass.getpass("Paste your OpenRouter key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nNo key provided — aborting.")
    if not api_key:
        raise SystemExit("No key provided — aborting.")

    os.environ["OPENROUTER_KEY"] = api_key
    try:
        save = input("Save to ./.env for next time? [Y/n]: ").strip().lower()
    except EOFError:
        save = ""
    if save in ("", "y", "yes"):
        env_path = Path(".env")
        line = f"OPENROUTER_KEY={api_key}\n"
        existing = env_path.read_text() if env_path.exists() else ""
        if "OPENROUTER_KEY=" in existing:
            existing = re.sub(r"^OPENROUTER_KEY=.*$", line.rstrip(), existing, flags=re.M)
            env_path.write_text(existing if existing.endswith("\n") else existing + "\n")
        else:
            sep = "" if existing == "" or existing.endswith("\n") else "\n"
            env_path.write_text(existing + sep + line)
        print(f"Saved to {env_path.resolve()}")
    return api_key


def _openrouter_text(model_name: str, audio_path: str, prompt: str,
                     max_new_tokens: int, temperature: float, timeout_s: float) -> dict:
    api_key = _resolve_openrouter_key()

    or_model = model_name[len(OPENROUTER_PREFIX):]
    upload_audio = _openrouter_upload_audio(audio_path, model_name)
    body = {
        "model": or_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {
                        "data":   upload_audio["b64"],
                        "format": "wav",
                    }},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens":  max_new_tokens,
        "temperature": temperature,
        # Asks OpenRouter to include per-call USD cost in `usage.cost`.
        "usage":       {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/vaclis-CNMAT/PitchBench",
        "X-Title":       "PitchBench",
    }

    last_err: str | None = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                json=body, headers=headers, timeout=timeout_s,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                time.sleep(2 ** attempt)
                continue
            if not resp.ok:
                # Surface the provider's error body (e.g. "model does not
                # support audio input") instead of requests' generic message.
                raise RuntimeError(
                    f"OpenRouter HTTP {resp.status_code} for model {or_model!r}: "
                    f"{resp.text[:500]}"
                )
            data    = resp.json()
            choice  = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            if isinstance(content, list):                # multipart content
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            usage   = cost_tracker.parse_openrouter_usage(data.get("usage"))
            if usage:
                cost_tracker.record(model_name, **usage)
            return {"result": content, "raw_response": data,
                    "top_tokens": None, "embedding": None,
                    "usage": usage or None,
                    "upload_audio": {
                        "uploaded_sha256": upload_audio["uploaded_sha256"],
                        "padding_ms": upload_audio["padding_ms"],
                        "min_duration_ms": upload_audio["min_duration_ms"],
                    }}
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter call failed after 3 retries: {last_err}")


# ── DashScope handler ─────────────────────────────────────────────────────────

def _resolve_dashscope_key() -> str:
    """Return the DashScope API key, prompting on a TTY if missing."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        return api_key

    if not sys.stdin.isatty():
        raise SystemExit(
            "DASHSCOPE_API_KEY not set. Add it to .env or export it before "
            "calling DashScope."
        )

    print("DASHSCOPE_API_KEY not found in environment or .env.")
    print("Get one at https://bailian.console.alibabacloud.com/")
    try:
        api_key = getpass.getpass("Paste your DashScope key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nNo key provided — aborting.")
    if not api_key:
        raise SystemExit("No key provided — aborting.")

    os.environ["DASHSCOPE_API_KEY"] = api_key
    try:
        save = input("Save to ./.env for next time? [Y/n]: ").strip().lower()
    except EOFError:
        save = ""
    if save in ("", "y", "yes"):
        env_path = Path(".env")
        line = f'DASHSCOPE_API_KEY="{api_key}"\n'
        existing = env_path.read_text() if env_path.exists() else ""
        if "DASHSCOPE_API_KEY=" in existing:
            existing = re.sub(
                r"^DASHSCOPE_API_KEY=.*$", line.rstrip(), existing, flags=re.M,
            )
            env_path.write_text(existing if existing.endswith("\n") else existing + "\n")
        else:
            sep = "" if existing == "" or existing.endswith("\n") else "\n"
            env_path.write_text(existing + sep + line)
        print(f"Saved to {env_path.resolve()}")
    return api_key


def _audio_format_from_path(audio_path: str) -> str:
    """Return the DashScope ``format`` field derived from the file extension."""
    ext = Path(audio_path).suffix.lower().lstrip(".") or "wav"
    # DashScope accepts "wav", "mp3", "aac", "amr", "3gp", "3gpp"; map common aliases.
    return {"wave": "wav", "m4a": "aac"}.get(ext, ext)


def _dashscope_audio_block(audio_path: str) -> dict:
    """Build an OpenAI-compatible ``input_audio`` block for DashScope.

    DashScope expects the base64 wrapped as a data URI (``data:;base64,...``),
    unlike OpenRouter which takes the raw base64 string.
    """
    return {
        "type": "input_audio",
        "input_audio": {
            "data":   f"data:;base64,{_audio_b64(audio_path)}",
            "format": _audio_format_from_path(audio_path),
        },
    }


def _dashscope_stream_text(
    body: dict, headers: dict, timeout_s: float, model_id: str,
) -> tuple[str, dict, dict | None]:
    """POST a streaming chat-completion to DashScope and assemble the text.

    Qwen-Omni models *require* ``stream=true``. We stream the SSE response,
    concatenate ``choices[0].delta.content`` deltas into the final string,
    and pull ``usage`` from the last chunk (DashScope honours
    ``stream_options.include_usage``). Returns ``(text, raw_response, usage)``.
    """
    last_err: str | None = None
    for attempt in range(3):
        try:
            with requests.post(
                f"{config.DASHSCOPE_BASE_URL}/chat/completions",
                json=body, headers=headers, timeout=timeout_s, stream=True,
            ) as resp:
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    time.sleep(2 ** attempt)
                    continue
                if not resp.ok:
                    raise RuntimeError(
                        f"DashScope HTTP {resp.status_code} for model {model_id!r}: "
                        f"{resp.text[:500]}"
                    )
                pieces: list[str] = []
                usage: dict | None = None
                last_chunk: dict | None = None
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    last_chunk = chunk
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        c = delta.get("content")
                        if isinstance(c, str):
                            pieces.append(c)
                        elif isinstance(c, list):
                            for p in c:
                                if isinstance(p, dict) and isinstance(p.get("text"), str):
                                    pieces.append(p["text"])
                # Use the final SSE chunk as the canonical raw_response — it carries
                # finish_reason and (with include_usage) the usage block.
                raw = last_chunk or {}
                return "".join(pieces), raw, usage
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DashScope call failed after 3 retries: {last_err}")


def _dashscope_text(model_name: str, audio_path: str, prompt: str,
                    max_new_tokens: int, temperature: float, timeout_s: float) -> dict:
    api_key = _resolve_dashscope_key()
    ds_model = model_name[len(DASHSCOPE_PREFIX):]

    body = {
        "model": ds_model,
        "messages": [{
            "role": "user",
            "content": [
                _dashscope_audio_block(audio_path),
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens":     max_new_tokens,
        "temperature":    temperature,
        "stream":         True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    text, raw, raw_usage = _dashscope_stream_text(body, headers, timeout_s, ds_model)
    usage = cost_tracker.parse_openrouter_usage(raw_usage)  # OpenAI-shape; cost defaults to 0
    if usage:
        cost_tracker.record(model_name, **usage)
    return {"result": text, "raw_response": raw,
            "top_tokens": None, "embedding": None,
            "usage": usage or None}


def _dashscope_text_multi(
    model_name: str, audio_paths: list[str], prompt: str,
    max_new_tokens: int, temperature: float, timeout_s: float,
) -> dict:
    """Multi-audio DashScope call — interleaves audio + text blocks via <AUDIO_N> markers."""
    api_key = _resolve_dashscope_key()
    ds_model = model_name[len(DASHSCOPE_PREFIX):]

    parts = _AUDIO_PLACEHOLDER_RE.split(prompt)
    content: list[dict] = []
    if parts[0]:
        content.append({"type": "text", "text": parts[0]})
    for i in range(1, len(parts), 2):
        idx = int(parts[i]) - 1
        if idx < 0 or idx >= len(audio_paths):
            raise ValueError(
                f"Prompt references <AUDIO{idx + 1}> but only "
                f"{len(audio_paths)} audio file(s) provided."
            )
        content.append(_dashscope_audio_block(audio_paths[idx]))
        if i + 1 < len(parts) and parts[i + 1]:
            content.append({"type": "text", "text": parts[i + 1]})
    if not any(b["type"] == "input_audio" for b in content):
        raise ValueError(
            "Prompt has no <AUDIO_N> placeholders — use query_alm for "
            "single-audio queries."
        )

    body = {
        "model":          ds_model,
        "messages":       [{"role": "user", "content": content}],
        "max_tokens":     max_new_tokens,
        "temperature":    temperature,
        "stream":         True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    try:
        text, raw, raw_usage = _dashscope_stream_text(body, headers, timeout_s, ds_model)
    except RuntimeError as exc:
        # DashScope returns HTTP 400 with messages like
        #   "Multiple inputs of the same modality or mixed modality inputs are
        #    currently not applicable to the omni model."
        # for omni models that can't accept multiple audios. Convert to
        # NotImplementedError so the d7b probe / capability-skip logic catches
        # it rather than aborting the whole run.
        msg_lower = str(exc).lower()
        if "http 400" in msg_lower and (
            "not applicable" in msg_lower
            or "multiple inputs" in msg_lower
            or "mixed modality" in msg_lower
        ):
            raise NotImplementedError(
                f"{model_name} does not support multi-audio queries: "
                f"{str(exc)[:300]}"
            ) from exc
        raise
    usage = cost_tracker.parse_openrouter_usage(raw_usage)
    if usage:
        cost_tracker.record(model_name, **usage)
    return {"result": text, "raw_response": raw,
            "top_tokens": None, "embedding": None,
            "usage": usage or None}


# ── public API ────────────────────────────────────────────────────────────────

def query_alm(
    model_name: str,
    audio_path: str | Path,
    prompt: str = "",
    *,
    max_new_tokens: int = 128,
    temperature: float  = 0.0,
    top_k: int | None   = None,
    timeout_s: float    = 120.0,
    mode: Literal["text", "probs", "embed"] = "text",
) -> dict:
    """Single ALM call — central dispatcher for local servers and OpenRouter.

    Args:
        model_name:     a local slug from ``config.MODEL_URLS`` (e.g.
                        ``"audio_flamingo_next_instruct"``), an OpenRouter slug
                        prefixed ``openrouter/`` (e.g.
                        ``"openrouter/google/gemini-2.5-flash"``), or a
                        DashScope slug prefixed ``dashscope/`` (e.g.
                        ``"dashscope/qwen3-omni-flash"``).
        audio_path:     path to a WAV file.
        prompt:         the text instruction. May be empty when ``mode='embed'``.
        max_new_tokens: generation budget.
        temperature:    sampling temperature (text + OpenRouter only).
        top_k:          top-k for ``mode='probs'``; ignored otherwise.
        timeout_s:      per-call HTTP timeout.
        mode:           "text" (default), "probs" (local only), or "embed"
                        (local only).

    Returns:
        dict with keys::

            result        str | None   cleaned text response (None for embed)
            raw_response  dict          provider's raw JSON
            top_tokens    list | None   per-step top-k token probs (probs mode only)
            embedding     list | None   mean-pooled audio embedding (embed mode only)
            usage         dict | None   token counts (OpenRouter only); None for local
            cost_usd      float | None  USD cost of this single call when the
                                        provider reports it (OpenRouter); ``None``
                                        for providers that don't expose cost
                                        (DashScope, local servers, manual)
            model_params  dict          everything needed to reproduce the call
            model_info    dict          /health response or synthetic OpenRouter info
            elapsed_s     float         wall-clock latency

    Raises:
        ValueError:           unknown local model_name.
        RuntimeError:         OpenRouter rejected the request (e.g. the model
                              doesn't accept audio); the provider's error body
                              is included in the message.
        NotImplementedError:  mode='probs' or 'embed' with OpenRouter.
        SystemExit:           local server unreachable, or OPENROUTER_KEY missing.
    """
    audio_path    = str(audio_path)
    is_manual     = model_name == MANUAL_MODEL_NAME
    is_openrouter = model_name.startswith(OPENROUTER_PREFIX)
    is_dashscope  = model_name.startswith(DASHSCOPE_PREFIX)
    info          = get_model_info(model_name)

    if is_manual:
        endpoint = "manual"
    elif is_openrouter:
        endpoint = "openrouter"
    elif is_dashscope:
        endpoint = config.DASHSCOPE_BASE_URL
    else:
        endpoint = config.MODEL_URLS.get(model_name)

    model_params: dict[str, Any] = {
        "model_name":     model_name,
        "mode":           mode,
        "max_new_tokens": max_new_tokens,
        "temperature":    temperature,
        "top_k":          top_k,
        "timeout_s":      timeout_s,
        "audio_file":     os.path.basename(audio_path),
        "audio_sha256":   _audio_sha256(audio_path),
        "prompt":         prompt,
        "endpoint":       endpoint,
    }
    if is_openrouter:
        upload_audio = _openrouter_upload_audio(audio_path, model_name)
        model_params["upload_audio_sha256"] = upload_audio["uploaded_sha256"]
        model_params["upload_audio_padding_ms"] = upload_audio["padding_ms"]
        if upload_audio["min_duration_ms"] is not None:
            model_params["upload_audio_min_duration_ms"] = upload_audio["min_duration_ms"]

    start = time.time()
    if is_manual:
        if mode != "text":
            raise NotImplementedError(
                f"Manual mode supports mode='text' only, got mode={mode!r}."
            )
        result = _manual_text(audio_path, prompt)
    elif is_openrouter:
        if mode == "embed":
            raise NotImplementedError("OpenRouter does not expose audio embeddings.")
        if mode == "probs":
            raise NotImplementedError("OpenRouter does not expose top-k token probabilities.")
        result = _openrouter_text(
            model_name, audio_path, prompt, max_new_tokens, temperature, timeout_s,
        )
    elif is_dashscope:
        if mode == "embed":
            raise NotImplementedError("DashScope does not expose audio embeddings.")
        if mode == "probs":
            raise NotImplementedError("DashScope does not expose top-k token probabilities.")
        result = _dashscope_text(
            model_name, audio_path, prompt, max_new_tokens, temperature, timeout_s,
        )
    else:
        if mode == "text":
            result = _local_text(model_name, audio_path, prompt, max_new_tokens, timeout_s)
        elif mode == "probs":
            if top_k is None:
                raise ValueError("mode='probs' requires top_k")
            result = _local_probs(
                model_name, audio_path, prompt, max_new_tokens, top_k, timeout_s,
            )
        elif mode == "embed":
            result = _local_embed(model_name, audio_path, timeout_s)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    elapsed = time.time() - start

    usage    = result.get("usage")
    raw_cost = (usage or {}).get("cost_usd")
    return {
        "result":       result["result"],
        "raw_response": result["raw_response"],
        "top_tokens":   result["top_tokens"],
        "embedding":    result["embedding"],
        "usage":        usage,
        "cost_usd":     None if raw_cost is None else float(raw_cost),
        "model_params": model_params,
        "model_info":   info,
        "elapsed_s":    round(elapsed, 3),
    }


def _print_running_cost(model_name: str) -> None:
    """Emit a one-line running-total for the current model (skipped for local servers).

    Hidden behind the ``PITCHBENCH_LIVE_COST=0`` env var for users who find the
    extra line noisy.
    """
    if os.environ.get("PITCHBENCH_LIVE_COST", "1") == "0":
        return
    u = cost_tracker.get(model_name)
    if not u["calls"]:
        return
    cost_part = f", ${u['cost_usd']:.4f}" if u.get("cost_reported") else ""
    print(f"        cost   → {u['calls']:,} calls, "
          f"{u['total_tokens']:,} tok{cost_part}")


_AUDIO_PLACEHOLDER_RE = re.compile(r"<AUDIO(\d+)>")


def _multi_audio_sha256(audio_paths: list[str]) -> list[str]:
    return [_audio_sha256(p) for p in audio_paths]


def _post_local_multi_with_retry(
    url: str, audio_paths: list[str], *, data: dict, timeout_s: float,
    accept_404_405_501: bool = False,
) -> requests.Response:
    """Multi-file variant of :func:`_post_local_with_retry`.

    POSTs ``files=[(name, fh, ct), ...]`` so each upload arrives as its own
    ``files`` form part. Mirrors the single-file retry / 5xx surfacing logic.

    When ``accept_404_405_501`` is set, those status codes return without
    raising so the caller can re-raise as :class:`NotImplementedError` —
    used to signal "this server's processor cannot handle multi-audio".
    """
    last_err: str | None = None
    for attempt in range(_LOCAL_RETRY_ATTEMPTS):
        opened: list = []
        try:
            files_payload = []
            for p in audio_paths:
                fh = open(p, "rb")
                opened.append(fh)
                files_payload.append(("files", (os.path.basename(p), fh, "audio/wav")))
            resp = requests.post(url, data=data, files=files_payload, timeout=timeout_s)
            if accept_404_405_501 and resp.status_code in (404, 405, 501):
                return resp
            if resp.status_code in _LOCAL_RETRY_STATUSES:
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if attempt < _LOCAL_RETRY_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Local server {url} failed after retries: {last_err}")
            if not resp.ok:
                raise RuntimeError(
                    f"Local server {url} HTTP {resp.status_code}: {resp.text[:500]}"
                )
            return resp
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            last_err = str(e)
            if attempt < _LOCAL_RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Local server {url} failed after retries: {last_err}")
        finally:
            for fh in opened:
                try:
                    fh.close()
                except OSError:
                    pass
    raise RuntimeError(f"Local server {url} failed after retries: {last_err}")


def _local_text_multi(
    model_name: str, audio_paths: list[str], prompt: str,
    max_new_tokens: int, timeout_s: float,
) -> dict:
    url = config.MODEL_URLS[model_name]
    try:
        resp = _post_local_multi_with_retry(
            f"{url}/analyze/upload_multi", audio_paths,
            data={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout_s=timeout_s, accept_404_405_501=True,
        )
    except RuntimeError as exc:
        # Fallback: if the local server is running an older build that doesn't
        # have the 501 short-circuit (e.g. AFN before /analyze/upload_multi
        # was added), the multi-audio call goes all the way into the processor
        # and surfaces as a 500 carrying the AFN-specific 1:1 alignment error.
        # Treat that as a capability signal too, so the probe skips cleanly
        # without the user needing to restart the server first.
        msg_lower = str(exc).lower()
        if "must match 1:1" in msg_lower or "they must match" in msg_lower:
            raise NotImplementedError(
                f"{model_name} does not support multi-audio queries "
                f"(processor enforces text:audio = 1:1): {str(exc)[:300]}"
            ) from exc
        raise
    if resp.status_code in (404, 405, 501):
        try:
            detail = resp.json().get("detail", resp.text[:200])
        except ValueError:
            detail = resp.text[:200]
        raise NotImplementedError(
            f"{model_name} does not support multi-audio queries "
            f"(HTTP {resp.status_code}): {detail}"
        )
    raw = resp.json()
    return {"result": raw.get("result", ""), "raw_response": raw,
            "top_tokens": None, "embedding": None, "usage": None}


def _openrouter_text_multi(
    model_name: str, audio_paths: list[str], prompt: str,
    max_new_tokens: int, temperature: float, timeout_s: float,
) -> dict:
    """OpenRouter variant with multiple ``input_audio`` blocks.

    Splits ``prompt`` on ``<AUDIO_N>`` placeholders and interleaves text /
    audio content blocks just like the local server does.
    """
    api_key = _resolve_openrouter_key()
    or_model = model_name[len(OPENROUTER_PREFIX):]
    upload_audios = [_openrouter_upload_audio(p, model_name) for p in audio_paths]

    parts = _AUDIO_PLACEHOLDER_RE.split(prompt)
    content: list[dict] = []
    if parts[0]:
        content.append({"type": "text", "text": parts[0]})
    for i in range(1, len(parts), 2):
        idx = int(parts[i]) - 1
        if idx < 0 or idx >= len(audio_paths):
            raise ValueError(
                f"Prompt references <AUDIO{idx + 1}> but only "
                f"{len(audio_paths)} audio file(s) provided."
            )
        content.append({"type": "input_audio", "input_audio": {
            "data":   upload_audios[idx]["b64"],
            "format": "wav",
        }})
        if i + 1 < len(parts) and parts[i + 1]:
            content.append({"type": "text", "text": parts[i + 1]})
    if not any(b["type"] == "input_audio" for b in content):
        raise ValueError(
            "Prompt has no <AUDIO_N> placeholders — use query_alm for "
            "single-audio queries."
        )

    body = {
        "model":       or_model,
        "messages":    [{"role": "user", "content": content}],
        "max_tokens":  max_new_tokens,
        "temperature": temperature,
        "usage":       {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/vaclis-CNMAT/PitchBench",
        "X-Title":       "PitchBench",
    }

    last_err: str | None = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                json=body, headers=headers, timeout=timeout_s,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                time.sleep(2 ** attempt)
                continue
            if not resp.ok:
                raise RuntimeError(
                    f"OpenRouter HTTP {resp.status_code} for model {or_model!r}: "
                    f"{resp.text[:500]}"
                )
            data    = resp.json()
            choice  = (data.get("choices") or [{}])[0]
            text    = (choice.get("message") or {}).get("content", "")
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            usage   = cost_tracker.parse_openrouter_usage(data.get("usage"))
            if usage:
                cost_tracker.record(model_name, **usage)
            return {"result": text, "raw_response": data,
                    "top_tokens": None, "embedding": None,
                    "usage": usage or None,
                    "upload_audio": [
                        {
                            "uploaded_sha256": item["uploaded_sha256"],
                            "padding_ms": item["padding_ms"],
                            "min_duration_ms": item["min_duration_ms"],
                        }
                        for item in upload_audios
                    ]}
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter call failed after 3 retries: {last_err}")


def query_alm_multi(
    model_name: str,
    audio_paths: list[str | Path],
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float  = 0.0,
    timeout_s: float    = 180.0,
) -> dict:
    """Multi-audio ALM call.

    The ``prompt`` must contain ``<AUDIO1>``, ``<AUDIO2>``, ... placeholders
    that mark where each audio in ``audio_paths`` should appear in the model's
    input (1-based, so ``<AUDIO1>`` = ``audio_paths[0]``).

    Routes to ``/analyze/upload_multi`` for local servers and to OpenRouter
    chat-completions with multiple ``input_audio`` blocks. Manual mode and
    ``mode='probs'`` / ``'embed'`` are not supported and raise
    ``NotImplementedError``.

    Returns the same dict shape as :func:`query_alm`.
    """
    paths_str: list[str] = [str(p) for p in audio_paths]
    if not paths_str:
        raise ValueError("query_alm_multi requires at least one audio path")

    is_manual     = model_name == MANUAL_MODEL_NAME
    is_openrouter = model_name.startswith(OPENROUTER_PREFIX)
    is_dashscope  = model_name.startswith(DASHSCOPE_PREFIX)
    info          = get_model_info(model_name)

    if is_manual:
        raise NotImplementedError(
            "Manual mode does not support multi-audio queries."
        )

    if is_openrouter:
        endpoint = "openrouter"
    elif is_dashscope:
        endpoint = config.DASHSCOPE_BASE_URL
    else:
        endpoint = config.MODEL_URLS.get(model_name)

    model_params: dict[str, Any] = {
        "model_name":      model_name,
        "mode":            "text",
        "max_new_tokens":  max_new_tokens,
        "temperature":     temperature,
        "top_k":           None,
        "timeout_s":       timeout_s,
        "audio_files":     [os.path.basename(p) for p in paths_str],
        "audio_sha256":    _multi_audio_sha256(paths_str),
        "prompt":          prompt,
        "endpoint":        endpoint,
        "multi_audio":     True,
        "n_audio":         len(paths_str),
    }
    if is_openrouter:
        upload_audio = [_openrouter_upload_audio(p, model_name) for p in paths_str]
        model_params["upload_audio_sha256"] = [a["uploaded_sha256"] for a in upload_audio]
        model_params["upload_audio_padding_ms"] = [a["padding_ms"] for a in upload_audio]
        model_params["upload_audio_min_duration_ms"] = [a["min_duration_ms"] for a in upload_audio]

    start = time.time()
    if is_openrouter:
        result = _openrouter_text_multi(
            model_name, paths_str, prompt, max_new_tokens, temperature, timeout_s,
        )
    elif is_dashscope:
        result = _dashscope_text_multi(
            model_name, paths_str, prompt, max_new_tokens, temperature, timeout_s,
        )
    else:
        result = _local_text_multi(
            model_name, paths_str, prompt, max_new_tokens, timeout_s,
        )
    elapsed = time.time() - start

    usage    = result.get("usage")
    raw_cost = (usage or {}).get("cost_usd")
    return {
        "result":       result["result"],
        "raw_response": result["raw_response"],
        "top_tokens":   None,
        "embedding":    None,
        "usage":        usage,
        "cost_usd":     None if raw_cost is None else float(raw_cost),
        "model_params": model_params,
        "model_info":   info,
        "elapsed_s":    round(elapsed, 3),
    }


def query_four_formats_multi(
    model_name: str,
    audio_paths: list[str | Path],
    prompt_midi:   str,
    prompt_spn:    str,
    prompt_doremi: str,
    prompt_hz:     str,
    *,
    verbose: bool = True,
) -> tuple[dict, dict, dict, dict]:
    """Multi-audio variant of :func:`query_four_formats`.

    Each prompt must reference the audios via ``<AUDIO1>``, ``<AUDIO2>``, ...
    placeholders. Returns four ``query_alm_multi`` result dicts in order:
    (MIDI, SPN, Doremi, Hz).
    """

    empty_result = {"result": None, "raw_response": "", "top_tokens": None, "embedding": None, "usage": 0}

    r_midi   = query_alm_multi(model_name, audio_paths, prompt_midi) if "midi" in config.pitchbench_general_NOTATION_FORMATS else empty_result
    r_spn    = query_alm_multi(model_name, audio_paths, prompt_spn) if "spn" in config.pitchbench_general_NOTATION_FORMATS else empty_result
    r_doremi = query_alm_multi(model_name, audio_paths, prompt_doremi) if "doremi" in config.pitchbench_general_NOTATION_FORMATS else empty_result
    r_hz     = query_alm_multi(model_name, audio_paths, prompt_hz) if "hz" in config.pitchbench_general_NOTATION_FORMATS else empty_result
    if verbose:
        names = ", ".join(Path(str(p)).name for p in audio_paths)
        print(f"      [{names}]")
        print(f"        MIDI   → {(r_midi['result']   or '').strip()!r}")
        print(f"        SPN    → {(r_spn['result']    or '').strip()!r}")
        print(f"        doremi → {(r_doremi['result'] or '').strip()!r}")
        print(f"        Hz     → {(r_hz['result']     or '').strip()!r}")
        _print_running_cost(model_name)
    return r_midi, r_spn, r_doremi, r_hz


def query_four_formats(
    model_name: str,
    audio_path: str | Path,
    prompt_midi:   str,
    prompt_spn:    str,
    prompt_doremi: str,
    prompt_hz:     str,
    *,
    verbose: bool = True,
) -> tuple[dict, dict, dict, dict]:
    """Query the model with the four standard pitch-naming formats.

    Returns four ``query_alm`` result dicts in order: (MIDI, SPN, Doremi, Hz).
    """
    r_midi   = query_alm(model_name, audio_path, prompt_midi)
    r_spn    = query_alm(model_name, audio_path, prompt_spn)
    r_doremi = query_alm(model_name, audio_path, prompt_doremi)
    r_hz     = query_alm(model_name, audio_path, prompt_hz)
    if verbose:
        fname = Path(str(audio_path)).name
        print(f"      [{fname}]")
        print(f"        MIDI   → {(r_midi['result']   or '').strip()!r}")
        print(f"        SPN    → {(r_spn['result']    or '').strip()!r}")
        print(f"        doremi → {(r_doremi['result'] or '').strip()!r}")
        print(f"        Hz     → {(r_hz['result']     or '').strip()!r}")
        _print_running_cost(model_name)
    return r_midi, r_spn, r_doremi, r_hz


def query_three_formats(
    model_name: str,
    audio_path: str | Path,
    prompt_midi:   str,
    prompt_spn:    str,
    prompt_doremi: str,
    *,
    verbose: bool = True,
) -> tuple[str, str, str]:
    """Three-prompt convenience for legacy pitch-ID experiments.

    Returns plain *string* responses (not the full dict) so the call shape
    matches the original v1 contract. New experiments should prefer
    :func:`query_four_formats` to also collect the Hz response.
    """
    s_midi   = query_alm(model_name, audio_path, prompt_midi)["result"] or ""
    s_spn    = query_alm(model_name, audio_path, prompt_spn)["result"] or ""
    s_doremi = query_alm(model_name, audio_path, prompt_doremi)["result"] or ""
    if verbose:
        fname = Path(str(audio_path)).name
        print(f"      [{fname}]")
        print(f"        MIDI   → {s_midi.strip()!r}")
        print(f"        SPN    → {s_spn.strip()!r}")
        print(f"        doremi → {s_doremi.strip()!r}")
        _print_running_cost(model_name)
    return s_midi, s_spn, s_doremi


