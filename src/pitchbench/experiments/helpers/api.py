"""
Central ALM (Audio Language Model) facade for PitchBench.

Single public entry point: ``query_alm`` — routes the call to either
  • a local FastAPI model server (registered in ``config.MODEL_URLS``), or
  • OpenRouter chat-completions API (model_name starts with ``openrouter/``).

Every call returns a structured dict capturing the cleaned text result, the
provider's raw JSON response, and a ``model_params`` block holding everything
needed to reproduce the call (model name, endpoint, max tokens, temperature,
audio sha256, prompt). This is the contract every experiment relies on for
academic-paper reproducibility.

A convenience wrapper ``query_four_formats`` issues the same audio through the
four standard pitch-naming prompts (MIDI, SPN, Doremi, Hz) and returns four
result dicts in that order.

There is no backwards-compatibility layer. The previous helpers
(``query_model_by_name``, ``query_three_formats``, ``query_model_with_probs``,
``embed_audio``) have been removed; every script calls ``query_alm`` directly.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()                            # picks up OPENROUTER_KEY etc.
except ImportError:                          # python-dotenv not installed; fall back to env
    pass

import pitchbench.config as config

OPENROUTER_PREFIX = "openrouter/"


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
    """Return /health for local models, or a synthetic info dict for OpenRouter."""
    if model_name.startswith(OPENROUTER_PREFIX):
        return {
            "model":               model_name,
            "checkpoint":          "openrouter",
            "provider":            "openrouter",
            "openrouter_model_id": model_name[len(OPENROUTER_PREFIX):],
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

def _local_text(model_name: str, audio_path: str, prompt: str,
                max_new_tokens: int, timeout_s: float) -> dict:
    url = config.MODEL_URLS[model_name]
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{url}/analyze/upload",
            data={"prompt": prompt, "max_new_tokens": max_new_tokens},
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=timeout_s,
        )
    resp.raise_for_status()
    raw = resp.json()
    return {"result": raw.get("result", ""), "raw_response": raw,
            "top_tokens": None, "embedding": None}


def _local_probs(model_name: str, audio_path: str, prompt: str,
                 max_new_tokens: int, top_k: int, timeout_s: float) -> dict:
    url = config.MODEL_URLS[model_name]
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{url}/generate_with_probs",
            data={"prompt": prompt, "max_new_tokens": max_new_tokens, "top_k": top_k},
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=timeout_s,
        )
    resp.raise_for_status()
    raw = resp.json()
    return {"result": raw.get("result", ""), "raw_response": raw,
            "top_tokens": raw.get("top_tokens"), "embedding": None}


def _local_embed(model_name: str, audio_path: str, timeout_s: float) -> dict:
    url = config.MODEL_URLS[model_name]
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{url}/embed",
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=timeout_s,
        )
    if resp.status_code in (404, 405):
        raise NotImplementedError(
            f"{model_name} does not expose /embed (HTTP {resp.status_code})"
        )
    resp.raise_for_status()
    raw = resp.json()
    return {"result": None, "raw_response": raw,
            "top_tokens": None, "embedding": raw.get("embedding")}


# ── OpenRouter handler ────────────────────────────────────────────────────────

def _openrouter_text(model_name: str, audio_path: str, prompt: str,
                     max_new_tokens: int, temperature: float, timeout_s: float) -> dict:
    api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_KEY not set. Add it to .env or export it before calling OpenRouter."
        )

    or_model = model_name[len(OPENROUTER_PREFIX):]
    if or_model not in config.OPENROUTER_AUDIO_MODELS:
        raise ValueError(
            f"OpenRouter model {or_model!r} not in audio-capable whitelist. "
            f"Allowed: {config.OPENROUTER_AUDIO_MODELS}"
        )

    body = {
        "model": or_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {
                        "data":   _audio_b64(audio_path),
                        "format": "wav",
                    }},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens":  max_new_tokens,
        "temperature": temperature,
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
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data    = resp.json()
            choice  = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "")
            if isinstance(content, list):                # multipart content
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return {"result": content, "raw_response": data,
                    "top_tokens": None, "embedding": None}
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter call failed after 3 retries: {last_err}")


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
                        ``"audio_flamingo_next_instruct"``) or an OpenRouter slug
                        prefixed ``openrouter/`` (e.g.
                        ``"openrouter/google/gemini-2.5-flash"``).
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
            model_params  dict          everything needed to reproduce the call
            model_info    dict          /health response or synthetic OpenRouter info
            elapsed_s     float         wall-clock latency

    Raises:
        ValueError:           unknown model_name, or OpenRouter slug not in the
                              audio-capable whitelist.
        NotImplementedError:  mode='probs' or 'embed' with OpenRouter.
        SystemExit:           local server unreachable, or OPENROUTER_KEY missing.
    """
    audio_path    = str(audio_path)
    is_openrouter = model_name.startswith(OPENROUTER_PREFIX)
    info          = get_model_info(model_name)

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
        "endpoint":       "openrouter" if is_openrouter else config.MODEL_URLS.get(model_name),
    }

    start = time.time()
    if is_openrouter:
        if mode == "embed":
            raise NotImplementedError("OpenRouter does not expose audio embeddings.")
        if mode == "probs":
            raise NotImplementedError("OpenRouter does not expose top-k token probabilities.")
        result = _openrouter_text(
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

    return {
        "result":       result["result"],
        "raw_response": result["raw_response"],
        "top_tokens":   result["top_tokens"],
        "embedding":    result["embedding"],
        "model_params": model_params,
        "model_info":   info,
        "elapsed_s":    round(elapsed, 3),
    }


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
    return s_midi, s_spn, s_doremi


