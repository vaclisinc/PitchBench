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
import os
import re
import sys
import time
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
MANUAL_MODEL_NAME = "manual"


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
            content = (choice.get("message") or {}).get("content", "")
            if isinstance(content, list):                # multipart content
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            usage   = cost_tracker.parse_openrouter_usage(data.get("usage"))
            if usage:
                cost_tracker.record(model_name, **usage)
            return {"result": content, "raw_response": data,
                    "top_tokens": None, "embedding": None,
                    "usage": usage or None}
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
            usage         dict | None   token counts (OpenRouter only); None for local
            cost_usd      float         USD cost of this single call (OpenRouter only; 0.0 otherwise)
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
    info          = get_model_info(model_name)

    if is_manual:
        endpoint = "manual"
    elif is_openrouter:
        endpoint = "openrouter"
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

    usage = result.get("usage")
    return {
        "result":       result["result"],
        "raw_response": result["raw_response"],
        "top_tokens":   result["top_tokens"],
        "embedding":    result["embedding"],
        "usage":        usage,
        "cost_usd":     float((usage or {}).get("cost_usd", 0.0)),
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
    print(f"        cost   → {u['calls']:,} calls, "
          f"{u['total_tokens']:,} tok, ${u['cost_usd']:.4f}")


_AUDIO_PLACEHOLDER_RE = re.compile(r"<AUDIO(\d+)>")


def _multi_audio_sha256(audio_paths: list[str]) -> list[str]:
    return [_audio_sha256(p) for p in audio_paths]


def _post_local_multi_with_retry(
    url: str, audio_paths: list[str], *, data: dict, timeout_s: float,
) -> requests.Response:
    """Multi-file variant of :func:`_post_local_with_retry`.

    POSTs ``files=[(name, fh, ct), ...]`` so each upload arrives as its own
    ``files`` form part. Mirrors the single-file retry / 5xx surfacing logic.
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
    resp = _post_local_multi_with_retry(
        f"{url}/analyze/upload_multi", audio_paths,
        data={"prompt": prompt, "max_new_tokens": max_new_tokens},
        timeout_s=timeout_s,
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
            "data":   _audio_b64(audio_paths[idx]),
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
                    "usage": usage or None}
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
    info          = get_model_info(model_name)

    if is_manual:
        raise NotImplementedError(
            "Manual mode does not support multi-audio queries."
        )

    if is_openrouter:
        endpoint = "openrouter"
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

    start = time.time()
    if is_openrouter:
        result = _openrouter_text_multi(
            model_name, paths_str, prompt, max_new_tokens, temperature, timeout_s,
        )
    else:
        result = _local_text_multi(
            model_name, paths_str, prompt, max_new_tokens, timeout_s,
        )
    elapsed = time.time() - start

    usage = result.get("usage")
    return {
        "result":       result["result"],
        "raw_response": result["raw_response"],
        "top_tokens":   None,
        "embedding":    None,
        "usage":        usage,
        "cost_usd":     float((usage or {}).get("cost_usd", 0.0)),
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


