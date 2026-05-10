"""FastAPI server wrapping moonshotai/Kimi-Audio-7B-Instruct.

Setup (NOT covered by ``pip install -e .[all]``):

    Kimi-Audio hard-imports flash-attn at module level (no fallback).
    flash-attn only ships prebuilt wheels for torch ≤ 2.6 + CUDA ≤ 12.4, and
    the main PitchBench env has a much newer torch, so a dedicated venv is
    required.

    The critical trick: install flash-attn from the GitHub release wheel URL
    *before* installing kimi-audio, so pip never tries to build it from source
    (which hangs for 15+ minutes and often fails)::

        uv venv --python 3.12 .venv-kimi
        source .venv-kimi/bin/activate
        pip install torch==2.6.0 torchaudio==2.6.0 \
            --index-url https://download.pytorch.org/whl/cu124
        # prebuilt wheel — instant download, no CUDA compilation:
        pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
        pip install git+https://github.com/MoonshotAI/Kimi-Audio.git
        pip install fastapi python-multipart uvicorn

    Adjust the wheel filename for your Python version (cp310/cp311/cp312).
    The CUDA tag is just ``cu12`` (covers all CUDA 12.x). If the TRUE variant
    fails with an ABI error, try ``cxx11abiFALSE``. All variants are at:
    https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1

    Use plain ``pip`` (not ``uv pip``) for the flash-attn and kimi-audio steps —
    uv's strict resolver will otherwise re-resolve flash-attn from PyPI and
    trigger a source build.

The server matches the contract used by the other PitchBench model servers:
``POST /analyze/upload`` (single audio) and ``POST /analyze/upload_multi``
(multiple audio files, for experiments like d7b) both return ``{"result": str}``.
``GET /health`` returns model info.
"""

import os
import tempfile
from typing import List

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from kimia_infer.api.kimia import KimiAudio

MODEL_ID = os.environ.get("KIMI_AUDIO_MODEL", "moonshotai/Kimi-Audio-7B-Instruct")

model = KimiAudio(model_path=MODEL_ID, load_detokenizer=False)

# Greedy text decoding by default — matches the deterministic-stimulus contract
# the rest of PitchBench relies on. Audio sampling params are unused (we only
# request output_type="text") but the API still requires them.
DEFAULT_SAMPLING = {
    "audio_temperature":            0.8,
    "audio_top_k":                  10,
    "text_temperature":             0.0,
    "text_top_k":                   5,
    "audio_repetition_penalty":     1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty":      1.0,
    "text_repetition_window_size":  16,
}

app = FastAPI()


def run_inference(prompt: str, audio_paths: list[str], max_new_tokens: int) -> str:
    messages = [{"role": "user", "message_type": "text", "content": prompt}]
    for path in audio_paths:
        messages.append({"role": "user", "message_type": "audio", "content": path})
    sampling = dict(DEFAULT_SAMPLING)
    if max_new_tokens:
        sampling["max_new_tokens"] = max_new_tokens
    _, text_output = model.generate(messages, **sampling, output_type="text")
    return text_output


class UrlRequest(BaseModel):
    prompt: str = "Describe this audio."
    audio_url: str
    max_new_tokens: int = 256


@app.post("/analyze/url")
def analyze_url(req: UrlRequest):
    result = run_inference(req.prompt, [req.audio_url], req.max_new_tokens)
    return {"result": result}


@app.post("/analyze/upload")
def analyze_upload(
    prompt: str = Form("Describe this audio."),
    max_new_tokens: int = Form(256),
    file: UploadFile = File(...),
):
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        result = run_inference(prompt, [tmp_path], max_new_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)
    return {"result": result}


@app.post("/analyze/upload_multi")
def analyze_upload_multi(
    prompt: str = Form("Describe this audio."),
    max_new_tokens: int = Form(256),
    files: List[UploadFile] = File(...),
):
    tmp_paths: list[str] = []
    try:
        for f in files:
            suffix = os.path.splitext(f.filename)[-1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(f.file.read())
                tmp_paths.append(tmp.name)
        result = run_inference(prompt, tmp_paths, max_new_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
    return {"result": result}


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      MODEL_ID,
        "checkpoint": "instruct",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8004"))
    uvicorn.run(app, host="0.0.0.0", port=port)
