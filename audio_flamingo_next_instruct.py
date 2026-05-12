"""
FastAPI server for Audio Flamingo Next Instruct (nvidia/audio-flamingo-next-hf).

Endpoints
---------
GET  /health                    — model info + device
POST /analyze/upload            — single-audio text generation
POST /analyze/url               — single-audio text generation from a URL path
POST /generate_with_probs       — generation with per-step top-k token probabilities
POST /embed                     — mean-pooled audio encoder embedding
POST /analyze/upload_multi      — always 501 (model does not support multi-audio)

Usage
-----
    python audio_flamingo_next_instruct.py

"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ── FastAPI / Pydantic ────────────────────────────────────────────────────────
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# ── Model (loaded lazily at server start, not at import time) ─────────────────
_processor = None
_model     = None
_model_id  = None


def _get_model():
    """Return (processor, model), loading on first call."""
    global _processor, _model
    if _model is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        mid = _model_id or os.environ.get(
            "PITCHBENCH_MODEL_ID", "nvidia/audio-flamingo-next-hf"
        )
        print(f"[server] Loading model: {mid}", flush=True)
        _processor = AutoProcessor.from_pretrained(mid)
        _model = AutoModel.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        print(f"[server] Model ready on {next(_model.parameters()).device}", flush=True)
    return _processor, _model


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Audio Flamingo Next Instruct", version="1.0")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tmp_save(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(upload.file.read())
        return f.name


def _run_inference(prompt: str, audio_path: str, max_new_tokens: int) -> str:
    import torch

    processor, model = _get_model()
    conversation = [[{
        "role": "user",
        "content": [
            {"type": "text",  "text":  prompt},
            {"type": "audio", "path":  audio_path},
        ],
    }]]
    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(model.dtype)

    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
        )

    prompt_len = batch["input_ids"].shape[1]
    return processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    processor, model = _get_model()
    mid = _model_id or os.environ.get("PITCHBENCH_MODEL_ID", "nvidia/audio-flamingo-next-hf")
    short = mid.split("/")[-1].removesuffix("-hf")
    return {
        "status":     "ok",
        "model":      mid,
        "checkpoint": short,
        "device":     str(next(model.parameters()).device),
    }


class UrlRequest(BaseModel):
    prompt:         str = "Describe this audio."
    audio_url:      str
    max_new_tokens: int = 256


@app.post("/analyze/url")
def analyze_url(req: UrlRequest):
    try:
        result = _run_inference(req.prompt, req.audio_url, req.max_new_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"result": result}


@app.post("/analyze/upload")
def analyze_upload(
    prompt:         str       = Form("Describe this audio."),
    max_new_tokens: int       = Form(256),
    file:           UploadFile = File(...),
):
    tmp_path = _tmp_save(file)
    try:
        result = _run_inference(prompt, tmp_path, max_new_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)
    return {"result": result}


@app.post("/analyze/upload_multi")
def analyze_upload_multi(
    prompt:         str              = Form(...),
    max_new_tokens: int              = Form(256),
    files:          list[UploadFile] = File(...),
):
    """Multi-audio queries are not supported by Audio Flamingo Next Instruct.

    The model's processor enforces a strict 1:1 text-to-audio alignment.
    Use ``/analyze/upload`` for single-audio queries, or route multi-audio
    experiments to a model that supports them (Qwen-Omni, Gemini, GPT-4o-audio).
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "audio_flamingo_next_instruct does not support multi-audio queries "
            "(processor enforces text:audio = 1:1). "
            "Use /analyze/upload for single-audio queries."
        ),
    )


@app.post("/generate_with_probs")
def generate_with_probs(
    prompt:         str       = Form("Describe this audio."),
    max_new_tokens: int       = Form(32),
    top_k:          int       = Form(50),
    file:           UploadFile = File(...),
):
    """Generate text and return per-step top-k token probabilities."""
    import torch
    import torch.nn.functional as F

    processor, model = _get_model()
    tmp_path = _tmp_save(file)
    try:
        conversation = [[{
            "role": "user",
            "content": [
                {"type": "text",  "text":  prompt},
                {"type": "audio", "path":  tmp_path},
            ],
        }]]
        batch = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(model.device)
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(model.dtype)

        with torch.inference_mode():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.2,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[:, batch["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        top_tokens_per_step = []
        for step_scores in outputs.scores:
            probs = F.softmax(step_scores[0], dim=-1)
            topk  = torch.topk(probs, k=min(top_k, probs.shape[-1]))
            top_tokens_per_step.append([
                {
                    "token":    processor.decode([tid.item()]),
                    "token_id": tid.item(),
                    "prob":     p.item(),
                }
                for tid, p in zip(topk.indices, topk.values)
            ])
        return {"result": decoded, "top_tokens": top_tokens_per_step}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)


@app.post("/embed")
def embed(file: UploadFile = File(...)):
    """Return the mean-pooled audio encoder embedding."""
    import torch

    processor, model = _get_model()
    tmp_path = _tmp_save(file)
    try:
        conversation = [[{
            "role": "user",
            "content": [
                {"type": "text",  "text":  "Describe this audio."},
                {"type": "audio", "path":  tmp_path},
            ],
        }]]
        batch = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(model.device)
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(model.dtype)

        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True)

        vec = out.encoder_last_hidden_state[0].mean(dim=0).float().cpu()
        return {"embedding": vec.tolist(), "dim": vec.shape[0]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _model_id

    parser = argparse.ArgumentParser(
        description="Serve Audio Flamingo Next Instruct via FastAPI.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("PITCHBENCH_MODEL_ID", "nvidia/audio-flamingo-next-hf"),
        help="HuggingFace model ID (default: nvidia/audio-flamingo-next-hf)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")),
                        help="Bind port (default: 8001)")
    args = parser.parse_args()

    _model_id = args.model_id

    # Eagerly load the model before accepting requests.
    _get_model()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
