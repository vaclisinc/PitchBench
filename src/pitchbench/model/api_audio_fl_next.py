"""Local AF-Next Instruct endpoint used by the canonical evaluator.

Run with the optional serving environment (torch, transformers, fastapi,
python-multipart, uvicorn). Set AF_NEXT_MODEL_PATH to a pinned HF snapshot.
"""
from __future__ import annotations

import io
import os
import threading

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from transformers import AutoModel, AutoProcessor

MODEL_PATH = os.environ["AF_NEXT_MODEL_PATH"]
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
).eval()
lock = threading.Lock()
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model": "nvidia/audio-flamingo-next-hf",
            "checkpoint": "audio_flamingo_next_instruct", "model_path": MODEL_PATH,
            "device": str(model.device), "dtype": str(model.dtype)}


@app.post("/analyze/upload")
def analyze(file: UploadFile = File(...), prompt: str = Form(""),
            max_new_tokens: int = Form(128)):
    audio, sr = sf.read(io.BytesIO(file.file.read()), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        divisor = np.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor)
    conversation = [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "audio", "audio": audio},
    ]}]
    with lock, torch.inference_mode():
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    return {"result": response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("AF_NEXT_PORT", "8017")))
