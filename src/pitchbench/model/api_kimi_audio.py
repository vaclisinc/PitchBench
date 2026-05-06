"""FastAPI server wrapping moonshotai/Kimi-Audio-7B-Instruct.

Uses the official `kimia_infer` package:

    pip install git+https://github.com/MoonshotAI/Kimi-Audio.git

The server matches the contract used by the other PitchBench model servers:
``POST /analyze/upload`` (multipart audio + prompt) returns ``{"result": str}``,
and ``GET /health`` returns the model id.
"""

import os
import tempfile

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from kimia_infer.api.kimia import KimiAudio

MODEL_ID = os.environ.get("KIMI_AUDIO_MODEL", "moonshotai/Kimi-Audio-7B-Instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = KimiAudio(model_path=MODEL_ID, load_detokenizer=True)
model.to(device)

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


def run_inference(prompt: str, audio_path: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "user", "message_type": "text",  "content": prompt},
        {"role": "user", "message_type": "audio", "content": audio_path},
    ]
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
    result = run_inference(req.prompt, req.audio_url, req.max_new_tokens)
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
        result = run_inference(prompt, tmp_path, max_new_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)
    return {"result": result}


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      MODEL_ID,
        "checkpoint": "instruct",
        "device":     device,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8004"))
    uvicorn.run(app, host="0.0.0.0", port=port)
