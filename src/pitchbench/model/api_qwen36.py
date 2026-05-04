# Qwen3.6 — open weights (Qwen/Qwen3.6-27B or Qwen/Qwen3.6-35B-A3B).
# WARNING: Qwen3.6 is text + vision only. It has NO audio encoder.
# Sending audio to this model will not produce meaningful pitch predictions.
# It is included here only so you can confirm that baseline with data.
#
# If you want an open-weight Qwen model that actually handles audio,
# use api_qwen3_omni.py (Qwen/Qwen3-Omni-30B-A3B-Instruct) instead.
#
# Set QWEN36_MODEL to override (default: Qwen/Qwen3.6-27B).

import os
import tempfile

import torch
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("QWEN36_MODEL", "Qwen/Qwen3.6-27B")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

app = FastAPI()


def run_inference(prompt: str, audio_path: str, max_new_tokens: int) -> str:
    # Qwen3.6 has no audio tower — the audio file is silently ignored.
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            thinking_budget=0,
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)


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
    from fastapi import HTTPException
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
        "status": "ok",
        "model": MODEL_ID,
        "checkpoint": "default",
        "audio_tower": False,
        "device": str(next(model.parameters()).device),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
