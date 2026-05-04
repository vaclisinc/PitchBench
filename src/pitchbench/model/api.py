import os
import sys
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
import pitchbench.config as config

app = FastAPI()

model_id = config.MODEL_MUSIC_FLAMINGO
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()



def run_inference(prompt: str, audio_path: str, max_new_tokens: int) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = {
        k: v.to(device=model.device, dtype=model.dtype) if v.dtype.is_floating_point else v.to(model.device)
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return decoded[0]


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
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        result = run_inference(prompt, tmp_path, max_new_tokens)
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)
    return {"result": result}


@app.post("/generate_with_probs")
def generate_with_probs(
    prompt: str = Form("Describe this audio."),
    max_new_tokens: int = Form(32),
    top_k: int = Form(50),
    file: UploadFile = File(...),
):
    """Generate a response and return the top-k token probabilities for each generated step."""
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": tmp_path},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        )
        inputs = {
            k: v.to(device=model.device, dtype=model.dtype) if v.dtype.is_floating_point else v.to(model.device)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        import torch.nn.functional as F
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        top_tokens_per_step = []
        for step_scores in outputs.scores:
            probs = F.softmax(step_scores[0], dim=-1)
            topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
            top_tokens_per_step.append([
                {
                    "token": processor.decode([tid.item()]),
                    "token_id": tid.item(),
                    "prob": p.item(),
                }
                for tid, p in zip(topk.indices, topk.values)
            ])
        return {"result": decoded, "top_tokens": top_tokens_per_step}
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)


@app.post("/embed")
def embed(
    file: UploadFile = File(...),
):
    """Return the mean-pooled audio encoder embedding for the uploaded file."""
    from fastapi import HTTPException
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        inputs = processor(
            audio=tmp_path,
            text="Describe this audio.",
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=model.device, dtype=model.dtype) if v.dtype.is_floating_point else v.to(model.device)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        vec = out.hidden_states[-1][0].mean(dim=0).float().cpu()
        return {"embedding": vec.tolist(), "dim": vec.shape[0]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok", "model": model_id, "device": str(next(model.parameters()).device)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
