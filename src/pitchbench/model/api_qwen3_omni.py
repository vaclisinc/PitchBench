import os
import tempfile

import torch
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_ID = os.environ.get("QWEN3_OMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "/tmp/qwen3_omni_offload")

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
    offload_folder=OFFLOAD_DIR,
).eval()

app = FastAPI()


def run_inference(prompt: str, audio_path: str, max_new_tokens: int) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_audio=False,
        )

    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        result[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


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


@app.post("/generate_with_probs")
def generate_with_probs(
    prompt: str = Form("Describe this audio."),
    max_new_tokens: int = Form(32),
    top_k: int = Form(50),
    file: UploadFile = File(...),
):
    """Generate a response and return the top-k token probabilities for each generated step."""
    import torch.nn.functional as F
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": tmp_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text, audio=audios, images=images or None,
            videos=videos or None, return_tensors="pt", padding=True,
        ).to(model.device).to(model.dtype)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_audio=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(
            outputs.sequences[:, input_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "checkpoint": "instruct",
        "device": str(next(model.parameters()).device),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
