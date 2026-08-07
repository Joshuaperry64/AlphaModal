"""
AlphaCore Modal Proxy — WebUI A1111-compatible API bridge
-------------------------------------------------------
Runs on localhost:7860 and translates WebUI-style /sdapi/v1/txt2img 
and /sdapi/v1/img2img requests to your Modal GPU endpoints.

Point HammerAI at: http://localhost:7860

Install deps:
    pip install fastapi uvicorn requests httpx pillow python-multipart

Run:
    python modal_proxy.py
"""

import base64
import json
import re
import time
from io import BytesIO

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TXT2IMG_URL  = "https://josh627764--text-to-image-sdxl-merger-inference-web.modal.run"
IMG2IMG_URL  = "https://josh627764--img2img-qwen-edit-plus-model-web.modal.run/stream"
PROXY_HOST   = "0.0.0.0"
PROXY_PORT   = 7860
REQUEST_TIMEOUT = 600   # seconds — Modal cold starts can be slow

# ─── STATE ───────────────────────────────────────────────────────────────────
CURRENT_MODEL = "Unholy [Modal]"

# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="AlphaCore Modal Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODELS ──────────────────────────────────────────────────────────────────
class Txt2ImgRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = "deformed, mutated, bad anatomy, bad proportions, disfigured, extra limbs, missing limbs, worst quality, low quality, blurry, distorted"
    steps: int = Field(default=20, alias="steps")
    cfg_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    n_iter: int = 1
    seed: int = -1
    sampler_name: str = "Euler"
    # extras ignored by Modal but accepted so HammerAI doesn't error
    restore_faces: bool = False
    tiling: bool = False
    enable_hr: bool = False
    enhance_prompt: bool = True
    override_settings: dict = None

    class Config:
        populate_by_name = True
        extra = "allow"


class Img2ImgRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = "low quality, blurry, distorted"
    init_images: list[str] = []   # base64-encoded
    steps: int = 20
    cfg_scale: float = 7.0
    denoising_strength: float = 0.75
    batch_size: int = 1
    n_iter: int = 1
    seed: int = -1
    sampler_name: str = "Euler"
    width: int = 1024
    height: int = 1024

    class Config:
        populate_by_name = True


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def decode_sse_images(sse_text: str) -> list[str]:
    """Parse SSE stream text and extract base64 image(s)."""
    images = []
    for line in sse_text.splitlines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if "image_b64" in data:
                    b64s = data["image_b64"]
                    if isinstance(b64s, list):
                        images.extend(b64s)
                    else:
                        images.append(b64s)
                elif "error" in data:
                    raise HTTPException(status_code=500, detail=data["error"])
            except (json.JSONDecodeError, KeyError):
                pass
    return images


def build_webui_response(b64_images: list[str], seed: int, info_extra: dict = None) -> dict:
    """Format response to match WebUI /sdapi/v1/txt2img response schema."""
    info = {
        "seed": seed,
        "all_seeds": [seed] * len(b64_images),
        "subseed": -1,
        "subseed_strength": 0,
        "info": "Generated via AlphaCore Modal Proxy",
        **(info_extra or {}),
    }
    return {
        "images": b64_images,
        "parameters": info,
        "info": json.dumps(info),
    }


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "AlphaCore Modal Proxy online", "port": PROXY_PORT}


@app.get("/sdapi/v1/options")
def get_options():
    """HammerAI calls this to discover capabilities."""
    global CURRENT_MODEL
    return {
        "sd_model_checkpoint": CURRENT_MODEL,
        "sd_vae": "Automatic",
        "samples_format": "png",
        "enable_pnginfo": True,
    }


@app.post("/sdapi/v1/options")
def set_options(body: dict):
    global CURRENT_MODEL
    if "sd_model_checkpoint" in body:
        CURRENT_MODEL = body["sd_model_checkpoint"]
        print(f"[options] Model checkpoint updated to: {CURRENT_MODEL}")
    return {}


@app.get("/sdapi/v1/sd-models")
def list_models():
    return [
        {"title": "JuggernautXL [Modal]", "model_name": "juggernautXL", "hash": "modal"},
        {"title": "CyberRealisticXL [Modal]", "model_name": "cyberrealisticXL", "hash": "modal"},
        {"title": "Unholy [Modal]", "model_name": "unholy", "hash": "modal"},
    ]


@app.get("/sdapi/v1/samplers")
def list_samplers():
    return [
        {"name": "Euler", "aliases": ["euler"], "options": {}},
        {"name": "DPM++ 2M", "aliases": ["dpm"], "options": {}},
        {"name": "Heun", "aliases": ["heun"], "options": {}},
    ]


@app.get("/sdapi/v1/loras")
def list_loras():
    return [
        {"name": "cunny", "alias": "cunny", "path": "/hf-hub-cache/loras/cunny.safetensors", "metadata": {}},
        {"name": "custom_training", "alias": "custom", "path": "/hf-hub-cache/loras/custom_training.safetensors", "metadata": {}},
    ]


@app.get("/sdapi/v1/progress")
def get_progress():
    """Stub — Modal handles progress internally via SSE."""
    return {"progress": 0, "eta_relative": 0, "state": {}, "current_image": None}


@app.post("/sdapi/v1/txt2img")
async def txt2img(req: Txt2ImgRequest):
    print(f"\n[txt2img] prompt='{req.prompt}' steps={req.steps} cfg={req.cfg_scale} batch={req.batch_size}")

    # Map sampler name to Modal's scheduler param
    scheduler_map = {"euler": "Euler", "dpm++ 2m": "DPM", "heun": "Heun"}
    scheduler = scheduler_map.get(req.sampler_name.lower(), "Euler")

    # Enforce a strong negative prompt to avoid mutations, bad anatomy, and low quality
    deformity_negative = f"{req.negative_prompt}, deformed, mutated, bad anatomy, bad proportions, disfigured, extra limbs, missing limbs, worst quality, low quality"

    global CURRENT_MODEL
    selected_model = CURRENT_MODEL
    if req.override_settings and "sd_model_checkpoint" in req.override_settings:
        selected_model = req.override_settings["sd_model_checkpoint"]

    model_lower = selected_model.lower()
    if "jugg" in model_lower:
        jugg_val = 1
        cyber_val = 0
    elif "cyber" in model_lower:
        jugg_val = 0
        cyber_val = 1
    else:  # unholy or fallback
        jugg_val = 0
        cyber_val = 0

    params = {
        "prompt": req.prompt,
        "negative_prompt": deformity_negative,
        "num_inference_steps": req.steps,
        "guidance_scale": str(req.cfg_scale),
        "batch_size": req.batch_size * req.n_iter,
        "seed": req.seed,
        "scheduler": scheduler,
        "lora": "none",
        "JuggernautXL": jugg_val,
        "CyberRealisticXL": cyber_val,
        "enhance_prompt": str(req.enhance_prompt)
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(TXT2IMG_URL, params=params)
            response.raise_for_status()
            b64_images = decode_sse_images(response.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Modal endpoint timed out. Cold start may need more time.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Modal Backend Error: {e.response.text}")

    if not b64_images:
        raise HTTPException(status_code=500, detail="No images returned from Modal endpoint.")

    print(f"[txt2img] Got {len(b64_images)} image(s) back.")
    return JSONResponse(build_webui_response(b64_images, req.seed))


@app.post("/sdapi/v1/img2img")
async def img2img(req: Img2ImgRequest):
    print(f"\n[img2img] prompt='{req.prompt}' steps={req.steps} cfg={req.cfg_scale}")

    if not req.init_images:
        raise HTTPException(status_code=400, detail="No init_images provided.")

    # Decode the first input image from base64 to raw bytes
    try:
        img_data = req.init_images[0]
        # Strip data URI prefix if present
        if img_data.startswith("data:"):
            img_data = img_data.split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode init_image: {e}")

    # Map sampler
    scheduler_map = {"euler": "Euler", "dpm++ 2m": "DPM", "heun": "Heun"}
    scheduler = scheduler_map.get(req.sampler_name.lower(), "Euler")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                IMG2IMG_URL,
                data={
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "num_inference_steps": req.steps,
                    "true_cfg_scale": str(req.cfg_scale),
                    "batch_size": req.batch_size * req.n_iter,
                    "seed": req.seed,
                    "lora": "none",
                },
                files={"image": ("input.png", img_bytes, "image/png")},
            )
            response.raise_for_status()
            b64_images = decode_sse_images(response.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Modal endpoint timed out.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

    if not b64_images:
        raise HTTPException(status_code=500, detail="No images returned from Modal endpoint.")

    print(f"[img2img] Got {len(b64_images)} image(s) back.")
    return JSONResponse(build_webui_response(b64_images, req.seed))


# ─── ENTRY ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AlphaCore Modal Proxy")
    print(f"  Listening on http://localhost:{PROXY_PORT}")
    print(f"  TXT2IMG → {TXT2IMG_URL}")
    print(f"  IMG2IMG → {IMG2IMG_URL}")
    print("  Point HammerAI at: http://localhost:7860")
    print("=" * 60)
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)
