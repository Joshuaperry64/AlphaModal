# ---
# output-directory: "/tmp/txt2vid-output"
# args: ["--prompt", "A cinematic video of a serene waterfall in a mystical forest at dusk"]
# ---

# # Text-to-Video Generator on Modal (Wan2.1 14B NSFW Video Model)
# Model: Slinkies86/NSFW_Wan_14b-video
# Powered by Modal H100 GPUs and Hugging Face Diffusers WanPipeline

import io
import os
import time
import random
from pathlib import Path
import tempfile

import modal

MINUTES = 60
CACHE_DIR = "/hf-hub-cache"
OUTPUTS_DIR = Path("/outputs")

# Build container image with CUDA 12.4+ and latest diffusers for WanPipeline support
cuda_version = "12.4.1"
tag = f"{cuda_version}-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
    )
    .uv_pip_install(
        "accelerate>=0.33.0",
        "fastapi[standard]>=0.115.4",
        "huggingface-hub>=0.36.0",
        "sentencepiece>=0.2.0",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.48.0",
        "safetensors>=0.4.5",
        "einops>=0.8.0",
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "numpy<2",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": CACHE_DIR,
            "HF_HOME": CACHE_DIR,
        }
    )
)

app = modal.App("txt2vid-wan-14b", image=image)

with image.imports():
    import torch
    from diffusers import WanPipeline, AutoencoderKLWan
    from diffusers.utils import export_to_video
    import numpy as np
    from PIL import Image

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("outputs", create_if_missing=True)

MODEL_ID = "Slinkies86/NSFW_Wan_14b-video"
BASE_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


@app.cls(
    gpu="H100",  # H100 GPU (80GB VRAM) for 14B video model inference
    timeout=60 * MINUTES,
    scaledown_window=20 * MINUTES,
    volumes={
        CACHE_DIR: cache_volume,
        OUTPUTS_DIR: outputs_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Model:
    @modal.enter()
    def setup(self):
        print(f"🚀 Loading Wan 14B Video Model ({MODEL_ID})...")
        try:
            self.pipe = WanPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR,
            )
        except Exception as e:
            print(f"⚠️ Direct load failed ({e}). Loading with fallbacks...")
            try:
                vae = AutoencoderKLWan.from_pretrained(
                    BASE_MODEL_ID,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                    cache_dir=CACHE_DIR,
                )
                self.pipe = WanPipeline.from_pretrained(
                    MODEL_ID,
                    vae=vae,
                    torch_dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR,
                )
            except Exception as ex:
                print(f"⚠️ Fallback load failed ({ex}). Loading base model {BASE_MODEL_ID}...")
                self.pipe = WanPipeline.from_pretrained(
                    BASE_MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR,
                )

        self.pipe.to("cuda")
        print("✅ Wan 14B Video Model loaded successfully into GPU VRAM!")

    @modal.method(is_generator=True)
    def run_stream(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, static, jittery, watermark",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: int = -1,
        fps: int = 16,
    ):
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        print(f"🎬 Generating video for prompt: '{prompt}' (Seed: {seed})")
        print(f"⚙️ Specs: {width}x{height}, {num_frames} frames, {num_inference_steps} steps, CFG {guidance_scale}")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        import queue
        import threading
        import base64

        q = queue.Queue()

        def callback(pipe, step_index, timestep, callback_kwargs):
            q.put({"step": step_index, "max_steps": num_inference_steps})
            return callback_kwargs

        def generate_task():
            try:
                start_time = time.time()
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback_on_step_end=callback
                ).frames[0]
                duration = time.time() - start_time
                print(f"✅ Video generation finished in {duration:.2f}s!")

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    export_to_video(output, tmp_path, fps=fps)
                    video_bytes = Path(tmp_path).read_bytes()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                # Save to outputs volume
                out_dir = OUTPUTS_DIR / "txt2vid"
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = f"vid_{int(time.time() * 1000)}_{seed}.mp4"
                (out_dir / filename).write_bytes(video_bytes)
                outputs_volume.commit()
                
                b64_video = base64.b64encode(video_bytes).decode("utf-8")
                
                torch.cuda.empty_cache()
                q.put({"video_b64": b64_video})
            except Exception as e:
                q.put({"error": str(e)})

        threading.Thread(target=generate_task).start()

        while True:
            msg = q.get()
            yield msg
            if "video_b64" in msg or "error" in msg:
                break

    @modal.method()
    def inference(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, static, jittery, watermark",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: int = -1,
        fps: int = 16,
    ) -> bytes:
        import base64
        msgs = list(self.run_stream.local(
            prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
            num_frames=num_frames, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
            seed=seed, fps=fps
        ))
        final_msg = msgs[-1]
        if "error" in final_msg:
            raise Exception(final_msg["error"])
        return base64.b64decode(final_msg["video_b64"])

    @modal.asgi_app()
    def web(self):
        import fastapi
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import Response
        import base64

        web_app = fastapi.FastAPI(title="AlphaCore Txt2Vid Wan-14B API", version="1.0.0")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/ping")
        def ping():
            return {"status": "ok", "model": MODEL_ID}

        @web_app.get("/v1/models")
        def get_models():
            return {
                "object": "list",
                "data": [
                    {"id": "wan-14b-nsfw", "object": "model", "owned_by": "Slinkies86"},
                ],
            }

        @web_app.post("/v1/videos/generations")
        @web_app.post("/generate")
        async def generate_video(request: fastapi.Request):
            body = await request.json()
            prompt = body.get("prompt", "A cinematic video")
            negative_prompt = body.get("negative_prompt", "low quality, blurry, distorted")
            height = int(body.get("height", 480))
            width = int(body.get("width", 832))
            num_frames = int(body.get("num_frames", 81))
            guidance_scale = float(body.get("guidance_scale", 5.0))
            num_inference_steps = int(body.get("num_inference_steps", 30))
            seed = int(body.get("seed", -1))
            fps = int(body.get("fps", 16))

            video_bytes = self.inference.local(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                fps=fps,
            )

            b64_video = base64.b64encode(video_bytes).decode("utf-8")
            return {
                "data": [
                    {
                        "b64_json": b64_video,
                        "content_type": "video/mp4",
                    }
                ]
            }

        @web_app.get("/stream")
        @web_app.get("/")
        def generate_endpoint(
            prompt: str,
            negative_prompt: str = "low quality, blurry, distorted, static, jittery",
            height: int = 480,
            width: int = 832,
            num_frames: int = 81,
            guidance_scale: float = 5.0,
            num_inference_steps: int = 30,
            seed: int = -1,
            fps: int = 16,
        ):
            from fastapi.responses import StreamingResponse
            import json

            def event_stream():
                try:
                    for msg in self.run_stream.local(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        fps=fps,
                    ):
                        # Force flush by yielding a 4KB comment padding along with the data
                        padding = ": " + " " * 4096 + "\n"
                        yield padding + f"data: {json.dumps(msg)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return web_app


@app.local_entrypoint()
def main(
    prompt: str = "A cinematic video of a majestic dragon flying over a sunset ocean",
    negative_prompt: str = "low quality, blurry, distorted, static, jittery",
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    guidance_scale: float = 5.0,
    steps: int = 30,
    seed: int = -1,
    fps: int = 16,
    output: str = "/tmp/txt2vid-output/output.mp4",
):
    output_path = Path(output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"🎬 Requesting video generation from Modal H100...")
    video_bytes = Model().inference.remote(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        seed=seed,
        fps=fps,
    )

    print(f"🎬 Saving generated MP4 video to {output_path}...")
    output_path.write_bytes(video_bytes)
    print("✅ Done!")
