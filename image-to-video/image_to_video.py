# ---
# output-directory: "/tmp/image_to_video"
# args: ["--prompt", "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background.", "--image-path", "https://modal-cdn.com/example_image_to_video_image.png"]
# ---

import io
import random
import time
from pathlib import Path
from typing import Annotated

import fastapi
import modal

app = modal.App("example-image-to-video-fixed")

# ### Configuring dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv")
    .uv_pip_install(
        "accelerate==1.4.0",
        "diffusers==0.32.2",
        "fastapi[standard]==0.115.8",
        "huggingface-hub==0.36.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "opencv-python==4.11.0.86",
        "pillow==11.1.0",
        "sentencepiece==0.2.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.49.0",
    )
)

MODEL_ID = "Lightricks/LTX-Video"
MODEL_REVISION_ID = "a6d59ee37c13c58261aa79027d3e41cd41960925"

model_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
MODEL_PATH = "/models"

image = image.env(
    {
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_CACHE": MODEL_PATH,
    }
)

OUTPUT_PATH = "/outputs"
output_volume = modal.Volume.from_name("outputs", create_if_missing=True)

with image.imports():
    import diffusers
    import torch
    from PIL import Image

MINUTES = 60


@app.cls(
    image=image,
    gpu="H100",
    timeout=20 * MINUTES,  # Extended timeout to 20 minutes to prevent heartbeat drops
    scaledown_window=10 * MINUTES,
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
)
class Inference:
    @modal.enter()
    def load_pipeline(self):
        self.pipe = diffusers.LTXImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    @modal.method()
    def run(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 25,
        num_inference_steps: int = 50,
        seed: int = 0,
    ) -> str:
        # Resolve clean string defaults to prevent parser issues
        if not negative_prompt:
            negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
        
        # Handle zero or unassigned seeds
        if seed <= 0:
            seed = random.randint(1, 2**32 - 1)
            
        print(f"Seeding RNG with: {seed}")
        torch.manual_seed(seed)

        width = 768
        height = 512

        image = diffusers.utils.load_image(Image.open(io.BytesIO(image_bytes)))

        video = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        ).frames[0]

        mp4_name = (
            f"{seed}_{''.join(c if c.isalnum() else '-' for c in prompt[:100])}.mp4"
        )
        diffusers.utils.export_to_video(
            video, f"{Path(OUTPUT_PATH) / mp4_name}", fps=24
        )
        output_volume.commit()
        torch.cuda.empty_cache()
        return mp4_name

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(
        self,
        image_bytes: Annotated[bytes, fastapi.File()],
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 25,
        num_inference_steps: int = 50,
        seed: int = 0,
    ) -> fastapi.Response:
        mp4_name = self.run.local(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return fastapi.responses.FileResponse(
            path=f"{Path(OUTPUT_PATH) / mp4_name}",
            media_type="video/mp4",
            filename=mp4_name,
        )


@app.local_entrypoint()
def entrypoint(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 25,
    num_inference_steps: int = 50,
    seed: int = 0,
    twice: bool = True,
):
    import os
    import urllib.request

    print(f"🎥 Generating a video from the image at {image_path}")
    print(f"🎥 using the prompt {prompt}")

    if image_path.startswith(("http://", "https://")):
        image_bytes = urllib.request.urlopen(image_path).read()
    elif os.path.isfile(image_path):
        image_bytes = Path(image_path).read_bytes()
    else:
        raise ValueError(f"{image_path} is not a valid file or URL.")

    inference_service = Inference()

    for _ in range(1 + twice):
        start = time.time()
        mp4_name = inference_service.run.remote(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        duration = time.time() - start
        print(f"🎥 Generated video in {duration:.3f}s")

        output_dir = Path("/tmp/image_to_video")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / mp4_name
        output_path.write_bytes(b"".join(output_volume.read_file(mp4_name)))
        print(f"🎥 Video saved to {output_path}")


frontend_path = Path(__file__).parent / "frontend"

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("jinja2==3.1.5", "fastapi[standard]==0.115.8")
    .add_local_dir(frontend_path, remote_path="/assets")
)


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    import fastapi.templating

    web_app = fastapi.FastAPI()
    templates = fastapi.templating.Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: fastapi.Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Inference().web.get_web_url(),
                "model_name": "LTX-Video Image to Video",
                "default_prompt": "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background.",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app