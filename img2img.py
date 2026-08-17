# ---
# output-directory: "/tmp/qwen-final-output"
# ---

# # Edit images with Qwen-Image-Edit-Plus and the NSFW LoRA (Final Version)

# This script correctly implements the Qwen-Image-Edit-2511 model and the
# ScottzillaSystems NSFW LoRA using the specific pipeline and loading method
# required for full functionality.

from io import BytesIO
from pathlib import Path

import modal

app = modal.App("img2img-qwen-edit-plus")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "Pillow",
        "accelerate",
        "fastapi[standard]",
        "git+https://github.com/huggingface/diffusers.git",
        "git+https://github.com/huggingface/transformers.git",
        "huggingface-hub",
        "safetensors",
        "torch",
        "einops", "timm", "transformers-stream-generator", # Qwen dependencies
        "peft", # LoRA dependency
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

# --- CORRECT MODEL & LORA SELECTION ---
BASE_MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_MODEL_REPO_ID = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
LORA_WEIGHT_NAME = "qwen-image-edit-plus-nsfw-lora.safetensors"
LORA_ADAPTER_NAME = "mcnl-nsfw-v1"
# --- END SELECTION ---

CACHE_DIR = "/hf-hub-cache"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("outputs", create_if_missing=True)
OUTPUTS_DIR = Path("/outputs")
volumes = {CACHE_DIR: cache_volume, OUTPUTS_DIR: outputs_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]

image = image.env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": str(CACHE_DIR), "FORCE_REBUILD": "2"})

with image.imports():
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.utils import load_image
    from PIL import Image


# --- MAXIMUM HARDWARE UPGRADE ---
# Upgrading to an A100 with 80GB of VRAM to overcome the OutOfMemoryError.
@app.cls(image=image, gpu="A100-80GB", volumes=volumes, secrets=secrets)
# --- END UPGRADE ---
class Model:
    @modal.enter()
    def enter(self):
        print(f"Loading base model {BASE_MODEL_NAME}...")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        ).to("cuda")

        print(f"Loading and setting LoRA adapter from {LORA_MODEL_REPO_ID}...")
        self.pipe.load_lora_weights(
            LORA_MODEL_REPO_ID,
            weight_name=LORA_WEIGHT_NAME,
            adapter_name=LORA_ADAPTER_NAME,
            cache_dir=CACHE_DIR,
        )
        self.pipe.set_adapters([LORA_ADAPTER_NAME])

        print("Disabling safety checker.")
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    @modal.method(is_generator=True)
    def run_stream(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str,
        true_cfg_scale: float,
        num_inference_steps: int,
        batch_size: int = 1,
        lora: str = "none",
        seed: int | None = None,
        image_bytes2: bytes | None = None,
        refs: list[bytes] = [],
    ):
        pil_images = [load_image(Image.open(BytesIO(image_bytes))).convert("RGB")]
        if image_bytes2:
            pil_images.append(load_image(Image.open(BytesIO(image_bytes2))).convert("RGB"))
        for ref_bytes in refs:
            if ref_bytes:
                pil_images.append(load_image(Image.open(BytesIO(ref_bytes))).convert("RGB"))
            
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

        import queue
        import threading
        import base64

        q = queue.Queue()

        def callback(pipe, step_index, timestep, callback_kwargs):
            q.put({"step": step_index, "max_steps": num_inference_steps})
            return callback_kwargs

        def generate_task():
            all_images_b64 = []
            try:
                if lora and lora != "none":
                    loras_to_load = [l.strip() for l in lora.split(",") if l.strip() and l.strip() != "none"]
                    loaded_adapters = []
                    for l in loras_to_load:
                        lora_path = Path("/hf-hub-cache/loras") / f"{l}.safetensors"
                        if lora_path.exists():
                            print(f"Loading LoRA: {lora_path}")
                            self.pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name, adapter_name=l)
                            loaded_adapters.append(l)
                        else:
                            print(f"LoRA {lora_path} not found!")
                    if loaded_adapters:
                        self.pipe.set_adapters(loaded_adapters)

                images_completed = 0
                while images_completed < batch_size:
                    current_batch_size = min(batch_size - images_completed, 10)

                    def chunk_callback(pipe, step_index, timestep, callback_kwargs):
                        q.put({"step": step_index, "max_steps": num_inference_steps, "images_completed": images_completed, "total_images": batch_size})
                        return callback_kwargs

                    chunk_images = self.pipe(
                        image=pil_images,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=current_batch_size,
                        num_inference_steps=num_inference_steps,
                        true_cfg_scale=true_cfg_scale,
                        generator=generator,
                        callback_on_step_end=chunk_callback
                    ).images

                    chunk_b64 = []
                    for image in chunk_images:
                        with BytesIO() as buf:
                            image.save(buf, format="PNG")
                            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                            chunk_b64.append(b64)
                            all_images_b64.append(b64)

                    images_completed += current_batch_size
                    q.put({"image_b64_partial": chunk_b64, "images_completed": images_completed, "total_images": batch_size})

                if lora and lora != "none":
                    self.pipe.unload_lora_weights()

                q.put({"image_b64": all_images_b64})
            except Exception as e:
                if lora and lora != "none":
                    try: self.pipe.unload_lora_weights()
                    except: pass
                q.put({"error": str(e)})

        threading.Thread(target=generate_task).start()

        while True:
            msg = q.get()
            yield msg
            if "image_b64" in msg or "error" in msg:
                break

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str,
        true_cfg_scale: float,
        num_inference_steps: int,
        batch_size: int = 1,
        lora: str = "none",
        seed: int | None = None,
    ) -> list[bytes]:
        import base64
        msgs = list(self.run_stream.local(
            image_bytes=image_bytes, prompt=prompt, negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale, num_inference_steps=num_inference_steps, batch_size=batch_size, lora=lora, seed=seed, image_bytes2=None, refs=[]
        ))
        final_msg = msgs[-1]
        if "error" in final_msg:
            raise Exception(final_msg["error"])
        return [base64.b64decode(b64) for b64 in final_msg["image_b64"]]

    @modal.asgi_app()
    def web(self):
        import fastapi
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi import Response, UploadFile, File, Form

        web_app = fastapi.FastAPI()

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/ping")
        def ping():
            return {"status": "ok"}

        @web_app.post("/shutdown")
        @web_app.get("/shutdown")
        def shutdown():
            import os
            import threading
            import time
            def kill():
                time.sleep(1)
                os._exit(0)
            threading.Thread(target=kill).start()
            return {"status": "shutting down"}

        @web_app.post("/v1/images/edits")
        async def openai_compatible_edits(request: fastapi.Request):
            import base64
            from io import BytesIO
            
            # The OpenAI spec for edits can be sent as form data or JSON depending on the client.
            # Open WebUI typically sends multipart/form-data for image edits.
            form = await request.form()
            
            prompt = form.get("prompt", "Apply standard edits")
            n = int(form.get("n", 1))
            
            # Extract the uploaded image file
            image_file = form.get("image")
            if not image_file:
                return fastapi.responses.JSONResponse(status_code=400, content={"error": "Image is required for editing."})
            
            image_bytes = await image_file.read()
            
            # Run your existing Modal inference function for Qwen Image Edit
            output_bytes_list = self.inference.local(
                image_bytes=image_bytes,
                prompt=prompt,
                negative_prompt="worst quality, low quality, censorship, text, watermark, signature, blur, bad anatomy, ugly, deformed",
                true_cfg_scale=4.0,
                num_inference_steps=20,
                batch_size=n,
                lora="none",
                seed=-1,
            )
            
            # Format the output exactly how Open WebUI expects it (OpenAI spec)
            b64_list = [{"b64_json": base64.b64encode(img).decode('utf-8')} for img in output_bytes_list]
            return {"data": b64_list}

        @web_app.post("/stream")
        def web_endpoint_stream(
            image: UploadFile = File(...),
            image2: UploadFile | None = File(None),
            ref1: UploadFile | None = File(None),
            ref2: UploadFile | None = File(None),
            ref3: UploadFile | None = File(None),
            ref4: UploadFile | None = File(None),
            ref5: UploadFile | None = File(None),
            prompt: str = Form("input requested image edits here."),
            negative_prompt: str = Form("worst quality, low quality, censorship, text, watermark, signature, blur, bad anatomy, ugly, deformed"),
            true_cfg_scale: float = Form(4.0),
            num_inference_steps: int = Form(20),
            batch_size: int = Form(1),
            lora: str = Form("none"),
            seed: int = Form(-1),
        ):
            from fastapi.responses import StreamingResponse
            import json
            import base64
            import time

            image_bytes = image.file.read()
            image_bytes2 = image2.file.read() if image2 else None
            refs_bytes = []
            for ref in [ref1, ref2, ref3, ref4, ref5]:
                if ref: refs_bytes.append(ref.file.read())
                
            seed_val = None if seed == -1 else seed
            
            def event_generator():
                for msg in self.run_stream.local(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    true_cfg_scale=true_cfg_scale,
                    num_inference_steps=num_inference_steps,
                    batch_size=batch_size,
                    lora=lora,
                    seed=seed_val,
                    image_bytes2=image_bytes2,
                    refs=refs_bytes,
                ):
                    yield f"data: {json.dumps(msg)}\n\n"
                    if "image_b64" in msg:
                        for idx, b64 in enumerate(msg["image_b64"]):
                            out_bytes = base64.b64decode(b64)
                            out_dir = OUTPUTS_DIR / "img2img"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            filename = f"img2img_{int(time.time() * 1000)}_{idx}.png"
                            (out_dir / filename).write_bytes(out_bytes)
                        outputs_volume.commit()
                        print(f"Saved {len(msg['image_b64'])} outputs to /outputs/img2img/")

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        @web_app.post("/")
        def web_endpoint(
            image: UploadFile = File(...),
            # --- PARAMETER DEFAULTS UPDATED ---
            prompt: str = Form("input requested image edits here."),
            negative_prompt: str = Form("worst quality, low quality, censorship, text, watermark, signature, blur, bad anatomy, ugly, deformed"),
            true_cfg_scale: float = Form(4.0),
            num_inference_steps: int = Form(20),
            batch_size: int = Form(1),
            lora: str = Form("none"),
            # --- END UPDATE ---
            seed: int = Form(-1),
        ):
            image_bytes = image.file.read()
            seed_val = None if seed == -1 else seed
            output_bytes_list = self.inference.local(
                image_bytes=image_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                batch_size=batch_size,
                lora=lora,
                seed=seed_val,
            )

            # Save to outputs volume
            import time
            out_dir = OUTPUTS_DIR / "img2img"
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, out_bytes in enumerate(output_bytes_list):
                filename = f"img2img_{int(time.time() * 1000)}_{idx}.png"
                (out_dir / filename).write_bytes(out_bytes)
            outputs_volume.commit()
            print(f"Saved {len(output_bytes_list)} outputs to /outputs/img2img/")

            return Response(
                content=output_bytes_list[0],
                media_type="image/png"
            )

        return web_app


@app.local_entrypoint()
def main(
    image_path=Path(__file__).parent / "demo_images/woman.png",
    output_path=Path("/tmp/qwen-final-output/output.png"),
    # --- PARAMETER DEFAULTS UPDATED ---
    prompt: str = "input requested image edits here.",
    negative_prompt: str = "worst quality, low quality, censorship, text, watermark, signature, blur, bad anatomy, ugly, deformed",
    num_steps: int = 20,
    # --- END UPDATE ---
):
    print(f"🎨 Reading input image from {image_path}")
    if not Path(image_path).exists():
        print(f"Error: Input image not found at {image_path}.")
        return

    input_image_bytes = Path(image_path).read_bytes()
    print(f"🎨 Editing image with instruction: '{prompt}'")
    output_image_bytes = Model().inference.remote(
        image_bytes=input_image_bytes,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=4.0,
        num_inference_steps=num_steps,
        batch_size=1,
    )[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎨 Saving output image to {output_path}")
    output_path.write_bytes(output_image_bytes)