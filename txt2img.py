# ---
# output-directory: "/tmp/stable-diffusion"
# args: ["--prompt", "A cinematic photo"]
# ---

import io
import random
import time
from pathlib import Path

import modal

MINUTES = 60

app = modal.App("text-to-image-sdxl-merger")

CACHE_DIR = "/hf-hub-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub==0.36.0",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
        "omegaconf>=2.3.0",
        "peft>=0.6.0",
        "safetensors>=0.4.5",  
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

with image.imports():
    from diffusers import StableDiffusionXLPipeline
    import diffusers
    import torch
    from safetensors.torch import load_file, save_file
    from fastapi import Response

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
)
class Inference:
    @modal.enter()
    def setup(self):
        self.current_model = None
        self.pipe = None

    def _load_model(self, model_filename: str):
        if self.current_model == model_filename and self.pipe is not None:
            return

        if self.pipe is not None:
            print(f"Unloading {self.current_model} from VRAM...")
            del self.pipe
            torch.cuda.empty_cache()

        print(f"Loading {model_filename} into VRAM...")
        model_path = Path(CACHE_DIR) / "checkpoints" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}. Did you upload it?")

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        self.current_model = model_filename

    def _merge_and_load(self, model1_name: str, model2_name: str, merged_name: str):
        merged_path = Path(CACHE_DIR) / "checkpoints" / merged_name
        
        if not merged_path.exists():
            print("🚀 BOTH MODELS SELECTED! Initiating 50/50 Checkpoint Merge...")
            print("⚠️ This will take ~60 seconds but will only happen ONCE. Caching result...")
            
            path1 = Path(CACHE_DIR) / "checkpoints" / model1_name
            path2 = Path(CACHE_DIR) / "checkpoints" / model2_name
            
            if not path1.exists() or not path2.exists():
                raise FileNotFoundError("Cannot merge: One or both source models are missing from the folder.")

            tensors1 = load_file(path1)
            tensors2 = load_file(path2)
            
            merged_tensors = {}
            for key in tensors1.keys():
                if key in tensors2:
                    merged_tensors[key] = (tensors1[key] * 0.5) + (tensors2[key] * 0.5)
                else:
                    merged_tensors[key] = tensors1[key]
            
            save_file(merged_tensors, merged_path)
            print(f"✅ Merge complete! Saved as {merged_name}")
            
            del tensors1, tensors2, merged_tensors
            
        self._load_model(merged_name)

    @modal.method()
    def run(
        self, 
        prompt: str, 
        JuggernautXL: int = 1,          
        CyberRealisticXL: int = 0,      
        negative_prompt: str = "low quality, blurry, distorted", 
        batch_size: int = 4, 
        guidance_scale: str = "7.0", 
        num_inference_steps: int = 25, 
        scheduler: str = "Euler", 
        seed: int = -1,
        lora: str = "none"
    ) -> list[bytes]:
        
        model_file = "unholyDesireMixSinister_v80.safetensors"
        
        if JuggernautXL == 1 and CyberRealisticXL == 0:
            model_file = "juggernautXL_ragnarok.safetensors"
        elif CyberRealisticXL == 1 and JuggernautXL == 0:
            model_file = "cyberrealisticXL_desireV30.safetensors"
            
        self._load_model(model_file)

        g_scale_float = float(guidance_scale)
        batch_size = min(batch_size, 10)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        print(f"Seeding RNG with: {seed}")
        torch.manual_seed(seed)

        if scheduler.lower() == "heun":
            self.pipe.scheduler = diffusers.HeunDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler.lower() == "dpm":
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # Load LoRAs if specified
        if lora and lora.lower() != "none":
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

        try:
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_images_per_prompt=batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=g_scale_float,
            ).images
        finally:
            if lora and lora.lower() != "none":
                try: self.pipe.unload_lora_weights()
                except: pass

        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        torch.cuda.empty_cache()
        return image_output

    @modal.asgi_app()
    def web(self):
        import fastapi
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi import Response
        import time

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
            def kill():
                time.sleep(1)
                os._exit(0)
            threading.Thread(target=kill).start()
            return {"status": "shutting down"}

        @web_app.post("/v1/images/generations")
        async def openai_compatible_generations(request: fastapi.Request):
            import base64
            
            body = await request.json()
            prompt = body.get("prompt", "A cinematic photo")
            n = int(body.get("n", 1))
            
            image_bytes_list = self.run.local(
                prompt=prompt,
                JuggernautXL=1,
                CyberRealisticXL=0,
                negative_prompt="illustration, 3d, render, anime, cartoon, painting, sketch, bad anatomy, deformed, disfigured, mutated, poorly drawn face, poorly drawn hands, mutated hands, missing limbs, extra limbs, extra fingers, distorted body, bad proportions, fused flesh, asymmetrical, plastic, airbrushed, overly smooth skin, oversaturated, blurry, worst quality, low quality, watermark, text, signature",
                batch_size=n,
                guidance_scale="7.0",
                num_inference_steps=25,
                scheduler="Euler",
                seed=-1,
                lora="none"
            )
            
            b64_list = [{"b64_json": base64.b64encode(img).decode('utf-8')} for img in image_bytes_list]
            return {"data": b64_list}

        @web_app.get("/stream")
        @web_app.get("/")
        def generate_endpoint(
            prompt: str, 
            JuggernautXL: int = 1,
            CyberRealisticXL: int = 0,
            negative_prompt: str = "illustration, 3d, render, anime, cartoon, painting, sketch, bad anatomy, deformed, disfigured, mutated, poorly drawn face, poorly drawn hands, mutated hands, missing limbs, extra limbs, extra fingers, distorted body, bad proportions, fused flesh, asymmetrical, plastic, airbrushed, overly smooth skin, oversaturated, blurry, worst quality, low quality, watermark, text, signature", 
            guidance_scale: str = "7.0", 
            num_inference_steps: int = 25, 
            scheduler: str = "Euler", 
            seed: int = -1,
            batch_size: int = 1,
            lora: str = "none"
        ):
            from starlette.responses import StreamingResponse
            import base64

            def event_stream():
                image_bytes_list = self.run.local(
                    prompt=prompt,
                    JuggernautXL=JuggernautXL,
                    CyberRealisticXL=CyberRealisticXL,
                    negative_prompt=negative_prompt,
                    batch_size=max(1, int(batch_size)),
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    scheduler=scheduler,
                    seed=seed,
                    lora=lora,
                )
                b64_list = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]
                import json
                yield f"data: {json.dumps({'image_b64': b64_list})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        return web_app


@app.local_entrypoint()
def entrypoint(
    samples: int = 4,
    prompt: str = "A cinematic photo",
    JuggernautXL: int = 1,
    CyberRealisticXL: int = 0,
    negative_prompt: str = "low quality, blurry, distorted",
    batch_size: int = 4,
    guidance_scale: str = "7.0",
    num_inference_steps: int = 25,
    scheduler: str = "Euler",
    seed: int = -1,
    lora: str = "none",
):
    output_dir = Path("/tmp/stable-diffusion")
    output_dir.mkdir(exist_ok=True, parents=True)

    inference_service = Inference()

    for sample_idx in range(samples):
        start = time.time()
        images = inference_service.run.remote(
            prompt=prompt,
            JuggernautXL=JuggernautXL,
            CyberRealisticXL=CyberRealisticXL,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            scheduler=scheduler,
            seed=seed,
            lora=lora,
        )
        duration = time.time() - start
        print(f"Run {sample_idx + 1} took {duration:.3f}s")
        for batch_idx, image_bytes in enumerate(images):
            output_path = output_dir / f"output_{slugify(prompt)[:64]}_{str(sample_idx).zfill(2)}_{str(batch_idx).zfill(2)}.png"
            output_path.write_bytes(image_bytes)

def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")