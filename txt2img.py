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
        JuggernautXL: int = 0,          
        CyberRealisticXL: int = 0,      
        EpicRealismXL: int = 0,
        UnholyDesireXL: int = 0,
        Lustify: int = 0,
        AutismPony: int = 0,
        model_name: str = "jugg",
        negative_prompt: str = "low quality, blurry, distorted", 
        batch_size: int = 4, 
        guidance_scale: str = "7.0", 
        num_inference_steps: int = 25, 
        scheduler: str = "Euler", 
        sampler: str = "",
        amateur: int = 0,
        seed: int = -1,
        lora: str = "none"
    ) -> list[bytes]:
        
        if model_name.endswith(".safetensors"):
            model_file = model_name
        elif EpicRealismXL == 1 or model_name.lower().startswith("epic"):
            model_file = "epicrealismXL_pureFix.safetensors"
        elif CyberRealisticXL == 1 or model_name.lower().startswith("cyber"):
            model_file = "cyberrealisticXL_desireV30.safetensors"
        elif UnholyDesireXL == 1 or model_name.lower().startswith("unholy"):
            model_file = "unholyDesireMixSinister_v80.safetensors"
        elif Lustify == 1 or model_name.lower().startswith("lust"):
            model_file = "lustifyNSFWCheckpoint_zenithV9.safetensors"
        elif AutismPony == 1 or model_name.lower().startswith("autism") or model_name.lower().startswith("pony"):
            model_file = "autismmixSDXL_autismmixPony.safetensors"
        elif model_name.lower().startswith("0x7"):
            model_file = "0x7RealisticFreedom_omegaSDXL.safetensors"
        else:
            model_file = "juggernautXL_ragnarok.safetensors"
            
        self._load_model(model_file)

        if not guidance_scale:
            guidance_scale = "7.0"
        g_scale_float = float(guidance_scale)
        batch_size = max(1, batch_size)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        print(f"Seeding RNG with: {seed}")
        torch.manual_seed(seed)

        # Combine sampler and scheduler strings to parse them comprehensively
        chosen_sampler = f"{sampler} {scheduler}".lower()

        if "euler a" in chosen_sampler:
            self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # Note: diffusers has a known bug with solver_order=3 and sde-dpmsolver++ (UnboundLocalError on x_t).
        # We catch 3M requests and safely route them through the stable 2nd-order SDE mathematics instead.
        elif "3m" in chosen_sampler and "sde" in chosen_sampler and "karras" in chosen_sampler:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        elif "3m" in chosen_sampler and "sde" in chosen_sampler:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
        elif "2m" in chosen_sampler and "sde" in chosen_sampler and "karras" in chosen_sampler:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        elif "dpm++ 2m karras" in chosen_sampler or "dpm++_2m_karras" in chosen_sampler or ("2m" in chosen_sampler and "karras" in chosen_sampler):
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        elif "dpm++ sde karras" in chosen_sampler or "dpm++_sde_karras" in chosen_sampler or ("sde" in chosen_sampler and "karras" in chosen_sampler):
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        elif "dpm++ 2m" in chosen_sampler or "dpm++_2m" in chosen_sampler:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif "dpm++ sde" in chosen_sampler or "dpm++_sde" in chosen_sampler:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
        elif "ddim" in chosen_sampler:
            self.pipe.scheduler = diffusers.DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif "lms" in chosen_sampler:
            self.pipe.scheduler = diffusers.LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif "heun" in chosen_sampler:
            self.pipe.scheduler = diffusers.HeunDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif "unipc" in chosen_sampler:
            self.pipe.scheduler = diffusers.UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif "dpm" in chosen_sampler:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # Handle Amateur UI toggle
        if amateur == 1:
            if lora and lora.lower() != "none":
                lora += ",New_Amateurs_XL"
            else:
                lora = "New_Amateurs_XL"

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
                images_completed = 0
                while images_completed < batch_size:
                    current_batch_size = min(batch_size - images_completed, 10)

                    def chunk_callback(pipe, step_index, timestep, callback_kwargs):
                        q.put({"step": step_index, "max_steps": num_inference_steps, "images_completed": images_completed, "total_images": batch_size})
                        return callback_kwargs

                    chunk_images = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        num_images_per_prompt=current_batch_size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=g_scale_float,
                        callback_on_step_end=chunk_callback
                    ).images
                    
                    chunk_b64 = []
                    for image in chunk_images:
                        with io.BytesIO() as buf:
                            image.save(buf, format="PNG")
                            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                            chunk_b64.append(b64)
                            all_images_b64.append(b64)
                    
                    images_completed += current_batch_size
                    q.put({"image_b64_partial": chunk_b64, "images_completed": images_completed, "total_images": batch_size})

                torch.cuda.empty_cache()
                if lora and lora.lower() != "none":
                    try: self.pipe.unload_lora_weights()
                    except: pass
                
                q.put({"image_b64": all_images_b64})
            except Exception as e:
                if lora and lora.lower() != "none":
                    try: self.pipe.unload_lora_weights()
                    except: pass
                q.put({"error": str(e)})

        threading.Thread(target=generate_task).start()

        while True:
            msg = q.get()
            yield msg
            if "image_b64" in msg or "error" in msg:
                break

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

        @web_app.get("/v1/models")
        def get_models():
            return {
                "object": "list",
                "data": [
                    {"id": "jugg", "object": "model", "owned_by": "user"},
                    {"id": "cyber", "object": "model", "owned_by": "user"},
                    {"id": "epic", "object": "model", "owned_by": "user"},
                    {"id": "unholy", "object": "model", "owned_by": "user"},
                    {"id": "lust", "object": "model", "owned_by": "user"},
                    {"id": "autism", "object": "model", "owned_by": "user"},
                ]
            }

        @web_app.post("/v1/images/generations")
        async def openai_compatible_generations(request: fastapi.Request):
            import base64
            
            body = await request.json()
            prompt = body.get("prompt", "A cinematic photo")
            n = int(body.get("n", 1))
            model_name = body.get("model", "jugg")
            sampler_val = body.get("sampler", "")
            scheduler_val = body.get("scheduler", "Euler")
            
            # Default massive negative prompt
            default_neg = "illustration, 3d, render, anime, cartoon, painting, sketch, bad anatomy, deformed, disfigured, mutated, poorly drawn face, poorly drawn hands, mutated hands, missing limbs, extra limbs, extra fingers, distorted body, bad proportions, fused flesh, asymmetrical, plastic, airbrushed, overly smooth skin, oversaturated, blurry, worst quality, low quality, watermark, text, signature"
            neg_prompt = body.get("negative_prompt", default_neg)
            # If they pass an empty string, we should respect it or use default? Usually, fallback if empty.
            if not neg_prompt:
                neg_prompt = default_neg
            
            msgs = list(self.run.local(
                prompt=prompt,
                model_name=model_name,
                negative_prompt=neg_prompt,
                batch_size=n,
                guidance_scale="7.0",
                num_inference_steps=25,
                scheduler=scheduler_val,
                sampler=sampler_val,
                amateur=0,
                seed=-1,
                lora="none"
            ))
            
            final_msg = msgs[-1]
            if "error" in final_msg:
                return {"error": final_msg["error"]}
            
            b64_list = [{"b64_json": b64} for b64 in final_msg["image_b64"]]
            return {"data": b64_list}

        @web_app.get("/stream")
        @web_app.get("/")
        def generate_endpoint(
            prompt: str, 
            model_name: str = "jugg",
            JuggernautXL: int = 0,
            CyberRealisticXL: int = 0,
            EpicRealismXL: int = 0,
            UnholyDesireXL: int = 0,
            Lustify: int = 0,
            AutismPony: int = 0,
            negative_prompt: str = "illustration, 3d, render, anime, cartoon, painting, sketch, bad anatomy, deformed, disfigured, mutated, poorly drawn face, poorly drawn hands, mutated hands, missing limbs, extra limbs, extra fingers, distorted body, bad proportions, fused flesh, asymmetrical, plastic, airbrushed, overly smooth skin, oversaturated, blurry, worst quality, low quality, watermark, text, signature", 
            guidance_scale: str = "7.0", 
            num_inference_steps: int = 25, 
            scheduler: str = "Euler", 
            sampler: str = "",
            amateur: int = 0,
            seed: int = -1,
            batch_size: int = 1,
            lora: str = "none"
        ):
            from starlette.responses import StreamingResponse
            import base64

            def event_stream():
                for msg in self.run.local(
                    prompt=prompt,
                    model_name=model_name,
                    JuggernautXL=JuggernautXL,
                    CyberRealisticXL=CyberRealisticXL,
                    EpicRealismXL=EpicRealismXL,
                    UnholyDesireXL=UnholyDesireXL,
                    Lustify=Lustify,
                    AutismPony=AutismPony,
                    negative_prompt=negative_prompt,
                    batch_size=max(1, int(batch_size)),
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    scheduler=scheduler,
                    sampler=sampler,
                    amateur=amateur,
                    seed=seed,
                    lora=lora,
                ):
                    import json
                    yield f"data: {json.dumps(msg)}\n\n"

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
    sampler: str = "",
    amateur: int = 0,
    seed: int = -1,
    lora: str = "none",
):
    output_dir = Path("/tmp/stable-diffusion")
    output_dir.mkdir(exist_ok=True, parents=True)

    inference_service = Inference()

    for sample_idx in range(samples):
        start = time.time()
        images_gen = inference_service.run.remote(
            prompt=prompt,
            JuggernautXL=JuggernautXL,
            CyberRealisticXL=CyberRealisticXL,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            scheduler=scheduler,
            sampler=sampler,
            amateur=amateur,
            seed=seed,
            lora=lora,
        )
        msgs = list(images_gen)
        final_msg = msgs[-1]
        if "error" in final_msg:
            print(f"Error: {final_msg['error']}")
            continue

        import base64
        images = [base64.b64decode(b64) for b64 in final_msg["image_b64"]]

        duration = time.time() - start
        print(f"Run {sample_idx + 1} took {duration:.3f}s")
        for batch_idx, image_bytes in enumerate(images):
            output_path = output_dir / f"output_{slugify(prompt)[:64]}_{str(sample_idx).zfill(2)}_{str(batch_idx).zfill(2)}.png"
            output_path.write_bytes(image_bytes)

def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")