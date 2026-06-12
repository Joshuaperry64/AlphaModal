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
    from diffusers import StableDiffusionXLPipeline  # FIXED: Explicitly use SDXL pipeline
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

        # FIXED: Use StableDiffusionXLPipeline instead of AutoPipeline
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
        seed: int = -1
    ) -> list[bytes]:
        
        model_1_file = "juggernautXL_ragnarokBy.safetensors"
        model_2_file = "cyberrealisticXL_desireV30.safetensors"
        
        if JuggernautXL == 1 and CyberRealisticXL == 1:
            self._merge_and_load(model_1_file, model_2_file, "merged_jugg_cyber.safetensors")
        elif CyberRealisticXL == 1:
            self._load_model(model_2_file)
        else:
            self._load_model(model_1_file)

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

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_images_per_prompt=batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=g_scale_float,
        ).images

        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        torch.cuda.empty_cache()
        return image_output

    @modal.fastapi_endpoint(docs=True)
    def web(
        self, 
        prompt: str, 
        JuggernautXL: int = 1,
        CyberRealisticXL: int = 0,
        negative_prompt: str = "low quality, blurry, distorted", 
        guidance_scale: str = "7.0", 
        num_inference_steps: int = 25, 
        scheduler: str = "Euler", 
        seed: int = -1
    ):
        return Response(
            content=self.run.local(
                prompt=prompt,
                JuggernautXL=JuggernautXL,
                CyberRealisticXL=CyberRealisticXL,
                negative_prompt=negative_prompt,
                batch_size=1,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                scheduler=scheduler,
                seed=seed,
            )[0],
            media_type="image/png",
        )


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
        )
        duration = time.time() - start
        print(f"Run {sample_idx + 1} took {duration:.3f}s")
        for batch_idx, image_bytes in enumerate(images):
            output_path = output_dir / f"output_{slugify(prompt)[:64]}_{str(sample_idx).zfill(2)}_{str(batch_idx).zfill(2)}.png"
            output_path.write_bytes(image_bytes)

def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")