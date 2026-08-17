"""
AlphaModal Batch Image-to-Image Editing (Qwen-Image-Edit-Plus)
================================================================
Processes an entire folder/batch of images using `Qwen/Qwen-Image-Edit-2511`
and optional LoRA adapters on a high-performance Modal GPU container.

Usage:
------
1. Modal Cloud Execution:
    modal run batch-img2img.py --input-dir ./downloads --output-dir ./edited_images --prompt "convert to cyberpunk synthwave aesthetic"

2. Python Direct Execution:
    python batch-img2img.py
"""

import argparse
import base64
import io
import os
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

# UTF-8 Encoding enforcement for Windows terminals
if sys.stdout and getattr(sys.stdout, 'encoding', '') != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and getattr(sys.stderr, 'encoding', '') != 'utf-8':
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# ─── CONFIGURATION (EDIT THESE SETTINGS) ──────────────────────────────────────
INPUT_DIR = "./cleaned_images"              # Path to folder containing images to edit
OUTPUT_DIR = "./training_images"          # Path where edited images will be saved
PROMPT = "photo- realistic, same image, just remove clothing from the girls and make them completly nude and exposed, nipples and vagina visible, accurate sized breasts and small vaginas, correct body porportions according to the original image, detailed nipples and vaginas."
NEGATIVE_PROMPT = "worst quality, low quality, censorship, text, watermark, signature, blur, bad anatomy, ugly, deformed, mature, aged, adult, clothes"

NUM_INFERENCE_STEPS = 30
TRUE_CFG_SCALE = 4.0
USE_LORA = True                        # Set True to load ScottzillaSystems LoRA, False for base Qwen only
SEED = -1                              # -1 for random seed (-1 = random)
GPU_TYPE = "A100-80GB"                 # Modal GPU hardware choice ("A100-80GB", "A10G", "L4")

# Model & LoRA Repositories
BASE_MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_MODEL_REPO_ID = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
LORA_WEIGHT_NAME = "qwen-image-edit-plus-nsfw-lora.safetensors"
LORA_ADAPTER_NAME = "mcnl-nsfw-v1"
# ──────────────────────────────────────────────────────────────────────────────



import modal

app = modal.App("batch-img2img-qwen")

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
        "einops", "timm", "transformers-stream-generator",
        "peft",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

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


# ─── MODAL GPU MODEL CLASS ───────────────────────────────────────────────────

@app.cls(image=image, gpu=GPU_TYPE, volumes=volumes, secrets=secrets, timeout=1800)
class BatchImg2ImgModel:
    @modal.enter()
    def enter(self):
        print(f"[*] Loading base model: {BASE_MODEL_NAME}...")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        ).to("cuda")

        if USE_LORA:
            print(f"[*] Loading LoRA adapter from {LORA_MODEL_REPO_ID}...")
            try:
                self.pipe.load_lora_weights(
                    LORA_MODEL_REPO_ID,
                    weight_name=LORA_WEIGHT_NAME,
                    adapter_name=LORA_ADAPTER_NAME,
                    cache_dir=CACHE_DIR,
                )
                self.pipe.set_adapters([LORA_ADAPTER_NAME])
                print("[✓] LoRA adapter set successfully.")
            except Exception as e:
                print(f"[!] Warning loading LoRA: {e}")


        print("[*] Disabling safety checker.")
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))


    @modal.method()
    def batch_process(
        self,
        image_bytes_dict: Dict[str, bytes],
        prompt: str,
        negative_prompt: str,
        true_cfg_scale: float,
        num_inference_steps: int,
        seed: int = -1,
    ) -> Dict[str, bytes]:
        """Processes a batch of images using a single prompt instruction."""
        results = {}
        total = len(image_bytes_dict)
        print(f"[*] Starting batch edit on {total} images...")
        print(f"    Prompt: '{prompt}'")
        print(f"    Steps: {num_inference_steps}, CFG: {true_cfg_scale}")

        generator = torch.Generator(device="cuda").manual_seed(seed) if seed != -1 else None

        out_dir = OUTPUTS_DIR / "batch_img2img"
        out_dir.mkdir(parents=True, exist_ok=True)



        for idx, (filename, img_bytes) in enumerate(image_bytes_dict.items(), start=1):
            print(f"[{idx}/{total}] Editing image: {filename}...")
            try:
                pil_img = load_image(Image.open(BytesIO(img_bytes))).convert("RGB")
                
                edited_images = self.pipe(
                    image=[pil_img],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=generator,
                ).images

                edited_img = edited_images[0]
                
                out_buffer = BytesIO()
                edited_img.save(out_buffer, format="PNG")
                out_bytes = out_buffer.getvalue()
                
                results[filename] = out_bytes

                # Persist copy to outputs volume
                save_filename = f"edited_{Path(filename).stem}_{int(time.time())}.png"
                (out_dir / save_filename).write_bytes(out_bytes)
            except Exception as err:
                print(f"[!] Error processing {filename}: {err}")

        outputs_volume.commit()
        print(f"[✓] Completed batch edit of {len(results)}/{total} images.")
        return results


# ─── LOCAL CLI ENTRYPOINT ────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    input_dir: str = "",
    output_dir: str = "",
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 0,
    cfg_scale: float = 0.0,
):
    """Modal CLI entrypoint: `modal run batch-img2img.py`"""
    in_path = Path(input_dir if input_dir else INPUT_DIR).resolve()
    out_path = Path(output_dir if output_dir else OUTPUT_DIR).resolve()
    target_prompt = prompt if prompt else PROMPT
    target_neg_prompt = negative_prompt if negative_prompt else NEGATIVE_PROMPT
    target_steps = steps if steps > 0 else NUM_INFERENCE_STEPS
    target_cfg = cfg_scale if cfg_scale > 0.0 else TRUE_CFG_SCALE

    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[!] Input directory does not exist: {in_path}")
        return

    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = [f for f in in_path.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

    if not image_files:
        print(f"[!] No valid images found in: {in_path}")
        return

    print(f"=== Starting Batch Image Editing on Modal Cloud ===")
    print(f"Input Directory: {in_path} ({len(image_files)} images)")
    print(f"Output Directory: {out_path}")
    print(f"Prompt Instruction: '{target_prompt}'")
    print(f"Inference Steps: {target_steps}, CFG Scale: {target_cfg}\n")

    # Package input images into byte payload
    payload = {}
    for f in image_files:
        rel_name = str(f.relative_to(in_path))
        payload[rel_name] = f.read_bytes()

    print("[*] Submitting job to Modal GPU (A100-80GB)...")
    results = BatchImg2ImgModel().batch_process.remote(
        image_bytes_dict=payload,
        prompt=target_prompt,
        negative_prompt=target_neg_prompt,
        true_cfg_scale=target_cfg,
        num_inference_steps=target_steps,
        seed=SEED,
    )


    for rel_name, out_bytes in results.items():
        stem = Path(rel_name).stem
        ext = Path(rel_name).suffix
        dest_filename = f"{stem}_edited.png"
        dest = out_path / Path(rel_name).parent / dest_filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(out_bytes)
        print(f"[✓] Saved edited image to: {dest}")

    print(f"\n==========================================")
    print(f"[✓] Batch processing complete! Saved {len(results)} images in: {out_path}")


def run_local():
    """Local Python script trigger."""
    parser = argparse.ArgumentParser(description="Batch Image-to-Image Editing (Qwen-Image-Edit-Plus)")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR, help=f"Input folder (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help=f"Output folder (default: {OUTPUT_DIR})")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Prompt editing instruction")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=TRUE_CFG_SCALE, help="True CFG scale")

    args = parser.parse_args()
    print("[*] Running Batch Image-to-Image via Modal local entrypoint...")
    # Invoke main with parsed args
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        steps=args.steps,
        cfg_scale=args.cfg,
    )


if __name__ == "__main__":
    if not any("modal" in arg for arg in sys.argv[:1]) and "MODAL_RUN_ENTRYPOINT" not in os.environ:
        run_local()
