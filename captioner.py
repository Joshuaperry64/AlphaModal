"""
AlphaModal Florence-2 Batch Image Captioner
==========================================
Uses Microsoft's `Florence-2-large` vision-language model to generate detailed
dataset captions and `.txt` sidecar files for image training/datasets.

Features:
- Web UI: Serve interactive Gradio app via `modal serve captioner.py`
- Batch Cloud Run: Process images via `modal run captioner.py` or `python captioner.py`
- Prominent configuration header for quick edits.
"""

import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

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
INPUT_DIR = "./training_images"                # Folder containing images to caption
OUTPUT_DIR = "./training_images"               # Target output folder for .txt sidecars
TRIGGER_WORD = ""                        # Optional trigger word prefix (e.g. "sks,")
CAPTION_TASK = "<MORE_DETAILED_CAPTION>" # Choices: "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"
GPU_TYPE = "L4"                          # Modal GPU choice: "L4", "a10g", or "a100"
OVERWRITE_EXISTING = True               # True to overwrite existing .txt files
# ──────────────────────────────────────────────────────────────────────────────

MODEL_ID = "microsoft/Florence-2-large"

import modal

app = modal.App("florence2-batch-captioner")

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID)

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "transformers==4.44.0",
        "torch>=2.2.0",
        "torchvision",
        "timm",
        "einops",
        "pillow",
        "gradio",
    )
    .run_function(download_model)
)

with modal_image.imports():
    import gradio as gr
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    from PIL import Image
    from transformers.dynamic_module_utils import get_imports

    def fixed_get_imports(filename):
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

outputs_vol = modal.Volume.from_name("images", create_if_missing=True)


# ─── CORE MODEL LOADER ────────────────────────────────────────────────────────

def load_florence_model():
    print("[*] Loading Florence-2 model into GPU...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            local_files_only=True,
        )
    print("[✓] Florence-2 model loaded.")
    return model, processor


# ─── BATCH REMOTE CAPTIONING FUNCTION ─────────────────────────────────────────

@app.function(
    image=modal_image,
    gpu=GPU_TYPE,
    timeout=1800,
    volumes={"/images": outputs_vol},
)
def batch_caption_remote(
    image_bytes_dict: Dict[str, bytes],
    task_prompt: str = CAPTION_TASK,
    trigger_word: str = TRIGGER_WORD,
    overwrite: bool = OVERWRITE_EXISTING,
) -> Dict[str, str]:
    """Remote Modal task: process dictionary of {rel_path: img_bytes} and return {rel_path: caption_text}."""
    model, processor = load_florence_model()
    results = {}
    total = len(image_bytes_dict)
    print(f"[*] Batch captioning {total} images on Modal GPU...")

    for idx, (filename, img_bytes) in enumerate(image_bytes_dict.items(), start=1):
        print(f"[{idx}/{total}] Captioning: {filename}...")
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            inputs = processor(text=task_prompt, images=img, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(img.width, img.height)
            )
            caption = parsed_answer[task_prompt]

            if trigger_word and trigger_word.strip():
                tw = trigger_word.strip()
                if not tw.endswith(","):
                    tw += ","
                caption = f"{tw} {caption}"

            results[filename] = caption.strip()
        except Exception as err:
            print(f"[!] Error captioning {filename}: {err}")

    return results


# ─── GRADIO WEB UI (MODAL SERVE) ─────────────────────────────────────────────

@app.function(image=modal_image, gpu=GPU_TYPE, timeout=3600, volumes={"/images": outputs_vol})
@modal.web_server(8000, startup_timeout=300)
def ui():
    model, processor = load_florence_model()

    def process_volume(task_prompt, trigger_word, progress=gr.Progress()):
        vol_path = Path("/images")
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            images.extend(vol_path.rglob(f"*{ext}"))
            images.extend(vol_path.rglob(f"*{ext.upper()}"))
            
        if not images:
            return "No images found in the 'images' Modal Volume."
            
        success_count = 0
        for i, img_path in enumerate(progress.tqdm(images, desc="Captioning Images")):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists() and not OVERWRITE_EXISTING:
                continue
                
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(text=task_prompt, images=img, return_tensors="pt")
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3
                    )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                parsed_answer = processor.post_process_generation(
                    generated_text, task=task_prompt, image_size=(img.width, img.height)
                )
                caption = parsed_answer[task_prompt]
                
                if trigger_word and trigger_word.strip():
                    tw = trigger_word.strip()
                    if not tw.endswith(","): tw += ","
                    caption = f"{tw} {caption}"
                    
                txt_path.write_text(caption.strip(), encoding="utf-8")
                success_count += 1
                
                if success_count > 0 and success_count % 50 == 0:
                    outputs_vol.commit()
            except Exception as e:
                print(f"Error on {img_path.name}: {e}")
                
        outputs_vol.commit()
        return f"Successfully processed and generated captions for {success_count} new images in the volume!"

    with gr.Blocks(title="Florence-2 Batch Captioner") as demo:
        gr.Markdown("# 🖼️ Florence-2 Batch Captioner")
        gr.Markdown("Scan and generate dataset `.txt` sidecars directly in the Modal volume or batch process folders.")
        
        with gr.Row():
            with gr.Column():
                task_prompt = gr.Dropdown(
                    choices=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                    value=CAPTION_TASK,
                    label="Detail Level"
                )
                trigger_word = gr.Textbox(label="Trigger Word (optional)", value=TRIGGER_WORD, placeholder="e.g. sks,")
                submit_btn = gr.Button("Start Captioning Volume", variant="primary")
            
            with gr.Column():
                status_text = gr.Textbox(label="Status", interactive=False)
                
        submit_btn.click(
            fn=process_volume,
            inputs=[task_prompt, trigger_word],
            outputs=[status_text]
        )
        
    demo.queue().launch(server_name="0.0.0.0", server_port=8000, share=True)


# ─── LOCAL / MODAL CLI ENTRYPOINTS ────────────────────────────────────────────

@app.local_entrypoint()
def main(input_dir: str = "", output_dir: str = "", trigger_word: str = "", task_prompt: str = ""):
    """Modal CLI entrypoint: `modal run captioner.py`"""
    in_path = Path(input_dir if input_dir else INPUT_DIR).resolve()
    out_path = Path(output_dir if output_dir else OUTPUT_DIR).resolve()
    target_trigger = trigger_word if trigger_word else TRIGGER_WORD
    target_task = task_prompt if task_prompt else CAPTION_TASK
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[!] Input directory does not exist: {in_path}")
        return

    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = [f for f in in_path.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

    if not image_files:
        print(f"[!] No valid images found in: {in_path}")
        return

    print(f"=== Starting Florence-2 Batch Captioner on Modal Cloud ===")
    print(f"Input Directory: {in_path} ({len(image_files)} images)")
    print(f"Output Directory: {out_path}")
    print(f"Caption Task: {target_task}")
    print(f"Trigger Word: '{target_trigger}'\n")

    payload = {}
    for f in image_files:
        txt_target = (out_path / f.relative_to(in_path)).with_suffix(".txt")
        if txt_target.exists() and not OVERWRITE_EXISTING:
            continue
        rel_name = str(f.relative_to(in_path))
        payload[rel_name] = f.read_bytes()

    if not payload:
        print("[✓] All images already have existing .txt caption sidecars!")
        return

    print(f"[*] Submitting {len(payload)} images to Modal GPU ({GPU_TYPE})...")
    captions = batch_caption_remote.remote(
        image_bytes_dict=payload,
        task_prompt=target_task,
        trigger_word=target_trigger,
        overwrite=OVERWRITE_EXISTING,
    )

    for rel_name, caption_text in captions.items():
        img_rel = Path(rel_name)
        txt_dest = (out_path / img_rel).with_suffix(".txt")
        txt_dest.parent.mkdir(parents=True, exist_ok=True)
        txt_dest.write_text(caption_text, encoding="utf-8")
        print(f"[✓] Saved caption: {txt_dest}")

    print(f"\n==========================================")
    print(f"[✓] Completed! Generated {len(captions)} caption .txt sidecars in: {out_path}")


def run_local():
    """Python direct execution trigger."""
    import argparse
    parser = argparse.ArgumentParser(description="Florence-2 Batch Captioner")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR, help=f"Input directory (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--trigger-word", type=str, default=TRIGGER_WORD, help=f"Trigger word prefix (default: '{TRIGGER_WORD}')")
    parser.add_argument("--task-prompt", type=str, default=CAPTION_TASK, help=f"Florence-2 task prompt (default: '{CAPTION_TASK}')")

    args = parser.parse_args()
    import subprocess
    cmd = [
        sys.executable, "-m", "modal", "run", "captioner.py",
        "--input-dir", args.input_dir,
        "--output-dir", args.output_dir,
        "--trigger-word", args.trigger_word,
        "--task-prompt", args.task_prompt,
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    if not any("modal" in arg for arg in sys.argv[:1]) and "MODAL_RUN_ENTRYPOINT" not in os.environ:
        run_local()
