"""
AlphaModal SDXL LoRA Trainer (Kohya-ss / sd-scripts)
=====================================================
Fine-tunes Stable Diffusion XL (SDXL 1.0) LoRA models on Modal cloud GPUs using
the industry-standard Kohya-ss `sdxl_train_network.py` pipeline.

Features:
- Cloud A100 GPU acceleration with high VRAM & bfloat16 mixed precision.
- Automatic dataset upload from local folder (`LOCAL_DATASET_DIR`).
- Prominent configuration block for hyperparameter tuning.
- Saves trained LoRA `.safetensors` files back to local folder & Modal volume.
"""

import argparse
import io
import os
import shutil
import sys
from pathlib import Path
from typing import Dict

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
PROJECT_NAME = "preload"            # Name of the output LoRA model file
LOCAL_DATASET_DIR = "./training_images"         # Local dataset folder containing images and .txt captions
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
REPEATS = 15                              # Kohya repeat count per image
EPOCHS = 10                               # Total training epochs
BATCH_SIZE = 4                            # Training batch size
NETWORK_DIM = 32                          # LoRA rank / dim (e.g. 16, 32, 64)
NETWORK_ALPHA = 16                        # LoRA alpha (typically dim/2)
UNET_LR = "1e-4"                          # UNet learning rate
TEXT_ENCODER_LR = "4e-5"                  # Text encoder learning rate
RESOLUTION = "768,768"                  # Training image resolution
GPU_TYPE = "A100"                         # Modal GPU choice: "A100", "H100", or "A10G"
# ──────────────────────────────────────────────────────────────────────────────

import modal

app = modal.App("sdxl-lora-training")
outputs_vol = modal.Volume.from_name("images", create_if_missing=True)

# Environment setup for kohya-ss/sd-scripts
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/kohya-ss/sd-scripts.git /sd-scripts",
        "cd /sd-scripts && pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118",
        "cd /sd-scripts && pip install --upgrade -r requirements.txt",
        "cd /sd-scripts && pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118",
        "pip install bitsandbytes"
    )
)

# ─── MODAL REMOTE TRAINING FUNCTION ─────────────────────────────────────────

@app.function(
    image=image, 
    gpu=GPU_TYPE,
    timeout=36000, 
    volumes={"/images": outputs_vol}
)
def train_remote(
    dataset_files: Dict[str, bytes],
    project_name: str = PROJECT_NAME,
    repeats: int = REPEATS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    network_dim: int = NETWORK_DIM,
    network_alpha: int = NETWORK_ALPHA,
    unet_lr: str = UNET_LR,
    text_encoder_lr: str = TEXT_ENCODER_LR,
) -> bytes:
    """Remote Modal task to execute Kohya-ss SDXL LoRA training."""
    import subprocess
    
    # 1. Prepare Kohya dataset structure
    dataset_dir = "/tmp/lora_dataset"
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        
    img_dir = os.path.join(dataset_dir, "img", f"{repeats}_{project_name}")
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"[*] Extracting dataset ({len(dataset_files)} files) to fast local storage...")
    image_count = 0
    for filename, file_bytes in dataset_files.items():
        filepath = os.path.join(img_dir, os.path.basename(filename))
        with open(filepath, "wb") as f:
            f.write(file_bytes)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_count += 1
                
    print(f"[✓] Prepared {image_count} training images ({len(dataset_files)} total dataset files).")
            
    output_dir = f"/images/trained_loras/{project_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Run the Kohya SDXL training script
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process=2",
        "/sd-scripts/sdxl_train_network.py",
        f"--pretrained_model_name_or_path={MODEL_BASE}",
        f"--train_data_dir={os.path.join(dataset_dir, 'img')}",
        f"--output_dir={output_dir}",
        f"--output_name={project_name}",
        f"--resolution={RESOLUTION}",
        f"--train_batch_size={batch_size}",
        "--save_model_as=safetensors",
        "--network_module=networks.lora",
        "--network_train_unet_only",
        "--caption_extension=.txt",
        f"--text_encoder_lr={text_encoder_lr}",
        f"--unet_lr={unet_lr}",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--optimizer_type=Adafactor",
        "--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False",
        "--lr_scheduler=constant_with_warmup",
        "--lr_warmup_steps=100",
        f"--max_train_epochs={epochs}",
        "--mixed_precision=bf16",
        "--save_precision=bf16",
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs"
    ]
    
    print(f"[*] Starting SDXL LoRA training for {epochs} epochs...")
    subprocess.run(cmd, check=True)
    
    print("[*] Training complete! Syncing volume...")
    outputs_vol.commit()
    
    lora_file_path = os.path.join(output_dir, f"{project_name}.safetensors")
    print(f"[✓] Saved LoRA to volume at {lora_file_path}")

    if os.path.exists(lora_file_path):
        with open(lora_file_path, "rb") as f:
            return f.read()
    return b""


# ─── LOCAL / MODAL CLI ENTRYPOINTS ────────────────────────────────────────────

@app.local_entrypoint()
def main(
    project_name: str = "",
    dataset_dir: str = "",
    epochs: int = 0,
    repeats: int = 0,
):
    """Modal CLI entrypoint: `modal run train_sdxl.py`"""
    proj_name = project_name if project_name else PROJECT_NAME
    data_path = Path(dataset_dir if dataset_dir else LOCAL_DATASET_DIR).resolve()
    num_epochs = epochs if epochs > 0 else EPOCHS
    num_repeats = repeats if repeats > 0 else REPEATS

    if not data_path.exists():
        print(f"[!] Dataset directory does not exist: {data_path}")
        return

    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt"}
    files = [f for f in data_path.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

    if not files:
        print(f"[!] No training images or .txt captions found in: {data_path}")
        return

    print(f"=== Kicking off SDXL LoRA Training on Modal Cloud ({GPU_TYPE} GPU) ===")
    print(f"Project Name: {proj_name}")
    print(f"Dataset Path: {data_path} ({len(files)} files)")
    print(f"Repeats per image: {num_repeats}, Epochs: {num_epochs}")
    print(f"Rank/Dim: {NETWORK_DIM}, Alpha: {NETWORK_ALPHA}\n")

    payload = {f.name: f.read_bytes() for f in files}

    lora_bytes = train_remote.remote(
        dataset_files=payload,
        project_name=proj_name,
        repeats=num_repeats,
        epochs=num_epochs,
        batch_size=BATCH_SIZE,
        network_dim=NETWORK_DIM,
        network_alpha=NETWORK_ALPHA,
        unet_lr=UNET_LR,
        text_encoder_lr=TEXT_ENCODER_LR,
    )

    if lora_bytes:
        local_output_path = Path(f"./{proj_name}.safetensors").resolve()
        local_output_path.write_bytes(lora_bytes)
        print(f"[✓] Saved local LoRA safetensors to: {local_output_path}")


def run_local():
    """Python direct trigger: dispatches execution to Modal Cloud GPU."""
    parser = argparse.ArgumentParser(description="SDXL LoRA Trainer (Modal Cloud GPU)")
    parser.add_argument("--project-name", type=str, default=PROJECT_NAME, help=f"Project name (default: {PROJECT_NAME})")
    parser.add_argument("--dataset-dir", type=str, default=LOCAL_DATASET_DIR, help=f"Dataset path (default: {LOCAL_DATASET_DIR})")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Epoch count (default: {EPOCHS})")
    parser.add_argument("--repeats", type=int, default=REPEATS, help=f"Repeat count (default: {REPEATS})")

    args = parser.parse_args()
    main(
        project_name=args.project_name,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    if not any("modal" in arg for arg in sys.argv[:1]) and "MODAL_RUN_ENTRYPOINT" not in os.environ:
        run_local()
