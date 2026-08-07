import modal
import os
import shutil

app = modal.App("sdxl-lora-training")
outputs_vol = modal.Volume.from_name("images")

# Setting up the environment for kohya-ss/sd-scripts
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

@app.function(
    image=image, 
    gpu="A100", # Using A100 for SDXL to ensure we have enough VRAM and fast training
    timeout=36000, 
    volumes={"/images": outputs_vol}
)
def train(project_name: str = "my_sdxl_lora", repeats: int = 15, epochs: int = 10):
    import subprocess
    
    # 1. Prepare Kohya dataset structure
    # Kohya expects a folder structure like: /data/img/<repeats>_<concept>/
    dataset_dir = "/tmp/lora_dataset"
    img_dir = os.path.join(dataset_dir, "img", f"{repeats}_concept")
    os.makedirs(img_dir, exist_ok=True)
    
    print("Copying dataset from volume to fast local /tmp storage...")
    image_count = 0
    for file in os.listdir("/images"):
        if file.endswith((".png", ".jpg", ".txt")):
            shutil.copy(os.path.join("/images", file), img_dir)
            if file.endswith(".png") or file.endswith(".jpg"):
                image_count += 1
                
    print(f"Prepared {image_count} images for training.")
            
    output_dir = f"/images/trained_loras/{project_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Run the Kohya training script
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process=2",
        "/sd-scripts/sdxl_train_network.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        f"--train_data_dir={os.path.join(dataset_dir, 'img')}",
        f"--output_dir={output_dir}",
        f"--output_name={project_name}",
        "--resolution=1024,1024",
        "--train_batch_size=4",
        "--save_model_as=safetensors",
        "--network_module=networks.lora",
        "--network_train_unet_only",
        "--caption_extension=.txt",
        "--text_encoder_lr=4e-5",
        "--unet_lr=1e-4",
        "--network_dim=32",
        "--network_alpha=16",
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
    
    print("Starting training! This may take a while depending on epochs...")
    subprocess.run(cmd, check=True)
    
    print("Training complete! Syncing volume...")
    outputs_vol.commit()
    print(f"LoRA saved to volume at /images/trained_loras/{project_name}")

@app.local_entrypoint()
def main():
    print("Kicking off SDXL LoRA training on Modal...")
    # You can tweak the project name, repeats per image, and epochs here
    train.remote(project_name="my_sdxl_lora", repeats=2, epochs=10)
