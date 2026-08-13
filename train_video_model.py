import modal
import os

# Create a Modal App for training
app = modal.App("train-video-model")

# Volumes for storing our dataset and the resulting LoRA weights
dataset_volume = modal.Volume.from_name("video-dataset", create_if_missing=True)
output_volume = modal.Volume.from_name("lora-outputs", create_if_missing=True)

# Define the image with all the crazy dependencies needed for Video LoRA training
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "diffusers>=0.30.0",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "decord",
        "imageio",
        "imageio-ffmpeg",
        "wandb",
        "sentencepiece",
        "tiktoken",
        "numpy<2.0.0"  # Training scripts can sometimes be picky about numpy
    )
    # Clone the diffusers repo to get the official training script
    .run_commands(
        "git clone https://github.com/huggingface/diffusers.git /diffusers",
        "cd /diffusers && pip install -e .",
        "pip install -r /diffusers/examples/cogvideo/requirements.txt"
    )
)

# We need a massive GPU for video training. Using an A100-80GB.
@app.function(
    image=image,
    gpu="A100",  # Training video models requires massive VRAM
    volumes={
        "/data": dataset_volume,
        "/outputs": output_volume
    },
    timeout=60 * 60 * 12, # Allow up to 12 hours for training
)
def train_lora(model_id="THUDM/CogVideoX-2b", prompt="your trigger word", max_train_steps=1000):
    import subprocess
    import os
    
    print(f"Starting LoRA training on {model_id}...")
    
    # The official CogVideoX training script path inside the cloned repo
    script_path = "/diffusers/examples/cogvideo/train_cogvideox_lora.py"
    
    # We set up accelerate config to use the GPU efficiently
    accelerate_cmd = [
        "accelerate", "launch",
        "--mixed_precision=bf16",
        script_path,
        f"--pretrained_model_name_or_path={model_id}",
        "--dataset_name=/data",
        "--dataset_config_name=default",
        "--dataloader_num_workers=4",
        "--resolution=720",
        "--fps=8",
        f"--max_train_steps={max_train_steps}",
        "--learning_rate=1e-4",
        "--lr_scheduler=cosine_with_restarts",
        "--lr_warmup_steps=100",
        "--mixed_precision=bf16",
        "--output_dir=/outputs/my-video-lora",
        "--checkpointing_steps=250",
        "--seed=42"
    ]
    
    # Run the training process
    subprocess.run(accelerate_cmd, check=True)
    
    print("Training complete! Your LoRA weights are saved in the /outputs volume.")
    output_volume.commit()

@app.local_entrypoint()
def main():
    print("Welcome to the Video LoRA Trainer!")
    print("Before running this, you need to upload your videos and a metadata.jsonl file to the 'video-dataset' Modal volume.")
    print("Example command to upload a folder:")
    print("  modal volume put video-dataset C:\\Path\\To\\My\\Videos /")
    print("\nStarting the training run on Modal's A100 GPUs...")
    
    # Kick off the remote training function
    # Note: We use CogVideoX-2b by default for training because 5B usually requires multiple GPUs to train!
    train_lora.remote(model_id="THUDM/CogVideoX-2b", max_train_steps=500)
