import os
import sys
import io
import time
import warnings
from pathlib import Path
from io import BytesIO

# Suppress annoying Gradio Blocks constructor warnings (since Modal handles launch() internally)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Blocks constructor.*")

import modal

# Force UTF-8 encoding for stdout and stderr to prevent encoding crashes on Windows console
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Define the Modal App
app = modal.App("multi-image-llm-editor")

# CUDA-enabled environment with all Qwen & Diffusers dependencies + Gradio
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
        "einops", "timm", "transformers-stream-generator", # Qwen requirements
        "peft", # LoRA requirements
        "gradio",
        "python-multipart",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

BASE_MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_MODEL_REPO_ID = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
LORA_WEIGHT_NAME = "qwen-image-edit-plus-nsfw-lora.safetensors"
LORA_ADAPTER_NAME = "mcnl-nsfw-v1"

CACHE_DIR = "/hf-hub-cache"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("outputs", create_if_missing=True)
images_volume = modal.Volume.from_name("images", create_if_missing=True)

volumes = {
    CACHE_DIR: cache_volume,
    "/outputs": outputs_volume,
    "/images": images_volume,
}

secrets = [modal.Secret.from_name("huggingface-secret")]

image = image.env({
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HOME": CACHE_DIR,
    "FORCE_REBUILD": "2"
})

with image.imports():
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.utils import load_image
    from PIL import Image
    import gradio as gr

@app.cls(image=image, gpu="A100-80GB", volumes=volumes, secrets=secrets, timeout=3600)
class Model:
    @modal.enter()
    def enter(self):
        print(f"Loading base Qwen-Image-Edit-2511 model...")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        ).to("cuda")

        print(f"Preloading default LoRA adapter from {LORA_MODEL_REPO_ID}...")
        try:
            self.pipe.load_lora_weights(
                LORA_MODEL_REPO_ID,
                weight_name=LORA_WEIGHT_NAME,
                adapter_name=LORA_ADAPTER_NAME,
                cache_dir=CACHE_DIR,
            )
            self.pipe.set_adapters([LORA_ADAPTER_NAME])
            print("LoRA adapter loaded successfully.")
        except Exception as e:
            print(f"Could not load pre-loaded LoRA: {e}")

        # Disable safety checker
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    @modal.method()
    def edit_images(
        self,
        image_list: list[bytes],
        prompt: str,
        negative_prompt: str,
        true_cfg_scale: float,
        num_inference_steps: int,
        lora_name: str = "none",
        seed: int = -1,
        save_volume: str = "images"
    ) -> tuple[bytes, str]:
        # Convert raw bytes back to PIL images
        pil_images = []
        for idx, img_bytes in enumerate(image_list):
            if img_bytes:
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                pil_images.append(pil_img)
        
        if not pil_images:
            raise ValueError("No valid input images were provided.")

        generator = torch.Generator(device="cuda").manual_seed(seed) if seed != -1 else None

        # Handle custom LoRA loading
        if lora_name and lora_name != "none":
            loras_to_load = [l.strip() for l in lora_name.split(",") if l.strip() and l.strip() != "none"]
            loaded_adapters = []
            for l in loras_to_load:
                lora_path = Path("/hf-hub-cache/loras") / f"{l}.safetensors"
                if lora_path.exists():
                    print(f"Loading Custom LoRA: {lora_path}")
                    self.pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name, adapter_name=l)
                    loaded_adapters.append(l)
                else:
                    if "/" in l:
                        try:
                            print(f"Attempting to download and load LoRA from repo: {l}")
                            self.pipe.load_lora_weights(l, cache_dir=CACHE_DIR, adapter_name=l.split("/")[-1])
                            loaded_adapters.append(l.split("/")[-1])
                        except Exception as e:
                            print(f"Failed to load LoRA from repo {l}: {e}")
                    else:
                        print(f"LoRA {lora_path} not found locally.")
            if loaded_adapters:
                self.pipe.set_adapters(loaded_adapters)
        else:
            try:
                self.pipe.set_adapters([LORA_ADAPTER_NAME])
            except Exception:
                pass

        print(f"Running inference with prompt: '{prompt}' and {len(pil_images)} images.")
        
        # Run Qwen Edit Pipeline
        output_images = self.pipe(
            image=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        ).images

        # Unload custom LoRAs if they were loaded
        if lora_name and lora_name != "none":
            self.pipe.unload_lora_weights()

        output_image = output_images[0]
        
        # Save output to volume
        timestamp = int(time.time() * 1000)
        filename = f"multi_edit_{timestamp}.png"
        
        if save_volume == "outputs":
            out_dir = Path("/outputs/multi_img2img")
            active_vol = outputs_volume
        else:
            out_dir = Path("/images/multi_img2img")
            active_vol = images_volume
            
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / filename
        output_image.save(file_path, format="PNG")
        
        print(f"Saving output to volume {save_volume} at {file_path}")
        active_vol.commit()

        # Return output as bytes
        with BytesIO() as buf:
            output_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            
        return img_bytes, filename

@app.function(image=image, volumes=volumes, timeout=3600)
@modal.web_server(8000, startup_timeout=300)
def ui():
    # Instantiate Model client
    model_client = Model()

    def process(img1, img2, img3, prompt, negative_prompt, cfg, steps, lora, seed, save_vol):
        image_list = []
        for img in [img1, img2, img3]:
            if img is not None:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                with BytesIO() as buf:
                    img.save(buf, format="PNG")
                    image_list.append(buf.getvalue())
        
        if not image_list:
            raise gr.Error("Please upload at least one image to edit.")

        print("Calling inference on Modal Model class...")
        output_bytes, filename = model_client.edit_images.remote(
            image_list=image_list,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            lora_name=lora,
            seed=seed,
            save_volume=save_vol
        )
        
        output_pil = Image.open(BytesIO(output_bytes))
        status_msg = f"✨ Done! Generated image saved to /{save_vol}/multi_img2img/{filename}"
        return output_pil, status_msg

    # Helper function to scan a volume for existing images
    def scan_volume_images(volume_name):
        path = Path(f"/{volume_name}")
        if not path.exists():
            return [], []
            
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        file_paths = []
        
        try:
            # Non-recursively scan the root
            for p in path.iterdir():
                if p.is_file() and p.suffix.lower() in valid_exts:
                    file_paths.append(p)
            
            # Non-recursively scan multi_img2img subfolder if it exists
            subfolder = path / "multi_img2img"
            if subfolder.exists():
                for p in subfolder.iterdir():
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        file_paths.append(p)
            
            # Sort the files by modification time (newest first)
            file_paths.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
            
            # Limit to 100 for UI performance
            result_paths = [str(p) for p in file_paths[:100]]
            return result_paths, result_paths
        except Exception as e:
            print(f"Error scanning volume {volume_name}: {e}")
            return [], []

    # Helper to load a selected image path into the editor input slots
    def load_selected_image(selected_file):
        if not selected_file:
            raise gr.Error("Please click and select an image from the gallery first.")
        try:
            return Image.open(selected_file)
        except Exception as e:
            raise gr.Error(f"Failed to load image: {e}")

    def get_select_path(evt: gr.SelectData, paths):
        if evt.index < len(paths):
            return paths[evt.index]
        return ""

    # CSS for premium dark-theme glassmorphism styling
    custom_css = """
    body { background-color: #0b0f19; color: #f3f4f6; font-family: 'Outfit', sans-serif; }
    .gradio-container { background-color: #0b0f19; border: none; }
    .glass-card { 
        background: rgba(17, 24, 39, 0.7); 
        backdrop-filter: blur(12px); 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        border-radius: 16px; 
        padding: 20px;
    }
    h1 { 
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #f43f5e 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800; 
        font-size: 2.2rem;
        text-align: center;
        margin-bottom: 5px;
    }
    .submit-btn {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        transition: transform 0.2s ease, filter 0.2s ease !important;
    }
    .submit-btn:hover {
        transform: translateY(-2px);
        filter: brightness(1.1);
    }
    .action-btn {
        background-color: rgba(99, 102, 241, 0.2) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        color: #e0e7ff !important;
        transition: background-color 0.2s ease !important;
    }
    .action-btn:hover {
        background-color: rgba(99, 102, 241, 0.4) !important;
    }
    """

    theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b0f19",
        block_background_fill="rgba(17, 24, 39, 0.7)",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.08)",
        block_title_text_color="#a78bfa",
    )

    with gr.Blocks(theme=theme, css=custom_css, title="AlphaCore Multi-Image LLM Editor") as demo:
        gr.Markdown("<h1>🖼️ Multi-Image LLM Editor (Qwen-Image-Edit-2511)</h1>")
        gr.Markdown(
            "<p style='text-align: center; color: #9ca3af; font-size: 1.1rem; margin-bottom: 20px;'>"
            "Upload images, or select from your Modal Volumes, write a semantic instruction referring to them as <b>image 1</b>, <b>image 2</b>, etc., "
            "and watch the vision-language model combine them!"
            "</p>"
        )

        # State storage for selection mechanism
        selected_file_path = gr.State("")
        gallery_paths_store = gr.State([])

        with gr.Tabs():
            # Creation Tab
            with gr.Tab("🎨 Studio & Generation"):
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="glass-card"):
                        gr.Markdown("### 📂 Input Images")
                        with gr.Row():
                            img1 = gr.Image(label="🖼️ Image 1 (Primary)", type="pil")
                            img2 = gr.Image(label="🖼️ Image 2 (Secondary)", type="pil")
                            img3 = gr.Image(label="🖼️ Image 3 (Optional)", type="pil")

                        gr.Markdown("### ✍️ Editing Instructions")
                        prompt = gr.Textbox(
                            label="Instruction Prompt",
                            placeholder="e.g., 'Take the face from image 1 and put it on the person in image 2'",
                            value="Place the person from image 1 into the background scene of image 2.",
                            lines=3
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="worst quality, low quality, censorship, text, watermark, signature, blur, bad anatomy, ugly, deformed",
                            lines=2
                        )
                        
                        with gr.Row():
                            cfg = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=4.0, label="True CFG Scale")
                            steps = gr.Slider(minimum=10, maximum=50, step=1, value=20, label="Steps")
                            seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                        
                        with gr.Row():
                            lora = gr.Textbox(
                                label="LoRA (optional)",
                                placeholder="e.g., my_trained_lora (looks in /hf-hub-cache/loras)",
                                value="none"
                            )
                            save_vol = gr.Dropdown(
                                choices=["images", "outputs"],
                                value="images",
                                label="Save Output Volume"
                            )

                        submit_btn = gr.Button("🚀 Generate Image", elem_classes="submit-btn")

                    with gr.Column(scale=2, elem_classes="glass-card"):
                        gr.Markdown("### 🎯 Result")
                        output_img = gr.Image(label="Generated Image", interactive=False)
                        status = gr.Textbox(label="Status & Save Location", interactive=False)
                        
                        gr.Markdown(
                            "#### 💡 How to use multiple images in prompts:\n"
                            "- **Reference them by index**: Use exactly the words `image 1`, `image 2`, and `image 3` in your prompt.\n"
                            "- *Example 1*: \"The cat from **image 1** should be sitting on the sofa in **image 2**.\"\n"
                            "- *Example 2*: \"Combine **image 1** and **image 2** in a split layout, but styled like **image 3**.\"\n"
                            "- *Example 3*: \"Make the car from **image 2** red, and place it in front of the building in **image 1**.\"\n"
                            "- Keep instructions clear, direct, and specify which details transfer from which image."
                        )

            # Volume Browser Tab
            with gr.Tab("📦 Volume Explorer"):
                gr.Markdown("### Browse, view, and select images directly from your mounted Modal Volumes")
                
                with gr.Row():
                    vol_select = gr.Dropdown(
                        choices=["images", "outputs"],
                        value="images",
                        label="Select Modal Volume to Browse"
                    )
                    refresh_btn = gr.Button("🔄 Scan Volume", elem_classes="action-btn")
                
                with gr.Row():
                    gallery = gr.Gallery(
                        label="Volume Gallery (Click an image to select)",
                        columns=6,
                        rows=3,
                        object_fit="contain",
                        height="450px"
                    )
                
                with gr.Row():
                    set_img1_btn = gr.Button("👈 Set Selected as Image 1", elem_classes="action-btn")
                    set_img2_btn = gr.Button("👈 Set Selected as Image 2", elem_classes="action-btn")
                    set_img3_btn = gr.Button("👈 Set Selected as Image 3", elem_classes="action-btn")

        # --- INTERACTION EVENT LISTENERS ---
        
        # Generation click
        submit_btn.click(
            fn=process,
            inputs=[img1, img2, img3, prompt, negative_prompt, cfg, steps, lora, seed, save_vol],
            outputs=[output_img, status]
        )

        # Automatically load gallery images on App Load (defaulting to 'images' volume)
        demo.load(
            fn=scan_volume_images,
            inputs=[vol_select],
            outputs=[gallery, gallery_paths_store]
        )

        # Reload gallery when volume selection changes
        vol_select.change(
            fn=scan_volume_images,
            inputs=[vol_select],
            outputs=[gallery, gallery_paths_store]
        )

        # Refresh button click
        refresh_btn.click(
            fn=scan_volume_images,
            inputs=[vol_select],
            outputs=[gallery, gallery_paths_store]
        )

        # Select image from gallery (stores index/filepath)
        gallery.select(
            fn=get_select_path,
            inputs=[gallery_paths_store],
            outputs=[selected_file_path]
        )

        # Transfer image buttons
        set_img1_btn.click(
            fn=load_selected_image,
            inputs=[selected_file_path],
            outputs=[img1]
        )
        set_img2_btn.click(
            fn=load_selected_image,
            inputs=[selected_file_path],
            outputs=[img2]
        )
        set_img3_btn.click(
            fn=load_selected_image,
            inputs=[selected_file_path],
            outputs=[img3]
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=8000, share=True, allowed_paths=["/outputs", "/images"])
