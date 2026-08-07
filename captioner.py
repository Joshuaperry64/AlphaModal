import modal
import os
import sys
import io
import zipfile
from pathlib import Path
import shutil

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

app = modal.App("gradio-captioner")

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="microsoft/Florence-2-large")

image = (
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

with image.imports():
    import gradio as gr
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    from PIL import Image
    from unittest.mock import patch
    from transformers.dynamic_module_utils import get_imports

    def fixed_get_imports(filename):
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

outputs_vol = modal.Volume.from_name("images")

@app.function(image=image, gpu="L4", timeout=3600, volumes={"/images": outputs_vol})
@modal.web_server(8000, startup_timeout=300)
def ui():
    print("Loading model into GPU...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            local_files_only=True,
        )
    print("Model loaded.")

    def process_volume(task_prompt, trigger_word, progress=gr.Progress()):
        vol_path = Path("/images")
        
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            images.extend(vol_path.rglob(f"*{ext}"))
            images.extend(vol_path.rglob(f"*{ext.upper()}"))
            
        if not images:
            return "No images found in the 'outputs' volume."
            
        success_count = 0
        for i, img_path in enumerate(progress.tqdm(images, desc="Captioning Images")):
            # Skip if caption already exists
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
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
                
                if trigger_word:
                    tw = trigger_word.strip()
                    if not tw.endswith(","): tw += ","
                    caption = f"{tw} {caption}"
                    
                txt_path.write_text(caption.strip(), encoding="utf-8")
                success_count += 1
                
                # Commit every 50 images to avoid losing progress
                if success_count > 0 and success_count % 50 == 0:
                    outputs_vol.commit()
            except Exception as e:
                print(f"Error on {img_path.name}: {e}")
                
        outputs_vol.commit()
        return f"Successfully processed and generated captions for {success_count} new images in the volume!"

    with gr.Blocks(title="Florence-2 Batch Captioner") as demo:
        gr.Markdown("# 🖼️ Florence-2 Batch Captioner (Volume Mode)")
        gr.Markdown("Click the button below to scan the `images` Modal Volume. It will find all images, generate captions, and save the `.txt` sidecar files directly back into the volume!")
        
        with gr.Row():
            with gr.Column():
                task_prompt = gr.Dropdown(
                    choices=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                    value="<MORE_DETAILED_CAPTION>",
                    label="Detail Level"
                )
                trigger_word = gr.Textbox(label="Trigger Word (optional)", placeholder="e.g. sks")
                submit_btn = gr.Button("Start Captioning Volume", variant="primary")
            
            with gr.Column():
                status_text = gr.Textbox(label="Status", interactive=False)
                
        submit_btn.click(
            fn=process_volume,
            inputs=[task_prompt, trigger_word],
            outputs=[status_text]
        )
        
    demo.queue().launch(server_name="0.0.0.0", server_port=8000, share=True)
