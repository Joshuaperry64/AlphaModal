# ---
# output-directory: "/tmp/framepack_studio"
# ---
import io
import os
import shutil
import random
import sys
from pathlib import Path
import modal

app = modal.App("framepack-studio-wsl-lifecycle")

# --- Persistent Storage Layout ---
model_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("outputs", create_if_missing=True)

VOL_MODELS = "/data/models"
VOL_OUTPUTS = "/data/outputs"

# --- Environment Architecture Configuration ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "python3-opencv", "ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .uv_pip_install(
        "fastapi",
        "accelerate",
        "diffusers==0.32.2",
        "transformers<=4.48.3",
        "tokenizers>=0.20.0,<0.21.0",
        "gradio>=4.0.0,<5.0.0",
        "sentencepiece",
        "huggingface-hub",
        "numpy<2.0.0",
        "pillow",
        "einops",
        "opencv-python",
        "decord",
        "imageio",
        "imageio-ffmpeg",
        "ffmpeg-python",
        "sageattention"
    )
    .run_commands(
        "git clone https://github.com/FP-Studio/framepack-studio.git /root/framepack-studio"
    )
    .run_commands(
        "cd /root/framepack-studio && pip install -r requirements.txt --ignore-installed || true"
    )
)


def wire_storage_volumes():
    """Executes once during container startup to align filesystem targets."""
    REPO_ROOT = "/root/framepack-studio"
    
    os.environ["HF_HOME"] = f"{VOL_MODELS}/framepack/hf_cache"
    os.environ["XDG_CACHE_HOME"] = f"{VOL_MODELS}/framepack/cache"
    
    os.makedirs(f"{VOL_MODELS}/framepack/hf_cache", exist_ok=True)
    os.makedirs(f"{VOL_MODELS}/framepack/ffmpeg_bin", exist_ok=True)
    os.makedirs(VOL_OUTPUTS, exist_ok=True)

    # Symlink the model and output dirs directly to persistent volumes
    repo_models = os.path.join(REPO_ROOT, "models")
    if not os.path.islink(repo_models):
        shutil.rmtree(repo_models, ignore_errors=True)
        os.symlink(f"{VOL_MODELS}/framepack", repo_models)

    repo_outputs = os.path.join(REPO_ROOT, "outputs")
    if not os.path.islink(repo_outputs):
        shutil.rmtree(repo_outputs, ignore_errors=True)
        os.symlink(VOL_OUTPUTS, repo_outputs)

    repo_loras = os.path.join(REPO_ROOT, "loras")
    os.makedirs(f"{VOL_MODELS}/framepack/loras", exist_ok=True)
    if not os.path.islink(repo_loras):
        shutil.rmtree(repo_loras, ignore_errors=True)
        os.symlink(f"{VOL_MODELS}/framepack/loras", repo_loras)

    toolbox = os.path.join(REPO_ROOT, "modules", "toolbox")
    os.makedirs(toolbox, exist_ok=True)
    toolbox_bin = os.path.join(toolbox, "bin")
    if not os.path.islink(toolbox_bin):
        shutil.rmtree(toolbox_bin, ignore_errors=True)
        os.symlink(f"{VOL_MODELS}/framepack/ffmpeg_bin", toolbox_bin)


class CleanGradioPathMiddleware:
    """
    ASGI Middleware that completely neutralizes routing loops and asset breaking.
    Forces root_path to empty to stop Gradio from calculating double slashes ('//assets'),
    and sanitizes paths inline to break 307 loops.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            # Force empty root path context across the entire ASGI state
            scope["root_path"] = ""
            
            # Collapse any double/triple slashes in the routing path down to a clean single slash
            path = scope.get("path", "")
            if "//" in path:
                while "//" in path:
                    path = path.replace("//", "/")
                scope["path"] = path
            
            raw_path = scope.get("raw_path", b"")
            if b"//" in raw_path:
                while b"//" in raw_path:
                    raw_path = raw_path.replace(b"//", b"/")
                scope["raw_path"] = raw_path
                
        await self.app(scope, receive, send)


# --- Container Class Lifecycle Layout ---
@app.cls(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={VOL_MODELS: model_volume, VOL_OUTPUTS: output_volume},
    max_containers=1,
    secrets=[modal.Secret.from_dict({"CACHE_BUSTER": "6"})]
)
@modal.concurrent(max_inputs=100)
class FramePackContainer:
    @modal.enter()
    def build_and_warmup_application(self):
        """Runs EXACTLY ONCE per container boot up."""
        print("ðŸš€ [Cold Start] Initializing container runtime environment...")
        
        REPO_ROOT = "/root/framepack-studio"
        
        import subprocess
        subprocess.run(["sed", "-i", "s/from __main__ import/from studio import/g", f"{REPO_ROOT}/modules/pipelines/worker.py"])
        subprocess.run(["sed", "-i", "s/from __main__ import/from studio import/g", f"{REPO_ROOT}/modules/generators/video_base_generator.py"])
        subprocess.run(["sed", "-i", "s/from __main__ import/from studio import/g", f"{REPO_ROOT}/modules/interface.py"])
        subprocess.run(["sed", "-i", "s/import __main__ as studio_module/import studio as studio_module/g", f"{REPO_ROOT}/modules/pipelines/worker.py"])
        subprocess.run(["sed", "-i", "s/sys.modules\\['__main__'\\]/sys.modules['studio']/g", f"{REPO_ROOT}/modules/toolbox_app.py"])
        subprocess.run(["sed", "-i", "s/'__main__' in sys.modules/'studio' in sys.modules/g", f"{REPO_ROOT}/modules/toolbox_app.py"])
        subprocess.run(["pip", "uninstall", "-y", "sageattention"])
        
        # Patch utils.py to use imageio instead of torchvision.io.write_video
        utils_path = f"{REPO_ROOT}/diffusers_helper/utils.py"
        if os.path.exists(utils_path):
            with open(utils_path, "r") as f:
                content = f.read()
            bad_code = "torchvision.io.write_video(output_filename, x, fps=fps, video_codec='libx264', options={'crf': str(int(crf))})"
            good_code = '''import imageio
    import numpy as np
    writer = imageio.get_writer(output_filename, fps=fps, codec="libx264", macro_block_size=8, quality=8)
    for frame in x.numpy():
        writer.append_data(frame)
    writer.close()'''
            content = content.replace(bad_code, good_code)
            with open(utils_path, "w") as f:
                f.write(content)





        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        
        MODULES_PATH = os.path.join(REPO_ROOT, "modules")
        if os.path.exists(MODULES_PATH) and MODULES_PATH not in sys.path:
            sys.path.insert(0, MODULES_PATH)
            
        os.chdir(REPO_ROOT)
        wire_storage_volumes()

        import gradio as gr

        # Block their blocking local .launch() chain execution thread
        gr.Blocks.launch = lambda *args, **kwargs: print("âš¡ Blocked native local server launch loop.")

        print("ðŸ“¦ [Cold Start] Importing studio codebase and parsing weights into GPU layer...")
        import studio as gradio_module
        
        if hasattr(gradio_module, "demo"):
            self.demo = gradio_module.demo
        elif hasattr(gradio_module, "interface"):
            self.demo = gradio_module.interface
        else:
            raise AttributeError("Failed to bind target Gradio blocks instance inside framework.")
        
        # Explicitly configure the underlying block schema to have no custom root prefix
        self.demo.root_path = ""
        
        # Commit volume configurations safely
        model_volume.commit()
        print("âœ… [Cold Start] Container environment fully warmed up. Models loaded.")

    @modal.asgi_app()
    def ui(self):
        """Instantiates the web app wrapped with path normalization to destroy proxy redirect chains."""
        import fastapi
        import gradio as gr
        
        web_app = fastapi.FastAPI()
        
        # 1. Initialize the functional queue chain
        demo = self.demo.queue()
        
        # 2. Mount Gradio to the FastAPI root instance
        return gr.mount_gradio_app(web_app, demo, path="/")

# Auto-generated class local_entrypoint wrappers
@app.local_entrypoint()
def entrypoint_FramePackContainer():
    """Auto-generated wrapper to instantiate FramePackContainer"""
    try:
        Cls = modal.Cls.from_name('framepack-studio-wsl-lifecycle', 'FramePackContainer')
    except Exception:
        try:
            Cls = modal.Cls.from_name(app.name, 'FramePackContainer')
        except Exception as e:
            raise RuntimeError('Could not resolve class FramePackContainer for local entrypoint: ' + str(e))
    inst = Cls()
    return 'instantiated FramePackContainer'
