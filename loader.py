"""
AlphaCore Model Loader — CivitAI & HuggingFace → Modal Volume
=================================================================
Downloads .safetensors (or any model file) from CivitAI or HuggingFace
directly into the 'hf-hub-cache' Modal volume at /hf-hub-cache/checkpoints/.

USAGE
-----
# Download from CivitAI by model version ID:
modal run civi-loader.py --source civitai --id 357609

# Download from CivitAI by full URL:
modal run civi-loader.py --source civitai --url "https://civitai.com/models/133005?modelVersionId=357609"

# Download from HuggingFace by repo + filename:
modal run civi-loader.py --source huggingface --repo "RunDiffusion/Juggernaut-XL-v9" --filename "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

# Download from a direct URL (any source):
modal run civi-loader.py --source url --url "https://example.com/model.safetensors" --filename "mymodel.safetensors"

# List what's already in the checkpoints folder:
modal run civi-loader.py --list

SECRETS
-------
For CivitAI (required for some models):
  modal secret create civitai-secret CIVITAI_API_KEY=your_key_here

For HuggingFace (required for gated models):
  modal secret create huggingface-secret HF_TOKEN=your_token_here
"""

import modal
from pathlib import Path

# ── Volume & App ─────────────────────────────────────────────────
CACHE_DIR    = "/hf-hub-cache"
CHECKPOINTS  = f"{CACHE_DIR}/checkpoints"

app          = modal.App("alphacore-model-loader")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "requests",
        "tqdm",
        "huggingface-hub>=0.24.0",
    )
)

# Optional secrets — load whichever exist, skip if missing
def _get_secrets():
    secrets = []
    try:
        secrets.append(modal.Secret.from_name("civitai-secret"))
    except Exception:
        pass
    try:
        secrets.append(modal.Secret.from_name("huggingface-secret"))
    except Exception:
        pass
    return secrets


# ── Core Download Function ────────────────────────────────────────
@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    timeout=60 * 60,          # 1 hour max for large models
    secrets=_get_secrets(),
    region="us-west",
)
def download_model(
    source: str,              # "civitai" | "huggingface" | "url"
    civitai_version_id: str = "",
    hf_repo: str = "",
    hf_filename: str = "",
    direct_url: str = "",
    output_filename: str = "",# override the saved filename (optional)
    subfolder: str = "checkpoints",
):
    import os
    import requests
    from tqdm import tqdm
    from pathlib import Path

    dest_dir = Path(CACHE_DIR) / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    source = source.lower().strip()

    # ── CivitAI ─────────────────────────────────────────────────
    if source == "civitai":
        api_key = os.environ.get("CIVITAI_API_KEY", "")

        if not civitai_version_id:
            raise ValueError("civitai_version_id is required for source=civitai")

        # Resolve metadata to get the real filename
        meta_url = f"https://civitai.com/api/v1/model-versions/{civitai_version_id}"
        headers  = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        print(f"🔍 Fetching CivitAI metadata for version {civitai_version_id}...")
        meta = requests.get(meta_url, headers=headers, timeout=30)
        if meta.status_code != 200:
            raise RuntimeError(f"CivitAI API error {meta.status_code}: {meta.text[:300]}")

        meta_json     = meta.json()
        model_name    = meta_json.get("model", {}).get("name", "unknown")
        files         = meta_json.get("files", [])

        # Pick the primary / largest file
        primary = next(
            (f for f in files if f.get("primary", False)),
            files[0] if files else None
        )
        if not primary:
            raise RuntimeError("No downloadable files found in CivitAI response")

        civitai_filename = primary["name"]
        download_url     = primary["downloadUrl"]
        size_kb          = primary.get("sizeKB", 0)

        # Append API key to URL if available
        if api_key:
            download_url += f"?token={api_key}"

        final_filename = output_filename or civitai_filename
        dest_path      = dest_dir / final_filename

        print(f"📦 Model   : {model_name}")
        print(f"📄 File    : {civitai_filename}")
        print(f"💾 Size    : {size_kb / 1024:.1f} MB")
        print(f"📥 Saving  : {dest_path}")

        _stream_download(download_url, dest_path, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    # ── HuggingFace ──────────────────────────────────────────────
    elif source == "huggingface":
        import huggingface_hub
        import os

        if not hf_repo or not hf_filename:
            raise ValueError("hf_repo and hf_filename are required for source=huggingface")

        hf_token = os.environ.get("HF_TOKEN", None)
        print(f"🤗 Downloading from HuggingFace: {hf_repo} / {hf_filename}")

        downloaded = huggingface_hub.hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            token=hf_token,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )

        final_filename = output_filename or Path(downloaded).name
        final_dest     = dest_dir / final_filename

        # Rename if override requested
        if output_filename and Path(downloaded) != final_dest:
            Path(downloaded).rename(final_dest)

        print(f"✅ Saved to: {final_dest}")

    # ── Direct URL ───────────────────────────────────────────────
    elif source == "url":
        if not direct_url:
            raise ValueError("direct_url is required for source=url")

        guessed_name   = direct_url.split("/")[-1].split("?")[0]
        final_filename = output_filename or guessed_name
        dest_path      = dest_dir / final_filename

        print(f"🌐 Downloading from URL: {direct_url}")
        print(f"📥 Saving to: {dest_path}")
        _stream_download(direct_url, dest_path)

    else:
        raise ValueError(f"Unknown source '{source}'. Use: civitai | huggingface | url")

    # Commit volume so files persist
    cache_volume.commit()
    print("✅ Volume committed — model is ready.")


# ── List Checkpoints ─────────────────────────────────────────────
@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
)
def list_checkpoints(subfolder: str = "checkpoints"):
    from pathlib import Path

    target = Path(CACHE_DIR) / subfolder
    if not target.exists():
        print(f"⚠  {target} does not exist yet.")
        return

    files = sorted(target.iterdir())
    if not files:
        print(f"📂 {target} is empty.")
        return

    print(f"\n📂 Contents of {target}:\n")
    for f in files:
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {'📄' if f.is_file() else '📁'} {f.name:<60} {size_mb:>8.1f} MB")
    print()


# ── Streaming Download Helper ─────────────────────────────────────
def _stream_download(url: str, dest: Path, headers: dict = {}):
    import requests
    from tqdm import tqdm

    resp = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunks

    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"✅ Download complete: {dest} ({dest.stat().st_size / 1024**2:.1f} MB)")


# ── Local Entrypoint — Interactive Menu ──────────────────────────
@app.local_entrypoint()
def main():
    import re
    import sys

    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    YELLOW= "\033[93m"
    RED   = "\033[91m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    RESET = "\033[0m"

    def banner():
        print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗
║       ALPHACORE // MODEL LOADER v1.0                 ║
║       CivitAI & HuggingFace → Modal Volume           ║
╚══════════════════════════════════════════════════════╝{RESET}
""")

    def prompt(msg, default=None):
        suffix = f" {DIM}[{default}]{RESET}" if default else ""
        val = input(f"  {CYAN}>{RESET} {msg}{suffix}: ").strip()
        return val if val else default

    def pick(msg, choices):
        """Keep asking until a valid choice is entered."""
        while True:
            val = input(f"  {CYAN}>{RESET} {msg}: ").strip()
            if val in choices:
                return val
            print(f"  {RED}Invalid choice. Enter one of: {', '.join(choices)}{RESET}")

    def pick_subfolder():
        print(f"\n  {BOLD}Select Model Type (Destination Folder):{RESET}")
        print(f"    {GREEN}[1]{RESET} Checkpoint / Base Model  (-> checkpoints)")
        print(f"    {GREEN}[2]{RESET} LoRA                     (-> loras)")
        print(f"    {GREEN}[3]{RESET} Textual Inversion        (-> embeddings)")
        print(f"    {GREEN}[4]{RESET} VAE                      (-> vae)")
        print(f"    {GREEN}[5]{RESET} ControlNet               (-> controlnet)")
        print(f"    {GREEN}[6]{RESET} FramePack LoRA           (-> framepack/loras)")
        print(f"    {GREEN}[7]{RESET} Custom / Other")
        
        c = pick("Enter type (1-7)", {"1", "2", "3", "4", "5", "6", "7"})
        if c == "1": return "checkpoints"
        if c == "2": return "loras"
        if c == "3": return "embeddings"
        if c == "4": return "vae"
        if c == "5": return "controlnet"
        if c == "6": return "framepack/loras"
        return prompt("Enter custom subfolder name", "checkpoints")

    banner()

    # ── Main menu ────────────────────────────────────────────────
    print(f"  {BOLD}Select source:{RESET}")
    print(f"    {GREEN}[1]{RESET}  HuggingFace")
    print(f"    {GREEN}[2]{RESET}  CivitAI")
    print(f"    {GREEN}[3]{RESET}  Direct URL")
    print(f"    {GREEN}[L]{RESET}  List current checkpoints")
    print(f"    {GREEN}[Q]{RESET}  Quit")
    print()

    choice = pick("Enter choice (1/2/3/L/Q)", {"1", "2", "3", "L", "l", "Q", "q"}).upper()

    if choice == "Q":
        print(f"\n  {DIM}Exiting.{RESET}\n")
        sys.exit(0)

    if choice == "L":
        subfolder = pick_subfolder()
        print(f"\n  {YELLOW}Fetching volume contents...{RESET}\n")
        list_checkpoints.remote(subfolder=subfolder)
        sys.exit(0)

    # ── HuggingFace ──────────────────────────────────────────────
    if choice == "1":
        print(f"\n  {BOLD}HuggingFace Download{RESET}")
        print(f"  {DIM}Paste a HuggingFace file URL or repo/filename.{RESET}")
        print(f"  {DIM}Examples:{RESET}")
        print(f"  {DIM}  https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9.safetensors{RESET}")
        print(f"  {DIM}  stabilityai/stable-diffusion-xl-base-1.0  (repo only, will ask filename){RESET}\n")

        url_or_repo = prompt("HuggingFace URL or repo ID")
        if not url_or_repo:
            print(f"  {RED}No input provided. Exiting.{RESET}")
            sys.exit(1)

        hf_repo = ""
        hf_filename = ""
        direct_url = ""
        source = "huggingface"

        # Detect if it's a full HF resolve URL
        hf_url_match = re.match(
            r"https://huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)",
            url_or_repo
        )
        if hf_url_match:
            hf_repo     = hf_url_match.group(1)
            hf_filename = hf_url_match.group(2)
            print(f"\n  {GREEN}✔ Parsed repo  : {hf_repo}{RESET}")
            print(f"  {GREEN}✔ Parsed file  : {hf_filename}{RESET}")
        elif "/" in url_or_repo and not url_or_repo.startswith("http"):
            # Looks like a repo ID
            hf_repo = url_or_repo
            hf_filename = prompt("Filename inside repo (e.g. model.safetensors)")
        else:
            # Fall back to direct URL download
            source     = "url"
            direct_url = url_or_repo
            guessed    = url_or_repo.split("/")[-1].split("?")[0]
            print(f"\n  {YELLOW}⚠ Could not parse as HF URL. Treating as direct download.{RESET}")
            print(f"  {DIM}Guessed filename: {guessed}{RESET}")

        out = prompt("Save as filename (leave blank to keep original)", "")
        subfolder = pick_subfolder()

        print(f"\n  {YELLOW}Dispatching to Modal...{RESET}\n")
        download_model.remote(
            source=source,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            direct_url=direct_url,
            output_filename=out or "",
            subfolder=subfolder,
        )

    # ── CivitAI ──────────────────────────────────────────────────
    elif choice == "2":
        print(f"\n  {BOLD}CivitAI Download{RESET}")
        print(f"  {DIM}Paste the CivitAI model page URL or just the modelVersionId.{RESET}")
        print(f"  {DIM}Example URL: https://civitai.com/models/133005?modelVersionId=357609{RESET}\n")

        user_input = prompt("CivitAI URL or version ID")
        if not user_input:
            print(f"  {RED}No input provided. Exiting.{RESET}")
            sys.exit(1)

        version_id = ""

        # Pure numeric = direct version ID
        if user_input.strip().isdigit():
            version_id = user_input.strip()
            print(f"  {GREEN}✔ Version ID   : {version_id}{RESET}")
        else:
            # Try to extract from URL
            match = re.search(r"modelVersionId=(\d+)", user_input)
            if match:
                version_id = match.group(1)
                print(f"  {GREEN}✔ Parsed version ID: {version_id}{RESET}")
            else:
                # Try /models/<id> as version id fallback
                match2 = re.search(r"/models/(\d+)", user_input)
                if match2:
                    print(f"\n  {YELLOW}⚠ No modelVersionId found in URL.{RESET}")
                    version_id = prompt("Enter the modelVersionId manually")
                else:
                    print(f"  {RED}Could not parse a CivitAI version ID from input.{RESET}")
                    sys.exit(1)

        out       = prompt("Save as filename (leave blank to use CivitAI's filename)", "")
        subfolder = pick_subfolder()

        print(f"\n  {YELLOW}Dispatching to Modal...{RESET}\n")
        download_model.remote(
            source="civitai",
            civitai_version_id=version_id,
            output_filename=out or "",
            subfolder=subfolder,
        )

    # ── Direct URL ───────────────────────────────────────────────
    elif choice == "3":
        print(f"\n  {BOLD}Direct URL Download{RESET}")
        print(f"  {DIM}Paste any direct download link to a model file.{RESET}\n")

        url = prompt("Direct download URL")
        if not url:
            print(f"  {RED}No URL provided. Exiting.{RESET}")
            sys.exit(1)

        guessed   = url.split("/")[-1].split("?")[0]
        out       = prompt(f"Save as filename", guessed)
        subfolder = pick_subfolder()

        print(f"\n  {YELLOW}Dispatching to Modal...{RESET}\n")
        download_model.remote(
            source="url",
            direct_url=url,
            output_filename=out,
            subfolder=subfolder,
        )
