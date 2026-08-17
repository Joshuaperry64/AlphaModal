import os
import sys
import subprocess
from pathlib import Path

# UTF-8 Output stream fix for Windows terminal
if sys.stdout and getattr(sys.stdout, 'encoding', '') != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def ask(prompt_text: str, default: str) -> str:
    """Prompt user for input with default fallback."""
    try:
        user_val = input(f"{prompt_text} [{default}]: ").strip()
        return user_val if user_val else default
    except (KeyboardInterrupt, EOFError):
        print("\n")
        return default


def main_menu():
    while True:
        clear_screen()
        print("===============================================================================")
        print("                          ALPHA IMAGE PROCESSOR MENU")
        print("===============================================================================")
        print("")
        print("   [1] Batch Image Downloader        (Multi-engine search and deduplication)")
        print("   [2] AI Watermark Remover          (FLUX.1-dev + LoRA watermark removal)")
        print("   [3] Batch Image-to-Image Edit     (Qwen / SDXL batch img2img)")
        print("   [4] Florence-2 Batch Captioner    (Dataset sidecar .txt generator)")
        print("   [5] SDXL LoRA Trainer             (Kohya-ss SDXL training on Modal A100)")
        print("")
        print(" -- WEB INTERFACES (MODAL SERVE) ---------------------------------------------")
        print("   [6] Serve Watermark Remover Web UI (Interactive Gradio App)")
        print("   [7] Serve Florence-2 Captioner Web UI (Interactive Gradio App)")
        print("")
        print("   [0] Exit")
        print("")
        print("===============================================================================")
        
        try:
            choice = input("Select an option [0-7]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AlphaModal. Goodbye!")
            sys.exit(0)

        if choice == "1":
            downloader_wizard()
        elif choice == "2":
            watermark_remover_wizard()
        elif choice == "3":
            img2img_wizard()
        elif choice == "4":
            captioner_wizard()
        elif choice == "5":
            trainer_wizard()
        elif choice == "6":
            serve_watermark_remover()
        elif choice == "7":
            serve_captioner()
        elif choice == "0":
            print("\nThank you for using AlphaModal! Goodbye.")
            sys.exit(0)
        else:
            print("\n[!] Invalid selection, try again.")
            input("Press Enter to continue...")


def downloader_wizard():
    clear_screen()
    print("===============================================================================")
    print("                         BATCH IMAGE DOWNLOADER CONFIG")
    print("===============================================================================")
    print("")
    queries = ask("[?] Enter search query terms (comma separated)", "cyberpunk city")
    limit = ask("[?] Clean images per query", "20")
    total_target = ask("[?] Total clean image target count (0 for auto)", "0")
    threads = ask("[?] Concurrent download threads", "10")
    output = ask("[?] Output folder path", "./pre-downloads")
    threshold = ask("[?] Similarity threshold (0.0 to 1.0)", "0.85")
    action = ask("[?] Action on duplicates (delete/move)", "delete")

    cmd = [
        sys.executable, "image_downloader.py",
        "--queries", queries,
        "--limit", limit,
        "--total-target", total_target,
        "--threads", threads,
        "--output", output,
        "--threshold", threshold,
        "--action", action
    ]

    print("\n-------------------------------------------------------------------------------")
    print(f"[>] Launching Image Downloader...")
    print(f"[>] Queries: {queries}")
    print(f"[>] Per Query: {limit} | Total Target: {total_target}")
    print(f"[>] Threads: {threads} | Output: {output}")
    print("-------------------------------------------------------------------------------\n")
    
    subprocess.run(cmd)
    input("\n[✓] Finished. Press Enter to return to menu...")


def watermark_remover_wizard():
    clear_screen()
    print("===============================================================================")
    print("                        AI WATERMARK REMOVER CONFIG")
    print("===============================================================================")
    print("")
    input_dir = ask("[?] Input image directory", "./pre-downloads")
    output_dir = ask("[?] Output cleaned directory", "./cleaned-images")
    prompt = ask("[?] Custom removal prompt", "remove any watermark text or logos from the image while preserving the background, texture, lighting, and overall realism. Ensure the edited areas blend seamlessly.")
    guidance = ask("[?] Guidance scale", "2.5")
    steps = ask("[?] Inference steps", "30")
    mode = ask("[?] Execution environment (1=Modal Cloud GPU, 2=Local GPU)", "1")

    cmd = [
        sys.executable, "watermark_remover.py",
        "--input", input_dir,
        "--output", output_dir,
        "--prompt", prompt,
        "--guidance-scale", guidance,
        "--steps", steps
    ]

    if mode == "2":
        cmd.append("--local-gpu")

    print("\n-------------------------------------------------------------------------------")
    print(f"[>] Launching AI Watermark Remover...")
    print(f"[>] Input: {input_dir}")
    print(f"[>] Output: {output_dir}")
    print(f"[>] Guidance: {guidance} | Steps: {steps}")
    print("-------------------------------------------------------------------------------\n")
    
    subprocess.run(cmd)
    input("\n[✓] Finished. Press Enter to return to menu...")


def img2img_wizard():
    clear_screen()
    print("===============================================================================")
    print("                         BATCH IMAGE-TO-IMAGE CONFIG")
    print("===============================================================================")
    print("")
    input_dir = ask("[?] Input image directory", "./cleaned-images")
    output_dir = ask("[?] Output edited directory", "./edited-images")
    prompt = ask("[?] Image edit prompt", "masterpiece, highly detailed, 8k resolution")
    neg_prompt = ask("[?] Negative prompt", "blurry, low quality, distortion, extra limbs")
    steps = ask("[?] Inference steps", "20")
    cfg = ask("[?] CFG scale", "4.0")

    cmd = [
        "modal", "run", "batch-img2img.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--prompt", prompt,
        "--negative-prompt", neg_prompt,
        "--steps", steps,
        "--cfg-scale", cfg
    ]

    print("\n-------------------------------------------------------------------------------")
    print(f"[>] Launching Batch Image-to-Image on Modal A100 GPU...")
    print(f"[>] Input: {input_dir}")
    print(f"[>] Output: {output_dir}")
    print("-------------------------------------------------------------------------------\n")
    
    subprocess.run(cmd, shell=True)
    input("\n[✓] Finished. Press Enter to return to menu...")


def captioner_wizard():
    clear_screen()
    print("===============================================================================")
    print("                    FLORENCE-2 BATCH CAPTIONER CONFIG")
    print("===============================================================================")
    print("")
    input_dir = ask("[?] Input image directory", "./cleaned-images")
    output_dir = ask("[?] Output caption sidecar directory", "./dataset-captioned")
    trigger = ask("[?] Trigger word prefix", "ohwx person")
    task = ask("[?] Task prompt", "<MORE_DETAILED_CAPTION>")

    cmd = [
        "modal", "run", "captioner.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--trigger-word", trigger,
        "--task-prompt", task
    ]

    print("\n-------------------------------------------------------------------------------")
    print(f"[>] Launching Florence-2 Batch Captioner on Modal Cloud...")
    print(f"[>] Input: {input_dir}")
    print(f"[>] Output: {output_dir}")
    print(f"[>] Trigger: {trigger}")
    print("-------------------------------------------------------------------------------\n")
    
    subprocess.run(cmd, shell=True)
    input("\n[✓] Finished. Press Enter to return to menu...")


def trainer_wizard():
    clear_screen()
    print("===============================================================================")
    print("                     SDXL LORA TRAINER CONFIG (A100)")
    print("===============================================================================")
    print("")
    project = ask("[?] Project name", "sdxl-lora-job")
    dataset = ask("[?] Dataset directory", "./dataset-captioned")
    epochs = ask("[?] Training Epochs", "10")
    repeats = ask("[?] Repeats per image", "10")

    cmd = [
        "modal", "run", "train_sdxl.py",
        "--project-name", project,
        "--dataset-dir", dataset,
        "--epochs", epochs,
        "--repeats", repeats
    ]

    print("\n-------------------------------------------------------------------------------")
    print(f"[>] Launching SDXL LoRA Trainer on Modal Cloud A100...")
    print(f"[>] Project: {project}")
    print(f"[>] Dataset: {dataset}")
    print(f"[>] Epochs: {epochs} | Repeats: {repeats}")
    print("-------------------------------------------------------------------------------\n")
    
    subprocess.run(cmd, shell=True)
    input("\n[✓] Finished. Press Enter to return to menu...")


def serve_watermark_remover():
    clear_screen()
    print("===============================================================================")
    print("                    WATERMARK REMOVER WEB UI (MODAL SERVE)")
    print("===============================================================================")
    print("Launching Gradio Web Server on Modal... Press Ctrl+C to stop.\n")
    try:
        subprocess.run(["modal", "serve", "watermark_remover.py"], shell=True)
    except KeyboardInterrupt:
        pass
    input("\n[✓] Stopped. Press Enter to return to menu...")


def serve_captioner():
    clear_screen()
    print("===============================================================================")
    print("                   FLORENCE-2 CAPTIONER WEB UI (MODAL SERVE)")
    print("===============================================================================")
    print("Launching Gradio Web Server on Modal... Press Ctrl+C to stop.\n")
    try:
        subprocess.run(["modal", "serve", "captioner.py"], shell=True)
    except KeyboardInterrupt:
        pass
    input("\n[✓] Stopped. Press Enter to return to menu...")


if __name__ == "__main__":
    main_menu()
