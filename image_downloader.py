"""
AlphaModal Batch Image Downloader & Integrated Deduplicator
============================================================
Downloads images from multiple search queries into a single flat output folder,
automatically detects and removes exact & visual perceptual duplicates after downloading,
and executes multi-provider top-up search reruns if purged duplicates cause the clean 
image count to fall short of your target goal.

Features:
- Multi-Engine Provider Rotation: Uses Bing, Flickr, and Wikimedia Open search engines
  across top-up reruns to eliminate duplicate search results from a single engine.
- Single flat downloads directory (no subfolder clutter).
- Built-in exact MD5 & visual dHash perceptual image deduplication.
- Auto-preserves the highest resolution copy of duplicate images.
- Smart target count top-up reruns to replace purged duplicates.
"""

import argparse
import concurrent.futures
import hashlib
import io
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
SEARCH_QUERIES = [
    "preteen girls",
    "candid preteen girls",
    "elementary school girls in cheerleading uniforms",
    "young preteen girls full body",
    "young preteen girls yoga stretching poses",
    "preteen girls wearing swimsuits full body",
    "candid preteen girls selfie's",
    "preteen girls cheerleading poses",
    "sexy elementary school girls",
    "sexy preteen girls poses",
]
TOTAL_TARGET_CLEAN_COUNT = 500            # Fallback total clean target (0 = calculate as len(queries) * IMAGES_PER_QUERY)
IMAGES_PER_QUERY = 50                   # Target clean image count per search query (if TOTAL_TARGET_CLEAN_COUNT = 0)
SIMILARITY_THRESHOLD = 0.95             # Visual similarity threshold for deduplication (0.80 - 0.95)
ACTION_ON_DUPLICATES = "delete"         # "delete" permanently or "move" to trash folder
OUTPUT_DIR = "./pre-downloads"              # Flat folder for ALL downloaded clean images
TRASH_DIR = "./duplicates_trash"        # Trash folder if ACTION_ON_DUPLICATES = "move"
MAX_TOPUP_ROUNDS = 15                   # Maximum top-up rerun rounds if duplicates are purged
NUM_THREADS = 15                         # Parallel download threads
# ──────────────────────────────────────────────────────────────────────────────


import requests
from PIL import Image
import modal

# ─── MODAL APP SETUP ──────────────────────────────────────────────────────────
app = modal.App("batch-image-downloader")
downloads_volume = modal.Volume.from_name("modal-image-downloads", create_if_missing=True)

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests>=2.31.0",
        "pillow>=10.0.0",
        "duckduckgo_search>=6.0.0",
    )
)

# ─── DEDUPLICATION ENGINE ─────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text into a safe filename slug."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_').lower()

def get_md5(file_path: Path) -> str:
    """Calculate MD5 byte hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> Optional[int]:
    """Compute 64-bit Difference Hash (dHash) for visual perceptual matching."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = list(getattr(img, "get_flattened_data", img.getdata)())
            difference = []
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    difference.append(pixels[row_start + col] > pixels[row_start + col + 1])
            decimal_val = 0
            for bit in difference:
                decimal_val = (decimal_val << 1) | bit
            return decimal_val
    except Exception:
        return None


def get_image_resolution(image_path: Path) -> Tuple[int, int, int]:
    """Return (width * height, width, height) to prioritize keeping higher resolution."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            return (w * h, w, h)
    except Exception:
        return (0, 0, 0)


def hamming_distance(hash1: int, hash2: int, bits: int = 64) -> int:
    """Calculate Hamming distance between two bitwise integer hashes."""
    return bin(hash1 ^ hash2).count('1')


def similarity_score(hash1: int, hash2: int, bits: int = 64) -> float:
    """Return normalized similarity percentage [0.0 to 1.0]."""
    return 1.0 - (hamming_distance(hash1, hash2, bits=bits) / float(bits))


def run_inline_deduplication(
    target_dir: Path,
    similarity_thresh: float = SIMILARITY_THRESHOLD,
    action: str = ACTION_ON_DUPLICATES,
    trash_dir: Path = Path(TRASH_DIR),
    workers: int = NUM_THREADS,
) -> int:
    """Detects and removes/moves duplicate images in target_dir. Returns count of purged duplicates."""
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    image_paths = [f for f in target_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

    if len(image_paths) < 2:
        return 0

    print(f"\n[*] Running Inline Deduplication on {len(image_paths)} images...")

    # Stage 1: MD5 Exact Matches
    md5_map: Dict[str, List[Path]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(get_md5, p): p for p in image_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            p = future_to_path[future]
            try:
                md5_map.setdefault(future.result(), []).append(p)
            except Exception:
                pass

    exact_duplicates: List[Path] = []
    remaining_paths: List[Path] = []
    for group in md5_map.values():
        if len(group) > 1:
            group.sort(key=lambda p: get_image_resolution(p)[0], reverse=True)
            remaining_paths.append(group[0])
            exact_duplicates.extend(group[1:])
        else:
            remaining_paths.extend(group)

    # Stage 2: Perceptual Visual Matches
    hash_data: List[Tuple[Path, int, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(compute_dhash, p): p for p in remaining_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            p = future_to_path[future]
            dh = future.result()
            if dh is not None:
                hash_data.append((p, dh, get_image_resolution(p)[0]))

    hash_data.sort(key=lambda item: item[2], reverse=True)
    perceptual_duplicates: List[Path] = []
    kept_images: List[Tuple[Path, int]] = []

    for path, dhash_val, res_area in hash_data:
        is_dup = False
        for keeper_path, keeper_hash in kept_images:
            if similarity_score(dhash_val, keeper_hash) >= similarity_thresh:
                is_dup = True
                perceptual_duplicates.append(path)
                break
        if not is_dup:
            kept_images.append((path, dhash_val))

    all_duplicates = exact_duplicates + perceptual_duplicates
    if not all_duplicates:
        print("[✓] Deduplication complete: 0 duplicates found.")
        return 0

    print(f"[✓] Identified {len(all_duplicates)} duplicate images ({len(exact_duplicates)} exact, {len(perceptual_duplicates)} visual). Purging...")

    if action == "move":
        trash_dir.mkdir(parents=True, exist_ok=True)
        for dup in all_duplicates:
            try:
                dest = trash_dir / dup.name
                if dest.exists():
                    dest = trash_dir / f"{dup.stem}_{get_md5(dup)[:4]}{dup.suffix}"
                shutil.move(str(dup), str(dest))
            except Exception:
                pass
    else:  # delete
        for dup in all_duplicates:
            try:
                dup.unlink()
            except Exception:
                pass

    print(f"[✓] Purged {len(all_duplicates)} duplicate files from output directory.")
    return len(all_duplicates)


# ─── MULTI-PROVIDER SEARCH ENGINES (ZERO API KEYS) ──────────────────────────

def fetch_bing_urls(query: str, limit: int, start_offset: int = 1) -> List[str]:
    """Provider 1: Bing Image Search."""
    urls = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    first = start_offset
    while len(urls) < limit and first <= start_offset + limit * 4:
        try:
            bing_url = f"https://www.bing.com/images/async?q={requests.utils.quote(query)}&first={first}&count=50"
            resp = requests.get(bing_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                found_urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
                if not found_urls:
                    break
                new_urls = 0
                for u in found_urls:
                    clean_u = u.replace("&amp;", "&")
                    if (clean_u.startswith("http://") or clean_u.startswith("https://")) and clean_u not in urls:
                        urls.append(clean_u)
                        new_urls += 1
                if new_urls == 0:
                    break
                first += len(found_urls)
            else:
                break
        except Exception:
            break
    return urls[:limit]


def fetch_flickr_urls(query: str, limit: int) -> List[str]:
    """Provider 2: Flickr Public Feed (Zero API Key, unique photography index)."""
    urls = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        url = f"https://www.flickr.com/services/feeds/photos_public.gne?tags={requests.utils.quote(query)}&format=json&nojsoncallback=1"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            for item in items:
                m_url = item.get("media", {}).get("m")
                if m_url:
                    full_url = m_url.replace("_m.jpg", "_b.jpg")
                    urls.append(full_url)
    except Exception:
        pass
    return urls[:limit]


def fetch_wikimedia_urls(query: str, limit: int) -> List[str]:
    """Provider 3: Wikimedia Commons Open Search API (Zero API key)."""
    urls = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        wiki_url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(limit * 2),
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }
        resp = requests.get(wiki_url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            pages = resp.json().get("query", {}).get("pages", {})
            for p in pages.values():
                for info in p.get("imageinfo", []):
                    u = info.get("url")
                    if u and u.lower().endswith((".jpg", ".png", ".webp", ".jpeg")):
                        urls.append(u)
    except Exception:
        pass
    return urls[:limit]


def fetch_duckduckgo_urls(query: str, limit: int) -> List[str]:
    """Provider 4: DuckDuckGo Image Search (Zero API Key)."""
    urls = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    }
    try:
        res = requests.get(f"https://duckduckgo.com/?q={requests.utils.quote(query)}", headers=headers, timeout=10)
        vqd_match = re.search(r'vqd=([\d-]+)', res.text)
        if vqd_match:
            vqd = vqd_match.group(1)
            i_res = requests.get(f"https://duckduckgo.com/i.js?q={requests.utils.quote(query)}&vqd={vqd}", headers=headers, timeout=10)
            if i_res.status_code == 200:
                results = i_res.json().get("results", [])
                for r in results:
                    img_url = r.get("image")
                    if img_url and img_url not in urls:
                        urls.append(img_url)
    except Exception:
        pass
    return urls[:limit]


def fetch_image_urls_rotated(query: str, limit: int, start_offset: int = 1, provider_round: int = 0) -> Tuple[List[str], str]:
    """Rotates search engines depending on round number to maximize unique images using exact search terms."""
    provider_name = "Bing Search Engine"
    urls = []

    if provider_round == 0:
        provider_name = "Bing Search Engine"
        urls = fetch_bing_urls(query, limit=limit, start_offset=start_offset)
    elif provider_round == 1:
        provider_name = "Bing Search (Offset)"
        urls = fetch_bing_urls(query, limit=limit, start_offset=start_offset + limit)
    elif provider_round == 2:
        provider_name = "Wikimedia Commons Engine"
        urls = fetch_wikimedia_urls(query, limit=limit)
        if len(urls) < limit:
            urls.extend(fetch_bing_urls(query, limit - len(urls), start_offset=start_offset + limit * 2))
    else:
        provider_name = "Multi-Engine Combined Aggregator"
        urls = fetch_bing_urls(query, limit=limit, start_offset=start_offset + limit * provider_round)
        if len(urls) < limit:
            urls.extend(fetch_wikimedia_urls(query, limit - len(urls)))

    return urls[:limit], provider_name


# ─── DOWNLOAD & TOPUP PIPELINE ────────────────────────────────────────────────

def download_single_image(url: str, output_dir: Path, query_prefix: str, index: int, timeout: int = 10) -> Optional[Path]:
    """Download, validate, and save a single image into the flat output directory."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return None
        
        content = resp.content
        if len(content) < 2048:
            return None

        img = Image.open(io.BytesIO(content))
        img.verify()
        img_format = (img.format or "JPEG").lower()
        ext = ".jpg" if img_format == "jpeg" else (f".{img_format}" if img_format in ("png", "webp", "gif") else ".jpg")

        img_hash = hashlib.md5(content).hexdigest()[:8]
        filename = output_dir / f"{query_prefix}_{index:03d}_{img_hash}{ext}"
        
        with open(filename, "wb") as f:
            f.write(content)
            
        return filename
    except Exception:
        return None


def count_clean_images(target_dir: Path) -> int:
    """Return current count of valid images in flat directory."""
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    return len([f for f in target_dir.glob("*") if f.is_file() and f.suffix.lower() in valid_exts])


def process_batch_download_and_dedupe(
    queries: List[str],
    images_per_query: int,
    base_dir: Path,
    similarity_thresh: float,
    action: str,
    trash_dir: Path,
    max_topup_rounds: int,
    threads: int,
    total_target_override: int = TOTAL_TARGET_CLEAN_COUNT,
) -> int:
    """Main batch engine: downloads flat images, runs deduplication, and performs multi-engine top-up reruns."""
    base_dir.mkdir(parents=True, exist_ok=True)
    total_target_clean = total_target_override if total_target_override > 0 else (len(queries) * images_per_query)
    seen_urls: Set[str] = set()
    query_offsets: Dict[str, int] = {q: 1 for q in queries}

    print(f"==========================================================")
    print(f"=== Starting Multi-Provider Image Downloader & Deduplicator ===")
    print(f"Queries: {queries}")
    print(f"Target Clean Images Goal: {total_target_clean} total (approx {images_per_query} per query)")
    print(f"Output Directory (Flat): {base_dir}")
    print(f"Deduplication Similarity Threshold: {similarity_thresh * 100:.1f}%")
    print(f"==========================================================\n")


    # Initial Pass (Round 0)
    for q in queries:
        prefix = slugify(q)
        urls, provider = fetch_image_urls_rotated(q, limit=images_per_query, start_offset=query_offsets[q], provider_round=0)
        query_offsets[q] += len(urls)
        
        new_urls = [u for u in urls if u not in seen_urls]
        seen_urls.update(new_urls)

        print(f"[+] Initial search via [{provider}] for '{q}' (target: {images_per_query})...")
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(download_single_image, u, base_dir, prefix, idx + 1)
                for idx, u in enumerate(new_urls)
            ]
            for f in concurrent.futures.as_completed(futures):
                if f.result():
                    success_count += 1
        print(f"    Saved {success_count} raw images for '{q}'.")

    # Run Deduplication on Initial Batch
    run_inline_deduplication(base_dir, similarity_thresh=similarity_thresh, action=action, trash_dir=trash_dir, workers=threads)
    current_clean = count_clean_images(base_dir)
    print(f"\n[📊 Status] Current clean image count: {current_clean}/{total_target_clean}")

    # Multi-Engine Top-Up Rerun Loop
    round_num = 1
    while current_clean < total_target_clean and round_num <= max_topup_rounds:
        deficit = total_target_clean - current_clean
        per_query_needed = math.ceil(deficit / len(queries))

        topup_downloaded = 0
        for q in queries:
            prefix = slugify(q)
            urls, provider = fetch_image_urls_rotated(
                q, limit=per_query_needed * 2, start_offset=query_offsets[q], provider_round=round_num
            )
            query_offsets[q] += len(urls)

            new_urls = [u for u in urls if u not in seen_urls]
            seen_urls.update(new_urls)

            if not new_urls:
                continue

            print(f"\n[🔄 Top-Up Round {round_num}/{max_topup_rounds}] Fetching ~{per_query_needed} images for '{q}' using Provider [{provider}]...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [
                    executor.submit(download_single_image, u, base_dir, f"{prefix}_r{round_num}", idx + 1)
                    for idx, u in enumerate(new_urls[:per_query_needed])
                ]
                for f in concurrent.futures.as_completed(futures):
                    if f.result():
                        topup_downloaded += 1

        print(f"  [✓] Downloaded {topup_downloaded} new top-up candidate images across providers.")
        
        # Run Deduplication on Top-Up Batch
        run_inline_deduplication(base_dir, similarity_thresh=similarity_thresh, action=action, trash_dir=trash_dir, workers=threads)
        current_clean = count_clean_images(base_dir)
        print(f"[📊 Status] Clean image count after Round {round_num}: {current_clean}/{total_target_clean}")
        round_num += 1

    print(f"\n==========================================================")
    print(f"[✓] Process Complete! Final clean dataset: {current_clean} images in: {base_dir}")
    return current_clean


# ─── MODAL REMOTE FUNCTION ───────────────────────────────────────────────────

@app.function(
    image=modal_image,
    volumes={"/downloads": downloads_volume},
    timeout=900,
)
def download_and_dedupe_remote(
    queries: List[str],
    images_per_query: int = IMAGES_PER_QUERY,
    similarity_thresh: float = SIMILARITY_THRESHOLD,
) -> bytes:
    """Remote Modal task to download flat images, deduplicate, and return ZIP archive."""
    base_dir = Path("/downloads/flat_batch")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    process_batch_download_and_dedupe(
        queries=queries,
        images_per_query=images_per_query,
        base_dir=base_dir,
        similarity_thresh=similarity_thresh,
        action="delete",
        trash_dir=Path("/tmp/trash"),
        max_topup_rounds=MAX_TOPUP_ROUNDS,
        threads=NUM_THREADS,
    )

    downloads_volume.commit()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in base_dir.glob("*"):
            if file.is_file():
                zf.write(file, file.name)

    return zip_buffer.getvalue()


# ─── ENTRYPOINTS ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(queries: str = "", limit: int = 0):
    """Modal CLI entrypoint (`modal run image_downloader.py`). Uses top config if flags omitted."""
    query_list = [q.strip() for q in queries.split(",") if q.strip()] if queries else SEARCH_QUERIES
    target_limit = limit if limit > 0 else IMAGES_PER_QUERY

    print(f"[*] Running Modal batch download & deduplication for: {query_list}")
    zip_bytes = download_and_dedupe_remote.remote(query_list, images_per_query=target_limit)
    
    output_zip = Path("downloaded_clean_images.zip")
    output_zip.write_bytes(zip_bytes)
    print(f"[✓] Successfully saved cloud package to: {output_zip.resolve()}")


def run_local():
    """Local Python script execution."""
    parser = argparse.ArgumentParser(description="Multi-Provider Image Downloader & Integrated Deduplicator")
    parser.add_argument("--queries", nargs="*", default=None, help=f"Search query terms, separated by commas (default: {SEARCH_QUERIES})")
    parser.add_argument("--limit", type=int, default=IMAGES_PER_QUERY, help=f"Clean images per query (default: {IMAGES_PER_QUERY})")
    parser.add_argument("--total-target", type=int, default=0, help="Overall total target clean images (0 for auto)")
    parser.add_argument("--threads", type=int, default=NUM_THREADS, help=f"Concurrent download threads (default: {NUM_THREADS})")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help=f"Flat output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help=f"Similarity threshold (default: {SIMILARITY_THRESHOLD})")
    parser.add_argument("--action", type=str, choices=["delete", "move"], default=ACTION_ON_DUPLICATES, help=f"Duplicate action (default: {ACTION_ON_DUPLICATES})")

    args = parser.parse_args()

    if args.queries is None:
        queries_list = SEARCH_QUERIES
    else:
        # Join any nargs space-split tokens back into a single string, then split strictly on commas
        raw_str = " ".join(args.queries) if isinstance(args.queries, list) else str(args.queries)
        queries_list = [q.strip() for q in raw_str.split(",") if q.strip()]
        if not queries_list:
            queries_list = SEARCH_QUERIES

    base_dir = Path(args.output).resolve()
    trash_dir = Path(TRASH_DIR).resolve()

    process_batch_download_and_dedupe(
        queries=queries_list,
        images_per_query=args.limit,
        base_dir=base_dir,
        similarity_thresh=args.threshold,
        action=args.action,
        trash_dir=trash_dir,
        max_topup_rounds=MAX_TOPUP_ROUNDS,
        threads=args.threads,
        total_target_override=args.total_target,
    )


if __name__ == "__main__":
    if not any("modal" in arg for arg in sys.argv[:1]) and "MODAL_RUN_ENTRYPOINT" not in os.environ:
        run_local()
