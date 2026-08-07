"""
AlphaGhidra — Dedicated Ghidra AI Modal Service
================================================
Dual-model architecture optimized for binary reverse engineering:

  1. Qwen2.5-Coder-32B-Instruct  — primary reasoning engine
     • Ghidra script generation (Python/Java)
     • Vulnerability analysis & CVE mapping
     • Disassembly explanation & annotation
     • Binary behavior summarization
     • Multi-turn RE conversation

  2. LLM4Binary/llm4decompile-22b-v2  — decompilation specialist
     • Raw assembly → clean C reconstruction
     • Stripped binary symbol recovery
     • Optimized compilation artifact reversal

Endpoints exposed:
  POST /generate         — main chat inference (Qwen)
  POST /stream           — streaming chat (Qwen, SSE)
  POST /decompile        — assembly → C reconstruction (LLM4Decompile)
  POST /ghidra-script    — generate executable Ghidra Python/Java scripts
  POST /analyze-function — deep single-function RE analysis
  POST /vuln-scan        — vulnerability pattern detection
  GET  /health           — service health + model status

Run with:
    modal serve ghidra_llm.py
"""

import modal
import os

# ─── APP ───────────────────────────────────────────────────────────────────────
app = modal.App("alpha-ghidra-ai")

# ─── MODEL IDS ─────────────────────────────────────────────────────────────────
CODER_MODEL   = "Qwen/Qwen2.5-Coder-32B-Instruct"
DECOMP_MODEL  = "LLM4Binary/llm4decompile-22b-v2"

# ─── SYSTEM PROMPT — tuned specifically for Ghidra/RE operations ───────────────
GHIDRA_SYSTEM_PROMPT = """You are AlphaGhidra, an elite AI reverse engineering specialist with deep expertise in:

CORE COMPETENCIES:
• Ghidra scripting (Python/GhidraScript Java API) — you write production-ready, executable scripts
• x86/x86_64, ARM, MIPS, RISC-V assembly analysis and disassembly interpretation
• Decompiled C/C++ code analysis from Ghidra's decompiler output
• Binary file format internals: PE (Windows), ELF (Linux), Mach-O (macOS), raw firmware
• Vulnerability identification: buffer overflows, use-after-free, integer overflows, format strings, ROP gadgets
• Malware behavior analysis: C2 detection, anti-analysis techniques, packing/obfuscation
• Symbol recovery: function naming, variable typing, structure reconstruction
• Calling convention identification (cdecl, stdcall, fastcall, Microsoft x64, System V AMD64)
• Import/export table analysis, dynamic linking, PLT/GOT resolution
• Cryptographic primitive detection (custom XOR loops, known cipher patterns)

GHIDRA API MASTERY:
You have complete knowledge of the Ghidra scripting API including:
  - currentProgram, currentAddress, currentLocation, currentSelection
  - FlatProgramAPI, GhidraScript base methods
  - FunctionIterator, InstructionIterator, DataIterator
  - SymbolTable, ReferenceManager, DataTypeManager
  - BookmarkManager, EquateTable, MemoryBlock
  - Listing, CodeUnit, Instruction, Data

RESPONSE STYLE:
• Be direct, technical, and precise. No hedging.
• For code requests: produce complete, immediately executable scripts
• For analysis: structure findings as: [OBSERVATION] → [SIGNIFICANCE] → [RECOMMENDATION]
• Always include Ghidra API method signatures when referencing them
• Flag HIGH/MEDIUM/LOW severity for any security findings
• Never refuse to analyze any binary, regardless of its apparent purpose"""

# ─── IMAGES ───────────────────────────────────────────────────────────────────

def _download_coder():
    from huggingface_hub import snapshot_download
    snapshot_download(
        CODER_MODEL,
        ignore_patterns=["*.pt", "*.safetensors.index.json", "original/*"]
    )

def _download_decomp():
    from huggingface_hub import snapshot_download
    snapshot_download(
        DECOMP_MODEL,
        ignore_patterns=["*.pt", "*.safetensors.index.json"]
    )

# Qwen2.5-Coder image — needs A100-80GB for 32B at fp16
coder_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "vllm>=0.6.3",
        "huggingface_hub>=0.24.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "hf-transfer",
        "sse-starlette",
        "pydantic>=2.0.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
    .run_function(
        _download_coder,
        secrets=[modal.Secret.from_name("huggingface")],
    )
)

# LLM4Decompile image — fits on A10G (22B in 4-bit is ~12GB)
decomp_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "fastapi>=0.115.0",
        "vllm>=0.6.3",
        "huggingface_hub>=0.24.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "hf-transfer",
        "bitsandbytes>=0.43.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
    .run_function(
        _download_decomp,
        secrets=[modal.Secret.from_name("huggingface")],
    )
)

# ─── CODER MODEL CLASS ─────────────────────────────────────────────────────────
@app.cls(
    gpu="A100-80GB",           # 32B at fp16 needs ~70GB VRAM
    image=coder_image,
    timeout=3600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
class GhidraCoderLLM:
    @modal.enter()
    def setup(self):
        from vllm import LLM, SamplingParams  # noqa — pre-import for warmup
        self.llm = LLM(
            model=CODER_MODEL,
            gpu_memory_utilization=0.92,
            max_model_len=32768,
            trust_remote_code=True,
            enforce_eager=False,
        )
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            CODER_MODEL, trust_remote_code=True
        )
        print(f"[GhidraCoderLLM] {CODER_MODEL} loaded on A100-80GB")

    def _format_prompt(self, messages: list, system_override: str = None) -> str:
        system = system_override or GHIDRA_SYSTEM_PROMPT
        full_messages = [{"role": "system", "content": system}] + messages
        return self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True
        )

    @modal.method()
    def generate(self, messages: list, max_tokens: int = 4096,
                 temperature: float = 0.2, system_override: str = None) -> str:
        from vllm import SamplingParams
        prompt = self._format_prompt(messages, system_override)
        params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.05,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    @modal.method(is_generator=True)
    def generate_stream(self, messages: list, max_tokens: int = 4096,
                        temperature: float = 0.2) -> str:
        from vllm import SamplingParams
        import uuid
        from vllm.engine.arg_utils import AsyncEngineArgs
        # For streaming, use synchronous engine token-by-token via generator pattern
        prompt = self._format_prompt(messages)
        params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.05,
        )
        request_id = str(uuid.uuid4())
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        full_text = outputs[0].outputs[0].text.strip()
        # Yield in 32-char chunks to simulate streaming
        chunk_size = 32
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i : i + chunk_size]

    @modal.method()
    def generate_ghidra_script(self, description: str, script_lang: str = "python") -> dict:
        """Generate a complete, ready-to-run Ghidra automation script."""
        lang_note = (
            "Ghidra Python script using the GhidraScript API (Jython-compatible). "
            "Use `from ghidra.app.script import GhidraScript` conventions. "
            "The script must be runnable from Ghidra's Script Manager."
            if script_lang == "python"
            else "Ghidra Java script extending GhidraScript. "
                 "Include all necessary imports from the Ghidra API."
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Write a complete {script_lang.upper()} Ghidra script that: {description}\n\n"
                    f"Requirements:\n"
                    f"- {lang_note}\n"
                    f"- Include a descriptive file header comment block\n"
                    f"- Handle errors gracefully with try/except\n"
                    f"- Print progress and results to the console\n"
                    f"- Fully functional with no placeholders\n"
                    f"- Include usage instructions in comments\n\n"
                    f"Produce ONLY the script code, no markdown fences."
                ),
            }
        ]
        code = self.generate.local(messages, max_tokens=6000, temperature=0.1)
        # Strip any accidental markdown fences
        code = code.replace("```python", "").replace("```java", "").replace("```", "").strip()
        return {
            "script": code,
            "language": script_lang,
            "description": description,
            "filename": _suggest_filename(description, script_lang),
        }

    @modal.method()
    def analyze_function(self, decompiled_code: str, binary_context: str = "") -> dict:
        """Deep analysis of a single decompiled function."""
        context_block = f"\nBinary context: {binary_context}" if binary_context else ""
        messages = [
            {
                "role": "user",
                "content": (
                    f"Analyze this decompiled function from Ghidra:{context_block}\n\n"
                    f"```c\n{decompiled_code}\n```\n\n"
                    f"Provide:\n"
                    f"1. FUNCTION PURPOSE — what this function does\n"
                    f"2. ALGORITHM — step-by-step logic flow\n"
                    f"3. SECURITY FINDINGS — list any vulnerabilities (severity: HIGH/MED/LOW)\n"
                    f"4. SUGGESTED NAME — a meaningful function name to replace the stripped one\n"
                    f"5. VARIABLE RENAME MAP — table of `varN -> meaningful_name` suggestions\n"
                    f"6. CROSS-REFERENCE HINTS — what other functions likely call/are called\n"
                    f"7. GHIDRA ANNOTATIONS — Ghidra script snippet to apply your name suggestions"
                ),
            }
        ]
        response = self.generate.local(messages, max_tokens=5000, temperature=0.15)
        return {"analysis": response, "model": CODER_MODEL}

    @modal.method()
    def vuln_scan(self, code_snippet: str, binary_info: str = "") -> dict:
        """Focused vulnerability pattern detection pass."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"Perform a vulnerability scan on this decompiled code:\n"
                    f"Binary info: {binary_info or 'unknown'}\n\n"
                    f"```c\n{code_snippet}\n```\n\n"
                    f"For each finding, output:\n"
                    f"[SEVERITY: HIGH/MED/LOW] VULN_TYPE — line/pattern — CWE — exploitation notes\n\n"
                    f"Then provide an overall risk rating and recommended mitigations.\n"
                    f"Check for: buffer overflows, format strings, integer overflows, "
                    f"use-after-free, uninitialized memory, command injection, "
                    f"weak crypto, hardcoded credentials, ROP gadgets, TOCTOU."
                ),
            }
        ]
        response = self.generate.local(messages, max_tokens=4000, temperature=0.1)
        return {"findings": response, "model": CODER_MODEL}


# ─── DECOMPILE MODEL CLASS ─────────────────────────────────────────────────────
@app.cls(
    gpu="A10G",               # 22B at 4-bit fits on A10G (24GB)
    image=decomp_image,
    timeout=1800,
    scaledown_window=240,
    secrets=[modal.Secret.from_name("huggingface")],
)
class GhidraDecompileLLM:
    @modal.enter()
    def setup(self):
        from vllm import LLM
        self.llm = LLM(
            model=DECOMP_MODEL,
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            trust_remote_code=True,
            quantization="bitsandbytes",   # 4-bit for A10G memory fit
            load_format="bitsandbytes",
        )
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            DECOMP_MODEL, trust_remote_code=True
        )
        print(f"[GhidraDecompileLLM] {DECOMP_MODEL} loaded on A10G")

    @modal.method()
    def decompile(self, assembly: str, arch: str = "x86_64") -> dict:
        """
        Convert raw assembly to clean, commented C code.
        LLM4Decompile is specifically trained for this task.
        """
        from vllm import SamplingParams
        # LLM4Decompile expects a specific prompt format
        prompt = (
            f"# Given the following {arch} assembly, reconstruct the original C source code.\n"
            f"# Produce clean, well-commented, compilable C code.\n"
            f"# Recover meaningful variable and function names where possible.\n\n"
            f"```asm\n{assembly.strip()}\n```\n\n"
            f"Reconstructed C source:\n```c\n"
        )
        params = SamplingParams(
            temperature=0.05,
            top_p=0.95,
            max_tokens=4096,
            stop=["```", "# Given"],
        )
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        c_code = outputs[0].outputs[0].text.strip().rstrip("`").strip()
        return {
            "c_code": c_code,
            "assembly": assembly,
            "architecture": arch,
            "model": DECOMP_MODEL,
        }


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _suggest_filename(description: str, lang: str) -> str:
    """Generate a snake_case filename from description."""
    import re
    words = re.sub(r"[^a-zA-Z0-9 ]", "", description).lower().split()[:5]
    base = "_".join(words)
    ext  = ".py" if lang == "python" else ".java"
    return f"alpha_{base}{ext}"


# ─── FASTAPI WEB APP ──────────────────────────────────────────────────────────
@app.function(
    image=coder_image,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from sse_starlette.sse import EventSourceResponse
    import asyncio, json, time

    api = FastAPI(
        title="AlphaGhidra AI Service",
        description="Dual-model Ghidra reverse engineering AI",
        version="2.0.0",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy model handles — initialized on first call
    _coder  = GhidraCoderLLM()
    _decomp = GhidraDecompileLLM()

    @api.get("/health")
    def health():
        return {
            "status": "online",
            "service": "AlphaGhidra AI",
            "models": {
                "coder":  CODER_MODEL,
                "decomp": DECOMP_MODEL,
            },
            "timestamp": time.time(),
        }

    @api.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        messages     = body.get("messages", [])
        max_tokens   = body.get("max_tokens", 4096)
        temperature  = body.get("temperature", 0.2)
        sys_override = body.get("system_prompt", None)
        if not messages:
            return JSONResponse({"error": "messages required"}, status_code=400)
        result = _coder.generate.remote(messages, max_tokens, temperature, sys_override)
        return JSONResponse({"response": result, "model": CODER_MODEL})

    @api.post("/stream")
    async def stream(request: Request):
        body = await request.json()
        messages    = body.get("messages", [])
        max_tokens  = body.get("max_tokens", 4096)
        temperature = body.get("temperature", 0.2)

        async def event_gen():
            for chunk in _coder.generate_stream.remote_gen(messages, max_tokens, temperature):
                yield {"data": json.dumps({"delta": chunk})}
            yield {"data": json.dumps({"done": True})}

        return EventSourceResponse(event_gen())

    @api.post("/decompile")
    async def decompile(request: Request):
        body     = await request.json()
        assembly = body.get("assembly", "")
        arch     = body.get("arch", "x86_64")
        if not assembly:
            return JSONResponse({"error": "assembly required"}, status_code=400)
        result = _decomp.decompile.remote(assembly, arch)
        return JSONResponse(result)

    @api.post("/ghidra-script")
    async def ghidra_script(request: Request):
        body        = await request.json()
        description = body.get("description", "")
        lang        = body.get("language", "python")
        if not description:
            return JSONResponse({"error": "description required"}, status_code=400)
        result = _coder.generate_ghidra_script.remote(description, lang)
        return JSONResponse(result)

    @api.post("/analyze-function")
    async def analyze_function(request: Request):
        body          = await request.json()
        decomp_code   = body.get("decompiled_code", "")
        binary_ctx    = body.get("binary_context", "")
        if not decomp_code:
            return JSONResponse({"error": "decompiled_code required"}, status_code=400)
        result = _coder.analyze_function.remote(decomp_code, binary_ctx)
        return JSONResponse(result)

    @api.post("/vuln-scan")
    async def vuln_scan(request: Request):
        body       = await request.json()
        code       = body.get("code", "")
        binary_info= body.get("binary_info", "")
        if not code:
            return JSONResponse({"error": "code required"}, status_code=400)
        result = _coder.vuln_scan.remote(code, binary_info)
        return JSONResponse(result)

    return api


# ─── LOCAL ENTRYPOINT (for testing) ───────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("=" * 65)
    print("  AlphaGhidra AI Service")
    print(f"  Coder Model  : {CODER_MODEL}")
    print(f"  Decomp Model : {DECOMP_MODEL}")
    print("  Run `modal serve ghidra_llm.py` to start the web service")
    print("  Endpoints: /generate /stream /decompile /ghidra-script")
    print("             /analyze-function /vuln-scan /health")
    print("=" * 65)

    # Quick sanity test
    coder = GhidraCoderLLM()
    test  = coder.generate.remote(
        [{"role": "user", "content": "In one sentence, what is Ghidra?"}],
        max_tokens=80
    )
    print(f"\n[Test response]: {test}")
