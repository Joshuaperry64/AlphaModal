import modal

app = modal.App("xtts-api-server")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "portaudio19-dev", "gcc", "g++", "clang")
    .pip_install(
        "torch==2.1.2+cu121", 
        "torchaudio==2.1.2+cu121", 
        extra_options="--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install("xtts-api-server", "pydantic==1.10.13")
    .run_commands("sed -i 's/if issubclass(field_type, Serializable):/if isinstance(field_type, type) and issubclass(field_type, Serializable):/g' /usr/local/lib/python3.10/site-packages/coqpit/coqpit.py")
    .env({"COQUI_TOS_AGREED": "1"})
)

volume = modal.Volume.from_name("xtts-voices", create_if_missing=True)

@app.cls(
    image=image,
    gpu="T4",
    volumes={"/voices": volume},
    timeout=20 * 60,
    scaledown_window=300,
)
class TTSModel:
    @modal.enter()
    def start_server(self):
        import subprocess
        import time
        import os
        import requests

        # Download a default female voice for lacey to use instantly
        default_voice_path = "/voices/lacey.wav"
        if not os.path.exists(default_voice_path):
            print(f"Downloading default voice sample to {default_voice_path}...")
            import urllib.request
            urllib.request.urlretrieve("https://github.com/daswer123/xtts-api-server/raw/main/example/female.wav", default_voice_path)
        
        # Start the official xtts-api-server CLI in the background
        cmd = [
            "python", "-m", "xtts_api_server",
            "-p", "8020",
            "-hs", "0.0.0.0",
            "-sf", "/voices",
            "-o", "/tmp"
        ]
        
        print("Starting XTTS API Server...")
        self.process = subprocess.Popen(cmd)
        
        # Wait for boot
        url = "http://localhost:8020/docs"
        for _ in range(120):
            try:
                if requests.get(url).status_code == 200:
                    print("XTTS server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        raise RuntimeError("XTTS failed to start within timeout.")

    @modal.asgi_app()
    def fastapi_app(self):
        import fastapi
        import httpx
        from starlette.requests import Request
        from starlette.responses import StreamingResponse

        web_app = fastapi.FastAPI()
        target_url = "http://localhost:8020"

        @web_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def proxy(request: Request, path: str):
            client = httpx.AsyncClient(base_url=target_url, timeout=120.0)
            url = f"/{path}"
            
            headers = dict(request.headers)
            headers.pop("host", None)
            
            body_bytes = await request.body()
            
            if path == "v1/audio/speech" and request.method == "POST":
                import json
                try:
                    body = json.loads(body_bytes)
                    text = body.get("input", "")
                    
                    xtts_payload = {
                        "text": text,
                        "speaker_wav": "lacey",
                        "language": "en"
                    }
                    req = client.build_request(
                        "POST",
                        "/tts_to_audio/",
                        json=xtts_payload,
                    )
                    response = await client.send(req, stream=True)
                    return StreamingResponse(
                        response.aiter_raw(),
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )
                except Exception as e:
                    import fastapi
                    return fastapi.responses.JSONResponse({"error": str(e)}, status_code=500)
            
            req = client.build_request(
                request.method,
                url,
                headers=headers,
                content=body_bytes,
                params=request.query_params,
            )
            
            response = await client.send(req, stream=True)
            
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=dict(response.headers),
            )

        return web_app
