import json
import httpx
from fastmcp import FastMCP

TXT2IMG_URL = "https://josh627764--text-to-image-sdxl-merger-inference-web.modal.run"

mcp = FastMCP("Modal MCP Proxy Server")

@mcp.tool()
async def generate_image(
    prompt: str, 
    negative_prompt: str = "deformed, mutated, bad anatomy, bad proportions, disfigured, extra limbs, missing limbs, worst quality, low quality, blurry, distorted", 
    steps: int = 20, 
    cfg_scale: float = 7.0
) -> str:
    """Generate an image using the Modal Stable Diffusion XL proxy"""
    
    print(f"[Modal MCP Proxy] Generating image for prompt: {prompt}")

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": str(cfg_scale),
        "batch_size": 1,
        "seed": -1,
        "scheduler": "Euler",
        "lora": "none",
        "JuggernautXL": 1,
        "CyberRealisticXL": 0,
        "enhance_prompt": "True"
    }

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.get(TXT2IMG_URL, params=params)
            response.raise_for_status()
            
            b64_image = None
            for line in response.text.splitlines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "image_b64" in data:
                            b64s = data["image_b64"]
                            b64_image = b64s[0] if isinstance(b64s, list) else b64s
                    except:
                        pass

            if not b64_image:
                return "Failed to generate image: No base64 image returned from Modal."

            # Return image as a base64 markdown string.
            # This is universally supported by all MCP clients as TextContent.
            return f"![Generated Image](data:image/png;base64,{b64_image})"

    except Exception as e:
        return f"Error generating image: {str(e)}"

if __name__ == "__main__":
    print("=" * 60)
    print("  AlphaCore Modal FastMCP Proxy")
    print("  Listening on port 7860")
    print("  Point Off Grid AI Connectors to: http://localhost:7860/sse")
    print("=" * 60)
    # Start the FastMCP server over SSE
    mcp.run(transport="stdio")
