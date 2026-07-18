import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny, EulerDiscreteScheduler
from PIL import Image
from typing import Optional

# Limit PyTorch thread count to (total cores - 1) to leave 1 core free for network processes (Uvicorn and Cloudflared)
# This prevents network packets from being delayed due to CPU starvation during generation.
cores: int = max(1, (os.cpu_count() or 2) - 1)
torch.set_num_threads(cores)
print(f"[Model] Configured PyTorch to use {cores} CPU thread(s).", flush=True)

MODEL_CODE: str = "white"
MODEL_ID: str = "etri-vilab/koala-lightning-700m"
_pipeline: Optional[StableDiffusionXLPipeline] = None

def get_pipeline() -> StableDiffusionXLPipeline:
    global _pipeline
    if _pipeline is None:
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # Always use float32 on CPU to leverage hardware AVX2/AVX-512 acceleration, avoiding slow bfloat16 emulated math.
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        print(f"[Model] Loading base model ID '{MODEL_ID}' on device '{device}'...", flush=True)
        _pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        # Configure EulerDiscreteScheduler with trailing timestep spacing for Koala-Lightning
        print("[Model] Configuring EulerDiscreteScheduler...", flush=True)
        _pipeline.scheduler = EulerDiscreteScheduler.from_config(
            _pipeline.scheduler.config,
            timestep_spacing="trailing"
        )
        
        # Load and set the Tiny VAE (TAESDXl) for SDXL to make VAE decoding instantaneous on CPU
        print("[Model] Loading Tiny AutoEncoder (TAESDXl) VAE for SDXL...", flush=True)
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            torch_dtype=dtype,
            use_safetensors=True
        )
        _pipeline.vae = vae
        _pipeline.to(device)
        print("[Model] Model loaded successfully with KOALA-Lightning-700M and Tiny VAE.", flush=True)
    return _pipeline

@torch.inference_mode()
def generate_image(prompt: str, num_inference_steps: int = 1, guidance_scale: float = 0.0, width: int = 512, height: int = 512) -> Image.Image:
    pipe: StableDiffusionXLPipeline = get_pipeline()
    
    # Auto-adjust defaults for KOALA-Lightning (4 steps, 0.0 guidance)
    steps = 4 if num_inference_steps <= 1 else num_inference_steps
    guidance = 0.0 if guidance_scale <= 0.0 else guidance_scale
    
    # Enforce 512x512 resolution for fast CPU runs
    w = 512 if width <= 0 else width
    h = 512 if height <= 0 else height
    
    print(f"[Model] Starting image generation for prompt: '{prompt}' (steps={steps}, guidance={guidance}, size={w}x{h})...", flush=True)
    
    # Run the model
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=w,
        height=h
    )
    
    image: Image.Image = result.images[0]
    print("[Model] Image generation complete.", flush=True)
    return image
