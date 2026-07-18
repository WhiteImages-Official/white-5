import os
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from PIL import Image
from typing import Optional
from huggingface_hub import hf_hub_download

# Limit PyTorch thread count to (total cores - 1) to leave 1 core free for network processes (Uvicorn and Cloudflared)
# This prevents network packets from being delayed due to CPU starvation during generation.
cores: int = max(1, (os.cpu_count() or 2) - 1)
torch.set_num_threads(cores)
print(f"[Model] Configured PyTorch to use {cores} CPU thread(s).", flush=True)

MODEL_CODE: str = "white"
MODEL_ID: str = "emilianJR/epiCRealism"
LORA_REPO: str = "ByteDance/Hyper-SD"
LORA_FILE: str = "Hyper-SD15-8steps-CFG-lora.safetensors"
_pipeline: Optional[AutoPipelineForText2Image] = None

def get_pipeline() -> AutoPipelineForText2Image:
    global _pipeline
    if _pipeline is None:
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # Always use float32 on CPU to leverage hardware AVX2/AVX-512 acceleration, avoiding slow bfloat16 emulated math.
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        print(f"[Model] Loading base model ID '{MODEL_ID}' on device '{device}'...", flush=True)
        _pipeline = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        # Load Hyper-SD LoRA
        print(f"[Model] Loading Hyper-SD LoRA weights '{LORA_FILE}' from repository '{LORA_REPO}'...", flush=True)
        lora_path = hf_hub_download(repo_id=LORA_REPO, filename=LORA_FILE)
        _pipeline.load_lora_weights(lora_path)
        _pipeline.fuse_lora()
        
        # Configure DDIMScheduler with trailing timestep spacing and disable sample clipping for Hyper-SD
        print("[Model] Configuring DDIMScheduler...", flush=True)
        _pipeline.scheduler = DDIMScheduler.from_config(
            _pipeline.scheduler.config,
            timestep_spacing="trailing",
            clip_sample=False
        )
        
        _pipeline.to(device)
        print("[Model] Model loaded successfully with epiCRealism (Native VAE) and 8-Step CFG LoRA.", flush=True)
    return _pipeline

@torch.inference_mode()
def generate_image(prompt: str, num_inference_steps: int = 1, guidance_scale: float = 0.0, width: int = 512, height: int = 512) -> Image.Image:
    pipe: AutoPipelineForText2Image = get_pipeline()
    
    # Auto-adjust defaults to work beautifully with 8-Step CFG LoRA (8 steps, 5.0 guidance)
    steps = 8 if num_inference_steps <= 1 else num_inference_steps
    guidance = 5.0 if guidance_scale <= 1.0 else guidance_scale
    
    # Force 512x512 resolution for fast CPU runs (avoids 1024x1024 latency)
    w = 512 if width <= 0 else width
    h = 512 if height <= 0 else height
    
    print(f"[Model] Starting image generation for prompt: '{prompt}' (steps={steps}, guidance={guidance}, size={w}x{h})...", flush=True)
    
    # Run the model with a default negative prompt to avoid distorted anatomy/noise
    result = pipe(
        prompt=prompt,
        negative_prompt="ugly, deformed, noisy, blurry, distorted, low quality, bad anatomy, bad hands, out of focus",
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=w,
        height=h
    )
    
    image: Image.Image = result.images[0]
    print("[Model] Image generation complete.", flush=True)
    return image
