import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny, EulerDiscreteScheduler
from PIL import Image
from typing import Optional
from huggingface_hub import hf_hub_download

# Limit PyTorch thread count to (total cores - 1) to leave 1 core free for network processes (Uvicorn and Cloudflared)
# This prevents network packets from being delayed due to CPU starvation during generation.
cores: int = max(1, (os.cpu_count() or 2) - 1)
torch.set_num_threads(cores)
print(f"[Model] Configured PyTorch to use {cores} CPU thread(s).", flush=True)

MODEL_CODE: str = "white"
MODEL_ID: str = "segmind/SSD-1B"
LORA_REPO: str = "ByteDance/SDXL-Lightning"
LORA_FILE: str = "sdxl_lightning_4step_lora.safetensors"
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
        
        # Load SDXL-Lightning LoRA
        print(f"[Model] Loading SDXL-Lightning LoRA weights '{LORA_FILE}'...", flush=True)
        lora_path = hf_hub_download(repo_id=LORA_REPO, filename=LORA_FILE)
        _pipeline.load_lora_weights(lora_path)
        _pipeline.fuse_lora()
        
        # Configure EulerDiscreteScheduler with trailing timestep spacing for SDXL-Lightning
        print("[Model] Configuring EulerDiscreteScheduler...", flush=True)
        _pipeline.scheduler = EulerDiscreteScheduler.from_config(
            _pipeline.scheduler.config,
            timestep_spacing="trailing"
        )
        
        # Load and set the Tiny VAE (TAESDXx) for SDXL to make VAE decoding instantaneous on CPU
        print("[Model] Loading Tiny AutoEncoder (TAESDXx) VAE for SDXL...", flush=True)
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxx",
            torch_dtype=dtype,
            use_safetensors=True
        )
        _pipeline.vae = vae
        _pipeline.to(device)
        print("[Model] Model loaded successfully with Segmind SSD-1B, SDXL-Lightning 4-Step LoRA and Tiny VAE.", flush=True)
    return _pipeline

@torch.inference_mode()
def generate_image(prompt: str, num_inference_steps: int = 1, guidance_scale: float = 0.0, width: int = 512, height: int = 512) -> Image.Image:
    pipe: StableDiffusionXLPipeline = get_pipeline()
    
    # Auto-adjust defaults for SDXL-Lightning (4 steps, 0.0 guidance)
    steps = 4 if num_inference_steps <= 1 else num_inference_steps
    guidance = 0.0 if guidance_scale <= 0.0 else guidance_scale
    
    # Map 512 defaults to 1024 for high-resolution SDXL output
    w = 1024 if width <= 512 else width
    h = 1024 if height <= 512 else height
    
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
