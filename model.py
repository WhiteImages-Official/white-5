import os
import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from PIL import Image
from typing import Optional

# Configure PyTorch to use all available CPU cores to maximize generation speed on the runner
cores: int = os.cpu_count() or 2
torch.set_num_threads(cores)
print(f"[Model] Configured PyTorch to use {cores} CPU thread(s).", flush=True)

MODEL_CODE: str = "white"
MODEL_ID: str = "emilianJR/epiCRealism"
LORA_REPO: str = "latent-consistency/lcm-lora-sdv1-5"
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
        
        # Load LCM-LoRA
        print(f"[Model] Loading LCM-LoRA weights from repository '{LORA_REPO}'...", flush=True)
        _pipeline.load_lora_weights(LORA_REPO)
        _pipeline.fuse_lora()
        
        # Configure LCMScheduler for latent consistency inference
        print("[Model] Configuring LCMScheduler...", flush=True)
        _pipeline.scheduler = LCMScheduler.from_config(_pipeline.scheduler.config)
        
        _pipeline.to(device)
        print("[Model] Model loaded successfully with epiCRealism (Native VAE) and LCM-LoRA.", flush=True)
    return _pipeline

@torch.inference_mode()
def generate_image(prompt: str, num_inference_steps: int = 1, guidance_scale: float = 0.0, width: int = 512, height: int = 512) -> Image.Image:
    pipe: AutoPipelineForText2Image = get_pipeline()
    
    # Auto-adjust defaults for LCM-LoRA (4 steps, 5.0 guidance)
    steps = 4 if num_inference_steps <= 1 else num_inference_steps
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
