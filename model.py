import os
from PIL import Image
from typing import Optional
from optimum.intel.openvino import OVStableDiffusionPipeline
from diffusers import LCMScheduler

MODEL_CODE: str = "white"
MODEL_ID: str = "rupeshs/sd15-lcm-square-openvino-int8"
_pipeline: Optional[OVStableDiffusionPipeline] = None

def get_pipeline() -> OVStableDiffusionPipeline:
    global _pipeline
    if _pipeline is None:
        print(f"[Model] Loading OpenVINO INT8 model ID '{MODEL_ID}' on CPU...", flush=True)
        
        # Configure OpenVINO inference settings (Threads: 2, Performance: Latency)
        ov_config = {
            "INFERENCE_NUM_THREADS": 2,
            "PERFORMANCE_HINT": "LATENCY"
        }
        
        # Load the pre-converted OpenVINO pipeline with safety checker disabled
        _pipeline = OVStableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            ov_config=ov_config,
            safety_checker=None,
            compile=False
        )
        
        # Configure LCMScheduler
        print("[Model] Configuring LCMScheduler...", flush=True)
        _pipeline.scheduler = LCMScheduler.from_config(_pipeline.scheduler.config)
        
        # Statically reshape the model shapes to 512x512 to optimize memory allocation and speed
        print("[Model] Reshaping OpenVINO model static shapes to 512x512...", flush=True)
        _pipeline.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
        
        # Load and inject the Tiny VAE (TAESD) pre-converted for OpenVINO to make VAE decoding instantaneous
        try:
            print("[Model] Injecting Tiny VAE (TAESD) OpenVINO decoder...", flush=True)
            from optimum.intel.openvino.modeling_diffusion import OVModelVaeDecoder, OVBaseModel
            from huggingface_hub import snapshot_download
            
            # Download TAESD OpenVINO weights (SD 1.5 variant)
            taesd_dir = snapshot_download(repo_id="deinferno/taesd-openvino")
            
            class CustomOVModelVaeDecoder(OVModelVaeDecoder):
                def __init__(self, model, parent_model, ov_config=None, model_dir=None):
                    super().__init__(model, parent_model, ov_config, "vae_decoder", model_dir)
            
            _pipeline.vae_decoder = CustomOVModelVaeDecoder(
                model=OVBaseModel.load_model(f"{taesd_dir}/vae_decoder/openvino_model.xml"),
                parent_model=_pipeline,
                model_dir=taesd_dir
            )
            print("[Model] Tiny VAE injected successfully.", flush=True)
        except Exception as vae_err:
            print(f"[Model] Warning: Failed to inject Tiny VAE: {vae_err}. Using default OpenVINO VAE.", flush=True)
        
        # Compile OpenVINO graph
        print("[Model] Compiling OpenVINO graph...", flush=True)
        _pipeline.compile()
        print("[Model] OpenVINO model loaded and compiled successfully.", flush=True)
    return _pipeline

def generate_image(prompt: str, num_inference_steps: int = 1, guidance_scale: float = 0.0, width: int = 512, height: int = 512) -> Image.Image:
    pipe = get_pipeline()
    
    # Auto-adjust defaults for LCM OpenVINO (4 steps, 1.0 guidance)
    steps = 4 if num_inference_steps <= 1 else num_inference_steps
    guidance = 1.0 if guidance_scale <= 0.0 else guidance_scale
    
    print(f"[Model] Starting OpenVINO INT8 generation for prompt: '{prompt}' (steps={steps}, guidance={guidance})...", flush=True)
    
    # Run OpenVINO inference (resolution locked at 512x512 for static shape optimizations)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=512,
        width=512,
        num_images_per_prompt=1
    )
    
    image: Image.Image = result.images[0]
    print("[Model] Image generation complete.", flush=True)
    return image
