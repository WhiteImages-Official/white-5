import os
from typing import Optional

from PIL import Image
from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionXLPipeline

# ---------------------------------------------------------------------------
# Pre-converted OpenVINO INT8 SDXL-Turbo. No export step -- this is a
# ready-to-load IR model published by rupeshs (same maintainer as the
# original working LCM-Dreamshaper model), so it carries over the exact
# pattern that's already proven to load without OOM on this box.
#
# Real reported reference point (not a guess): ~5s per image on an Intel
# Core i7 desktop CPU at 512x512, 1 step, per the model's own usage notes.
# Your 2 vCPU cloud runner is very likely weaker than a full i7, so expect
# somewhat more than 5s -- but this is a real anchor, unlike SANA-Sprint
# where no CPU number exists anywhere.
#
# Quality note: SDXL-Turbo at 1-4 steps was preferred by human evaluators
# over 4-step LCM-XL, and even over a full 50-step SDXL run at just 4 steps
# -- a genuine quality step up from the SD1.5-LCM setup, not a lateral move.
# ---------------------------------------------------------------------------
MODEL_ID: str = "rupeshs/sdxl-turbo-openvino-int8"

WIDTH: int = 512
HEIGHT: int = 512
NUM_THREADS: int = 2

_pipeline: Optional[OVStableDiffusionXLPipeline] = None


def get_pipeline() -> OVStableDiffusionXLPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print(f"[Model] Loading pre-converted OpenVINO INT8 model '{MODEL_ID}'...", flush=True)

    ov_config = {
        "INFERENCE_NUM_THREADS": NUM_THREADS,
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": "",
    }

    _pipeline = OVStableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        ov_config=ov_config,
        compile=False,
    )

    print(f"[Model] Reshaping static shapes to {WIDTH}x{HEIGHT}...", flush=True)
    _pipeline.reshape(batch_size=1, height=HEIGHT, width=WIDTH, num_images_per_prompt=1)

    print("[Model] Compiling OpenVINO graph...", flush=True)
    _pipeline.compile()
    print("[Model] SDXL-Turbo OpenVINO INT8 pipeline ready.", flush=True)

    return _pipeline


def generate_image(
    prompt: str,
    num_inference_steps: int = 2,
    guidance_scale: float = 1.0,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Image.Image:
    """
    SDXL-Turbo was trained WITHOUT classifier-free guidance -- keep
    guidance_scale=1.0 (i.e. effectively disabled), matching both the
    model's own example code and its training setup. Don't reintroduce
    a higher CFG value here; it wasn't trained for it and quality will
    likely get worse, not better.

    1 step is the model's native fast mode. Try 2-4 steps (see the
    SDXL-Lightning-2steps-openvino-int8 sibling model for a 2-step-native
    alternative) if 1-step output looks too soft for your use case --
    each extra step costs roughly proportional extra time on CPU.
    """
    pipe = get_pipeline()

    if width != WIDTH or height != HEIGHT:
        print(
            f"[Model] Warning: requested {width}x{height} differs from the "
            f"statically reshaped {WIDTH}x{HEIGHT}. Ignoring override.",
            flush=True,
        )

    print(
        f"[Model] Generating for prompt: '{prompt}' "
        f"(steps={num_inference_steps}, guidance={guidance_scale})...",
        flush=True,
    )

    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=HEIGHT,
        width=WIDTH,
        num_images_per_prompt=1,
    )

    image: Image.Image = result.images[0]
    print("[Model] Image generation complete.", flush=True)
    return image