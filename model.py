import os
from pathlib import Path
from typing import Optional

from PIL import Image

# Sana-Sprint requires diffusers>=0.33.0 for OVSanaSprintPipeline support in optimum-intel.
from optimum.intel.openvino import OVSanaSprintPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# IMPORTANT: use the "_diffusers" suffixed repo, not the bare NVLabs-format
# repo. The diffusers-native `SanaSprintPipeline` (and therefore optimum-intel's
# export=True path, which loads through diffusers under the hood) is confirmed
# to work against this repo specifically -- the un-suffixed repo may be in the
# original NVLabs checkpoint format instead, which diffusers' auto-export
# likely can't consume directly. If export=True fails on this repo, that's
# the first thing to check.
MODEL_ID: str = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"

# Where we cache the converted OpenVINO IR model after first export.
# Persisted via the GitHub Actions cache step (path: ov_cache) so we only
# pay the export cost once.
OV_MODEL_DIR: str = os.environ.get("OV_MODEL_DIR", "./ov_cache/sana-sprint-0.6b-int8")

WIDTH: int = 512
HEIGHT: int = 512
NUM_THREADS: int = 2

_pipeline: Optional[OVSanaSprintPipeline] = None


def get_pipeline() -> OVSanaSprintPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    ov_config = {
        "INFERENCE_NUM_THREADS": NUM_THREADS,
        "PERFORMANCE_HINT": "LATENCY",
    }

    if Path(OV_MODEL_DIR).exists() and any(Path(OV_MODEL_DIR).iterdir()):
        # Fast path: already converted + quantized on a previous run.
        print(f"[Model] Loading cached OpenVINO INT8 model from '{OV_MODEL_DIR}'...", flush=True)
        _pipeline = OVSanaSprintPipeline.from_pretrained(
            OV_MODEL_DIR,
            ov_config=ov_config,
            compile=False,
        )
    else:
        # First run: export from the original PyTorch weights straight to
        # OpenVINO IR with INT8 weight compression, then persist to disk.
        print(f"[Model] No OpenVINO cache found. Exporting '{MODEL_ID}' to OpenVINO INT8...", flush=True)
        _pipeline = OVSanaSprintPipeline.from_pretrained(
            MODEL_ID,
            export=True,
            weight_format="int8",  # nncf-backed weight-only INT8 quantization
            ov_config=ov_config,
            compile=False,
        )
        print(f"[Model] Saving converted model to '{OV_MODEL_DIR}' for future runs...", flush=True)
        Path(OV_MODEL_DIR).mkdir(parents=True, exist_ok=True)
        _pipeline.save_pretrained(OV_MODEL_DIR)

    # Static reshape locks input/output shapes, which lets OpenVINO skip
    # re-planning the graph on every call -- meaningful latency win, but
    # means every generation from this point on MUST use WIDTH x HEIGHT.
    print(f"[Model] Reshaping static shapes to {WIDTH}x{HEIGHT}...", flush=True)
    _pipeline.reshape(batch_size=1, height=HEIGHT, width=WIDTH, num_images_per_prompt=1)

    print("[Model] Compiling OpenVINO graph...", flush=True)
    _pipeline.compile()
    print("[Model] Sana-Sprint OpenVINO INT8 pipeline ready.", flush=True)

    return _pipeline


def generate_image(
    prompt: str,
    num_inference_steps: int = 4,
    guidance_scale: float = 4.5,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Image.Image:
    """
    Sana-Sprint is a consistency-distilled model supporting 1-4 step
    inference. The official demo/reference default is 4 steps -- drop to
    2 for a speed test, but expect a visible quality dip below 4.
    guidance_scale=4.5 mirrors the default exposed in the official Gradio
    demo's "Advanced Settings" -- unlike vanilla LCM, Sana-Sprint keeps real
    classifier-free guidance active rather than pinning guidance_scale=1.0,
    so don't copy that convention over from the old SD1.5-LCM setup.
    """
    pipe = get_pipeline()

    if width != WIDTH or height != HEIGHT:
        print(
            f"[Model] Warning: requested {width}x{height} differs from the "
            f"statically reshaped {WIDTH}x{HEIGHT}. Ignoring override -- "
            f"reshape the pipeline again if you truly need a different size.",
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