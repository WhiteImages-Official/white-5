import os
import time
import json
import base64
import re
import io
import subprocess
import asyncio
import threading
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from github import Github, Auth
from github.GithubException import UnknownObjectException

from model import generate_image, MODEL_ID

app: FastAPI = FastAPI(title="White Cloud Image Generation API Server")

# Enable CORS for all origins to allow browser clients to communicate via Cloudflare tunnel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 2
    guidance_scale: float = 0.0
    guidance: float = 0.0
    width: int = 512
    height: int = 512

# Global handle for cloudflared process
tunnel_process: Optional[subprocess.Popen] = None

# Helper function to convert PIL Image to base64
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str: str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# ---------------------------------------------------------------------------
# Cloudflare Tunnel Manager
# ---------------------------------------------------------------------------
def start_cloudflare_tunnel() -> Optional[str]:
    global tunnel_process
    cmd: str = "./cloudflared" if os.path.exists("./cloudflared") else "cloudflared"
    
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
    except Exception as e:
        print(f"cloudflared binary not found or not working: {e}. Running without tunnel.", flush=True)
        return None

    print(f"Starting cloudflared tunnel using: {cmd}", flush=True)
    try:
        log_file = open("tunnel.log", "w")
        tunnel_process = subprocess.Popen(
            [cmd, "tunnel", "--url", "http://localhost:8000"],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        # Wait up to 15 seconds to extract the trycloudflare.com URL
        url: Optional[str] = None
        for i in range(15):
            time.sleep(1)
            if os.path.exists("tunnel.log"):
                with open("tunnel.log", "r") as f:
                    content: str = f.read()
                    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                    if match:
                        url = match.group(0)
                        break
        log_file.close()
        
        if url:
            return url
        else:
            print("Failed to extract Cloudflare tunnel URL from tunnel.log.", flush=True)
            return None
    except Exception as ex:
        print(f"Failed to start cloudflared tunnel process: {ex}", flush=True)
        return None

# ---------------------------------------------------------------------------
# GitHub DNS Updater & Dispatcher
# ---------------------------------------------------------------------------
def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    import requests
    import random
    worker_url = "https://dns-manager.aakashmishra2050880.workers.dev/update"
    print(f"[DNS] Registering tunnel URL '{public_url}' for repo '{repo_name}' via CF Worker...", flush=True)
    max_attempts: int = 5
    for attempt in range(1, max_attempts + 1):
        try:
            res = requests.post(
                worker_url,
                json={"key": repo_name, "value": public_url},
                timeout=15
            )
            if res.status_code == 200:
                print(f"[DNS] DNS successfully registered!", flush=True)
                return
            else:
                print(f"[DNS] Worker returned error {res.status_code}: {res.text}", flush=True)
        except Exception as e:
            print(f"[DNS] Network error connecting to Worker: {e}", flush=True)
        
        if attempt < max_attempts:
            sleep_time = random.uniform(2.0, 7.0)
            print(f"[DNS] Retrying DNS update in {sleep_time:.2f} seconds...", flush=True)
            time.sleep(sleep_time)
            
    print("[DNS] Warning: Failed to update registry after all attempts.", flush=True)

def clear_github_dns(repo_name: str) -> None:
    import requests
    worker_url = "https://dns-manager.aakashmishra2050880.workers.dev/update"
    print(f"[DNS] Clearing tunnel URL registry for '{repo_name}'...", flush=True)
    try:
        res = requests.post(
            worker_url,
            json={"key": repo_name, "value": "EMPTY"},
            timeout=10
        )
        if res.status_code == 200:
            print(f"[DNS] DNS registry cleared successfully for '{repo_name}'.", flush=True)
        else:
            print(f"[DNS] Worker returned error clearing DNS: {res.status_code}", flush=True)
    except Exception as e:
        print(f"[DNS] Failed to clear URL on shutdown: {e}", flush=True)

def trigger_self_workflow(pat: str, org: str, repo_name: str) -> None:
    print(f"Triggering self workflow dispatch for repository {repo_name}...", flush=True)
    try:
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        repo = g.get_repo(f"{org}/{repo_name}")
        default_branch: str = repo.default_branch
        
        # Trigger standard workflow.yml on the default branch
        wf = repo.get_workflow("workflow.yml")
        wf.create_dispatch(default_branch)
        print("Self workflow dispatch triggered successfully.", flush=True)
    except Exception as e:
        print(f"Failed to trigger self workflow: {e}", flush=True)

def shutdown_timer(pat: str, org: str, repo_name: str, duration_hours: float) -> None:
    duration_seconds: float = duration_hours * 3600
    print(f"Graceful shutdown timer started: Server will run for {duration_hours} hours ({duration_seconds} seconds).", flush=True)
    
    time.sleep(duration_seconds)
    
    print("Timer expired. Initiating graceful shutdown and restart...", flush=True)
    
    # 1. Trigger next workflow run
    if pat and repo_name != "test":
        trigger_self_workflow(pat, org, repo_name)
    else:
        print("Local mode or GH_PAT missing, skipping self-dispatch trigger.", flush=True)
        
    # 2. Short wait to allow dispatch request to register
    time.sleep(5)
    
    if repo_name != "test":
        clear_github_dns(repo_name)
        
    # 3. Kill cloudflared tunnel
    global tunnel_process
    if tunnel_process:
        try:
            tunnel_process.terminate()
            tunnel_process.wait(timeout=5)
            print("cloudflared tunnel terminated.", flush=True)
        except Exception as te:
            print(f"Error terminating cloudflared: {te}", flush=True)
        
    print("Exiting server process gracefully with code 0.", flush=True)
    os._exit(0)

# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    pat: str = os.getenv("GH_PAT", "")
    org: str = os.getenv("GH_ORG", "")

    # Resolve repo name from standard environment variable or GH_REPOSITORY
    repo_full: str = os.getenv("GITHUB_REPOSITORY", "") or os.getenv("GH_REPOSITORY", "")
    repo_name: str = repo_full.split("/")[-1] if "/" in repo_full else "test"

    # Start the shutdown timer thread
    duration_str: str = os.getenv("RUN_DURATION_HOURS", "4.0")
    try:
        duration_hours: float = float(duration_str)
    except ValueError:
        duration_hours = 4.0

    t: threading.Thread = threading.Thread(
        target=shutdown_timer,
        args=(pat, org, repo_name, duration_hours),
        daemon=True
    )
    t.start()

    # Warm up model weights
    print("[Warmup] Loading model pipeline on startup...", flush=True)
    try:
        from model import get_pipeline
        get_pipeline()
        print("[Warmup] Model pipeline warmed up successfully.", flush=True)
    except Exception as warmup_err:
        print(f"[Warmup] Warning: model warmup failed: {warmup_err}", flush=True)

    # Start Cloudflare Quick Tunnel
    public_url: Optional[str] = start_cloudflare_tunnel()
    if public_url:
        print(f"==================================================", flush=True)
        print(f"CLOUDFLARE TUNNEL ESTABLISHED SUCCESSFULLY!", flush=True)
        print(f"Public API Address: {public_url}", flush=True)
        print(f"==================================================", flush=True)
        
        # Write back tunnel DNS to config.json
        if pat and org:
            update_github_dns(pat, org, public_url, repo_name)
        else:
            print("Warning: GH_PAT or GH_ORG not configured. Skipping DNS config.json registration.", flush=True)
    else:
        print("Running server without public tunnel.", flush=True)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/image")
async def generate_image_endpoint(req: ImageRequest) -> dict:
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt parameter is required.")
    
    try:
        # Extract the guidance scale, supporting both standard API naming and new frontend custom naming
        guidance_val = req.guidance if req.guidance > 0.0 else req.guidance_scale
        
        # Run directly on the main thread to utilize all available CPU cores without OpenMP deadlock
        image = generate_image(
            req.prompt, 
            req.num_inference_steps, 
            guidance_val,
            req.width,
            req.height
        )
        img_b64: str = image_to_base64(image)
        return {
            "image_base64": img_b64,
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
