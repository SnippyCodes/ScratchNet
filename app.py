import io
import os
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from model import (
    FlowerClassifier,
    CATEGORIES,
    FLOWER_DETAILS,
    predict_flower,
    load_model,
    get_device,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the PyTorch model on startup."""
    try:
        model, device = load_model()
        print(f"[*] Model loaded successfully on {device}")
    except Exception as e:
        print(f"[!] Warning during model load: {e}")
    yield


app = FastAPI(
    title="ScratchNet Flower Classifier",
    description="Deep Learning CNN UI for 7-Class Flower Detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local testing & embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main interactive web UI."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it through the trained CNN model,
    and returns the predicted flower class, confidence, ranked probabilities, and botanical metadata.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be an image. Received: {file.content_type}",
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = predict_flower(image_bytes)
        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                **result,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/api/samples")
async def get_samples() -> Dict[str, Any]:
    """Returns the list of sample flowers available for quick 1-click testing."""
    samples = []
    samples_dir = STATIC_DIR / "samples"
    for cat in CATEGORIES:
        sample_img = samples_dir / f"{cat}.jpg"
        if sample_img.is_file():
            info = FLOWER_DETAILS.get(cat, {})
            samples.append({
                "category": cat,
                "display_name": cat.capitalize(),
                "icon": info.get("icon", "🌸"),
                "color": info.get("color", "#10b981"),
                "url": f"/static/samples/{cat}.jpg",
            })
    return {"samples": samples}


@app.get("/api/health")
async def health_check():
    """Model status and hardware metadata."""
    device = get_device()
    model, _ = load_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "status": "online",
        "device": str(device),
        "cuda_available": device.type == "cuda",
        "classes": CATEGORIES,
        "num_classes": len(CATEGORIES),
        "input_resolution": "150x150 RGB",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "architecture": "3-Block CNN with BatchNorm and Dropout",
    }


def find_available_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Finds an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ScratchNet Flower Detection Web App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=None, help="Port to run server on")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    host = args.host
    port = args.port if args.port is not None else find_available_port(8000)
    url = f"http://{host}:{port}"

    print("\n" + "=" * 60)
    print("  🌸 ScratchNet Flower Detection Web App")
    print(f"  🚀 Server running at: {url}")
    print("  📸 Upload or drag-and-drop images to detect the 7 flowers")
    print("=" * 60 + "\n")

    if not args.no_browser:
        try:
            import threading
            def open_browser():
                import time
                time.sleep(1.2)
                webbrowser.open(url)
            threading.Thread(target=open_browser, daemon=True).start()
        except Exception:
            pass

    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
