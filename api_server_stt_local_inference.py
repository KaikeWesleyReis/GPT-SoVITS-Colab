"""
Harbinger E2E — Local STT API Server
Loads distil-large-v3 (English-only) once at startup from local disk,
then serves transcription requests over HTTP.

Usage:
    uv run stt-local-inference-api-server.py

Endpoints:
    GET  /health    -> {"status": "ok"}
    POST /transcribe -> multipart/form-data with "audio" file field
                         returns {"text": "...", "duration": ..., "elapsed": ...}
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel

MODEL_DIR = Path(__file__).parent.parent / "data" / "models" / "distil-large-v3"
LANGUAGE = "en"          # always English
COMPUTE_TYPE = "int8"    # fastest on CPU
CPU_THREADS = 4          # adjust based on benchmarking vs your TTS server

model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_DIR}. Download it manually."
        )
    print(f"[stt-server] loading model from {MODEL_DIR} (compute_type={COMPUTE_TYPE}, cpu_threads={CPU_THREADS})...")
    t0 = time.time()
    model = WhisperModel(
        str(MODEL_DIR),
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
    )
    print(f"[stt-server] model loaded in {time.time() - t0:.2f}s — ready to accept requests")

    yield  # server runs here

    print("[stt-server] shutting down")


app = FastAPI(title="Harbinger STT API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Save upload to a temp file, since faster-whisper needs a path or file-like seekable object
    tmp_path = Path(f"/tmp_{audio.filename}") if False else Path(f"./_tmp_{audio.filename}")
    contents = await audio.read()
    tmp_path.write_bytes(contents)

    try:
        t0 = time.time()
        segments, info = model.transcribe(
            str(tmp_path),
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
        )
        text_parts = [seg.text.strip() for seg in segments]
        elapsed = time.time() - t0
        full_text = " ".join(text_parts).strip()

        return {
            "text": full_text,
            "duration_seconds": info.duration,
            "elapsed_seconds": elapsed,
            "realtime_factor": (info.duration / elapsed) if elapsed > 0 else None,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766)