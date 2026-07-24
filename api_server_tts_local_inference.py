"""
Local TTS API Server
=================================================================
Loads all GPT-SoVITS models once at startup and exposes a single
endpoint to generate audio from text.

Run with:
    uv run tts-api.py

Endpoint:
    POST /generate
    Body: { "text": "Text to synthesize." }
    Returns: WAV audio file (audio/wav)
"""

import os
import sys
import io
import pathlib
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- Path setup (mirrors tts-local-inference.py) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPT_SOVITS_DIR = os.path.join(SCRIPT_DIR, "GPT_SoVITS")
if GPT_SOVITS_DIR not in sys.path:
    sys.path.insert(0, GPT_SOVITS_DIR)

if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# --- Import inference module ---
from tts_local_inference import (
    load_ssl_model,
    load_sovits_model,
    load_gpt_model,
    load_sv_model,
    generate_tts_on_cpu,
    MODELS_DIR,
    REFERENCE_AUDIOS_DIR,
)

######################################################################################
# STARTUP — load models once
######################################################################################

cnhubert_base_path   = os.path.join(MODELS_DIR, "chinese-hubert-base")
sovits_ckpt_path     = os.path.join(MODELS_DIR, "sovits", "SOVITS_GENERATOR.pth")
sovits_config_path   = os.path.join(MODELS_DIR, "sovits", "config.json")
gpt_ckpt_path        = os.path.join(MODELS_DIR, "gpt", "GPT.ckpt")
gpt_config_path      = os.path.join(MODELS_DIR, "gpt", "config.json")
sv_ckpt_path         = os.path.join(MODELS_DIR, "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
reference_wav_path   = os.path.join(REFERENCE_AUDIOS_DIR, "ref.wav")
reference_text_path  = os.path.join(REFERENCE_AUDIOS_DIR, "ref.txt")

print("Loading models...")
ssl_model            = load_ssl_model(cnhubert_base_path)
vq_model, hps        = load_sovits_model(sovits_ckpt_path, sovits_config_path)
t2s_model, max_sec   = load_gpt_model(gpt_ckpt_path, gpt_config_path)
sv_model             = load_sv_model(sv_ckpt_path)
print("All models loaded.")

with open(reference_text_path, "r", encoding="utf-8") as f:
    reference_text = f.read().strip()

######################################################################################
# API
######################################################################################

app = FastAPI(title="GPT-SoVITS Local TTS", version="1.0.0")


class TTSRequest(BaseModel):
    text: str
    top_k: int = 12
    top_p: float = 1.0
    temperature: float = 0.9
    speed: float = 1.0
    pause_seconds: float = 0.3


@app.post("/generate")
def generate(request: TTSRequest):
    '''
    Generates TTS audio for the given text using the preloaded models
    and fixed reference voice. Returns a WAV file stream.
    '''
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text field is required and cannot be empty.")

    try:
        sample_rate, audio = generate_tts_on_cpu(
            reference_wav_path=reference_wav_path,
            reference_text=reference_text,
            text_to_generate=request.text,
            ssl_model=ssl_model,
            vq_model=vq_model,
            t2s_model=t2s_model,
            sv_model=sv_model,
            hps=hps,
            max_sec=max_sec,
            text_language="en",
            reference_language="en",
            parameter_top_k=request.top_k,
            parameter_top_p=request.top_p,
            parameter_temperature=request.temperature,
            parameter_audio_speed=request.speed,
            parameter_pause_seconds=request.pause_seconds,
            parameter_reuse_gpt_tokens=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Write audio to an in-memory buffer as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


######################################################################################
# ENTRYPOINT
######################################################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)