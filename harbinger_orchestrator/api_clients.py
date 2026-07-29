"""
Harbinger E2E — API Clients
Thin wrappers around the chat (LLM) and TTS HTTP services.
"""

import time
import uuid
from pathlib import Path

import requests

CHAT_API_URL = "http://localhost:8767"
TTS_API_URL = "http://localhost:8765"

AUDIO_STORE_DIR = Path(__file__).parent / "audio_store"


def get_chat_response(messages: list[dict[str, str]]) -> str:
    """Send full conversation history to the stateless /chat API, get reply text."""
    response = requests.post(
        f"{CHAT_API_URL}/chat",
        json={"messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["text"]


def generate_audio(text: str) -> str:
    """
    Send text to the TTS /generate API, save the returned audio locally,
    and return the local file path (as a string) for playback.
    """
    AUDIO_STORE_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.post(
        f"{TTS_API_URL}/generate",
        json={"text": text},
        timeout=120,
    )
    response.raise_for_status()

    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    file_path = AUDIO_STORE_DIR / filename
    file_path.write_bytes(response.content)

    return str(file_path)
