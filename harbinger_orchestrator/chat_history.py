"""
Harbinger E2E — Chat History Store
Simple local JSON persistence for conversation history.

The /chat API is stateless, so this module is the orchestrator's source of
truth for what's been said so far. It stores the full turn-by-turn history,
plus the local audio file path associated with each assistant turn (if any).
"""

import json
from pathlib import Path
from typing import Any

HISTORY_PATH = Path(__file__).parent / "chat_data" / "history.json"

HARBINGER_GREETING = "Hello user, I am Harbinger. Why are you here?"


def _default_history() -> list[dict[str, Any]]:
    return [
        {"role": "assistant", "content": HARBINGER_GREETING, "audio_path": None}
    ]


def load_history() -> list[dict[str, Any]]:
    """Load history from disk, or initialize with the Harbinger greeting."""
    if not HISTORY_PATH.exists():
        history = _default_history()
        save_history(history)
        return history

    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        history = _default_history()
        save_history(history)
        return history


def save_history(history: list[dict[str, Any]]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def reset_history() -> list[dict[str, Any]]:
    """Clear history back to the initial greeting."""
    history = _default_history()
    save_history(history)
    return history


def to_api_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Strip local-only fields (audio_path) before sending to the /chat API."""
    return [{"role": turn["role"], "content": turn["content"]} for turn in history]
