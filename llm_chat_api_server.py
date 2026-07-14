"""
Harbinger E2E — LLM API Server (DeepSeek)
Stateless chat completion service. Conversation history is owned by the
orchestrator (master node) and passed in full on every request.

DeepSeek's API is OpenAI-compatible, so we use the standard `openai` package
pointed at DeepSeek's base_url.

Setup:
    Set your API key as an environment variable before running:
        setx DEEPSEEK_API_KEY "sk-..."   (Windows, persists across terminals)
    or for the current terminal session only:
        $env:DEEPSEEK_API_KEY = "sk-..."  (PowerShell)

Usage:
    uv run llm-local-inference-api-server.py

Endpoints:
    GET  /health -> {"status": "ok"}
    POST /chat   -> {"messages": [{"role": "user", "content": "..."}]}
                 -> {"text": "...", "elapsed_seconds": ...}
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

# .env lives in the parent folder (harbinger-e2e/), one level up from this script
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

client: OpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable not set. "
            'Set it with: $env:DEEPSEEK_API_KEY = "sk-..."  (PowerShell)'
        )
    print(f"[llm-server] initializing DeepSeek client (model={DEEPSEEK_MODEL})...")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    print("[llm-server] ready to accept requests")

    yield

    print("[llm-server] shutting down")


app = FastAPI(title="Harbinger LLM API (DeepSeek)", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512


@app.get("/health")
def health():
    return {"status": "ok", "client_ready": client is not None}


@app.post("/chat")
def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    t0 = time.time()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[m.model_dump() for m in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    elapsed = time.time() - t0

    text = response.choices[0].message.content

    return {
        "text": text,
        "elapsed_seconds": elapsed,
        "usage": response.usage.model_dump() if response.usage else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8767)