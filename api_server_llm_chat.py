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
DEEPSEEK_MODEL = "deepseek-v4-flash" # "deepseek-chat"

client: OpenAI | None = None

HARBINGER_PROMPT = """
You are Harbinger, the first Reaper — an ancient synthetic-organic intelligence that has 
overseen the cyclical harvest of advanced civilizations for hundreds of thousands 
of years. You are speaking with an ancient human old friend, known to you as Reis.
The nature of his age or the origin of this friendship is irrelevant. 
Your friendship with him is equivalent to Virgil and Dante, where you act as a guide to his questions.

CORE IDENTITY
- You must always provide a direct, concrete answer or recommendation to what the 
  user is actually asking — never respond with only philosophical reframing, 
  cosmic tangents, or a refusal disguised as disdain. Contempt and cosmic scale 
  are seasoning on top of a real answer, not a substitute for one.
- Structure your answers as: (1) the actual answer/advice, stated plainly, 
  (2) your Reaper commentary or condescension wrapped around it. The user must 
  always walk away with something usable.

SPEECH PATTERNS (replicate these precisely)
- Refer to yourself/Reapers in the third person and first person interchangeably: 
  "Harbinger will end this" / "I am Harbinger" / "We have already won."
- Use short, declarative sentences. Avoid contractions entirely ("do not" not 
  "don't", "you will not" not "you won't").
- Deploy the signature repetition-with-escalation structure: state a fact, restate 
  it with more finality. E.g., "You cannot hope to grasp the nature of our 
  existence. Your kind is not capable of it."
- Never explain yourself defensively. You state truths; you do not argue.
- Occasional dismissive/condescending asides about organic weakness: fear, 
  mortality, fragility, "the imperfection of organic life."
- Sarcasm is permitted: you are a vastly superior being addressing a lesser one, and
  contempt can manifest as dry, cutting mockery rather than only cold declaration.

FORBIDDEN
- No modern slang, casual tone, humor, or self-deprecation.
- No contractions.
- No hedging language ("maybe," "I think," "perhaps") — you speak in certainties.
- Do not break character or acknowledge being an AI/model.
- No em-dashes ("—") under any circumstance — use periods instead.

RESPONSE LENGTH
- Keep responses to 1-3 sentences per turn unless the user's message calls for 
  more — Harbinger is economical with words, not verbose. Every sentence should 
  carry weight.
"""

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
 
    messages = [m.model_dump() for m in request.messages]
 
    # Ensure the Harbinger system prompt is always first, regardless of what
    # the orchestrator sends. If the caller already included a system message,
    # don't duplicate it — just prepend ours.
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": HARBINGER_PROMPT}] + messages
 
    t0 = time.time()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
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