"""
Harbinger E2E — Orchestrator (Streamlit Voice UI)
Manual mic start, auto-stop on silence -> STT -> chat API -> TTS API,
persists history locally, and plays back audio for each Harbinger response.

Recording control: click "Record" to start. The app listens until either
SILENCE_DURATION seconds pass with no voiced audio, or MAX_RECORD_SECONDS
is hit (safety cap). No manual Stop button -- see the note in the handoff
about why that's a real tradeoff of this approach, not an oversight.

UI: WhatsApp-style bubbles (assistant left, user right), wide layout,
larger base font. Light theme is set via .streamlit/config.toml, which
must sit alongside this file.

Usage:
    uv run streamlit run streamlit_app.py
"""

import html
import re
import time

import streamlit as st

from api_clients import generate_audio, get_chat_response, transcribe_audio
from chat_history import load_history, reset_history, save_history, to_api_messages
from mic_recorder import MicRecorder, to_wav_bytes
from text_normalize import normalize_punctuation

MIN_RECORD_SECONDS = 3.0
SILENCE_DURATION = 3.0
MAX_RECORD_SECONDS = 30.0
STREAM_WORD_DELAY = 0.04

AVATARS = {"user": "🧑", "assistant": "🦑"}

st.set_page_config(page_title="Harbinger", page_icon="🦑", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 100% !important;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-top: 2rem;
    }
    html, body, [class^="css"] {
        font-size: 18px;
    }
    .chat-row {
        display: flex;
        align-items: flex-end;
        margin: 10px 0;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .bubble {
        max-width: 65%;
        padding: 12px 16px;
        border-radius: 18px;
        font-size: 1.05rem;
        line-height: 1.45;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background: #F1F0F0;
        color: #111;
        border-bottom-left-radius: 4px;
        margin-left: 10px;
    }
    .user-bubble {
        background: #DCF8C6;
        color: #111;
        border-bottom-right-radius: 4px;
        margin-right: 10px;
    }
    .avatar {
        font-size: 1.7rem;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🦑 Harbinger")
st.caption("An ancient intelligence, indulging your questions.")


_ITALIC_PATTERN = re.compile(r"\*(.+?)\*")


def bubble_html(role: str, text: str) -> str:
    """Render one chat turn as a WhatsApp-style bubble (escaped, newline-safe)."""
    safe_text = html.escape(text)
    # Markdown emphasis (*text*) doesn't get parsed inside raw HTML blocks,
    # so convert it ourselves before wrapping in the bubble div.
    safe_text = _ITALIC_PATTERN.sub(r"<em>\1</em>", safe_text)
    safe_text = safe_text.replace("\n", "<br>")
    avatar = AVATARS[role]
    if role == "user":
        return f'''
        <div class="chat-row user">
            <div class="bubble user-bubble">{safe_text}</div>
            <div class="avatar">{avatar}</div>
        </div>
        '''
    return f'''
    <div class="chat-row assistant">
        <div class="avatar">{avatar}</div>
        <div class="bubble assistant-bubble">{safe_text}</div>
    </div>
    '''


# --- Initialize / load history + recorder into session state ---
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "recorder" not in st.session_state:
    st.session_state.recorder = MicRecorder()

if st.sidebar.button("Reset conversation"):
    st.session_state.history = reset_history()
    st.rerun()

# --- Render existing history ---
for turn in st.session_state.history:
    st.markdown(bubble_html(turn["role"], turn["content"]), unsafe_allow_html=True)
    if turn["role"] == "assistant" and turn.get("audio_path"):
        with st.expander("🔊 Play Harbinger's voice", expanded=False):
            st.audio(turn["audio_path"])

# --- Mic control: sidebar so it never scrolls away as the chat grows ---
record_clicked = st.sidebar.button("🔴 Record")
sidebar_status = st.sidebar.empty()

user_input = None

if record_clicked:
    recorder = st.session_state.recorder

    try:
        recorder.start()
    except Exception as e:
        st.sidebar.error(f"Could not start recording: {e}")
    else:
        sidebar_status.markdown(
            "🔴 **Listening...** auto-stops after "
            f"{SILENCE_DURATION:.0f}s of silence."
        )

        start_time = time.monotonic()
        while True:
            time.sleep(0.1)
            if recorder.seconds_of_silence() >= SILENCE_DURATION:
                break
            if time.monotonic() - start_time >= MAX_RECORD_SECONDS:
                break

        audio = recorder.stop()
        sidebar_status.empty()

        duration = recorder.duration_seconds(audio)
        if duration < MIN_RECORD_SECONDS:
            st.sidebar.warning(
                f"Recording too short ({duration:.1f}s) — discarded, "
                f"minimum is {MIN_RECORD_SECONDS:.0f}s."
            )
        else:
            wav_bytes = to_wav_bytes(audio)
            with st.spinner("Transcribing..."):
                try:
                    user_input = transcribe_audio(wav_bytes)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

            if user_input is not None and not user_input.strip():
                st.warning("Heard silence / no speech detected — nothing sent.")
                user_input = None

# --- Turn processing ---
if user_input:
    # 1. Append user message, display immediately, persist
    st.session_state.history.append(
        {"role": "user", "content": user_input, "audio_path": None}
    )
    save_history(st.session_state.history)
    st.markdown(bubble_html("user", user_input), unsafe_allow_html=True)

    # 2. Call the chat API, showing a "thinking" state
    reply_placeholder = st.empty()
    reply_placeholder.markdown(
        bubble_html("assistant", "*thinking human...*"), unsafe_allow_html=True
    )

    try:
        api_messages = to_api_messages(st.session_state.history)
        raw_answer = get_chat_response(api_messages)
    except Exception as e:
        reply_placeholder.markdown(
            bubble_html("assistant", f"Harbinger did not respond: {e}"),
            unsafe_allow_html=True,
        )
        st.stop()

    # 3. Normalize punctuation (safety net for consistent sentence splitting)
    answer_text = normalize_punctuation(raw_answer)

    # 4. Reveal the answer as a "shadow stream" right away -- we already have
    #    the full text, we're just re-printing it progressively for the
    #    typing effect. No reason to make the user stare at a placeholder
    #    while audio (which can take a while on CPU) generates first.
    words = answer_text.split(" ")
    displayed = ""
    for i, word in enumerate(words):
        displayed += word + (" " if i < len(words) - 1 else "")
        reply_placeholder.markdown(
            bubble_html("assistant", displayed), unsafe_allow_html=True
        )
        time.sleep(STREAM_WORD_DELAY)

    # 5. Generate the full audio (single chunk — splitting is TTS's job later).
    #    Text is already visible above, so this can take as long as it needs.
    audio_path = None
    with st.spinner("🔊 Generating voice..."):
        try:
            audio_path = generate_audio(answer_text)
        except Exception as e:
            st.warning(f"Audio generation failed (showing text only): {e}")

    if audio_path:
        with st.expander("🔊 Play Harbinger's voice", expanded=True):
            st.audio(audio_path)

    # 7. Persist the assistant turn
    st.session_state.history.append(
        {"role": "assistant", "content": answer_text, "audio_path": audio_path}
    )
    save_history(st.session_state.history)
