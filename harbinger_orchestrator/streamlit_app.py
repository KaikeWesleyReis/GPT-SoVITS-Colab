"""
Harbinger E2E — Orchestrator (Streamlit Chat UI)
Text-to-text chatbot (STT ignored for now). Talks to the stateless /chat API
and the /generate TTS API, persists history locally, and plays back audio
for each Harbinger response.

Usage:
    uv run streamlit run streamlit_app.py
"""

import time

import streamlit as st

from api_clients import generate_audio, get_chat_response
from chat_history import load_history, reset_history, save_history, to_api_messages
from text_normalize import normalize_punctuation

st.set_page_config(page_title="Harbinger", page_icon="🦑", layout="centered")

st.title("🦑 Harbinger")
st.caption("An ancient intelligence, indulging your questions.")

# --- Initialize / load history into session state ---
if "history" not in st.session_state:
    st.session_state.history = load_history()

if st.sidebar.button("Reset conversation"):
    st.session_state.history = reset_history()
    st.rerun()

# --- Render existing history ---
for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and turn.get("audio_path"):
            with st.expander("🔊 Play Harbinger's voice", expanded=False):
                st.audio(turn["audio_path"])

# --- User input ---
user_input = st.chat_input("Speak to Harbinger...")

if user_input:
    # 1. Append user message, display immediately, persist
    st.session_state.history.append(
        {"role": "user", "content": user_input, "audio_path": None}
    )
    save_history(st.session_state.history)

    with st.chat_message("user"):
        st.write(user_input)

    # 2. Call the chat API, showing a "thinking" state
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("*thinking human...*")

        try:
            api_messages = to_api_messages(st.session_state.history)
            raw_answer = get_chat_response(api_messages)
        except Exception as e:
            thinking_placeholder.error(f"Harbinger did not respond: {e}")
            st.stop()

        # 3. Normalize punctuation (safety net for consistent sentence splitting)
        answer_text = normalize_punctuation(raw_answer)

        # 4. Simulate the TTS "chunk" delay before revealing the answer
        thinking_placeholder.markdown("*I shall speak now...*")
        time.sleep(7)

        # 5. Generate the full audio (single chunk — splitting is TTS's job later)
        audio_path = None
        try:
            audio_path = generate_audio(answer_text)
        except Exception as e:
            st.warning(f"Audio generation failed (showing text only): {e}")

        # 6. Reveal the answer
        thinking_placeholder.empty()
        st.write(answer_text)
        if audio_path:
            with st.expander("🔊 Play Harbinger's voice", expanded=True):
                st.audio(audio_path)

    # 7. Persist the assistant turn
    st.session_state.history.append(
        {"role": "assistant", "content": answer_text, "audio_path": audio_path}
    )
    save_history(st.session_state.history)
