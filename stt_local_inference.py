"""
Harbinger E2E — Local STT Module
File-based transcription using faster-whisper (CTranslate2 backend).

CPU-only (AMD GPU, no CUDA support in faster-whisper).
Model: distil-large-v3 by default — good accuracy/speed tradeoff on CPU.
Fallback: pass --model large-v3 to benchmark full accuracy, or
--model medium / small for more speed.

Usage:
    python stt_local_inference.py path/to/audio.wav
    python stt_local_inference.py path/to/audio.wav --model medium
    python stt_local_inference.py path/to/audio.wav --language en

First run will download the model from Hugging Face (cached locally after).
"""

import argparse
import time
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(
    audio_path: str,
    model_size: str = "distil-large-v3",
    language: str | None = "en",
    compute_type: str = "int8",
    num_workers: int = 1,
    cpu_threads: int = 4,
):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[stt] loading model '{model_size}' (compute_type={compute_type}, cpu_threads={cpu_threads})...")
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )
    print(f"[stt] model loaded in {time.time() - t0:.2f}s")

    print(f"[stt] transcribing: {audio_path}")
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,  # trims leading/trailing silence + internal long silences
    )

    full_text_parts = []
    for seg in segments:
        print(f"  [{seg.start:6.2f}s -> {seg.end:6.2f}s] {seg.text.strip()}")
        full_text_parts.append(seg.text.strip())

    elapsed = time.time() - t0
    full_text = " ".join(full_text_parts).strip()

    print(f"\n[stt] detected language: {info.language} (p={info.language_probability:.2f})")
    print(f"[stt] transcription took {elapsed:.2f}s for {info.duration:.2f}s of audio "
          f"({info.duration / elapsed:.2f}x realtime)" if elapsed > 0 else "")
    print(f"\n[stt] FULL TEXT:\n{full_text}")

    return full_text


def main():
    parser = argparse.ArgumentParser(description="Local file-based STT via faster-whisper")
    parser.add_argument("audio_path", help="Path to input audio file (wav/mp3/etc.)")
    parser.add_argument(
        "--model",
        default="distil-large-v3",
        help="Model size/name: tiny, base, small, medium, large-v3, distil-large-v3 (default)",
    )
    parser.add_argument("--language", default="en", help="Language code, or 'auto' to detect")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="int8 (fastest on CPU), int8_float16, float32 (most accurate, slowest)",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    language = None if args.language == "auto" else args.language

    transcribe(
        audio_path=args.audio_path,
        model_size=args.model,
        language=language,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
    )


if __name__ == "__main__":
    main()