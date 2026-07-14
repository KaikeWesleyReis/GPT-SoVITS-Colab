"""
Harbinger E2E — STT API Test Client
Sends a local wav file to the running STT server and prints the result.

Usage:
    uv run test_stt_api.py
    uv run test_stt_api.py --file stt_test.wav --url http://localhost:8766
"""

import argparse
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Test client for the local STT API")
    parser.add_argument("--file", default="stt_test.wav", help="Path to the audio file to send")
    parser.add_argument("--url", default="http://localhost:8766", help="Base URL of the STT server")
    args = parser.parse_args()

    audio_path = Path(args.file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path.resolve()}")

    # Health check first
    health = requests.get(f"{args.url}/health", timeout=10)
    print(f"[health] {health.status_code} -> {health.json()}")

    # Send the file for transcription
    with audio_path.open("rb") as f:
        files = {"audio": (audio_path.name, f, "audio/wav")}
        response = requests.post(f"{args.url}/transcribe", files=files, timeout=120)

    response.raise_for_status()
    result = response.json()

    print("\n[transcribe] result:")
    print(f"  text            : {result['text']}")
    print(f"  duration_seconds: {result['duration_seconds']:.2f}")
    print(f"  elapsed_seconds : {result['elapsed_seconds']:.2f}")
    print(f"  realtime_factor : {result['realtime_factor']:.2f}x")


if __name__ == "__main__":
    main()