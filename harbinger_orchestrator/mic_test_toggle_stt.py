"""
Harbinger E2E — Manual Mic Toggle -> STT Test
Standalone test: press ENTER to start recording, ENTER again to stop.
If the recording is longer than --min-duration seconds, sends it to the
STT server for transcription. No orchestrator / chat / TTS involved --
this only exercises mic capture + STT.

Usage:
    uv run test_mic_toggle_stt.py
    uv run test_mic_toggle_stt.py --url http://localhost:8766 --min-duration 3.0

Controls:
    ENTER      -> start recording / stop recording (toggle)
    q + ENTER  -> quit
"""

import argparse
import io

import requests

from mic_recorder import SAMPLE_RATE, MicRecorder, to_wav_bytes

DEFAULT_MIN_DURATION_SECONDS = 3.0


def send_to_stt(wav_bytes: bytes, url: str) -> dict:
    files = {"audio": ("utterance.wav", io.BytesIO(wav_bytes), "audio/wav")}
    response = requests.post(f"{url}/transcribe", files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Manual mic toggle -> STT test")
    parser.add_argument("--url", default="http://localhost:8766", help="STT server base URL")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION_SECONDS,
        help="Minimum recording duration (seconds) required to send to STT",
    )
    args = parser.parse_args()

    # Health check first, fail fast (but non-fatally) if the STT server isn't up
    try:
        health = requests.get(f"{args.url}/health", timeout=10)
        health.raise_for_status()
        print(f"[health] {health.status_code} -> {health.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[warn] STT server not reachable at {args.url}: {e}")
        print("[warn] continuing anyway -- recording will still work, transcription will fail\n")

    recorder = MicRecorder()

    print("Manual mic toggle test")
    print(f"  Recordings under {args.min_duration:.1f}s are discarded (not sent to STT).")
    print("  Press ENTER to start recording, ENTER again to stop. Type 'q' + ENTER to quit.\n")

    while True:
        cmd = input("[idle] press ENTER to start recording (or 'q' to quit)... ")
        if cmd.strip().lower() == "q":
            break

        print("[recording] speak now... press ENTER to stop")
        recorder.start()
        input()
        audio = recorder.stop()

        duration = recorder.duration_seconds(audio)
        print(f"[stopped] captured {duration:.2f}s of audio")

        if duration < args.min_duration:
            print(f"[skip] under {args.min_duration:.1f}s minimum -- discarding, not sending to STT\n")
            continue

        wav_bytes = to_wav_bytes(audio, samplerate=SAMPLE_RATE)

        print("[stt] sending to transcription server...")
        try:
            result = send_to_stt(wav_bytes, args.url)
        except requests.exceptions.RequestException as e:
            print(f"[error] STT request failed: {e}\n")
            continue

        print(f"[stt] text            : {result['text']}")
        print(f"[stt] duration_seconds: {result['duration_seconds']:.2f}")
        print(f"[stt] elapsed_seconds : {result['elapsed_seconds']:.2f}")
        rtf = result.get("realtime_factor")
        if rtf:
            print(f"[stt] realtime_factor : {rtf:.2f}x")
        print()

    print("bye.")


if __name__ == "__main__":
    main()
