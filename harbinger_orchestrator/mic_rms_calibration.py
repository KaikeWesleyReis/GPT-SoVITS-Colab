"""
Harbinger E2E — Mic RMS Calibration
Standalone diagnostic: records two short clips (silence, then normal speech)
and prints RMS values per time window, so you can pick a real, measured
silence_rms_threshold for MicRecorder instead of guessing or relying on
live auto-calibration.

Usage:
    uv run mic_rms_calibration.py
"""

import numpy as np

from mic_recorder import SAMPLE_RATE, MicRecorder

WINDOW_SECONDS = 1.0  # size of each analysis window

def record_clip(prompt: str) -> np.ndarray:
    recorder = MicRecorder()
    input(f"\n{prompt}\nPress ENTER to start recording...")
    recorder.start()
    print("Recording... press ENTER to stop.")
    input()
    audio = recorder.stop()
    print(f"Captured {recorder.duration_seconds(audio):.2f}s")
    return audio


def windowed_rms(audio: np.ndarray, samplerate: int, window_seconds: float) -> list[float]:
    window_size = int(samplerate * window_seconds)
    if window_size <= 0 or len(audio) == 0:
        return []
    values = []
    for start in range(0, len(audio), window_size):
        chunk = audio[start:start + window_size]
        if len(chunk) == 0:
            continue
        values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    return values


def print_timeline(values: list[float], window_seconds: float, label: str):
    print(f"\n{label} — RMS per {window_seconds * 1000:.0f}ms window:")
    for i, v in enumerate(values):
        t = i * window_seconds
        bar = "#" * int(v * 200)  # crude visual scale, just for eyeballing
        print(f"  {t:5.1f}s  {v:.4f}  {bar}")


def main():
    print("Mic RMS calibration")
    print("Records two short clips: one of silence, one of you speaking normally.")

    silence_audio = record_clip(
        "Clip 1: stay SILENT (just ambient room noise) for ~3-5 seconds."
    )
    speech_audio = record_clip(
        "Clip 2: speak NORMALLY, like you would to Harbinger, for ~3-5 seconds."
    )

    silence_windows = windowed_rms(silence_audio, SAMPLE_RATE, WINDOW_SECONDS)
    speech_windows = windowed_rms(speech_audio, SAMPLE_RATE, WINDOW_SECONDS)

    print_timeline(silence_windows, WINDOW_SECONDS, "SILENCE clip")
    print_timeline(speech_windows, WINDOW_SECONDS, "SPEECH clip")

    silence_max = max(silence_windows) if silence_windows else 0.0
    speech_min = min(speech_windows) if speech_windows else 0.0
    speech_mean = float(np.mean(speech_windows)) if speech_windows else 0.0

    print("\n--- Summary ---")
    print(f"Silence clip max RMS : {silence_max:.4f}")
    print(f"Speech clip min RMS  : {speech_min:.4f}")
    print(f"Speech clip mean RMS : {speech_mean:.4f}")

    if speech_min <= silence_max:
        print(
            "\n[warn] Your quietest speech window overlaps your loudest silence "
            "window -- there's no clean gap between them. Try speaking a bit "
            "louder/closer to the mic, or reducing background noise, then "
            "re-run this before picking a threshold."
        )
    else:
        suggested = (silence_max + speech_min) / 2
        print(f"\nSuggested silence_rms_threshold: {suggested:.4f}")
        print("(sits between your silence max and speech min, with margin on both sides)")


if __name__ == "__main__":
    main()
