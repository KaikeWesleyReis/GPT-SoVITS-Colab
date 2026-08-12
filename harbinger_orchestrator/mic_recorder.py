"""
Harbinger E2E — Mic Recorder (manual start, silence-based auto-stop)
Toggle-style recorder: call start() to begin capturing. The recorder tracks
audio energy internally so the caller can poll seconds_of_silence() to decide
when to stop -- this module does NOT stop itself; policy (how much silence
is "done talking", max duration safety cap) lives in the orchestrator.

Silence detection is a simple RMS energy threshold, not a trained VAD model
(e.g. Silero/webrtcvad). This keeps things dependency-free and easy to
reason about, at the cost of needing the threshold tuned to your room/mic --
a loud fan or hum can sit above the default threshold and prevent silence
from ever being detected, so treat silence_rms_threshold as something you
may need to adjust after a first real-world test.

Requires:
    pip install sounddevice numpy
"""

from __future__ import annotations

import io
import threading
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_SILENCE_RMS_THRESHOLD = 0.0013  # measured via mic_rms_calibration.py


class MicRecorder:
    """Manual-start recorder that tracks speech/silence via RMS energy."""

    def __init__(
        self,
        samplerate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        silence_rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.silence_rms_threshold = silence_rms_threshold

        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.is_recording = False

        self._voice_detected = False
        self._last_voice_ts: float | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] stream status: {status}")

        with self._lock:
            self._frames.append(indata.copy())

        rms = float(np.sqrt(np.mean(np.square(indata))))
        if rms >= self.silence_rms_threshold:
            self._voice_detected = True
            self._last_voice_ts = time.monotonic()

    def start(self) -> None:
        """Begin capturing from the default input device."""
        if self.is_recording:
            return
        self._frames = []
        self._voice_detected = False
        self._last_voice_ts = None
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self.is_recording = True

    def stop(self) -> np.ndarray:
        """Stop capture and return the recorded audio as a 1D float32 array."""
        if not self.is_recording:
            return np.zeros((0,), dtype="float32")

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.is_recording = False

        with self._lock:
            if not self._frames:
                return np.zeros((0,), dtype="float32")
            audio = np.concatenate(self._frames, axis=0)

        return audio.flatten()

    def duration_seconds(self, audio: np.ndarray) -> float:
        return len(audio) / self.samplerate

    def has_voice(self) -> bool:
        """Whether any audio above the silence threshold has been seen yet this recording."""
        return self._voice_detected

    def calibrate_silence_threshold(self, multiplier: float = 2.5, minimum: float = 0.005) -> float:
        """
        Measure ambient noise from audio captured so far (call this shortly
        after start(), before the user is expected to speak) and set the
        silence threshold just above that room/mic's noise floor. Resets
        voice-detection state afterward so the calibration window itself
        doesn't count as "silence already seen".
        """
        with self._lock:
            frames_snapshot = list(self._frames)

        if frames_snapshot:
            ambient = np.concatenate(frames_snapshot, axis=0)
            ambient_rms = float(np.sqrt(np.mean(np.square(ambient))))
            self.silence_rms_threshold = max(minimum, ambient_rms * multiplier)

        self._voice_detected = False
        self._last_voice_ts = None
        return self.silence_rms_threshold

    def seconds_of_silence(self) -> float:
        """
        Seconds since voice was last detected above the RMS threshold.
        Returns 0.0 if no voice has been detected yet this recording --
        this avoids auto-stopping before the user has even started speaking.
        """
        if not self._voice_detected or self._last_voice_ts is None:
            return 0.0
        return time.monotonic() - self._last_voice_ts


def to_wav_bytes(audio: np.ndarray, samplerate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 [-1, 1] mono array to 16-bit PCM WAV bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue()
