"""Real-time audio stream analysis for Frigate RTSP sources.

Extracts audio from an RTSP stream via ffmpeg, runs faster-whisper for speech
transcription, and applies a lightweight spectral acoustic event classifier to
detect alarms, impacts, engines, and loud anomalies.

Results are emitted as AcousticObservation objects suitable for ingestion into
the coop SiteStateAggregator or the selfsuvis realtime pipeline.

Requires: ffmpeg on PATH, faster-whisper installed.
"""

import asyncio
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

# Acoustic event labels and their simple spectral signatures.
# Each entry: (label, min_freq_hz, max_freq_hz, energy_ratio_threshold)
_ACOUSTIC_SIGNATURES: list[tuple[str, float, float, float]] = [
    ("alarm", 2000.0, 4000.0, 0.40),  # sharp tonal alarm bursts
    ("engine", 80.0, 400.0, 0.35),  # low-frequency motor rumble
    ("impact", 200.0, 2000.0, 0.50),  # broadband transient spike
    ("glass", 3000.0, 8000.0, 0.30),  # high-frequency glass break
]

_SAMPLE_RATE = 16_000
_CHUNK_SEC = 4
_CHUNK_SAMPLES = _SAMPLE_RATE * _CHUNK_SEC


@dataclass
class AcousticObservation:
    """One analysed audio chunk from a live stream."""

    camera: str
    recorded_at: datetime
    chunk_duration_sec: float
    speech_transcript: str | None
    acoustic_events: list[dict[str, Any]] = field(default_factory=list)
    rms_db: float = -60.0
    silence: bool = True


class SoundAnalyzer:
    """Analyse audio from a Frigate RTSP stream in rolling chunks.

    Args:
        camera:   Camera name (matches Frigate / SiteState camera key).
        rtsp_url: Full RTSP URL (e.g. ``rtsp://frigate:8554/entrance``).
        on_observation: Async callback invoked after each analysed chunk.
    """

    def __init__(
        self,
        camera: str,
        rtsp_url: str,
        on_observation=None,
        chunk_sec: float = _CHUNK_SEC,
    ) -> None:
        self._camera = camera
        self._rtsp_url = rtsp_url
        self._on_observation = on_observation
        self._chunk_sec = chunk_sec
        self._stop = asyncio.Event()
        self._whisper = None

    async def run(self) -> None:
        """Stream audio, analyse each chunk, call on_observation. Runs until stop()."""
        logger.info("SoundAnalyzer starting: camera=%s url=%s", self._camera, self._rtsp_url)
        while not self._stop.is_set():
            try:
                await self._process_chunk()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SoundAnalyzer error for camera=%s, retrying in 5s", self._camera)
                await asyncio.sleep(5.0)

    def stop(self) -> None:
        self._stop.set()

    # -- Chunk processing ------------------------------------------------------

    async def _process_chunk(self) -> None:
        audio = await asyncio.to_thread(self._capture_audio_chunk)
        if audio is None or len(audio) == 0:
            await asyncio.sleep(self._chunk_sec)
            return

        recorded_at = datetime.now(timezone.utc)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        # Plain Python scalars: numpy ones do not serialize to JSON.
        rms_db = float(20.0 * np.log10(max(rms, 1e-6)) - 90.0)  # rough dBFS
        silence = bool(rms_db < -45.0)

        acoustic_events: list[dict[str, Any]] = []
        if not silence:
            acoustic_events = self._classify_acoustic_events(audio)

        transcript: str | None = None
        if not silence:
            transcript = await asyncio.to_thread(self._transcribe, audio)

        obs = AcousticObservation(
            camera=self._camera,
            recorded_at=recorded_at,
            chunk_duration_sec=self._chunk_sec,
            speech_transcript=transcript,
            acoustic_events=acoustic_events,
            rms_db=rms_db,
            silence=silence,
        )

        if self._on_observation:
            try:
                await self._on_observation(obs)
            except Exception:
                logger.exception("SoundAnalyzer: on_observation callback error")

    def _capture_audio_chunk(self) -> "np.ndarray | None":
        """Extract one chunk of raw PCM audio from the RTSP stream via ffmpeg."""
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-i",
                self._rtsp_url,
                "-t",
                str(self._chunk_sec),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(_SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "s16le",
                "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=self._chunk_sec + 10)
            if result.returncode != 0 or not result.stdout:
                return None
            return np.frombuffer(result.stdout, dtype=np.int16)
        except subprocess.TimeoutExpired:
            logger.debug("SoundAnalyzer: ffmpeg capture timeout for %s", self._camera)
            return None
        except FileNotFoundError:
            logger.warning("SoundAnalyzer: ffmpeg not found — audio analysis disabled")
            self._stop.set()
            return None
        except Exception as exc:
            logger.debug("SoundAnalyzer capture error: %s", exc)
            return None

    def _transcribe(self, audio: "np.ndarray") -> str | None:
        """Run faster-whisper on raw PCM. Returns transcript or None."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError:
            return None
        try:
            if self._whisper is None:
                self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
            float_audio = audio.astype(np.float32) / 32768.0
            segments, _ = self._whisper.transcribe(float_audio, language=None, beam_size=1)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text if text else None
        except Exception as exc:
            logger.debug("SoundAnalyzer: transcription error: %s", exc)
            return None

    def _classify_acoustic_events(self, audio: "np.ndarray") -> list[dict[str, Any]]:
        """Simple FFT-based acoustic event classification."""
        events: list[dict[str, Any]] = []
        try:
            float_audio = audio.astype(np.float64) / 32768.0
            spectrum = np.abs(np.fft.rfft(float_audio))
            freqs = np.fft.rfftfreq(len(float_audio), d=1.0 / _SAMPLE_RATE)
            total_energy = float(np.sum(spectrum**2)) or 1.0

            for label, f_lo, f_hi, threshold in _ACOUSTIC_SIGNATURES:
                mask = (freqs >= f_lo) & (freqs <= f_hi)
                band_energy = float(np.sum(spectrum[mask] ** 2))
                ratio = band_energy / total_energy
                if ratio >= threshold:
                    events.append({"event": label, "energy_ratio": round(ratio, 3)})
        except Exception:
            pass
        return events
