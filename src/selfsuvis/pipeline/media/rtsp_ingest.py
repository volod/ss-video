"""Live RTSP/RTMP stream ingestion via ffmpeg.

Records a live RTSP or RTMP stream to a local MP4 file, which the worker
then indexes normally via the standard video indexing pipeline.

Supports:
  - rtsp://  — RTSP over TCP (default) or UDP
  - rtmp://  — RTMP (e.g. from MediaMTX re-streams)

Security:
  - Only rtsp:// and rtmp:// schemes are accepted.
  - Hostname must resolve to a non-private IP unless ALLOW_PRIVATE_URLS=true.

Usage (in worker):
    from selfsuvis.pipeline.media.rtsp_ingest import validate_rtsp_url, record_rtsp
    validate_rtsp_url(url)           # raises ValueError on bad URL
    record_rtsp(url, output_path, duration_sec=300)
"""

import ipaddress
import socket
from urllib.parse import urlparse

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.media.subprocess_common import run_checked

logger = get_logger(__name__)

_ALLOWED_SCHEMES = {"rtsp", "rtmp"}


def validate_rtsp_url(url: str) -> None:
    """Validate that a URL is a safe RTSP/RTMP stream endpoint.

    Raises:
        ValueError: if the URL is invalid or points to a disallowed host.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme must be one of {sorted(_ALLOWED_SCHEMES)}; got '{parsed.scheme}'"
        )
    if not parsed.hostname:
        raise ValueError("RTSP/RTMP URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("RTSP/RTMP URL must not contain credentials in the URL")

    if not settings.ALLOW_PRIVATE_URLS:
        try:
            infos = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as exc:
            if parsed.hostname.endswith(".local"):
                logger.debug(
                    "RTSP validation: allowing unresolved mDNS hostname '%s'",
                    parsed.hostname,
                )
                return
            raise ValueError(f"Cannot resolve RTSP host '{parsed.hostname}': {exc}") from exc
        for info in infos:
            ip_str = info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise ValueError(
                    f"RTSP host '{parsed.hostname}' resolves to a private/loopback IP "
                    f"({ip_str}). Set ALLOW_PRIVATE_URLS=true to allow local streams."
                )


def record_rtsp(
    url: str,
    output_path: str,
    duration_sec: int | None = None,
    timeout_sec: int | None = None,
) -> None:
    """Record a live RTSP/RTMP stream to an MP4 file using ffmpeg.

    Args:
        url:          rtsp:// or rtmp:// stream URL.
        output_path:  Destination MP4 file path.
        duration_sec: Stop after this many seconds. If None, records until
                      the stream ends or timeout_sec is reached.
        timeout_sec:  Subprocess timeout. Defaults to RTSP_MAX_DURATION_SEC + 30s
                      to allow for connection overhead.

    Raises:
        subprocess.TimeoutExpired: if recording exceeds timeout_sec.
        subprocess.CalledProcessError: if ffmpeg exits with non-zero status.
        FileNotFoundError: if ffmpeg is not installed.
    """
    max_dur = settings.RTSP_MAX_DURATION_SEC
    if duration_sec is not None:
        effective_duration = min(duration_sec, max_dur)
    else:
        effective_duration = None

    if timeout_sec is None:
        # Give ffmpeg max_duration + 60s overhead for connection / teardown
        timeout_sec = max_dur + 60

    cmd = [
        "ffmpeg",
        "-y",
        "-rtsp_transport",
        "tcp",  # prefer TCP for reliability
        "-i",
        url,
        "-c",
        "copy",  # stream copy, no re-encoding
    ]
    if effective_duration is not None:
        cmd += ["-t", str(effective_duration)]
    cmd.append(output_path)

    logger.info(
        "RTSP ingest: recording url=%s duration=%s output=%s",
        url,
        effective_duration,
        output_path,
    )
    run_checked(cmd, timeout=timeout_sec)
    logger.info("RTSP ingest: recording complete output=%s", output_path)
