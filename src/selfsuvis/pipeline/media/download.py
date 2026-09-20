from selfsuvis.pipeline.core import settings

from .fs_common import ensure_parent_dir, remove_if_exists
from .network import safe_request


def download_url(
    url: str,
    dest_path: str,
    chunk_size: int = 1024 * 1024,
    max_bytes: int | None = None,
) -> None:
    """Download URL to dest_path. Stops after max_bytes if set."""
    max_bytes = max_bytes if max_bytes is not None else settings.MAX_DOWNLOAD_BYTES
    ensure_parent_dir(dest_path)
    with safe_request("GET", url, timeout=60, stream=True) as r:
        r.raise_for_status()
        content_length = r.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(
                f"Content-Length {content_length} exceeds max download size {max_bytes}"
            )
        written = 0
        try:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"Download exceeded max size {max_bytes} bytes")
                    f.write(chunk)
        except Exception:
            remove_if_exists(dest_path)
            raise
