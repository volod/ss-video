import hashlib
import io
import os

from fastapi import UploadFile


async def write_upload_to_path(
    file: UploadFile,
    dest_path: str,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Stream upload to dest_path, enforcing max_bytes. Returns total bytes written.
    On overflow, closes and removes the partial file, then raises ValueError."""
    total = 0
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Upload exceeds max size {max_bytes} bytes")
                f.write(chunk)
    except ValueError:
        try:
            os.remove(dest_path)
        except OSError:
            pass
        raise
    return total


async def read_upload_limited(
    file: UploadFile, max_bytes: int, chunk_size: int = 1024 * 1024
) -> bytes:
    buf = io.BytesIO()
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Upload exceeds max size {max_bytes} bytes")
        buf.write(chunk)
    return buf.getvalue()


async def hash_upload_limited(
    file: UploadFile, max_bytes: int, chunk_size: int = 1024 * 1024
) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Upload exceeds max size {max_bytes} bytes")
        h.update(chunk)
    return h.hexdigest(), total
