"""Private Supabase Storage helpers for Face Access.

All calls must run on the trusted backend. Never expose SUPABASE_SERVICE_ROLE_KEY
or this module to browser/client code.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Iterable

from supabase import Client, create_client

BUCKET_EMPLOYEE_FACES = "employee-faces"
BUCKET_ATTENDANCE_EVIDENCE = "attendance-evidence"
BUCKET_CORRECTION_DOCUMENTS = "correction-documents"

_ALLOWED_BUCKETS = {
    BUCKET_EMPLOYEE_FACES,
    BUCKET_ATTENDANCE_EVIDENCE,
    BUCKET_CORRECTION_DOCUMENTS,
}


@lru_cache(maxsize=1)
def _client() -> Client:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured."
        )
    return create_client(url, key)


def _safe_object_path(path: str) -> str:
    value = str(path or "").replace("\\", "/").strip("/")
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
        raise ValueError("Invalid storage object path.")
    return str(parsed)


def _bucket(name: str):
    if name not in _ALLOWED_BUCKETS:
        raise ValueError("Unknown Face Access storage bucket.")
    return _client().storage.from_(name)


def upload_bytes(
    bucket: str,
    path: str,
    content: bytes,
    content_type: str,
    *,
    upsert: bool = False,
) -> str:
    """Upload bytes and return the private object path, not a public URL."""
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise ValueError("Storage upload must contain non-empty bytes.")
    object_path = _safe_object_path(path)
    _bucket(bucket).upload(
        object_path,
        bytes(content),
        file_options={
            "content-type": content_type,
            "upsert": "true" if upsert else "false",
        },
    )
    return object_path


def download_bytes(bucket: str, path: str) -> bytes:
    """Download an object through the backend's privileged storage client."""
    result = _bucket(bucket).download(_safe_object_path(path))
    return bytes(result)


def delete_objects(bucket: str, paths: Iterable[str]) -> None:
    """Delete one or more objects; an empty list is a no-op."""
    safe_paths = [_safe_object_path(path) for path in paths if path]
    if safe_paths:
        _bucket(bucket).remove(safe_paths)


def create_signed_url(bucket: str, path: str, expires_in: int = 60) -> str:
    """Create a short-lived URL for an authenticated backend response."""
    if not 1 <= int(expires_in) <= 3600:
        raise ValueError("Signed URL expiry must be between 1 and 3600 seconds.")
    result = _bucket(bucket).create_signed_url(
        _safe_object_path(path), int(expires_in)
    )
    signed = result.get("signedURL") or result.get("signedUrl")
    if not signed:
        raise RuntimeError("Supabase did not return a signed storage URL.")
    return signed


def list_objects(bucket: str, prefix: str = "") -> list[dict]:
    """List private objects under a validated folder prefix."""
    if prefix:
        clean = _safe_object_path(prefix)
    else:
        clean = ""
    return _bucket(bucket).list(clean)


def move_object(bucket: str, source: str, destination: str) -> str:
    """Move an object within the same private bucket."""
    source_path = _safe_object_path(source)
    destination_path = _safe_object_path(destination)
    _bucket(bucket).move(source_path, destination_path)
    return destination_path
