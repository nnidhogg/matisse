"""Fetching the pinned rank files into tiktoken's cache directory, each held to its SHA-256.

The cache is tiktoken's own: the directory TIKTOKEN_CACHE_DIR names, else the one DATA_GYM_CACHE_DIR names, else
data-gym-cache under the system's temporary directory, each file named by the SHA-1 of its URL, so that a file either
package fetched serves both. An empty TIKTOKEN_CACHE_DIR turns the cache off, as it does for tiktoken. A cached file
whose SHA-256 is not the pinned one is fetched again and replaced. A request carries the User-Agent
matisse-tokenizer/VERSION and nothing else of the caller's.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import urllib.request
import uuid
from collections.abc import Iterator

USER_AGENT = "matisse-tokenizer/0.1.0"
"""The User-Agent every request carries: the package and its version, and nothing of the caller's."""

TIMEOUT = 60
"""The seconds a request may take to connect or to deliver its next bytes."""


def cache_dir() -> tuple[str, bool]:
    """Returns the cache directory as tiktoken chooses it.

    Returns:
        The directory, empty when caching is off, and whether the caller's environment named it.
    """
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        return os.environ["TIKTOKEN_CACHE_DIR"], True
    if "DATA_GYM_CACHE_DIR" in os.environ:
        return os.environ["DATA_GYM_CACHE_DIR"], True
    return os.path.join(tempfile.gettempdir(), "data-gym-cache"), False


def cache_path(url: str) -> str | None:
    """Returns the path a URL's file is cached at.

    Args:
        url: The URL.

    Returns:
        The path, tiktoken's cache key under the cache directory, or None when caching is off.
    """
    directory, _ = cache_dir()
    if directory == "":
        return None
    return os.path.join(directory, hashlib.sha1(url.encode()).hexdigest())


def sha256_of_file(path: str | os.PathLike[str]) -> str:
    """Returns the SHA-256 of a file's bytes.

    Args:
        path: The file's path.

    Returns:
        The digest, in hex.
    """
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(url: str, expected_hash: str) -> bytes:
    """Fetches a URL's bytes and holds them to a SHA-256.

    Args:
        url: The URL.
        expected_hash: The SHA-256 the bytes must have.

    Returns:
        The bytes.

    Raises:
        ValueError: If the bytes have another SHA-256.
        urllib.error.URLError: If the request fails.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(
            f"Hash mismatch for data downloaded from {url} (expected {expected_hash}). This may indicate a corrupted "
            f"download. Please try again."
        )
    return data


@contextlib.contextmanager
def fetched(url: str, expected_hash: str) -> Iterator[str]:
    """Yields the path of a URL's file held to its SHA-256, from the cache when it holds the file and fetched otherwise.

    A fetched file is written to the cache; when caching is off, or the default cache directory cannot be written, it is
    written to a temporary file that is removed once the caller is done with it.

    Args:
        url: The URL.
        expected_hash: The SHA-256 the file must have.

    Yields:
        The file's path.

    Raises:
        ValueError: If the fetched bytes have another SHA-256.
        OSError: If the cache directory the environment names cannot be written.
    """
    cached = cache_path(url)

    if cached is not None and os.path.exists(cached):
        if sha256_of_file(cached) == expected_hash:
            yield cached
            return
        with contextlib.suppress(OSError):
            os.remove(cached)

    data = download(url, expected_hash)

    if cached is not None and stored(cached, data):
        yield cached
        return

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, os.path.basename(url))
        with open(path, "wb") as file:
            file.write(data)
        yield path


def stored(cached: str, data: bytes) -> bool:
    """Writes a fetched file to the cache, through a temporary file renamed into place, as tiktoken writes it.

    Args:
        cached: The file's path in the cache.
        data: The file's bytes.

    Returns:
        True when the file was written, False when the default cache directory cannot be written.

    Raises:
        OSError: If a cache directory the environment names cannot be written.
    """
    _, named = cache_dir()
    try:
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        partial = f"{cached}.{uuid.uuid4()}.tmp"
        with open(partial, "wb") as file:
            file.write(data)
        os.replace(partial, cached)
    except OSError:
        if named:
            raise
        return False
    return True
