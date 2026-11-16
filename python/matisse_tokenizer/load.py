"""Reading rank files through tiktoken's cache directory, as tiktoken's load module reads them: read_file(),
read_file_cached() and load_tiktoken_bpe(), and fetched(), which gives matisse's loader a pinned file's bytes.

The cache is tiktoken's own: the directory TIKTOKEN_CACHE_DIR names, else the one DATA_GYM_CACHE_DIR names, else
data-gym-cache under the system's temporary directory, each file named by the SHA-1 of its URL, so that a file either
package fetched serves both. An empty TIKTOKEN_CACHE_DIR turns the cache off, as it does for tiktoken. A cached file
whose SHA-256 is not the pinned one is fetched again and replaced, or left as it is in a default directory that cannot
be written. A request carries the User-Agent matisse-tokenizer/VERSION and nothing else of the caller's.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import tempfile
import urllib.request
import uuid
from collections.abc import Iterator

from matisse_tokenizer import __version__

USER_AGENT = f"matisse-tokenizer/{__version__}"
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


def read_file(blobpath: str) -> bytes:
    """Returns the bytes of a local file or of an http or https URL, as tiktoken's read_file() reads them.

    Args:
        blobpath: The path or the URL.

    Returns:
        The bytes.

    Raises:
        ValueError: If the URL is of another scheme, which tiktoken reads through blobfile and matisse_tokenizer does
            not read.
        OSError: If the file cannot be read or the request fails, urllib's errors among them.
    """
    if "://" not in blobpath:
        with open(blobpath, "rb", buffering=0) as file:
            return file.read()
    if not blobpath.startswith(("http://", "https://")):
        raise ValueError(f"matisse_tokenizer reads local files and http and https URLs, and not {blobpath}")
    request = urllib.request.Request(blobpath, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def read_file_cached(blobpath: str, expected_hash: str | None = None) -> bytes:
    """Returns the bytes of a local file or a URL through the cache, held to a SHA-256 when one is given, as tiktoken's
    read_file_cached() returns them.

    A cached file is returned when it has the SHA-256 or none is given; one with another SHA-256 is removed and read
    again, and left as it is in a default directory that cannot be written. A file read is written to the cache, a local
    file as well as a URL's, as tiktoken writes it.

    Args:
        blobpath: The path or the URL.
        expected_hash: The SHA-256 the bytes must have, or None.

    Returns:
        The bytes.

    Raises:
        ValueError: If the bytes have another SHA-256, with tiktoken's message.
        OSError: If the file cannot be read, or a cache directory the environment names cannot be written.
    """
    cached = cache_path(blobpath)
    if cached is None:
        return read_file(blobpath)

    if os.path.exists(cached):
        with open(cached, "rb", buffering=0) as file:
            data = file.read()
        if expected_hash is None or hashlib.sha256(data).hexdigest() == expected_hash:
            return data
        with contextlib.suppress(OSError):
            os.remove(cached)

    data = read_file(blobpath)
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(
            f"Hash mismatch for data downloaded from {blobpath} (expected {expected_hash}). This may indicate a "
            f"corrupted download. Please try again."
        )
    stored(cached, data)
    return data


def load_tiktoken_bpe(tiktoken_bpe_file: str, expected_hash: str | None = None) -> dict[bytes, int]:
    """Returns the ranks of a tiktoken rank file, each token's bytes with its rank, as tiktoken's load_tiktoken_bpe()
    reads them, through the cache.

    Args:
        tiktoken_bpe_file: The rank file's path or URL.
        expected_hash: The SHA-256 the file must have, or None.

    Returns:
        The ranks.

    Raises:
        ValueError: If the file has another SHA-256, or a line is not a base64 token and a rank, with tiktoken's
            messages.
    """
    contents = read_file_cached(tiktoken_bpe_file, expected_hash)
    ranks = {}
    for line in contents.splitlines():
        if not line:
            continue
        try:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
        except Exception as error:
            raise ValueError(f"Error parsing line {line!r} in {tiktoken_bpe_file}") from error
    return ranks


@contextlib.contextmanager
def fetched(url: str, expected_hash: str) -> Iterator[str]:
    """Yields the path of a file holding a URL's bytes, read through the cache by read_file_cached() and held to their
    SHA-256 whether caching is on or off.

    The file is a temporary one, written from the bytes whose SHA-256 was checked and removed once the caller is done
    with it, so that matisse's loader reads those bytes whatever the cache holds.

    Args:
        url: The URL.
        expected_hash: The SHA-256 the file must have.

    Yields:
        The file's path.

    Raises:
        ValueError: If the bytes have another SHA-256, with tiktoken's message.
        OSError: If the cache directory the environment names cannot be written.
    """
    data = read_file_cached(url, expected_hash)
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(
            f"Hash mismatch for data downloaded from {url} (expected {expected_hash}). This may indicate a corrupted "
            f"download. Please try again."
        )

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, os.path.basename(url))
        with open(path, "wb") as file:
            file.write(data)
        yield path


def stored(cached: str, data: bytes) -> bool:
    """Writes a file to the cache, through a temporary file renamed into place, as tiktoken writes it.

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
