"""The encodings matisse_tokenizer knows, under tiktoken's names, and get_encoding(), load_encoding() and
list_encoding_names() over them.

Each encoding is a tiktoken rank file read under one of the patterns matisse describes, gpt2, cl100k, o200k and llama3,
with the vendor's special tokens. GPT-2's rank file is r50k_base.tiktoken, which tiktoken builds its gpt2 encoding
from as well: the same ranks, pattern and special token. The rank files of GPT-2, cl100k_base and o200k_base are
fetched from the URLs tiktoken reads them from and held to the SHA-256 tiktoken pins; Llama 3's is distributed by Meta
under its own license and is read from a file the caller names, with load_encoding().
"""

from __future__ import annotations

import dataclasses
import threading

from matisse_tokenizer import core, load


@dataclasses.dataclass(frozen=True)
class Source:
    """Where an encoding's rank file comes from, what pins it, and how matisse reads it."""

    tokenizer: str
    """The pattern's name in matisse, as lookup::tokenizer() knows it."""

    file: str
    """The rank file's name, the last part of its URL."""

    url: str | None
    """The URL tiktoken reads the rank file from, None when no vendor publishes one to fetch."""

    sha256: str
    """The SHA-256 of the pinned rank file, which the shipped report is held to."""

    width: int
    """The widest window of the pattern's report, as the shipped report is decided."""

    pat_str: str
    """The pattern as tiktoken's Encoding takes it, spelled as its vendor publishes it."""


R50K_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
"""GPT-2's pattern as tiktoken spells it for gpt2 and r50k_base, equivalent to the spelling of GPT-2's encoder.py."""

CL100K_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|"""
    r"""\s+(?!\S)|\s"""
)
"""cl100k_base's pattern, as tiktoken's openai_public.py spells it."""

O200K_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"""
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"""
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
"""o200k_base's pattern, as tiktoken's openai_public.py spells it."""

LLAMA3_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"""
    r"""\s+(?!\S)|\s+"""
)
"""Llama 3's pattern, as meta-llama/llama3's llama/tokenizer.py spells it."""

SOURCES: dict[str, Source] = {
    "gpt2": Source(
        tokenizer="gpt2",
        file="r50k_base.tiktoken",
        url="https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        sha256="306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        width=5,
        pat_str=R50K_PATTERN,
    ),
    "cl100k_base": Source(
        tokenizer="cl100k",
        file="cl100k_base.tiktoken",
        url="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        sha256="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        width=5,
        pat_str=CL100K_PATTERN,
    ),
    "o200k_base": Source(
        tokenizer="o200k",
        file="o200k_base.tiktoken",
        url="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        sha256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        width=3,
        pat_str=O200K_PATTERN,
    ),
    "llama3": Source(
        tokenizer="llama3",
        file="llama3.tiktoken",
        url=None,
        sha256="82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
        width=5,
        pat_str=LLAMA3_PATTERN,
    ),
}
"""Every encoding matisse_tokenizer knows, by its name."""

_lock = threading.RLock()
"""Held while an encoding is built, so that each is built once."""

_encodings: dict[str, core.Encoding] = {}
"""The encodings get_encoding() has built, by name."""


def source_of(name: str) -> Source:
    """Returns the source of an encoding.

    Args:
        name: The encoding's name.

    Returns:
        The source.

    Raises:
        ValueError: If matisse_tokenizer knows no encoding of the name.
    """
    if name not in SOURCES:
        raise ValueError(
            f"Unknown encoding {name}. matisse_tokenizer encodes {', '.join(list_encoding_names())}; tiktoken's other "
            f"encodings have no pattern description in matisse."
        )
    return SOURCES[name]


def get_encoding(encoding_name: str) -> core.Encoding:
    """Returns an encoding by its name, built once per process and shared, as tiktoken's get_encoding() returns it.

    The rank file is fetched from tiktoken's URL into tiktoken's cache directory unless it is there already, and held
    to its SHA-256.

    Args:
        encoding_name: gpt2, cl100k_base or o200k_base.

    Returns:
        The encoding.

    Raises:
        ValueError: If matisse_tokenizer knows no encoding of the name, if the name is llama3, whose rank file no vendor
            publishes to fetch, or if the file fetched is not the pinned one.
    """
    if not isinstance(encoding_name, str):
        raise ValueError(f"Expected a string in get_encoding, got {type(encoding_name)}")

    with _lock:
        if encoding_name in _encodings:
            return _encodings[encoding_name]

        source = source_of(encoding_name)

        if source.url is None:
            raise ValueError(
                f"The rank file of {encoding_name} is distributed by its vendor under its own license and is not "
                f"fetched; read a copy with load_encoding({encoding_name!r}, path)."
            )

        with load.fetched(source.url, source.sha256) as path:
            encoding = core.Encoding._of_file(encoding_name, source, path, source.sha256)

        encoding._registered = True
        _encodings[encoding_name] = encoding
        return encoding


def load_encoding(encoding_name: str, path: str) -> core.Encoding:
    """Returns an encoding read from a local rank file under the pattern and special tokens of a known encoding.

    A file whose SHA-256 is the pinned file's takes the report shipped with the package; any other file is a table
    matisse has not seen, and its pattern's report is decided when it is read, which takes a second or two.

    Args:
        encoding_name: gpt2, cl100k_base, o200k_base or llama3.
        path: The rank file's path.

    Returns:
        The encoding, not shared.

    Raises:
        ValueError: If matisse_tokenizer knows no encoding of the name, or the file is not a rank file matisse reads.
        RuntimeError: If the file cannot be read.
    """
    source = source_of(encoding_name)

    digest = load.sha256_of_file(path)

    encoding = core.Encoding._of_file(encoding_name, source, path, digest)
    encoding._path = str(path)
    return encoding


def list_encoding_names() -> list[str]:
    """Returns the names of every encoding matisse_tokenizer knows.

    Returns:
        The names, gpt2, cl100k_base, o200k_base and llama3.
    """
    return list(SOURCES)
