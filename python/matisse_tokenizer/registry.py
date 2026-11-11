"""The encodings matisse_tokenizer knows, under tiktoken's names, and get_encoding(), load_encoding(),
load_tokenizer_json() and list_encoding_names() over them.

Each encoding is a tiktoken rank file read under one of the patterns matisse describes, gpt2, cl100k, o200k and llama3,
with the vendor's special tokens, and o200k_harmony is o200k_base's rank file and pattern with tiktoken's special tokens
of it. GPT-2's rank file is r50k_base.tiktoken, which tiktoken builds its gpt2 encoding from as well: the same ranks,
pattern and special token, which the package also serves as r50k_base, as tiktoken does. The rank files of GPT-2,
cl100k_base and o200k_base are fetched from the URLs tiktoken reads them from and held to the SHA-256 tiktoken pins;
Llama 3's is distributed by Meta under its own license and is read from a file the caller names, with load_encoding().
A Hugging Face tokenizer.json is read under the pattern its pre-tokenizer runs, with load_tokenizer_json().
"""

from __future__ import annotations

import dataclasses
import threading

import os

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

    special_tokens: dict[str, int] | None = None
    """The special tokens with their ids, in the order tiktoken lists them, or None for the vendor's special tokens of
    the pattern, as the tokenizer's library supplies them."""


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

O200K_HARMONY_SPECIALS: dict[str, int] = {
    "<|endoftext|>": 199999,
    "<|endofprompt|>": 200018,
    "<|startoftext|>": 199998,
    "<|reserved_200000|>": 200000,
    "<|reserved_200001|>": 200001,
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|reserved_200004|>": 200004,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|reserved_200009|>": 200009,
    "<|reserved_200010|>": 200010,
    "<|reserved_200011|>": 200011,
    "<|call|>": 200012,
} | {f"<|reserved_{i}|>": i for i in range(200013, 201088)}
"""o200k_harmony's special tokens as tiktoken 0.14.0's openai_public.o200k_harmony() lists them, o200k_base's two
first, so that `<|endofprompt|>` stands before `<|reserved_200018|>`, which shares its id 200018 and decodes as it, as
tiktoken's decoder gives it."""

SOURCES: dict[str, Source] = {
    "gpt2": Source(
        tokenizer="gpt2",
        file="r50k_base.tiktoken",
        url="https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        sha256="306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        width=5,
        pat_str=R50K_PATTERN,
    ),
    "r50k_base": Source(
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
    "o200k_harmony": Source(
        tokenizer="o200k",
        file="o200k_base.tiktoken",
        url="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        sha256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        width=3,
        pat_str=O200K_PATTERN,
        special_tokens=O200K_HARMONY_SPECIALS,
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

TOKENIZER_JSONS: dict[str, str] = {
    "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6": "gpt2",
    "239eb2359f79c38497476671aaa835e01fb43d42743c612a8514a0dfa2ac93a2": "llama3",
    "43a3ad4618a6a938f8c2614154ea10c31ad53a62d5683ae0b0e6133575cef07e": "o200k",
    "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7": "llama3",
}
"""The SHA-256 of each pinned tokenizer.json with the pattern it is read under, in matisse's names:
openai-community/gpt2's at commit 607a30d, Xenova/gpt-4's at 1d9f1f1, whose conversion of cl100k carries Llama 3's
pattern, Xenova/gpt-4o's at 7956d98, and Meta's of Llama 3, as matisse's design.md pins them. A file with one of them
takes the shipped report."""

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
            f"Unknown encoding {name}. matisse_tokenizer knows the encodings {', '.join(list_encoding_names())}; "
            f"Encoding() builds another from its ranks under one of their patterns."
        )
    return SOURCES[name]


def get_encoding(encoding_name: str) -> core.Encoding:
    """Returns an encoding by its name, built once per process and shared, as tiktoken's get_encoding() returns it.

    The rank file is fetched from tiktoken's URL into tiktoken's cache directory unless it is there already, and held
    to its SHA-256.

    Args:
        encoding_name: gpt2, r50k_base, cl100k_base, o200k_base or o200k_harmony.

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


def load_tokenizer_json(path: str | os.PathLike[str]) -> core.Encoding:
    """Returns an encoding read from a Hugging Face tokenizer.json, under the pattern its pre-tokenizer runs and with
    the special tokens of its added tokens.

    matisse's loader recognizes the pre-tokenizer by the published spelling of each pattern matisse describes, and the
    encoding is named by that pattern's encoding: gpt2, o200k_base or llama3, and llama3 for the cl100k conversions on
    the Hugging Face Hub, which carry Llama 3's pattern; no tokenizer.json is read under cl100k_base's, which Hugging
    Face's engine reads otherwise than tiktoken. Its ids are those Hugging Face tokenizers gives the file with
    add_special_tokens false, a special token's spelling cut out where it is allowed, as tiktoken's calls cut it. A file
    whose SHA-256 is a pinned tokenizer.json's takes the shipped report; any other has its pattern's report decided when
    it is read.

    Args:
        path: The tokenizer.json's path.

    Returns:
        The encoding, not shared.

    Raises:
        ValueError: If matisse's loader refuses the file, with its message naming the member and what it holds, or no
            pattern matisse describes is the one the pre-tokenizer runs.
        RuntimeError: If the file cannot be read.
    """
    return core.Encoding._of_tokenizer_json(path)


def encoding_of_tokenizer(tokenizer: str) -> tuple[str, Source]:
    """Returns the encoding a pattern names, the first the registry lists under it.

    Args:
        tokenizer: The pattern's name in matisse, gpt2, cl100k, o200k or llama3.

    Returns:
        The encoding's name and its source.
    """
    return next((name, source) for name, source in SOURCES.items() if source.tokenizer == tokenizer)


def list_encoding_names() -> list[str]:
    """Returns the names of every encoding matisse_tokenizer knows.

    Returns:
        The names, gpt2, r50k_base, cl100k_base, o200k_base, o200k_harmony and llama3.
    """
    return list(SOURCES)
