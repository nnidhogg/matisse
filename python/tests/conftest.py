"""What the tests share: the pinned vocabularies served to both matisse_tokenizer and tiktoken through one cache
directory, the two packages' encodings side by side, and matisse's fixtures and corpora.

The vocabulary comes from the directory MATISSE_VOCAB_DIR names, as matisse's CMake option names it, holding
cl100k_base.tiktoken, o200k_base.tiktoken and, for Llama 3, llama3.tiktoken, and GPT-2's gpt2-vocab.json and
gpt2-merges.txt, from which GPT-2's rank file is written with tiktoken's own conversion and held to tiktoken's
r50k_base digest. With MATISSE_TEST_DOWNLOAD=1 and no directory, the files are fetched from tiktoken's URLs instead,
Llama 3's left out. With neither, every test that needs a vocabulary is skipped. MATISSE_CORPUS_DIR names the corpora of
matisse's docs/benchmarks.md, and the tests of a corpus the directory lacks are skipped.
"""

from __future__ import annotations

import base64
import hashlib
import os
import pathlib
import shutil

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
"""The repository's root, where matisse's fixtures are."""

FIXTURES = REPOSITORY / "libs" / "load" / "tests" / "data"
"""The directory of the known-answer and vendor fixtures."""

NAMES = ["gpt2", "cl100k_base", "o200k_base", "llama3"]
"""Every encoding, by its name."""

UNIVERSES = {"gpt2": "gpt2", "cl100k_base": "cl100k", "o200k_base": "o200k", "llama3": "llama3"}
"""The universe of each encoding in matisse's fixtures."""

GPT2_URLS = {
    "gpt2-merges.txt": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
    "gpt2-vocab.json": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
}
"""The URLs of GPT-2's two files, which tiktoken's own gpt2 encoding reads."""

LLAMA3_SPECIALS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>",
] + [f"<|reserved_special_token_{index}|>" for index in range(5, 256 - 5)]
"""Llama 3's special tokens in id order from 128000, as meta-llama/llama3's llama/tokenizer.py names them."""


def sha1_name(url: str) -> str:
    """Returns the name tiktoken's cache gives a URL's file.

    Args:
        url: The URL.

    Returns:
        The SHA-1 of the URL, in hex.
    """
    return hashlib.sha1(url.encode()).hexdigest()


def sha256_of(path: pathlib.Path) -> str:
    """Returns the SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        The digest, in hex.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_gpt2_ranks(vocab: pathlib.Path, target: pathlib.Path) -> None:
    """Writes GPT-2's rank file from encoder.json and vocab.bpe with tiktoken's own conversion.

    Args:
        vocab: The directory holding gpt2-vocab.json and gpt2-merges.txt.
        target: The rank file to write.
    """
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks

    ranks = data_gym_to_mergeable_bpe_ranks(str(vocab / "gpt2-merges.txt"), str(vocab / "gpt2-vocab.json"))
    with open(target, "wb") as file:
        for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
            file.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")


@pytest.fixture(scope="session")
def vocabulary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pathlib.Path]:
    """Serves the pinned rank files to both packages through one cache directory, set as TIKTOKEN_CACHE_DIR.

    Returns:
        The path of each encoding's rank file, Llama 3's only when the vocabulary directory holds it.
    """
    from matisse_tokenizer.registry import SOURCES

    cache = tmp_path_factory.mktemp("cache")
    files = tmp_path_factory.mktemp("files")
    named = os.environ.get("MATISSE_VOCAB_DIR")
    paths: dict[str, pathlib.Path] = {}

    if named:
        vocab = pathlib.Path(named)
        write_gpt2_ranks(vocab, files / "r50k_base.tiktoken")
        for file, url in GPT2_URLS.items():
            shutil.copyfile(vocab / file, cache / sha1_name(url))
        for name, source in SOURCES.items():
            path = files / source.file if name == "gpt2" else vocab / source.file
            if not path.exists():
                continue
            assert sha256_of(path) == source.sha256, f"{path} is not the pinned file"
            paths[name] = path
            if source.url is not None:
                shutil.copyfile(path, cache / sha1_name(source.url))
    elif os.environ.get("MATISSE_TEST_DOWNLOAD") == "1":
        from matisse_tokenizer import load

        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
        for name, source in SOURCES.items():
            if source.url is None:
                continue
            with load.fetched(source.url, source.sha256) as path:
                paths[name] = pathlib.Path(path)
    else:
        pytest.skip("no vocabulary: set MATISSE_VOCAB_DIR, or MATISSE_TEST_DOWNLOAD=1 to fetch it")

    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    return paths


@pytest.fixture(scope="session")
def pairs(vocabulary: dict[str, pathlib.Path]) -> dict[str, tuple[object, object]]:
    """Returns each encoding the vocabulary serves under both packages, matisse_tokenizer's first.

    Llama 3's tiktoken encoding is built as meta-llama/llama3's tokenizer.py builds it, from the rank file, Meta's
    pattern and its 256 special tokens.

    Returns:
        The two encodings of each name.
    """
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    import matisse_tokenizer
    from matisse_tokenizer.registry import LLAMA3_PATTERN

    built: dict[str, tuple[object, object]] = {}
    for name in NAMES:
        if name not in vocabulary:
            continue
        if name == "llama3":
            path = str(vocabulary[name])
            ranks = load_tiktoken_bpe(path)
            specials = {spelling: len(ranks) + index for index, spelling in enumerate(LLAMA3_SPECIALS)}
            theirs = tiktoken.Encoding("llama3", pat_str=LLAMA3_PATTERN, mergeable_ranks=ranks, special_tokens=specials)
            built[name] = (matisse_tokenizer.load_encoding("llama3", path), theirs)
            continue
        built[name] = (matisse_tokenizer.get_encoding(name), tiktoken.get_encoding(name))
    return built


@pytest.fixture(params=NAMES)
def pair(request: pytest.FixtureRequest, pairs: dict[str, tuple[object, object]]) -> tuple[object, object]:
    """Returns one encoding under both packages, each test run once per encoding the vocabulary serves.

    Returns:
        matisse_tokenizer's encoding and tiktoken's.
    """
    if request.param not in pairs:
        pytest.skip(f"the vocabulary does not hold {request.param}")
    return pairs[request.param]


def corpus_path(name: str) -> pathlib.Path:
    """Returns a corpus of the directory MATISSE_CORPUS_DIR names, skipping the test when it is not there.

    Args:
        name: The corpus's file name.

    Returns:
        The path.
    """
    named = os.environ.get("MATISSE_CORPUS_DIR")
    if not named:
        pytest.skip("no corpus directory: set MATISSE_CORPUS_DIR")
    path = pathlib.Path(named) / name
    if not path.exists():
        pytest.skip(f"the corpus directory does not hold {name}")
    return path
