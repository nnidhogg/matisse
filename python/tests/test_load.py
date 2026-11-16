"""Fetching and caching a rank file: tiktoken's cache directory and its keys, the digest every file is held to, a cached
file that is not the pinned one, also where it cannot be replaced, caching turned off, and the request's User-Agent, all
against a server on the loopback interface, so that no test reaches the network."""

from __future__ import annotations

import hashlib
import http.server
import os
import threading
from collections.abc import Iterator

import pytest

import matisse_tokenizer
from matisse_tokenizer import load

PAYLOAD = b"a rank file's bytes\n"
"""What the server serves."""


class Handler(http.server.BaseHTTPRequestHandler):
    """Serves PAYLOAD at every path and records each request's headers."""

    requests: list[dict[str, str]] = []
    """The headers of every request served."""

    def do_GET(self) -> None:
        """Serves PAYLOAD and records the request's headers."""
        Handler.requests.append(dict(self.headers))
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, format: str, *args: object) -> None:
        """Writes nothing, so the tests' output stays clean.

        Args:
            format: The message's format.
            args: Its arguments.
        """


@pytest.fixture
def server() -> Iterator[str]:
    """Runs the server on a free port of the loopback interface for one test.

    Yields:
        The URL of a file on it.
    """
    Handler.requests.clear()
    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/encodings/test.tiktoken"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def cache(tmp_path, monkeypatch) -> str:
    """Names an empty cache directory with TIKTOKEN_CACHE_DIR for one test.

    Returns:
        The directory.
    """
    directory = str(tmp_path / "cache")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", directory)
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    return directory


DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
"""The SHA-256 of PAYLOAD."""


def refused(*args: object) -> None:
    """Refuses to change a file, as a directory that cannot be written refuses it, whoever the process runs as.

    Args:
        args: The call's arguments.

    Raises:
        PermissionError: Always.
    """
    raise PermissionError("the directory cannot be written")


def test_a_fetched_file_is_held_to_its_digest_and_cached_under_tiktoken_s_key(server, cache):
    with load.fetched(server, DIGEST) as path:
        assert open(path, "rb").read() == PAYLOAD
    key = hashlib.sha1(server.encode()).hexdigest()
    assert os.listdir(cache) == [key]
    assert open(os.path.join(cache, key), "rb").read() == PAYLOAD
    with load.fetched(server, DIGEST) as again:
        assert open(again, "rb").read() == PAYLOAD
    assert len(Handler.requests) == 1


def test_a_request_carries_the_package_s_user_agent_and_nothing_of_the_caller_s(server, cache):
    with load.fetched(server, DIGEST):
        pass
    headers = {name.lower(): value for name, value in Handler.requests[0].items()}
    assert headers["user-agent"] == f"matisse-tokenizer/{matisse_tokenizer.__version__}"
    assert set(headers) <= {"user-agent", "host", "accept-encoding", "connection"}


def test_a_file_with_another_digest_is_refused(server, cache):
    with pytest.raises(ValueError, match="Hash mismatch"):
        with load.fetched(server, "0" * 64):
            pass
    assert not os.path.exists(os.path.join(cache, hashlib.sha1(server.encode()).hexdigest()))


def test_a_cached_file_that_is_not_the_pinned_one_is_fetched_again(server, cache):
    os.makedirs(cache)
    cached = os.path.join(cache, hashlib.sha1(server.encode()).hexdigest())
    with open(cached, "wb") as file:
        file.write(b"something else")
    with load.fetched(server, DIGEST) as path:
        assert open(path, "rb").read() == PAYLOAD
    assert open(cached, "rb").read() == PAYLOAD
    assert len(Handler.requests) == 1


def test_an_empty_cache_directory_turns_the_cache_off(server, monkeypatch, tmp_path):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    with load.fetched(server, DIGEST) as path:
        assert open(path, "rb").read() == PAYLOAD
    assert not os.path.exists(path)
    with load.fetched(server, DIGEST):
        pass
    assert len(Handler.requests) == 2


def test_with_the_cache_off_a_file_with_another_digest_is_refused_though_read_file_cached_returns_it(
    server, monkeypatch
):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    assert load.read_file_cached(server, "0" * 64) == PAYLOAD
    with pytest.raises(ValueError, match="Hash mismatch"):
        with load.fetched(server, "0" * 64):
            pass


def test_the_fetched_bytes_are_read_where_a_stale_file_in_the_default_directory_cannot_be_replaced(
    server, monkeypatch, tmp_path
):
    directory = tmp_path / "default"
    directory.mkdir()
    cached = directory / hashlib.sha1(server.encode()).hexdigest()
    cached.write_bytes(b"something else")
    monkeypatch.setattr(load, "cache_dir", lambda: (str(directory), False))
    monkeypatch.setattr(load.os, "remove", refused)
    monkeypatch.setattr(load.os, "replace", refused)
    with load.fetched(server, DIGEST) as path:
        assert open(path, "rb").read() == PAYLOAD
    assert cached.read_bytes() == b"something else"


def test_data_gym_s_directory_serves_when_tiktoken_s_is_not_named(server, monkeypatch, tmp_path):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "gym"))
    with load.fetched(server, DIGEST):
        pass
    assert os.listdir(tmp_path / "gym") == [hashlib.sha1(server.encode()).hexdigest()]


def test_load_tiktoken_bpe_reads_ranks_as_tiktoken_s_does(vocabulary, cache, server):
    import tiktoken.load

    path = str(vocabulary["cl100k_base"])
    assert load.load_tiktoken_bpe(path) == tiktoken.load.load_tiktoken_bpe(path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert load.load_tiktoken_bpe(path, digest) == tiktoken.load.load_tiktoken_bpe(path, digest)
    # As tiktoken does, a local file is cached too, under the SHA-1 of its path.
    assert os.path.exists(os.path.join(cache, hashlib.sha1(path.encode()).hexdigest()))
    with pytest.raises(ValueError, match="Hash mismatch"):
        load.load_tiktoken_bpe(server, "0" * 64)
    with pytest.raises(ValueError, match="Error parsing line"):
        load.load_tiktoken_bpe(server)


def test_meta_s_llama_3_tokenizer_builds_its_encoding_from_load_tiktoken_bpe(vocabulary, pairs):
    if "llama3" not in pairs:
        pytest.skip("the vocabulary does not hold llama3")
    from conftest import LLAMA3_SPECIALS
    from matisse_tokenizer.registry import LLAMA3_PATTERN

    _, theirs = pairs["llama3"]
    ranks = load.load_tiktoken_bpe(str(vocabulary["llama3"]))
    specials = {spelling: len(ranks) + index for index, spelling in enumerate(LLAMA3_SPECIALS)}
    ours = matisse_tokenizer.Encoding("llama3", pat_str=LLAMA3_PATTERN, mergeable_ranks=ranks, special_tokens=specials)
    assert ours._report_shipped
    text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello, Llama!<|eot_id|>"
    assert ours.encode(text, allowed_special="all") == theirs.encode(text, allowed_special="all")
