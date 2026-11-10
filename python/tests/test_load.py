"""Fetching and caching a rank file: tiktoken's cache directory and its keys, the digest every file is held to, a
cached file that is not the pinned one, caching turned off, and the request's User-Agent, all against a server on the
loopback interface, so that no test reaches the network."""

from __future__ import annotations

import hashlib
import http.server
import os
import threading
from collections.abc import Iterator

import pytest

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


def test_a_fetched_file_is_held_to_its_digest_and_cached_under_tiktoken_s_key(server, cache):
    with load.fetched(server, DIGEST) as path:
        assert open(path, "rb").read() == PAYLOAD
    key = hashlib.sha1(server.encode()).hexdigest()
    assert path == os.path.join(cache, key)
    assert os.listdir(cache) == [key]
    with load.fetched(server, DIGEST) as again:
        assert again == path
    assert len(Handler.requests) == 1


def test_a_request_carries_the_package_s_user_agent_and_nothing_of_the_caller_s(server, cache):
    with load.fetched(server, DIGEST):
        pass
    headers = {name.lower(): value for name, value in Handler.requests[0].items()}
    assert headers["user-agent"] == "matisse-tokenizer/0.1.0"
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


def test_data_gym_s_directory_serves_when_tiktoken_s_is_not_named(server, monkeypatch, tmp_path):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "gym"))
    with load.fetched(server, DIGEST) as path:
        assert os.path.dirname(path) == str(tmp_path / "gym")
