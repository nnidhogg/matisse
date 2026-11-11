"""The Unicode version: matisse's class tables are pinned to Unicode 16.0.0, the version of the tables tiktoken 0.14.0's
regex engine reads, so that the two class every code point alike. These tests hold matisse-tokenizer's ids to tiktoken's
at every Unicode scalar value, and hold tiktoken to matisse's tables through matisse's own check,
tools/unicode/check_tiktoken.py, which fails when a tiktoken reads a code point as a class the tables do not give it, as
tables of a later Unicode version would.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io

import pytest

from conftest import REPOSITORY, UNIVERSES

SCALARS = [*range(0xD800), *range(0xE000, 0x110000)]
"""Every Unicode scalar value, the code points but the surrogates."""

PLANES = [point for point in SCALARS if point < 0x40000 or 0xE0000 <= point < 0xF0000]
"""The scalar values of the planes Unicode assigns letters, numbers and marks in, 0 to 3, and 14."""


def check_tiktoken() -> object:
    """Returns matisse's check of tiktoken's classes against its tables, tools/unicode/check_tiktoken.py, as a module.

    Returns:
        The module.
    """
    path = REPOSITORY / "tools" / "unicode" / "check_tiktoken.py"
    if not path.exists():
        pytest.skip(f"{path} is not here")
    spec = importlib.util.spec_from_file_location("check_tiktoken", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_class_tables_are_those_of_unicode_16(pair):
    ours, _ = pair
    version, _ = check_tiktoken().classed(UNIVERSES[ours.name])
    assert version == "16.0.0"


def test_tiktoken_reads_every_code_point_the_tables_leave_out_as_the_other_class(pair):
    ours, theirs = pair
    # The check prints a line per tokenizer, which the test keeps out of its output.
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        departed = check_tiktoken().check_tables(UNIVERSES[ours.name], theirs)
    assert departed == 0, printed.getvalue()


def test_matisse_and_tiktoken_agree_at_every_scalar_value(pair):
    ours, theirs = pair
    texts = [chr(point) + "'s" for point in SCALARS]
    assert ours.encode_ordinary_batch(texts) == theirs.encode_ordinary_batch(texts)


def test_matisse_and_tiktoken_agree_at_every_code_point_of_the_assigned_planes_in_three_more_contexts(pair):
    ours, theirs = pair
    contexts = ("x{}1", " {0}{0}", "a{}A")
    texts = [context.format(chr(point)) for point in PLANES for context in contexts]
    assert ours.encode_ordinary_batch(texts) == theirs.encode_ordinary_batch(texts)
