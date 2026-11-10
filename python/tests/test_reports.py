"""The shipped reports: each is the report validate() decides of its pattern at the width the registry names, and each
is read back only under its own pattern."""

from __future__ import annotations

from importlib import resources

import pytest

from matisse_tokenizer import _core
from matisse_tokenizer.registry import SOURCES


def shipped(tokenizer: str) -> bytes:
    """Returns the text of a shipped report.

    Args:
        tokenizer: The tokenizer's name in matisse.

    Returns:
        The text.
    """
    return (resources.files("matisse_tokenizer") / "reports" / f"{tokenizer}.report").read_bytes()


@pytest.mark.parametrize("name", list(SOURCES))
def test_a_shipped_report_is_the_report_its_pattern_decides_at_the_registry_s_width(name):
    source = SOURCES[name]
    decided = _core.validate(source.tokenizer, source.width)
    assert decided.width == source.width
    assert decided.serialize().encode() == shipped(source.tokenizer)
    read = _core.deserialize(source.tokenizer, shipped(source.tokenizer))
    assert read.width == source.width
    assert read.serialize() == decided.serialize()


def test_a_report_is_refused_under_a_pattern_of_other_class_symbols():
    with pytest.raises(ValueError, match="class symbols"):
        _core.deserialize("cl100k", shipped("gpt2"))
    with pytest.raises(ValueError, match="line 1"):
        _core.deserialize("gpt2", b"not a report\n")
