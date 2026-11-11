"""validate_merges(), matisse's certification of the windows of a byte-level merge table without a pre-tokenizer, with
the MergeWindow each candidate is decided as and the Refutation of a window that certifies no cut.

A Hugging Face tokenizer.json whose pre-tokenizer is a ByteLevel without its regex splits nothing, so its merge table
merges a whole text as one piece and has no pattern to encode under; its cuts are the windows the table certifies
itself. matisse's validate_merges() builds munch's token automaton of the table once and decides each candidate the
caller gives on it, exactly, under the pair rule, which is Hugging Face's merge.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable

from matisse_tokenizer import _core


@dataclasses.dataclass(frozen=True)
class Refutation:
    """A string on which a window certifies no cut: the merge's tokens of it, and where the window occurs in it with
    the token covering its last byte beginning elsewhere than the origin."""

    tokens: tuple[bytes, ...]
    """The encoder's segmentation of the string, which the tokens spell."""

    occurrence: int
    """The index of the window's first byte in the string."""


@dataclasses.dataclass(frozen=True)
class MergeWindow:
    """A candidate window decided: its bytes, the origin it is decided at, and its refutation or None."""

    window: bytes
    """The window's bytes."""

    origin: int
    """Where the window's own encoding begins the token covering its last byte, the only origin that can be
    certified, since the window occurs in itself."""

    refutation: Refutation | None
    """None when the window is certified, the token covering its last byte beginning at the origin at every occurrence
    in every byte string; otherwise a string that refutes it, so that it certifies no cut at any origin."""

    @property
    def certified(self) -> bool:
        """Whether the window is certified at its origin."""
        return self.refutation is None


def validate_merges(path: str | os.PathLike[str], candidates: Iterable[bytes]) -> list[MergeWindow]:
    """Returns each candidate window of the merge table of a tokenizer.json whose pre-tokenizer splits nothing, decided
    by matisse's validate_merges().

    The file is read by matisse's loader under the empty spelling, which takes a ByteLevel pre-tokenizer without its
    regex, and every window is decided at the origin where its own encoding begins the token covering its last byte, by
    munch's miscovering() on the table's token automaton, built once for all of them. The verdicts are of the merge
    alone, on a text with no special token cut out. The interpreter lock is released while the file is read and the
    windows are decided.

    Args:
        path: The tokenizer.json's path.
        candidates: The candidate windows, each nonempty bytes.

    Returns:
        Each candidate decided, in the order given.

    Raises:
        TypeError: If a candidate is not bytes.
        ValueError: If matisse's loader refuses the file under the empty spelling, a pre-tokenizer that splits a text
            among its refusals, with its message naming the member, or matisse refuses the table, with munch's message
            naming a merge whose part no earlier merge makes or that makes a token again, or the whole-piece lookup the
            file's ignore_merges gives, or an empty candidate, naming its index.
        RuntimeError: If the file cannot be read.
    """
    windows = list(candidates)
    for window in windows:
        if not isinstance(window, bytes):
            raise TypeError(f"a candidate window is bytes, not '{type(window).__name__}'")

    decided = _core.validate_merges(os.fspath(path), windows)

    return [
        MergeWindow(
            window=window,
            origin=origin,
            refutation=None if refuted is None else Refutation(tokens=tuple(refuted[0]), occurrence=refuted[1]),
        )
        for window, origin, refuted in decided
    ]
