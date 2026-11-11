"""Encoding, tiktoken's interface over matisse's encoder: the ids of a text under a rank file and a pattern matisse
describes, held to tiktoken's on every call tiktoken offers that matisse_tokenizer offers too.

An Encoding holds one matisse encoder, built with no special token allowed. A call that allows special tokens cuts the
text at their spellings here, as matisse's encoder cuts at those it is built with: from the end of each, the earliest
next one, the longest among those that begin at one byte. Each stretch between them is encoded as a complete input and
each special token emits its id, which is the encoding matisse's encoder gives with those tokens allowed, so that any
set of special tokens is served by the one encoder. A thread keeps a piece cache of its own across its calls, which
never changes an id. The checks tiktoken makes in Python, the disallowed special tokens and the lone surrogates, are
made here as tiktoken makes them.
"""

from __future__ import annotations

import base64
import functools
import math
import os
import re
import tempfile
import threading
from collections.abc import Collection, Iterable, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from typing import TYPE_CHECKING, Literal, NoReturn

from matisse_tokenizer import _core

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from matisse_tokenizer.registry import Source


class Encoding:
    """An encoding: a rank file read under one of the patterns matisse describes, with its special tokens."""

    def __init__(
        self,
        name: str,
        *,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
        explicit_n_vocab: int | None = None,
    ) -> None:
        """Creates an encoding from ranks and special tokens, as tiktoken's Encoding is created.

        The pattern must be one of the four matisse describes, spelled as tiktoken spells gpt2's, cl100k_base's and
        o200k_base's and as Meta spells Llama 3's, since matisse encodes only under a pattern it has a verified
        description of. The ranks are written to a rank file in rank order and read back by matisse's loader; when that
        file is the pinned one byte for byte, the shipped report serves it, and otherwise the pattern's report is
        decided here.

        Args:
            name: The encoding's name.
            pat_str: The pattern.
            mergeable_ranks: The bytes of every ordinary token with its rank, which is its id.
            special_tokens: The special tokens' spellings with their ids.
            explicit_n_vocab: The number of tokens, checked against the ranks and the special tokens when given.

        Raises:
            ValueError: If the pattern is not one matisse describes, if one special token's spelling begins another's,
                or if matisse's loader or encoder refuses the ranks or the special tokens, two of which sharing an id
                among them.
        """
        from matisse_tokenizer.registry import SOURCES

        known = [source for source in SOURCES.values() if source.pat_str == pat_str]
        if not known:
            raise ValueError(
                "matisse encodes only under the patterns of gpt2, cl100k_base, o200k_base and llama3, spelled as "
                "tiktoken and Meta publish them; this pattern has no description in matisse"
            )

        if explicit_n_vocab:
            assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
            assert max(max(mergeable_ranks.values()), max(special_tokens.values(), default=0)) == explicit_n_vocab - 1

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ranks.tiktoken")
            with open(path, "wb") as file:
                for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1]):
                    file.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")
            from matisse_tokenizer.load import sha256_of_file

            self._build(name, known[0], path, sha256_of_file(path), special_tokens)

    @classmethod
    def _of_file(cls, name: str, source: Source, path: str, digest: str) -> Encoding:
        """Returns an encoding read from a rank file with the vendor's special tokens of its pattern.

        Args:
            name: The encoding's name.
            source: The encoding's source.
            path: The rank file's path.
            digest: The rank file's SHA-256.

        Returns:
            The encoding.
        """
        encoding = cls.__new__(cls)
        specials = dict(_core.vendor_specials(source.tokenizer))
        encoding._build(name, source, path, digest, specials)
        return encoding

    def _build(self, name: str, source: Source, path: str, digest: str, special_tokens: dict[str, int]) -> None:
        """Reads the rank file and takes the pattern's report, the shipped one when the file is the pinned one.

        Args:
            name: The encoding's name.
            source: The source of the encoding whose pattern the file is read under.
            path: The rank file's path.
            digest: The rank file's SHA-256.
            special_tokens: The special tokens' spellings with their ids.
        """
        self.name = name
        self._pat_str = source.pat_str
        self._special_tokens = dict(special_tokens)
        self._registered = False
        self._path: str | None = None
        self._ranks: dict[bytes, int] | None = None

        by_id = sorted(self._special_tokens.items(), key=lambda item: item[1])
        self._definition = _core.read_tiktoken(os.fspath(path), by_id)

        ordinary = self._definition.ordinary_count
        for spelling, token in by_id:
            if not spelling or token < ordinary:
                raise ValueError(
                    f"the special token {spelling!r} with the id {token} is empty or has an ordinary token's id, which "
                    f"matisse's encoder refuses"
                )
        # In sorted order a spelling that begins another is directly followed by one it begins.
        spellings = sorted(self._special_tokens)
        for shorter, longer in zip(spellings, spellings[1:]):
            if longer.startswith(shorter):
                raise ValueError(
                    f"the special token {shorter!r} begins the special token {longer!r}: tiktoken chooses between two "
                    f"such spellings at one place in the order of its table, which matisse does not follow"
                )
        self.max_token_value = max(ordinary - 1, max(self._special_tokens.values(), default=0))
        self._special_token_values = set(self._special_tokens.values())

        self._report_shipped = digest == source.sha256
        if self._report_shipped:
            text = (resources.files("matisse_tokenizer") / "reports" / f"{source.tokenizer}.report").read_bytes()
            self._report = _core.deserialize(source.tokenizer, text)
        else:
            self._report = _core.validate(source.tokenizer, source.width)

        self._encoder = _core.Encoder(self._definition, self._report, [])
        self._local = threading.local()

    def __repr__(self) -> str:
        """Returns the encoding as tiktoken shows it.

        Returns:
            `<Encoding 'name'>`.
        """
        return f"<Encoding {self.name!r}>"

    # ====================
    # Encoding
    # ====================

    def encode_ordinary(self, text: str) -> list[int]:
        """Returns the ids of a text, every special token's spelling read as text.

        Equivalent to `encode(text, disallowed_special=())`. A lone surrogate is replaced by U+FFFD first, as tiktoken
        replaces it.

        Args:
            text: The text.

        Returns:
            The ids.
        """
        return self._encode(text, frozenset())

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Returns the ids of a text, with tiktoken's handling of special tokens.

        An allowed special token's spelling is cut out of the text and encoded as its id. By default every other
        special token is disallowed, and a text holding the spelling of a disallowed one is refused; a spelling that
        is neither allowed nor disallowed is encoded as text. A lone surrogate is replaced by U+FFFD, as tiktoken
        replaces it.

        Args:
            text: The text.
            allowed_special: The special tokens cut out of the text, or "all".
            disallowed_special: The special tokens whose spelling refuses the text, or "all", every special token not
                allowed.

        Returns:
            The ids.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set, as tiktoken refuses it.
            ValueError: If the text holds the spelling of a disallowed special token.
        """
        self._check(text, self._disallowed(allowed_special, disallowed_special))
        allowed = self._allowed(allowed_special)
        return self._encode(text, allowed)

    def encode_to_numpy(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> npt.NDArray[np.uint32]:
        """Returns the ids of a text as a NumPy array of unsigned 32-bit integers, as encode() returns them.

        The array owns the encoder's buffer, so no Python integer is made per id. A lone surrogate is refused with
        UnicodeEncodeError, as tiktoken's encode_to_numpy() refuses it.

        Args:
            text: The text.
            allowed_special: As for encode().
            disallowed_special: As for encode().

        Returns:
            The ids.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set, as tiktoken refuses it.
            ValueError: If the text holds the spelling of a disallowed special token.
            UnicodeEncodeError: If the text holds a lone surrogate.
        """
        self._check(text, self._disallowed(allowed_special, disallowed_special))
        allowed = self._allowed(allowed_special)
        return self._encode_array(text, allowed)

    def encode_ordinary_batch(self, text: list[str], *, num_threads: int = 8) -> list[list[int]]:
        """Returns the ids of every text, each encoded as encode_ordinary() encodes it, by a pool of threads.

        Args:
            text: The texts.
            num_threads: The threads of the pool.

        Returns:
            The ids of each text, in the texts' order.

        Raises:
            ValueError: If num_threads is not positive.
        """
        threads = _threads(num_threads)
        return self._encode_batch(list(text), frozenset(), threads)

    def encode_batch(
        self,
        text: list[str],
        *,
        num_threads: int = 8,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[list[int]]:
        """Returns the ids of every text, each encoded as encode() encodes it, by a pool of threads.

        The texts are checked for disallowed special tokens in order before any is encoded, and the first one holding a
        spelling of one is refused, as tiktoken refuses the first in order.

        Args:
            text: The texts.
            num_threads: The threads of the pool.
            allowed_special: As for encode().
            disallowed_special: As for encode().

        Returns:
            The ids of each text, in the texts' order.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set, as tiktoken refuses it.
            ValueError: If a text holds the spelling of a disallowed special token, or num_threads is not positive.
        """
        texts, allowed, threads = self._batch_arguments(text, num_threads, allowed_special, disallowed_special)
        return self._encode_batch(texts, allowed, threads)

    def encode_batch_to_numpy(
        self,
        text: list[str],
        *,
        num_threads: int = 8,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[npt.NDArray[np.uint32]]:
        """Returns the ids of every text as encode_batch() returns them, each as a NumPy array of unsigned 32-bit
        integers.

        This call is matisse's and not tiktoken's. No Python integer is made per id, which on a batch costs more than
        the encoding on many threads. A lone surrogate is refused, as encode_to_numpy() refuses it.

        Args:
            text: The texts.
            num_threads: The threads of the pool.
            allowed_special: As for encode().
            disallowed_special: As for encode().

        Returns:
            An array of the ids of each text, in the texts' order.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set, as tiktoken refuses it.
            ValueError: If a text holds the spelling of a disallowed special token, or num_threads is not positive.
            UnicodeEncodeError: If a text holds a lone surrogate.
        """
        texts, allowed, threads = self._batch_arguments(text, num_threads, allowed_special, disallowed_special)
        return self._encode_batch_stretches(texts, allowed, threads, arrays=True)

    def encode_parallel(
        self,
        text: str,
        chunks: int,
        *,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Returns the ids of a text encoded in chunks on as many threads, which are the ids encode() returns.

        This call is matisse's and not tiktoken's. The text is cut only where matisse's certified windows place a cut,
        so a text with few cuts is encoded in fewer chunks than asked for, down to one.

        Args:
            text: The text.
            chunks: The number of chunks aimed for, one thread each; zero behaves as one.
            allowed_special: As for encode().
            disallowed_special: As for encode().

        Returns:
            The ids.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set, as tiktoken refuses it.
            ValueError: If the text holds the spelling of a disallowed special token, or chunks is negative.
        """
        self._check(text, self._disallowed(allowed_special, disallowed_special))
        allowed = self._allowed(allowed_special)
        if chunks < 0:
            raise ValueError(f"chunks must not be negative, not {chunks}")
        try:
            return self._encode_parallel(text, allowed, chunks)
        except UnicodeEncodeError:
            return self._encode_parallel(_without_surrogates(text), allowed, chunks)

    def encode_parallel_to_numpy(
        self,
        text: str,
        chunks: int,
        *,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> npt.NDArray[np.uint32]:
        """Returns the ids of a text encoded in chunks as encode_parallel() encodes it, as a NumPy array.

        This call is matisse's and not tiktoken's. A lone surrogate is refused, as encode_to_numpy() refuses it.

        Args:
            text: The text.
            chunks: As for encode_parallel().
            allowed_special: As for encode().
            disallowed_special: As for encode().

        Returns:
            The ids.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set, as tiktoken refuses it.
            ValueError: If the text holds the spelling of a disallowed special token, or chunks is negative.
            UnicodeEncodeError: If the text holds a lone surrogate.
        """
        self._check(text, self._disallowed(allowed_special, disallowed_special))
        allowed = self._allowed(allowed_special)
        if chunks < 0:
            raise ValueError(f"chunks must not be negative, not {chunks}")
        import numpy

        stretches, specials = self._split(text, allowed)
        parts = [self._encoder.encode_parallel_to_numpy(stretches[0], self._share(chunks, stretches[0], text))]
        for special, stretch in zip(specials, stretches[1:]):
            parts.append(numpy.array([special], dtype=numpy.uint32))
            parts.append(self._encoder.encode_parallel_to_numpy(stretch, self._share(chunks, stretch, text)))
        return parts[0] if len(parts) == 1 else numpy.concatenate(parts)

    def encode_with_unstable(self, text: str, **kwargs: object) -> NoReturn:
        """Refuses: matisse has no list of the completions a text's last piece can take.

        matisse's streaming encoder returns the ids that are final whatever follows, which is not this call's result.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "matisse_tokenizer does not offer encode_with_unstable(): matisse lists no completions of a text's last "
            "piece"
        )

    def encode_single_token(self, text_or_bytes: str | bytes) -> int:
        """Returns the id of the one token a text or its bytes spell, ordinary or special.

        Args:
            text_or_bytes: The text or its bytes.

        Returns:
            The id.

        Raises:
            KeyError: If no token is spelled so.
        """
        if isinstance(text_or_bytes, str):
            text_or_bytes = text_or_bytes.encode("utf-8")
        token = self._mergeable_ranks.get(text_or_bytes)
        if token is not None:
            return token
        try:
            special = self._special_tokens.get(text_or_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            special = None
        if special is None:
            raise KeyError(text_or_bytes)
        return special

    # ====================
    # Decoding
    # ====================

    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        """Returns the bytes a run of ids spells, ordinary and special.

        Args:
            tokens: The ids.

        Returns:
            The bytes.

        Raises:
            KeyError: If an id names no token.
            OverflowError: If an id is negative or does not fit 32 bits.
        """
        return self._encoder.decode_bytes(tokens)

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Returns the text a run of ids spells, its bytes decoded as UTF-8 under an error handler.

        Args:
            tokens: The ids.
            errors: The handler, "replace" unless given, so that a run cut inside a code point decodes.

        Returns:
            The text.

        Raises:
            KeyError: If an id names no token.
        """
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Returns the bytes of one token, ordinary or special.

        Args:
            token: The id.

        Returns:
            The bytes.

        Raises:
            KeyError: If the id names no token.
        """
        try:
            return self.decode_bytes((token,))
        except KeyError:
            raise KeyError(str(token)) from None

    def decode_tokens_bytes(self, tokens: Sequence[int]) -> list[bytes]:
        """Returns the bytes of every token of a run.

        Args:
            tokens: The ids.

        Returns:
            Each token's bytes.
        """
        return [self.decode_single_token_bytes(token) for token in tokens]

    def decode_with_offsets(self, tokens: Sequence[int]) -> tuple[str, list[int]]:
        """Returns the text a run of ids spells and the index of the code point each token begins in.

        A token that begins inside a code point's sequence is given the index of that code point. The bytes must be
        UTF-8, as tiktoken requires.

        Args:
            tokens: The ids.

        Returns:
            The text and the offsets.

        Raises:
            UnicodeDecodeError: If the bytes are not UTF-8.
        """
        spelled = self.decode_tokens_bytes(tokens)
        points = 0
        offsets = []
        for token in spelled:
            inside = bool(token) and 0x80 <= token[0] < 0xC0
            offsets.append(max(0, points - inside))
            points += sum(1 for byte in token if not 0x80 <= byte < 0xC0)
        return b"".join(spelled).decode("utf-8", errors="strict"), offsets

    def decode_batch(
        self, batch: Sequence[Sequence[int]], *, errors: str = "replace", num_threads: int = 8
    ) -> list[str]:
        """Returns the text of every run of ids, by a pool of threads.

        Args:
            batch: The runs.
            errors: As for decode().
            num_threads: The threads of the pool.

        Returns:
            Each run's text, in the runs' order.
        """
        decoder = functools.partial(self.decode, errors=errors)
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(decoder, batch))

    def decode_bytes_batch(self, batch: Sequence[Sequence[int]], *, num_threads: int = 8) -> list[bytes]:
        """Returns the bytes of every run of ids, by a pool of threads.

        Args:
            batch: The runs.
            num_threads: The threads of the pool.

        Returns:
            Each run's bytes, in the runs' order.
        """
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(self.decode_bytes, batch))

    # ====================
    # Miscellaneous
    # ====================

    def token_byte_values(self) -> list[bytes]:
        """Returns the bytes of every ordinary token, ascending bytewise.

        Returns:
            The bytes.
        """
        return sorted(self._definition.spellings())

    @property
    def eot_token(self) -> int:
        """The id of `<|endoftext|>`.

        Raises:
            KeyError: If the encoding has no such special token, as Llama 3 has none.
        """
        return self._special_tokens["<|endoftext|>"]

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        """The spellings of the special tokens."""
        return set(self._special_tokens)

    def is_special_token(self, token: int) -> bool:
        """Returns whether an id is a special token's.

        Args:
            token: The id.

        Returns:
            True for a special token's id.
        """
        assert isinstance(token, int)
        return token in self._special_token_values

    @property
    def n_vocab(self) -> int:
        """One more than the greatest id, ordinary or special."""
        return self.max_token_value + 1

    @property
    def _mergeable_ranks(self) -> dict[bytes, int]:
        """The bytes of every ordinary token with its id, as tiktoken's Encoding holds them, built on first use."""
        if self._ranks is None:
            self._ranks = {spelling: token for token, spelling in enumerate(self._definition.spellings())}
        return self._ranks

    def __getstate__(self) -> object:
        """Returns what pickles the encoding: its name when get_encoding() gives it, its name and path when
        load_encoding() read it, and its ranks and special tokens otherwise, as tiktoken pickles an Encoding.

        Returns:
            The state.
        """
        if self._registered:
            return self.name
        if self._path is not None:
            return {"name": self.name, "path": self._path}
        return {
            "name": self.name,
            "pat_str": self._pat_str,
            "mergeable_ranks": self._mergeable_ranks,
            "special_tokens": self._special_tokens,
        }

    def __setstate__(self, value: object) -> None:
        """Rebuilds the encoding from what __getstate__() returned.

        Args:
            value: The state.
        """
        from matisse_tokenizer import registry

        if isinstance(value, str):
            self.__dict__ = registry.get_encoding(value).__dict__
            return
        if "path" in value:
            self.__dict__ = registry.load_encoding(value["name"], value["path"]).__dict__
            return
        self.__init__(**value)

    # ====================
    # Private
    # ====================

    def _disallowed(
        self, allowed_special: Literal["all"] | Set[str], disallowed_special: Literal["all"] | Collection[str]
    ) -> Collection[str]:
        """Returns the special tokens a call disallows as tiktoken resolves them: for "all", every special token not
        allowed, the set difference refusing an allowed_special that is no set as tiktoken's refuses it.

        Args:
            allowed_special: The special tokens allowed, or "all".
            disallowed_special: The special tokens disallowed, or "all".

        Returns:
            The spellings disallowed.

        Raises:
            TypeError: If disallowed_special is "all" and allowed_special is neither "all" nor a set.
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        return disallowed_special

    def _allowed(self, allowed_special: Literal["all"] | Set[str]) -> frozenset[str]:
        """Returns the special tokens a call allows: every one for "all", and otherwise the ones given that are special
        tokens of the encoding, a spelling that is not one changing nothing, as for tiktoken.

        Args:
            allowed_special: The special tokens allowed, "all" or a set.

        Returns:
            The spellings.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set nor a frozenset, as tiktoken's core refuses it.
        """
        if allowed_special == "all":
            return frozenset(self._special_tokens)
        if not isinstance(allowed_special, (set, frozenset)):
            raise TypeError(f"{_described(allowed_special)} is not an instance of 'set'")
        return frozenset(allowed_special).intersection(self._special_tokens)

    def _check(self, text: str, disallowed: Collection[str]) -> None:
        """Refuses a text holding the spelling of a disallowed special token, the first in the text, as tiktoken does.

        Args:
            text: The text.
            disallowed: The special tokens disallowed.

        Raises:
            ValueError: If the text holds such a spelling.
        """
        if not disallowed:
            return
        match = _special_token_regex(frozenset(disallowed)).search(text)
        if match:
            raise_disallowed_special_token(match.group())

    def _batch_arguments(
        self,
        text: Iterable[str],
        num_threads: float | None,
        allowed_special: Literal["all"] | Set[str],
        disallowed_special: Literal["all"] | Collection[str],
    ) -> tuple[list[str], frozenset[str], int]:
        """Reads a batch call's arguments in the order tiktoken's encode_batch() reads them: the disallowed special
        tokens, then the threads, then each text in turn, checked for a disallowed special token and, at the first, the
        allowed special tokens.

        Args:
            text: The texts, any iterable, read once.
            num_threads: The threads of the pool.
            allowed_special: The special tokens allowed, or "all".
            disallowed_special: The special tokens disallowed, or "all".

        Returns:
            The texts, the special tokens allowed and the threads.

        Raises:
            TypeError: If allowed_special is neither "all" nor a set and a text is given.
            ValueError: If a text holds the spelling of a disallowed special token, or num_threads is not positive.
        """
        disallowed = frozenset(self._disallowed(allowed_special, disallowed_special))
        threads = _threads(num_threads)
        texts = list(text)
        allowed = frozenset()
        for index, each in enumerate(texts):
            self._check(each, disallowed)
            if index == 0:
                allowed = self._allowed(allowed_special)
        return texts, allowed, threads

    def _cache(self) -> _core.Piece_cache:
        """Returns the calling thread's piece cache, made on its first call.

        Returns:
            The cache.
        """
        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = _core.Piece_cache(self._encoder)
            self._local.cache = cache
        return cache

    def _split(self, text: str, allowed: frozenset[str]) -> tuple[list[str], list[int]]:
        """Cuts a text at the spellings of the allowed special tokens, as matisse's encoder cuts at those it allows:
        from the end of each, the earliest next one, the longest among those that begin at one byte.

        Args:
            text: The text.
            allowed: The spellings allowed, special tokens of the encoding.

        Returns:
            The stretches between the special tokens, one more than the special tokens, and the special tokens' ids.
        """
        if not allowed:
            return [text], []
        stretches = []
        specials = []
        start = 0
        for match in _allowed_regex(allowed).finditer(text):
            stretches.append(text[start : match.start()])
            specials.append(self._special_tokens[match.group()])
            start = match.end()
        stretches.append(text[start:])
        return stretches, specials

    def _encode(self, text: str, allowed: frozenset[str]) -> list[int]:
        """Returns the ids of a text with special tokens allowed, a lone surrogate replaced by U+FFFD first, as tiktoken
        replaces it.

        Args:
            text: The text.
            allowed: The spellings allowed.

        Returns:
            The ids.
        """
        try:
            return self._encode_stretches(text, allowed)
        except UnicodeEncodeError:
            return self._encode_stretches(_without_surrogates(text), allowed)

    def _encode_stretches(self, text: str, allowed: frozenset[str]) -> list[int]:
        """Returns the ids of a text with special tokens allowed, each stretch between them encoded as a complete input.

        Args:
            text: The text.
            allowed: The spellings allowed.

        Returns:
            The ids.

        Raises:
            UnicodeEncodeError: If the text holds a lone surrogate.
        """
        stretches, specials = self._split(text, allowed)
        cache = self._cache()
        ids = self._encoder.encode(stretches[0], cache)
        for special, stretch in zip(specials, stretches[1:]):
            ids.append(special)
            ids += self._encoder.encode(stretch, cache)
        return ids

    def _encode_array(self, text: str, allowed: frozenset[str]) -> npt.NDArray[np.uint32]:
        """Returns the ids of a text with special tokens allowed as a NumPy array.

        Args:
            text: The text.
            allowed: The spellings allowed.

        Returns:
            The ids.

        Raises:
            UnicodeEncodeError: If the text holds a lone surrogate.
        """
        stretches, specials = self._split(text, allowed)
        cache = self._cache()
        first = self._encoder.encode_to_numpy(stretches[0], cache)
        if not specials:
            return first
        import numpy

        parts = [first]
        for special, stretch in zip(specials, stretches[1:]):
            parts.append(numpy.array([special], dtype=numpy.uint32))
            parts.append(self._encoder.encode_to_numpy(stretch, cache))
        return numpy.concatenate(parts)

    def _encode_parallel(self, text: str, allowed: frozenset[str], chunks: int) -> list[int]:
        """Returns the ids of a text with special tokens allowed, each stretch encoded in its share of the chunks.

        Args:
            text: The text.
            allowed: The spellings allowed.
            chunks: The number of chunks aimed for over the whole text.

        Returns:
            The ids.

        Raises:
            UnicodeEncodeError: If the text holds a lone surrogate.
        """
        stretches, specials = self._split(text, allowed)
        ids = self._encoder.encode_parallel(stretches[0], self._share(chunks, stretches[0], text))
        for special, stretch in zip(specials, stretches[1:]):
            ids.append(special)
            ids += self._encoder.encode_parallel(stretch, self._share(chunks, stretch, text))
        return ids

    @staticmethod
    def _share(chunks: int, stretch: str, text: str) -> int:
        """Returns a stretch's share of the chunks asked for a text, by its length, at least one.

        Args:
            chunks: The chunks asked for the text.
            stretch: The stretch.
            text: The text.

        Returns:
            The stretch's chunks.
        """
        return max(1, chunks * len(stretch) // max(1, len(text)))

    def _encode_batch(self, texts: list[str], allowed: frozenset[str], num_threads: int) -> list[list[int]]:
        """Returns the ids of every text with special tokens allowed, by a pool of threads, a lone surrogate replaced by
        U+FFFD first, as tiktoken replaces it.

        Args:
            texts: The texts.
            allowed: The spellings allowed.
            num_threads: The threads of the pool.

        Returns:
            The ids of each text.

        """
        try:
            return self._encode_batch_stretches(texts, allowed, num_threads)
        except UnicodeEncodeError:
            return self._encode_batch_stretches([_without_surrogates(each) for each in texts], allowed, num_threads)

    def _encode_batch_stretches(
        self, texts: list[str], allowed: frozenset[str], num_threads: int, arrays: bool = False
    ) -> list[list[int]] | list[npt.NDArray[np.uint32]]:
        """Returns the ids of every text with special tokens allowed, the stretches of every text encoded as one batch.

        Args:
            texts: The texts.
            allowed: The spellings allowed.
            num_threads: The threads of the pool.
            arrays: Whether each text's ids are a NumPy array rather than a list.

        Returns:
            The ids of each text.

        Raises:
            UnicodeEncodeError: If a text holds a lone surrogate.
        """
        batch = self._encoder.encode_batch_to_numpy if arrays else self._encoder.encode_batch
        if not allowed:
            return batch(texts, num_threads)
        splits = [self._split(text, allowed) for text in texts]
        flat = [stretch for stretches, _ in splits for stretch in stretches]
        encoded = iter(batch(flat, num_threads))
        result = []
        for _, specials in splits:
            parts = [next(encoded)]
            for special in specials:
                parts.append([special])
                parts.append(next(encoded))
            result.append(_joined(parts, arrays))
        return result


def _joined(parts: list[object], arrays: bool) -> list[int] | npt.NDArray[np.uint32]:
    """Returns the ids of a text's stretches and special tokens joined, as a list or as a NumPy array.

    Args:
        parts: The ids of each stretch and each special token, in order.
        arrays: Whether the ids are joined into a NumPy array rather than a list.

    Returns:
        The ids.
    """
    if not arrays:
        return [token for part in parts for token in part]
    import numpy

    return numpy.concatenate([numpy.asarray(part, dtype=numpy.uint32) for part in parts])


def _threads(num_threads: float | None) -> int:
    """Returns the threads a pool of num_threads starts at most, as tiktoken's ThreadPoolExecutor reads the number:
    Python's default for None, and a fraction rounded up.

    Args:
        num_threads: The threads asked for.

    Returns:
        The threads.

    Raises:
        ValueError: If the number is not positive, with the executor's message.
        TypeError: If it is not a number, as the executor's comparison refuses it.
    """
    if num_threads is None:
        counted = getattr(os, "process_cpu_count", os.cpu_count)()
        return min(32, (counted or 1) + 4)
    if num_threads <= 0:
        raise ValueError("max_workers must be greater than 0")
    return math.ceil(num_threads)


def _described(value: object) -> str:
    """Returns a value as tiktoken's core names it when it refuses its type.

    Args:
        value: The value.

    Returns:
        `'None'` for None and `'T' object` for a value of the type T.
    """
    return "'None'" if value is None else f"'{type(value).__name__}' object"


def _without_surrogates(text: str) -> str:
    """Returns a text with every lone surrogate replaced by U+FFFD and every surrogate pair joined, as tiktoken repairs
    a text before it encodes it.

    Args:
        text: The text.

    Returns:
        The repaired text.
    """
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


@functools.lru_cache(maxsize=128)
def _allowed_regex(tokens: frozenset[str]) -> re.Pattern[str]:
    """Returns the expression that finds the next spelling of a set of special tokens, the longest first among those
    that begin at one place.

    Args:
        tokens: The spellings.

    Returns:
        The compiled expression.
    """
    ordered = sorted(tokens, key=lambda token: (-len(token), token))
    return re.compile("|".join(re.escape(token) for token in ordered))


@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> re.Pattern[str]:
    """Returns the expression that finds the first spelling of any of a set of special tokens.

    Args:
        tokens: The spellings.

    Returns:
        The compiled expression.
    """
    inner = "|".join(re.escape(token) for token in tokens)
    return re.compile(f"({inner})")


def raise_disallowed_special_token(token: str) -> NoReturn:
    """Refuses a text for holding a disallowed special token, with tiktoken's message.

    Args:
        token: The spelling.

    Raises:
        ValueError: Always.
    """
    raise ValueError(
        f"Encountered text corresponding to disallowed special token {token!r}.\n"
        "If you want this text to be encoded as a special token, "
        f"pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\n"
        f"If you want this text to be encoded as normal text, disable the check for this token "
        f"by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\n"
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
    )
