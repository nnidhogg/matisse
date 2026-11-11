"""validate_merges() held to Hugging Face tokenizers' own encoding of the same merge table without a pre-tokenizer: on
small tables over three letters at every window of up to three bytes and every string up to the length past which no
refutation is needed, and on GPT-2's merge table at the frequent windows of twitter.json and at every occurrence of
each certified one there; and its refusals, which are matisse's."""

from __future__ import annotations

import collections
import functools
import itertools
import json
import pathlib

import pytest

import matisse_tokenizer
from conftest import corpus_path

tokenizers = pytest.importorskip("tokenizers")

ALPHABET = b"abc"
"""The bytes the small tables merge and the windows are drawn from; a string refuting a window over them is spelled by
them alone, since no merge reads another byte."""

WIDEST_WINDOW = 3
"""The widest window decided on the small tables."""

TABLES = [
    [],
    [("a", "b")],
    [("a", "b"), ("ab", "a")],
    [("a", "a"), ("aa", "a")],
    [("a", "b"), ("b", "c"), ("ab", "c")],
    [("b", "a"), ("a", "b"), ("ab", "a"), ("ba", "ba")],
]
"""The small tables: none, one merge, a merge that extends its own product, a self-pair, two overlapping merges and
their join, and a table whose products overlap both ways."""

MERGE_TABLE_PRE_TOKENIZER = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}
"""A ByteLevel pre-tokenizer without its regex, which splits nothing."""

DRAWN = 1000
"""How many of the corpus's most frequent pairs and of its most frequent triples are drawn, as matisse-validate draws
them unless told."""


@functools.cache
def byte_map() -> dict[int, str]:
    """Returns GPT-2's byte-to-character map, under which a ByteLevel tokenizer.json spells its bytes.

    Returns:
        The character of every byte.
    """
    printable = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    others = [byte for byte in range(256) if byte not in printable]
    return {byte: chr(byte) for byte in printable} | {byte: chr(0x100 + n) for n, byte in enumerate(others)}


@functools.cache
def undone_map() -> dict[str, int]:
    """Returns GPT-2's byte map undone.

    Returns:
        The byte of every character the map spells.
    """
    return {character: byte for byte, character in byte_map().items()}


def spelled(text: str) -> bytes:
    """Returns the bytes a token's spelling under GPT-2's byte map stands for.

    Args:
        text: The spelling.

    Returns:
        The bytes.
    """
    return bytes(undone_map()[character] for character in text)


def written_table(
    directory: pathlib.Path,
    merges: list[tuple[str, str]],
    ignore_merges: bool = False,
    pre_tokenizer: dict[str, object] = MERGE_TABLE_PRE_TOKENIZER,
) -> pathlib.Path:
    """Writes a tokenizer.json of a merge table over ASCII letters, with no pre-tokenizer unless told: the 256 single
    bytes under GPT-2's byte map, each product once after them, and one special token.

    Args:
        directory: The directory.
        merges: The merges, in rank order, each two parts of letters.
        ignore_merges: The model's ignore_merges, whether a text that is a token is looked up whole.
        pre_tokenizer: The pre-tokenizer.

    Returns:
        The file's path.
    """
    vocab = {character: byte for byte, character in byte_map().items()}
    for left, right in merges:
        vocab.setdefault(left + right, len(vocab))
    document = {
        "normalizer": None,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": None,
        "added_tokens": [
            {
                "id": len(vocab),
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [f"{left} {right}" for left, right in merges],
            "ignore_merges": ignore_merges,
        },
    }
    path = directory / f"table-{len(list(directory.iterdir()))}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def hugging_face_tokens(tokenizer, texts: list[bytes]) -> list[list[bytes]]:
    """Returns Hugging Face tokenizers' tokens of texts, each as the bytes it stands for.

    Args:
        tokenizer: The tokenizer.
        texts: The texts, each UTF-8.

    Returns:
        The tokens of each text.
    """
    encodings = tokenizer.encode_batch([text.decode("utf-8") for text in texts], add_special_tokens=False)
    return [[spelled(token) for token in encoding.tokens] for encoding in encodings]


def covering_starts(tokens: list[bytes]) -> list[int]:
    """Returns where the token covering each byte of a text begins.

    Args:
        tokens: The text's tokens.

    Returns:
        The start of the covering token of every byte, by the byte's index.
    """
    starts: list[int] = []
    for token in tokens:
        starts += [len(starts)] * len(token)
    return starts


def every_string(longest: int) -> list[bytes]:
    """Returns every string over the alphabet of one to a length of bytes.

    Args:
        longest: The longest length.

    Returns:
        The strings, shorter first.
    """
    return [bytes(string) for length in range(1, longest + 1) for string in itertools.product(ALPHABET, repeat=length)]


def most_frequent(text: bytes, width: int, count: int) -> list[bytes]:
    """Returns a text's windows of one width that occur most often, the most frequent first and a tie in the order of
    the bytes, as matisse-validate draws them.

    Args:
        text: The text.
        width: The windows' width.
        count: How many are drawn at most.

    Returns:
        The windows.
    """
    occurrences = collections.Counter(text[at : at + width] for at in range(len(text) - width + 1))
    ranked = sorted(occurrences.items(), key=lambda item: (-item[1], item[0]))
    return [window for window, _ in ranked[:count]]


@pytest.fixture
def gpt2_merge_table(tmp_path, tokenizer_jsons) -> pathlib.Path:
    """Writes GPT-2's pinned tokenizer.json with its pre-tokenizer a ByteLevel without its regex, so that its merge
    table merges a whole text.

    Returns:
        The file's path.
    """
    if "gpt2-tokenizer.json" not in tokenizer_jsons:
        pytest.skip("the vocabulary does not hold gpt2-tokenizer.json")
    document = json.loads(tokenizer_jsons["gpt2-tokenizer.json"].read_text(encoding="utf-8"))
    document["pre_tokenizer"] = MERGE_TABLE_PRE_TOKENIZER
    path = tmp_path / "gpt2-merges.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize("merges", TABLES)
def test_every_window_is_decided_as_hugging_face_s_encoding_of_every_string_up_to_the_bound_decides_it(
    tmp_path, merges
):
    path = written_table(tmp_path, merges)
    theirs = tokenizers.Tokenizer.from_file(str(path))
    windows = [
        bytes(window) for width in range(1, WIDEST_WINDOW + 1) for window in itertools.product(ALPHABET, repeat=width)
    ]
    longest_token = max([1] + [len(left) + len(right) for left, right in merges])
    strings = every_string(2 * longest_token + WIDEST_WINDOW - 2)

    starts: dict[bytes, set[int]] = collections.defaultdict(set)
    for string, tokens in zip(strings, hugging_face_tokens(theirs, strings)):
        covering = covering_starts(tokens)
        for width in range(1, WIDEST_WINDOW + 1):
            for at in range(len(string) - width + 1):
                starts[string[at : at + width]].add(covering[at + width - 1] - at)

    decided = matisse_tokenizer.validate_merges(path, windows)

    assert [window.window for window in decided] == windows
    for window in decided:
        assert window.certified == (starts[window.window] == {window.origin}), window
        own = covering_starts(hugging_face_tokens(theirs, [window.window])[0])
        assert window.origin == own[-1], window
        if window.refutation is not None:
            text = b"".join(window.refutation.tokens)
            assert [list(window.refutation.tokens)] == hugging_face_tokens(theirs, [text]), window
            occurrence = window.refutation.occurrence
            assert text[occurrence : occurrence + len(window.window)] == window.window
            assert covering_starts(list(window.refutation.tokens))[occurrence + len(window.window) - 1] != (
                occurrence + window.origin
            )


def test_gpt2_s_table_certifies_twitter_json_s_frequent_windows_at_every_occurrence_hugging_face_encodes(
    gpt2_merge_table,
):
    text = corpus_path("twitter.json").read_bytes()
    candidates = most_frequent(text, 1, 256) + most_frequent(text, 2, DRAWN) + most_frequent(text, 3, DRAWN)
    theirs = tokenizers.Tokenizer.from_file(str(gpt2_merge_table))

    decided = matisse_tokenizer.validate_merges(gpt2_merge_table, candidates)

    certified = collections.Counter(len(window.window) for window in decided if window.certified)
    assert len(decided) == 2177
    assert certified == {1: 4, 2: 488, 3: 527}

    # The corpus encoded as one piece, as a merge table without a pre-tokenizer encodes it.
    starts = covering_starts(hugging_face_tokens(theirs, [text])[0])
    for window in decided:
        if window.refutation is None:
            at = text.find(window.window)
            while at != -1:
                assert starts[at + len(window.window) - 1] == at + window.origin, (window.window, at)
                at = text.find(window.window, at + 1)
            continue
        refuting = b"".join(window.refutation.tokens)
        # Hugging Face reads a text, so a refutation that is not UTF-8 is held to the window alone.
        occurrence = window.refutation.occurrence
        assert refuting[occurrence : occurrence + len(window.window)] == window.window
        try:
            refuting.decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert [list(window.refutation.tokens)] == hugging_face_tokens(theirs, [refuting]), window.window


def test_the_refusals_are_matisse_s(tmp_path):
    recreating = written_table(tmp_path, [("c", "c"), ("c", "cc"), ("ccc", "a"), ("cc", "c")])
    with pytest.raises(ValueError) as refusal:
        matisse_tokenizer.validate_merges(recreating, [b"c"])
    assert str(refusal.value) == (
        'token_automaton: the merge ("cc", "c") at rank 3 recreates the token "ccc", which the alphabet or an earlier '
        "merge already carries"
    )

    looked_up = written_table(tmp_path, [("a", "b")], ignore_merges=True)
    with pytest.raises(ValueError, match="validate_merges: the definition looks a whole piece up"):
        matisse_tokenizer.validate_merges(looked_up, [b"ab"])

    table = written_table(tmp_path, [("a", "b")])
    with pytest.raises(ValueError, match="validate_merges: candidate 1 is empty"):
        matisse_tokenizer.validate_merges(table, [b"ab", b""])
    with pytest.raises(TypeError, match="a candidate window is bytes, not 'str'"):
        matisse_tokenizer.validate_merges(table, ["ab"])
    with pytest.raises(RuntimeError, match="cannot read"):
        matisse_tokenizer.validate_merges(tmp_path / "missing.json", [b"a"])
    assert matisse_tokenizer.validate_merges(table, []) == []

    # GPT-2's ByteLevel with its regex runs a pattern, which the empty spelling does not read.
    patterned = written_table(tmp_path, [("a", "b")], pre_tokenizer={"type": "ByteLevel", "add_prefix_space": False})
    with pytest.raises(ValueError, match="pre_tokenizer of"):
        matisse_tokenizer.validate_merges(patterned, [b"ab"])


def test_a_decided_window_carries_its_origin_and_its_refutation_s_tokens(tmp_path):
    table = written_table(tmp_path, [("a", "b"), ("ab", "a")])

    ab, b = matisse_tokenizer.validate_merges(table, [b"ab", b"b"])

    assert ab == matisse_tokenizer.MergeWindow(window=b"ab", origin=0, refutation=None)
    assert ab.certified
    assert b == matisse_tokenizer.MergeWindow(
        window=b"b", origin=0, refutation=matisse_tokenizer.Refutation(tokens=(b"ab",), occurrence=1)
    )
    assert not b.certified
