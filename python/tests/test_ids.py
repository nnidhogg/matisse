"""Every id held to tiktoken's: matisse's known answers, its corpora, seeded random texts and adversarial ones, under
all four encodings, through encode(), encode_ordinary(), encode_to_numpy(), the batch calls and encode_parallel()."""

from __future__ import annotations

import hashlib
import random

import numpy
import pytest

from conftest import FIXTURES, UNIVERSES, corpus_path

SEED = 20261110
"""The seed of the random texts."""

RANDOM_TEXTS = 3000
"""The random texts each encoding is held to tiktoken on."""

WHITE_SPACE = [" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x85", "\xa0", " ", " ", " ", " ", "　"]
"""White space the patterns read apart: ASCII's, the next line, the no-break space and Unicode's other White_Space."""

PIECES = (
    WHITE_SPACE
    + ["\r\n", "\n\n", "  ", "\t\t", " \n", "\n\t"]
    + ["'s", "'S", "'t", "'T", "'re", "'RE", "'ve", "'m", "'ll", "'LL", "'d", "'D", "'ſ", "'x"]
    + ["a", "Z", "the", " The", "HTTP", "camelCase", "é", "naïve", "ß", "İ", "ſ"]
    + ["αβγ", "Жизнь", "مرحبا", "नमस्ते"]
    + ["日本語", "한국어", "ไทย", "ǅ", "ʰ", "Ⅰ", "½", "²"]
    + ["́", "é", "ً", "ि", "‍", "​", "﻿", "�", "", "\U0001f600"]
    + ["\U0001f468‍\U0001f469‍\U0001f467", "\U0001f1f8\U0001f1ea", "\U00010400", "\U00020000", "\U000e0001"]
    + ["0", "7", "42", "123", "4567", "89012", "٣", "１", "3.14", "1,000"]
    + ["!", "?", ".", ",", "...", "--", "/", "//", "\\", "{", "}", "(", ")", "<", ">", "|", "<|", "|>", "\"", "'"]
    + ["<|endoftext|>", "<|endofprompt|>", "<|fim_prefix|>", "<|begin_of_text|>", "<|eot_id|>", "<|endoftext|"]
)
"""The pieces random texts are drawn from: white space, contractions in both cases, letters of many scripts, title
case and modifier letters, marks, joiners, emoji, astral letters, numbers, punctuation and special tokens' spellings."""


def digest_of(ids: numpy.ndarray | list[int]) -> str:
    """Returns the SHA-256 of ids as little-endian 32-bit words, as matisse's fixtures digest them.

    Args:
        ids: The ids.

    Returns:
        The digest, in hex.
    """
    return hashlib.sha256(numpy.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def random_texts(count: int) -> list[str]:
    """Returns seeded random texts drawn from the pieces, with random code points of every plane among them.

    Args:
        count: The number of texts.

    Returns:
        The texts.
    """
    generator = random.Random(SEED)
    texts = []
    for _ in range(count):
        parts = []
        for _ in range(generator.randrange(0, 40)):
            if generator.random() < 0.1:
                point = generator.randrange(0, 0x110000)
                parts.append(chr(point if not 0xD800 <= point < 0xE000 else 0x20))
            else:
                parts.append(generator.choice(PIECES))
        texts.append("".join(parts))
    return texts


def adversarial_texts() -> list[str]:
    """Returns texts made to stress the patterns and the merge: long runs of white space, letters and digits, white
    space before letters and punctuation, newlines before runs of tabs, long single pieces, and special tokens'
    spellings against letters and against each other.

    Returns:
        The texts.
    """
    texts = [" " * count + tail for count in range(0, 70) for tail in ("", "a", "!", "\n", "1")]
    texts += ["\n" + "\t" * count + tail for count in (1, 2, 3, 17, 4000) for tail in ("", "a", " ", "\n")]
    texts += [" " * 100000, " " * 100000 + "x", "\n" * 50000, "\r\n" * 30000, "\t" * 70000 + "!"]
    texts += ["a" * 100000, "A" * 50000 + "b", "1" * 100001, "ab" * 50000, "日" * 60000, "é" * 40000]
    texts += ["x" * 128 + " ", "x" * 129, "Ж" * 64, "Ж" * 65, ".:" * 5000, "́" * 3000]
    texts += ["  " * 3000, "　" * 2000 + "a", "'s" * 9999, "'LL" * 3000, "!" * 30000 + "\r\n" * 10]
    texts += ["x<|endoftext|>y", "<<|endoftext|>>", "<|endoftext|><|endoftext|>", "<|endoftext|", "|endoftext|>"]
    texts += ["<|fim_prefix|>a<|fim_middle|>b<|fim_suffix|>", "\n<|endofprompt|>\n", "<|begin_of_text|><|eot_id|>"]
    return texts


def known_answers() -> dict[str, list[tuple[str, list[int]]]]:
    """Returns the known answers of matisse's fixture, each universe's texts with the reference's ids.

    Returns:
        The answers by universe.
    """
    answers: dict[str, list[tuple[str, list[int]]]] = {}
    universe = None
    for line in (FIXTURES / "known_answers_reference.txt").read_text().splitlines():
        fields = line.split()
        if fields[0] == "universe":
            universe = fields[1]
            answers[universe] = []
        elif fields[0] == "encode":
            text = "" if fields[1] == "-" else bytes.fromhex(fields[1]).decode("utf-8")
            answers[universe].append((text, [int(field) for field in fields[2:]]))
    return answers


def vendor_ids() -> dict[str, dict[str, tuple[int, str]]]:
    """Returns the vendors' ids of matisse's vendor fixture, the count and digest of each corpus under each universe.

    Returns:
        The count and digest by universe and corpus file name.
    """
    ids: dict[str, dict[str, tuple[int, str]]] = {}
    universe = None
    for line in (FIXTURES / "vendors_reference.txt").read_text().splitlines():
        fields = line.split()
        if fields[0] == "universe":
            universe = fields[1]
            ids[universe] = {}
        elif fields[0] == "ids":
            ids[universe][fields[1]] = (int(fields[2]), fields[3])
    return ids


def expect_same(ours: object, theirs: object, text: str) -> None:
    """Holds every encoding call of matisse_tokenizer on a text to tiktoken's, and decode() of the ids to the text.

    Args:
        ours: matisse_tokenizer's encoding.
        theirs: tiktoken's encoding.
        text: The text.
    """
    ordinary = theirs.encode_ordinary(text)
    assert ours.encode_ordinary(text) == ordinary, repr(text[:80])
    assert ours.encode(text, disallowed_special=()) == ordinary, repr(text[:80])
    every = theirs.encode(text, allowed_special="all")
    assert ours.encode(text, allowed_special="all") == every, repr(text[:80])
    assert ours.decode_bytes(every) == text.encode("utf-8")


def test_the_known_answers_are_the_references_and_tiktoken_s(pair):
    ours, theirs = pair
    answers = known_answers()[UNIVERSES[ours.name]]
    assert len(answers) >= 15
    for text, ids in answers:
        assert ours.encode_ordinary(text) == ids, repr(text)
        assert theirs.encode_ordinary(text) == ids, repr(text)


def test_seeded_random_texts_encode_as_tiktoken_encodes_them(pair):
    ours, theirs = pair
    texts = random_texts(RANDOM_TEXTS)
    for text in texts:
        expect_same(ours, theirs, text)
    assert ours.encode_ordinary_batch(texts, num_threads=4) == theirs.encode_ordinary_batch(texts, num_threads=4)
    every = theirs.encode_batch(texts, allowed_special="all")
    assert ours.encode_batch(texts, allowed_special="all") == every
    arrays = ours.encode_batch_to_numpy(texts, allowed_special="all", num_threads=4)
    assert [array.tolist() for array in arrays] == every


def test_adversarial_texts_encode_as_tiktoken_encodes_them(pair):
    ours, theirs = pair
    for text in adversarial_texts():
        expect_same(ours, theirs, text)


def test_every_special_token_alone_and_between_letters_is_its_id_when_allowed_and_text_when_not(pair):
    ours, theirs = pair
    for spelling in sorted(theirs.special_tokens_set):
        for text in (spelling, f"a{spelling}b", f" {spelling}\n", spelling * 3):
            expect_same(ours, theirs, text)
            allowed = {spelling}
            assert ours.encode(text, allowed_special=allowed) == theirs.encode(text, allowed_special=allowed)


def test_the_numpy_call_and_the_parallel_calls_return_encode_s_ids(pair):
    ours, theirs = pair
    text = "".join(random_texts(400)) * 3
    expected = theirs.encode(text, disallowed_special=())
    array = ours.encode_to_numpy(text, disallowed_special=())
    assert array.dtype == numpy.uint32
    assert array.tolist() == expected
    assert theirs.encode_to_numpy(text, disallowed_special=()).tolist() == expected
    for chunks in (0, 1, 2, 3, 8, 64):
        assert ours.encode_parallel(text, chunks, disallowed_special=()) == expected
    assert ours.encode_parallel_to_numpy(text, 8, disallowed_special=()).tolist() == expected
    every = theirs.encode(text, allowed_special="all")
    assert ours.encode_parallel(text, 16, allowed_special="all") == every


CORPORA = [
    "twitter.json",
    "munch-python@15b2863c.txt",
    "munch-cpp@15b2863c.txt",
    "gutenberg.txt",
    "wiki-zh.txt",
    "wiki-ar.txt",
    "wiki-ru.txt",
    "wiki-hi.txt",
    "wiki-sv.txt",
    "prompts.txt",
]
"""The corpora of matisse's docs/benchmarks.md the tests encode, all but enwik8 and the three files of a gigabyte or
more."""


@pytest.mark.parametrize("corpus", CORPORA)
def test_the_corpora_encode_to_the_vendors_ids(pair, corpus):
    ours, theirs = pair
    data = corpus_path(corpus).read_bytes()
    documents = [document.decode("utf-8") for document in data.split(b"\0")]
    count, digest = vendor_ids()[UNIVERSES[ours.name]][corpus]

    if len(documents) == 1:
        ids = ours.encode_to_numpy(documents[0], disallowed_special=())
        assert ours.encode_parallel_to_numpy(documents[0], 12, disallowed_special=()).tolist() == ids.tolist()
    else:
        encoded = ours.encode_ordinary_batch(documents)
        ids = numpy.concatenate([numpy.asarray(each, dtype=numpy.uint32) for each in encoded])
    assert (len(ids), digest_of(ids)) == (count, digest)

    theirs_ids = numpy.concatenate(
        [theirs.encode_to_numpy(document, disallowed_special=()) for document in documents]
    )
    assert (len(theirs_ids), digest_of(theirs_ids)) == (count, digest)
