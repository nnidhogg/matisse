"""load_tokenizer_json() held to Hugging Face tokenizers' own encoding of the same file: the pinned tokenizer.json files
of GPT-2, Xenova's conversions of cl100k and o200k and Llama 3's, on matisse's known answers, seeded random texts,
adversarial texts, special tokens and the corpora of the test strategy, and its refusals, which are matisse's loader's."""

from __future__ import annotations

import json
import pickle

import numpy
import pytest

import matisse_tokenizer
from conftest import corpus_path
from test_ids import adversarial_texts, digest_of, known_answers, random_texts, vendor_ids

tokenizers = pytest.importorskip("tokenizers")

FILES = {
    "gpt2-tokenizer.json": ("gpt2", "gpt2"),
    "cl100k-tokenizer.json": ("llama3", "cl100k"),
    "o200k-tokenizer.json": ("o200k_base", "o200k"),
    "llama3-tokenizer.json": ("llama3", "llama3"),
}
"""Every pinned tokenizer.json with the encoding it reads as and the universe of matisse's fixtures its ids are of:
Xenova's conversion of cl100k carries Llama 3's pattern and cl100k's vocabulary."""

RANDOM_TEXTS = 1000
"""The random texts each file is held to Hugging Face tokenizers on."""

STRATEGY_CORPORA = ["twitter.json", "munch-python@15b2863c.txt", "munch-cpp@15b2863c.txt"]
"""The corpora of the test strategy, on which the vendor fixture's ids are Hugging Face's."""

CL100K_PATTERN_JSON = json.dumps(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|"""
    r"""\s+(?!\S)|\s"""
)
"""cl100k_base's published pattern as a JSON string."""


@pytest.fixture(params=list(FILES))
def read(request: pytest.FixtureRequest, tokenizer_jsons):
    """Returns one pinned tokenizer.json read by both packages, each test run once per file the vocabulary holds.

    Returns:
        The file's name, matisse_tokenizer's encoding, and Hugging Face's tokenizer.
    """
    if request.param not in tokenizer_jsons:
        pytest.skip(f"the vocabulary does not hold {request.param}")
    path = tokenizer_jsons[request.param]
    return request.param, matisse_tokenizer.load_tokenizer_json(path), tokenizers.Tokenizer.from_file(str(path))


def hugging_face_ids(tokenizer, text: str, specials: bool) -> list[int]:
    """Returns Hugging Face tokenizers' ids of a text, with add_special_tokens false.

    Args:
        tokenizer: The tokenizer.
        text: The text.
        specials: Whether a special token's spelling is cut out as the token, or read as text.

    Returns:
        The ids.
    """
    tokenizer.encode_special_tokens = not specials
    return tokenizer.encode(text, add_special_tokens=False).ids


def written(directory, pre_tokenizer: str, normalizer: str = "null"):
    """Writes a tokenizer.json of the 256 single bytes under GPT-2's byte map and one special token.

    Args:
        directory: The directory.
        pre_tokenizer: The pre-tokenizer, as JSON text.
        normalizer: The normalizer, as JSON text.

    Returns:
        The file's path.
    """
    printable = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    others = [byte for byte in range(256) if byte not in printable]
    characters = {byte: chr(byte) for byte in printable} | {byte: chr(0x100 + n) for n, byte in enumerate(others)}
    vocab = {characters[byte]: byte for byte in range(256)}
    document = (
        f'{{"normalizer": {normalizer}, "pre_tokenizer": {pre_tokenizer}, "added_tokens": '
        f'[{{"id": 256, "content": "<|endoftext|>", "special": true}}], '
        f'"model": {{"type": "BPE", "vocab": {json.dumps(vocab)}, "merges": []}}}}'
    )
    path = directory / "tokenizer.json"
    path.write_text(document, encoding="utf-8")
    return path


def test_each_file_reads_under_the_pattern_it_carries_with_its_added_special_tokens(read):
    file, ours, theirs = read
    name, _ = FILES[file]
    added = {token.content for token in theirs.get_added_tokens_decoder().values() if token.special}

    assert ours.name == name
    assert ours.special_tokens_set == added
    assert ours.max_token_value == max(theirs.get_vocab(with_added_tokens=True).values())
    assert ours._report_shipped


def test_the_known_answers_random_and_adversarial_texts_encode_as_hugging_face_encodes_them(read):
    _, ours, theirs = read
    texts = [text for answers in known_answers().values() for text, _ in answers]
    texts += random_texts(RANDOM_TEXTS) + adversarial_texts()

    for text in texts:
        assert ours.encode_ordinary(text) == hugging_face_ids(theirs, text, specials=False), text[:80]


def test_every_special_token_is_cut_out_as_hugging_face_cuts_it_when_allowed(read):
    _, ours, theirs = read
    specials = sorted(ours.special_tokens_set)[:32]

    for special in specials:
        for text in (special, f"a{special}b", f" {special}\n", special * 2, f"{special} {specials[0]}x"):
            assert ours.encode(text, allowed_special="all") == hugging_face_ids(theirs, text, specials=True), text
            assert ours.encode_ordinary(text) == hugging_face_ids(theirs, text, specials=False), text


@pytest.mark.parametrize("corpus", STRATEGY_CORPORA)
def test_the_corpora_encode_to_hugging_face_s_ids_and_the_vendors(read, corpus):
    file, ours, theirs = read
    _, universe = FILES[file]
    text = corpus_path(corpus).read_text(encoding="utf-8")

    ids = ours.encode_to_numpy(text, disallowed_special=())

    assert (len(ids), digest_of(ids)) == vendor_ids()[universe][corpus]
    assert ids.tolist() == hugging_face_ids(theirs, text, specials=False)
    assert ours.encode_parallel_to_numpy(text, 12, disallowed_special=()).tolist() == ids.tolist()


def test_an_encoding_read_from_a_tokenizer_json_pickles_by_its_path(read):
    _, ours, _ = read
    text = "Hello, world! 123 <|endoftext|>"

    restored = pickle.loads(pickle.dumps(ours))

    assert restored.name == ours.name
    assert restored.encode_ordinary(text) == ours.encode_ordinary(text)


def test_a_file_that_is_not_a_pinned_one_has_its_pattern_s_report_decided(tmp_path, tokenizer_jsons):
    if "gpt2-tokenizer.json" not in tokenizer_jsons:
        pytest.skip("the vocabulary does not hold gpt2-tokenizer.json")
    pinned = tokenizer_jsons["gpt2-tokenizer.json"]
    copy = tmp_path / "tokenizer.json"
    copy.write_bytes(pinned.read_bytes() + b"\n")
    text = "a text whose ids the two reports give alike, 'til 2026"

    ours = matisse_tokenizer.load_tokenizer_json(copy)

    assert not ours._report_shipped
    assert ours.encode_ordinary(text) == matisse_tokenizer.load_tokenizer_json(pinned).encode_ordinary(text)


def test_the_refusals_are_matisse_s_loader_s(tmp_path):
    byte_level = '{"type": "ByteLevel", "add_prefix_space": false}'
    split = (
        '{{"type": "Sequence", "pretokenizers": [{{"type": "Split", "pattern": {{"Regex": {}}}, "behavior": "Isolated",'
        ' "invert": false}}, {{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}}]}}'
    )

    normalized = written(tmp_path, byte_level, normalizer='{"type": "NFC"}')
    expected = f'read_tokenizer_json: normalizer of {normalized} is a "NFC" object, where the reader takes null'
    with pytest.raises(ValueError) as refusal:
        matisse_tokenizer.load_tokenizer_json(normalized)
    assert str(refusal.value) == expected

    unknown = written(tmp_path, split.format(json.dumps(r"\p{L}+")))
    with pytest.raises(ValueError, match="and no tokenizer matisse describes has that pattern"):
        matisse_tokenizer.load_tokenizer_json(unknown)

    cl100k = written(tmp_path, split.format(CL100K_PATTERN_JSON))
    with pytest.raises(ValueError, match="Oniguruma, Hugging Face's engine, reads as the interval repeated"):
        matisse_tokenizer.load_tokenizer_json(cl100k)

    with pytest.raises(RuntimeError, match="cannot read"):
        matisse_tokenizer.load_tokenizer_json(tmp_path / "missing.json")


def test_a_file_of_the_bytes_alone_reads_under_gpt2_s_pattern_and_encodes_byte_by_byte(tmp_path):
    ours = matisse_tokenizer.load_tokenizer_json(written(tmp_path, '{"type": "ByteLevel", "add_prefix_space": false}'))
    text = "naïve 'til"

    assert ours.name == "gpt2"
    assert ours.encode_ordinary(text) == list(text.encode("utf-8"))
    assert ours.encode(text + "<|endoftext|>", allowed_special="all")[-1] == 256
    assert numpy.array_equal(ours.encode_to_numpy(text), numpy.frombuffer(text.encode("utf-8"), dtype=numpy.uint8))
