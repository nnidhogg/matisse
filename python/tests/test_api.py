"""tiktoken's interface as matisse_tokenizer gives it: the special-token rules and their refusals, the vocabulary's
figures, the single-token calls, decoding and its errors, the batch calls, lone surrogates, pickling, the constructor
from ranks, and the calls matisse_tokenizer refuses."""

from __future__ import annotations

import pickle
import shutil
import threading

import pytest
import tiktoken

import matisse_tokenizer


def raised(call: object) -> tuple[type, str]:
    """Returns the type and message of the exception a call raises.

    Args:
        call: The call, taking no argument.

    Returns:
        The exception's type and its message.
    """
    with pytest.raises(Exception) as caught:
        call()
    return type(caught.value), str(caught.value)


def outcome(call: object) -> object:
    """Returns what a call returns as a list, or the type of the exception it raises.

    Args:
        call: The call, taking no argument.

    Returns:
        The list of what it returns, or the exception's type.
    """
    try:
        return list(call())
    except Exception as error:
        return type(error)


def test_a_disallowed_special_token_is_refused_with_tiktoken_s_error_and_message(pair):
    ours, theirs = pair
    spelling = sorted(theirs.special_tokens_set)[0]
    text = f"hello {spelling} world"
    assert raised(lambda: ours.encode(text)) == raised(lambda: theirs.encode(text))
    assert raised(lambda: ours.encode_to_numpy(text)) == raised(lambda: theirs.encode_to_numpy(text))
    assert raised(lambda: ours.encode_batch(["a", text])) == raised(lambda: theirs.encode_batch(["a", text]))
    with pytest.raises(ValueError):
        ours.encode_parallel(text, 4)


def test_the_special_token_arguments_combine_as_tiktoken_s_do(pair):
    ours, theirs = pair
    spellings = sorted(theirs.special_tokens_set)
    first = spellings[0]
    text = "".join(f"x{spelling} y" for spelling in spellings[:5])
    combinations = [
        {"allowed_special": "all"},
        {"disallowed_special": ()},
        {"allowed_special": {first}, "disallowed_special": ()},
        {"allowed_special": {first, "not a special token"}, "disallowed_special": ()},
        {"allowed_special": set(spellings[:5])},
        {"disallowed_special": {"x"}},
    ]
    for arguments in combinations:
        expected = None
        try:
            expected = theirs.encode(text, **arguments)
        except ValueError:
            assert raised(lambda: ours.encode(text, **arguments)) == raised(lambda: theirs.encode(text, **arguments))
            continue
        assert ours.encode(text, **arguments) == expected, arguments
        assert ours.encode_to_numpy(text, **arguments).tolist() == expected, arguments
        assert ours.encode_batch([text, text], **arguments) == [expected, expected], arguments
        arrays = ours.encode_batch_to_numpy([text, text], **arguments)
        assert [array.tolist() for array in arrays] == [expected, expected], arguments
        assert ours.encode_parallel(text, 3, **arguments) == expected, arguments
        assert ours.encode_parallel_to_numpy(text, 3, **arguments).tolist() == expected, arguments


def test_the_vocabulary_s_figures_are_tiktoken_s(pair):
    ours, theirs = pair
    assert ours.n_vocab == theirs.n_vocab
    assert ours.max_token_value == theirs.max_token_value
    assert ours.special_tokens_set == theirs.special_tokens_set
    assert repr(ours) == repr(theirs)
    assert ours.name == theirs.name
    if "<|endoftext|>" in theirs.special_tokens_set:
        assert ours.eot_token == theirs.eot_token
    else:
        assert raised(lambda: ours.eot_token)[0] is KeyError
    for token in (0, 255, theirs.max_token_value, theirs.max_token_value - 1):
        assert ours.is_special_token(token) == theirs.is_special_token(token)


def test_single_tokens_encode_and_decode_as_tiktoken_s_and_refuse_as_they_do(pair):
    ours, theirs = pair
    for text in ("hello", " the", b"\xff", "<|endoftext|>", "<|eot_id|>"):
        try:
            expected = theirs.encode_single_token(text)
        except KeyError:
            assert raised(lambda: ours.encode_single_token(text)) == raised(lambda: theirs.encode_single_token(text))
            continue
        assert ours.encode_single_token(text) == expected
    for token in (0, 300, theirs.max_token_value, theirs.max_token_value + 1, 10**7):
        try:
            expected = theirs.decode_single_token_bytes(token)
        except KeyError:
            mine = raised(lambda: ours.decode_single_token_bytes(token))
            assert mine == raised(lambda: theirs.decode_single_token_bytes(token))
            continue
        assert ours.decode_single_token_bytes(token) == expected
    assert ours.token_byte_values() == theirs.token_byte_values()
    assert ours._mergeable_ranks == theirs._mergeable_ranks


def test_decoding_is_tiktoken_s_and_refuses_an_unknown_or_negative_id_with_tiktoken_s_error(pair):
    ours, theirs = pair
    text = "Hello, wörld! 日本 \U0001f600 <|endoftext|>"
    ids = theirs.encode(text, disallowed_special=())
    assert ours.decode(ids) == theirs.decode(ids) == text
    assert ours.decode_bytes(ids) == theirs.decode_bytes(ids)
    assert ours.decode_tokens_bytes(ids) == theirs.decode_tokens_bytes(ids)
    assert ours.decode_with_offsets(ids) == theirs.decode_with_offsets(ids)
    cut = ids[: len(ids) // 2] + [ours.encode_single_token(b"\xe6")]
    assert ours.decode(cut) == theirs.decode(cut)
    assert ours.decode(cut, errors="ignore") == theirs.decode(cut, errors="ignore")
    batch = [ids, ids[:3], []]
    assert ours.decode_batch(batch) == theirs.decode_batch(batch)
    assert ours.decode_bytes_batch(batch) == theirs.decode_bytes_batch(batch)
    unknown = theirs.max_token_value + 1
    assert raised(lambda: ours.decode([unknown]))[0] is raised(lambda: theirs.decode([unknown]))[0] is KeyError
    assert raised(lambda: ours.decode([-1]))[0] is raised(lambda: theirs.decode([-1]))[0] is OverflowError
    assert raised(lambda: ours.decode([2**40]))[0] is raised(lambda: theirs.decode([2**40]))[0] is OverflowError


def test_a_lone_surrogate_is_replaced_as_tiktoken_replaces_it_and_refused_where_tiktoken_refuses_it(pair):
    ours, theirs = pair
    for text in ("a\ud800b", "\udfff", "x😀y", "\ud800" * 10):
        assert ours.encode(text) == theirs.encode(text)
        assert ours.encode_ordinary(text) == theirs.encode_ordinary(text)
        assert ours.encode_batch([text, "ok"]) == theirs.encode_batch([text, "ok"])
        assert ours.encode_ordinary_batch([text]) == theirs.encode_ordinary_batch([text])
        assert ours.encode_parallel(text, 2) == theirs.encode(text)
        assert outcome(lambda: ours.encode_to_numpy(text)) == outcome(lambda: theirs.encode_to_numpy(text))


def test_a_batch_refuses_no_thread_as_tiktoken_s_pool_does(pair):
    ours, theirs = pair
    assert raised(lambda: ours.encode_batch(["a"], num_threads=0)) == raised(
        lambda: theirs.encode_batch(["a"], num_threads=0)
    )
    assert ours.encode_batch([]) == []
    texts = ["a", "", "b c"]
    assert ours.encode_ordinary_batch(texts, num_threads=64) == theirs.encode_ordinary_batch(texts)


def test_threads_encoding_at_once_each_get_encode_s_ids(pair):
    ours, theirs = pair
    texts = [f"thread {index}: " + "the quick brown fox " * (index + 1) for index in range(8)]
    expected = [theirs.encode_ordinary(text) for text in texts]
    results: dict[int, list[list[int]]] = {}

    def work(index: int) -> None:
        results[index] = [ours.encode_ordinary(text) for text in texts for _ in range(20)]

    threads = [threading.Thread(target=work, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for index in range(8):
        assert results[index] == [ids for ids in expected for _ in range(20)]


def test_an_encoding_pickles_by_its_name_or_its_path_and_back(pair, vocabulary):
    ours, _ = pair
    copied = pickle.loads(pickle.dumps(ours))
    assert copied.encode("pickled text") == ours.encode("pickled text")
    if ours.name != "llama3":
        assert pickle.dumps(ours) == pickle.dumps(matisse_tokenizer.get_encoding(ours.name))


def test_the_constructor_from_ranks_takes_a_known_pattern_and_extends_its_special_tokens(pairs):
    if "cl100k_base" not in pairs:
        pytest.skip("the vocabulary does not hold cl100k_base")
    ours, theirs = pairs["cl100k_base"]
    specials = {**theirs._special_tokens, "<|im_start|>": 100264, "<|im_end|>": 100265}
    arguments = {"pat_str": theirs._pat_str, "mergeable_ranks": theirs._mergeable_ranks, "special_tokens": specials}
    mine = matisse_tokenizer.Encoding("cl100k_im", **arguments)
    reference = tiktoken.Encoding("cl100k_im", **arguments)
    text = "<|im_start|>user\nhello<|im_end|>"
    assert mine.encode(text, allowed_special="all") == reference.encode(text, allowed_special="all")
    assert mine._report_shipped
    assert pickle.loads(pickle.dumps(mine)).encode(text, allowed_special="all") == mine.encode(
        text, allowed_special="all"
    )
    with pytest.raises(ValueError, match="no description in matisse"):
        matisse_tokenizer.Encoding("other", pat_str=r"\w+", mergeable_ranks={b"a": 0}, special_tokens={})


def test_the_constructor_s_single_tokens_are_its_ranks_whatever_the_caller_changes_in_them_after(pairs):
    if "gpt2" not in pairs:
        pytest.skip("the vocabulary does not hold gpt2")
    _, theirs = pairs["gpt2"]
    ranks = dict(theirs._mergeable_ranks)
    arguments = {"pat_str": theirs._pat_str, "mergeable_ranks": ranks, "special_tokens": theirs._special_tokens}
    mine = matisse_tokenizer.Encoding("x", **arguments)
    reference = tiktoken.Encoding("x", **arguments)
    ranks[b"a"], ranks[b"b"] = ranks[b"b"], ranks[b"a"]
    for spelling in ("a", b"b", " the"):
        assert mine.encode_single_token(spelling) == reference.encode_single_token(spelling)
    assert mine.encode_ordinary("a b") == reference.encode_ordinary("a b")
    assert mine.token_byte_values() == reference.token_byte_values()
    assert mine._mergeable_ranks == theirs._mergeable_ranks
    assert pickle.loads(pickle.dumps(mine)).encode_single_token("a") == reference.encode_single_token("a")


def test_a_rank_file_that_is_not_the_pinned_one_has_its_report_decided(vocabulary, tmp_path, pairs):
    if "gpt2" not in pairs:
        pytest.skip("the vocabulary does not hold gpt2")
    ours, theirs = pairs["gpt2"]
    # An empty line more, which read_tiktoken() skips, makes another file of the same table.
    other = tmp_path / "r50k_base.tiktoken"
    shutil.copyfile(vocabulary["gpt2"], other)
    with open(other, "ab") as file:
        file.write(b"\n")
    decided = matisse_tokenizer.load_encoding("gpt2", str(other))
    assert not decided._report_shipped
    assert ours._report_shipped
    text = "A table matisse has not seen, under GPT-2's pattern."
    assert decided.encode_parallel(text * 50, 4) == theirs.encode(text * 50)


def test_the_names_and_the_refusals_of_the_registry():
    assert matisse_tokenizer.list_encoding_names() == ["gpt2", "cl100k_base", "o200k_base", "llama3"]
    with pytest.raises(ValueError, match="Unknown encoding p50k_base"):
        matisse_tokenizer.get_encoding("p50k_base")
    with pytest.raises(ValueError, match="load_encoding"):
        matisse_tokenizer.get_encoding("llama3")
    with pytest.raises(NotImplementedError):
        matisse_tokenizer.encoding_for_model("gpt-4o")
    with pytest.raises(NotImplementedError):
        matisse_tokenizer.encoding_name_for_model("gpt-4o")


def test_encode_with_unstable_is_refused(pair):
    ours, _ = pair
    with pytest.raises(NotImplementedError):
        ours.encode_with_unstable("hello fanta")
