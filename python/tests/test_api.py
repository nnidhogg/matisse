"""tiktoken's interface as matisse_tokenizer gives it: the special-token rules and their refusals, the vocabulary's
figures, the single-token calls, decoding and its errors, the batch calls, lone surrogates, pickling, the constructor
from ranks, and the calls matisse_tokenizer refuses."""

from __future__ import annotations

import contextlib
import pickle
import shutil
import threading

import numpy
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
    names = ["gpt2", "r50k_base", "cl100k_base", "o200k_base", "o200k_harmony", "llama3"]
    assert matisse_tokenizer.list_encoding_names() == names
    with pytest.raises(ValueError, match="Unknown encoding p50k_base"):
        matisse_tokenizer.get_encoding("p50k_base")
    with pytest.raises(ValueError, match="load_encoding"):
        matisse_tokenizer.get_encoding("llama3")


def test_encode_with_unstable_is_refused(pair):
    ours, _ = pair
    with pytest.raises(NotImplementedError):
        ours.encode_with_unstable("hello fanta")


def test_a_batch_takes_any_iterable_of_texts_as_tiktoken_s_does(pair):
    ours, theirs = pair
    texts = ["hello world", "foo <|endoftext|>", ""]
    expected = theirs.encode_batch(texts, disallowed_special=())
    for made in (lambda: iter(texts), lambda: tuple(texts), lambda: (text for text in texts)):
        assert ours.encode_batch(made(), disallowed_special=()) == expected
        assert ours.encode_ordinary_batch(made()) == theirs.encode_ordinary_batch(made())
        arrays = ours.encode_batch_to_numpy(made(), disallowed_special=())
        assert [array.tolist() for array in arrays] == expected


def test_a_text_that_is_not_a_string_is_refused_with_tiktoken_s_type_error(pair):
    ours, theirs = pair
    for texts in (["a", 5], [b"a"], [None]):
        assert raised(lambda: ours.encode_ordinary_batch(texts)) == raised(lambda: theirs.encode_ordinary_batch(texts))
        mine = raised(lambda: ours.encode_batch(texts, disallowed_special=()))
        assert mine == raised(lambda: theirs.encode_batch(texts, disallowed_special=()))
    for text in (b"abc", None, 5):
        assert raised(lambda: ours.encode_ordinary(text)) == raised(lambda: theirs.encode_ordinary(text))
        # encode() searches the text for disallowed special tokens first, tiktoken's with the regex package.
        assert raised(lambda: ours.encode(text))[0] is raised(lambda: theirs.encode(text))[0] is TypeError


def test_allowed_special_is_a_set_or_all_and_anything_else_is_refused_as_tiktoken_refuses_it(pair):
    ours, theirs = pair
    spelling = sorted(theirs.special_tokens_set)[0]
    text = f"hi{spelling}"
    for allowed in (spelling, [spelling], (spelling,), {spelling: 1}):
        for disallowed in ("all", (), {"zzz"}):
            arguments = {"allowed_special": allowed, "disallowed_special": disallowed}
            assert outcome(lambda: ours.encode(text, **arguments)) == outcome(lambda: theirs.encode(text, **arguments))
            mine = outcome(lambda: ours.encode_to_numpy(text, **arguments))
            assert mine == outcome(lambda: theirs.encode_to_numpy(text, **arguments))
            mine = outcome(lambda: ours.encode_batch([text], **arguments))
            assert mine == outcome(lambda: theirs.encode_batch([text], **arguments))
            assert outcome(lambda: ours.encode_batch([], **arguments)) == outcome(
                lambda: theirs.encode_batch([], **arguments)
            )
    for allowed in (frozenset({spelling}), {spelling}, set()):
        assert outcome(lambda: ours.encode(text, allowed_special=allowed)) == outcome(
            lambda: theirs.encode(text, allowed_special=allowed)
        )


def test_num_threads_takes_what_tiktoken_s_pool_takes(pair):
    ours, theirs = pair
    texts = ["a b", "c d e", "f"]
    for count in (None, 2.0, 2.5, True, 1, 64):
        assert ours.encode_ordinary_batch(texts, num_threads=count) == theirs.encode_ordinary_batch(
            texts, num_threads=count
        )
        assert ours.encode_batch(texts, num_threads=count) == theirs.encode_batch(texts, num_threads=count)
    for count in ("2", 0, -1, 0.0):
        mine = raised(lambda: ours.encode_ordinary_batch(texts, num_threads=count))
        assert mine == raised(lambda: theirs.encode_ordinary_batch(texts, num_threads=count))
        mine = raised(lambda: ours.encode_batch(texts, num_threads=count))
        assert mine == raised(lambda: theirs.encode_batch(texts, num_threads=count))


def test_special_tokens_of_which_one_begins_another_or_two_share_an_id_are_refused(pairs):
    if "cl100k_base" not in pairs:
        pytest.skip("the vocabulary does not hold cl100k_base")
    _, theirs = pairs["cl100k_base"]
    ranks = theirs._mergeable_ranks
    begins = {"<|a|>": len(ranks), "<|a|>b": len(ranks) + 1, "x<|a": len(ranks) + 2}
    with pytest.raises(ValueError, match="begins"):
        matisse_tokenizer.Encoding("overlap", pat_str=theirs._pat_str, mergeable_ranks=ranks, special_tokens=begins)
    # tiktoken decodes an id two spellings share as one of them by the hash order of its table, whichever they are
    # listed in, so a shared id is refused outside o200k_harmony's table.
    for shared in ({"<|a|>": len(ranks), "<|b|>": len(ranks)}, {"<|b|>": len(ranks), "<|a|>": len(ranks)}):
        reference = tiktoken.Encoding("shared", pat_str=theirs._pat_str, mergeable_ranks=ranks, special_tokens=shared)
        assert reference.decode([len(ranks)]) == "<|b|>"
        with pytest.raises(ValueError, match="share the id"):
            matisse_tokenizer.Encoding("shared", pat_str=theirs._pat_str, mergeable_ranks=ranks, special_tokens=shared)
    # Overlapping without one beginning the other is taken, and its ids are tiktoken's.
    overlapping = {"<|a|>": len(ranks), "x<|a": len(ranks) + 1}
    mine = matisse_tokenizer.Encoding("ok", pat_str=theirs._pat_str, mergeable_ranks=ranks, special_tokens=overlapping)
    reference = tiktoken.Encoding("ok", pat_str=theirs._pat_str, mergeable_ranks=ranks, special_tokens=overlapping)
    for text in ("x<|a|>", "qx<|a|>b", "<|a|>x<|a"):
        for allowed in ("all", {"<|a|>"}, {"x<|a"}):
            arguments = {"allowed_special": allowed, "disallowed_special": ()}
            assert mine.encode(text, **arguments) == reference.encode(text, **arguments)


def test_the_package_s_version_is_its_metadata_s_and_the_user_agent_s():
    from importlib.metadata import version

    from matisse_tokenizer import load

    assert matisse_tokenizer.__version__ == version("matisse-tokenizer")
    assert load.USER_AGENT == f"matisse-tokenizer/{matisse_tokenizer.__version__}"


def test_allowed_special_holds_strings_alone_as_tiktoken_requires(pair):
    ours, theirs = pair
    spelling = sorted(theirs.special_tokens_set)[0]
    for allowed in ({1}, {None}, {spelling, 1}):
        for text in ("a", f"a{spelling}"):
            for disallowed in ("all", ()):
                arguments = {"allowed_special": allowed, "disallowed_special": disallowed}
                assert outcome(lambda: ours.encode(text, **arguments)) == outcome(
                    lambda: theirs.encode(text, **arguments)
                )
                assert raised_or_ids(lambda: ours.encode(text, **arguments)) == raised_or_ids(
                    lambda: theirs.encode(text, **arguments)
                )
                assert raised_or_ids(lambda: ours.encode_batch([text], **arguments)) == raised_or_ids(
                    lambda: theirs.encode_batch([text], **arguments)
                )


def raised_or_ids(call: object) -> object:
    """Returns what a call returns, or the type and message of the exception it raises.

    Args:
        call: The call, taking no argument.

    Returns:
        The result, or the exception's type and message.
    """
    try:
        return call()
    except Exception as error:
        return type(error), str(error)


def test_decoding_takes_a_sequence_as_tiktoken_requires(pair):
    ours, theirs = pair
    for tokens in ({1, 2}, frozenset({1}), {1: 2}, iter([1]), (token for token in [1]), "ab", [1, 2], (1, 2), b"ab",
                   range(3), numpy.array([1, 2], dtype=numpy.uint32)):
        rebuilt = (lambda: (token for token in [1])) if hasattr(tokens, "__next__") else (lambda: tokens)
        assert raised_or_ids(lambda: ours.decode(rebuilt())) == raised_or_ids(lambda: theirs.decode(rebuilt()))
        assert raised_or_ids(lambda: ours.decode_bytes(rebuilt())) == raised_or_ids(
            lambda: theirs.decode_bytes(rebuilt())
        )
    assert raised_or_ids(lambda: ours.decode_batch([{1}])) == raised_or_ids(lambda: theirs.decode_batch([{1}]))
    assert raised_or_ids(lambda: ours.decode_bytes_batch([{1}])) == raised_or_ids(
        lambda: theirs.decode_bytes_batch([{1}])
    )


def test_a_single_token_is_a_string_or_bytes_as_tiktoken_requires(pair):
    ours, theirs = pair
    for value in (None, 5, 1.0, memoryview(b"a"), bytearray(b"a"), "\ud800", b"a", "a"):
        assert raised_or_ids(lambda: ours.encode_single_token(value)) == raised_or_ids(
            lambda: theirs.encode_single_token(value)
        )


def test_the_constructor_refuses_special_tokens_as_tiktoken_s_conversion_refuses_them(pairs):
    if "gpt2" not in pairs:
        pytest.skip("the vocabulary does not hold gpt2")
    _, theirs = pairs["gpt2"]
    ranks = theirs._mergeable_ranks
    for specials in ({"<|x|>": 2**32}, {"<|x|>": -1}, {"<|x|>": 2**64}, {"<|x|>": 1.5}, {"\ud800": 50300},
                     {5: 50300}):
        arguments = {"pat_str": theirs._pat_str, "mergeable_ranks": ranks, "special_tokens": specials}
        assert raised(lambda: matisse_tokenizer.Encoding("x", **arguments)) == raised(
            lambda: tiktoken.Encoding("x", **arguments)
        )


def test_the_text_is_read_before_the_allowed_special_tokens_as_tiktoken_s_core_reads_them(pair):
    ours, theirs = pair
    spelling = sorted(theirs.special_tokens_set)[0]
    for text in ("a\ud800", None, 5, "a"):
        for call in ("encode", "encode_to_numpy"):
            arguments = {"allowed_special": [spelling], "disallowed_special": ()}
            assert raised(lambda: getattr(ours, call)(text, **arguments)) == raised(
                lambda: getattr(theirs, call)(text, **arguments)
            )


def test_every_array_is_read_only_as_tiktoken_s_is(pair):
    ours, theirs = pair
    spelling = sorted(theirs.special_tokens_set)[0]
    text = f"a{spelling}b"
    assert theirs.encode_to_numpy(text, allowed_special="all").flags.writeable is False
    assert ours.encode_to_numpy(text, allowed_special="all").flags.writeable is False
    assert ours.encode_to_numpy("ab").flags.writeable is False
    assert ours.encode_parallel_to_numpy(text, 2, allowed_special="all").flags.writeable is False
    assert all(not array.flags.writeable for array in ours.encode_batch_to_numpy([text, "ab"], allowed_special="all"))


def test_an_unbounded_pool_runs_as_tiktoken_s_does(pair):
    ours, theirs = pair
    texts = ["a b", "c"]
    for count in (float("inf"), 10**30):
        assert ours.encode_ordinary_batch(texts, num_threads=count) == theirs.encode_ordinary_batch(
            texts, num_threads=count
        )
        assert ours.encode_batch(texts, num_threads=count) == theirs.encode_batch(texts, num_threads=count)


def test_o200k_harmony_is_tiktoken_s_its_shared_id_decoded_as_tiktoken_decodes_it(vocabulary):
    if "o200k_base" not in vocabulary:
        pytest.skip("the vocabulary does not hold o200k_base")
    from test_ids import random_texts

    ours = matisse_tokenizer.get_encoding("o200k_harmony")
    theirs = tiktoken.get_encoding("o200k_harmony")
    assert ours._report_shipped
    figures = ("n_vocab", "max_token_value", "eot_token")
    assert [getattr(ours, figure) for figure in figures] == [getattr(theirs, figure) for figure in figures]
    assert ours.special_tokens_set == theirs.special_tokens_set
    specials = sorted(theirs.special_tokens_set)
    texts = [f"{text}{specials[index % len(specials)]}{text}" for index, text in enumerate(random_texts(400))]
    texts += ["<|endofprompt|><|reserved_200018|>", "<|start|>user<|message|>hi<|end|><|start|>assistant"]
    for text in texts:
        for arguments in ({"allowed_special": "all"}, {"disallowed_special": ()}, {"allowed_special": {specials[0]}}):
            assert outcome(lambda: ours.encode(text, **arguments)) == outcome(lambda: theirs.encode(text, **arguments))
    for token in (200018, 199998, 200007, 201087):
        assert ours.decode_single_token_bytes(token) == theirs.decode_single_token_bytes(token)
        assert ours.decode([token]) == theirs.decode([token])
    for spelling in ("<|endofprompt|>", "<|reserved_200018|>", "<|call|>"):
        assert ours.encode_single_token(spelling) == theirs.encode_single_token(spelling)
    assert pickle.loads(pickle.dumps(ours)) is not None


def test_tiktoken_s_own_o200k_harmony_table_builds_through_the_constructor_in_any_order(vocabulary):
    if "o200k_base" not in vocabulary:
        pytest.skip("the vocabulary does not hold o200k_base")
    theirs = tiktoken.get_encoding("o200k_harmony")
    for specials in (theirs._special_tokens, dict(reversed(theirs._special_tokens.items()))):
        arguments = {"pat_str": theirs._pat_str, "mergeable_ranks": theirs._mergeable_ranks, "special_tokens": specials}
        ours = matisse_tokenizer.Encoding("o200k_harmony", **arguments)
        assert ours.decode([200018]) == theirs.decode([200018])
        text = "<|start|>user<|message|>x<|reserved_200018|>y<|endofprompt|><|end|>"
        assert ours.encode(text, allowed_special="all") == theirs.encode(text, allowed_special="all")


def test_a_stream_returns_final_ids_that_are_a_prefix_of_tiktoken_s_and_all_of_them_at_finish(pair):
    import random

    from test_ids import adversarial_texts, random_texts

    ours, theirs = pair
    generator = random.Random(7)
    texts = random_texts(300) + [text for text in adversarial_texts() if len(text) < 20000]
    stream = ours.stream()
    for text in texts:
        expected = theirs.encode_ordinary(text)
        data = text.encode("utf-8")
        returned: list[int] = []
        start = 0
        while start < len(data):
            end = min(len(data), start + generator.randrange(1, 70))
            part = data[start:end]
            # Half the parts are pushed as text where they hold whole code points, the rest as bytes.
            if generator.random() < 0.5:
                with contextlib.suppress(UnicodeDecodeError):
                    part = part.decode("utf-8")
            returned += stream.push(part)
            assert expected[: len(returned)] == returned
            start = end
        returned += stream.finish()
        assert returned == expected
    assert stream.push("hello ") + stream.push("world") + stream.finish() == theirs.encode_ordinary("hello world")


def test_a_stream_refuses_bytes_that_are_not_utf_8_and_then_every_further_call(pair):
    ours, _ = pair
    stream = ours.stream()
    with pytest.raises(ValueError):
        stream.push(b"ab\xffcd")
        stream.finish()
    with pytest.raises(ValueError, match="refused"):
        stream.push("more")
    unfinished = ours.stream()
    unfinished.push(b"a\xc3")
    with pytest.raises(ValueError):
        unfinished.finish()
    with pytest.raises(TypeError):
        ours.stream().push(5)


def test_streams_on_threads_each_return_tiktoken_s_ids(pair):
    ours, theirs = pair
    texts = [f"thread {index}: " + "streamed text, piece by piece. " * (index + 3) for index in range(8)]
    results: dict[int, list[int]] = {}

    def work(index: int) -> None:
        stream = ours.stream()
        ids: list[int] = []
        for start in range(0, len(texts[index]), 5):
            ids += stream.push(texts[index][start : start + 5])
        results[index] = ids + stream.finish()

    threads = [threading.Thread(target=work, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [results[index] for index in range(8)] == [theirs.encode_ordinary(text) for text in texts]


def test_r50k_base_is_gpt2_s_rank_file_pattern_and_special_token_under_tiktoken_s_name(pairs):
    if "gpt2" not in pairs:
        pytest.skip("the vocabulary does not hold gpt2")
    from test_ids import adversarial_texts, random_texts

    ours = matisse_tokenizer.get_encoding("r50k_base")
    theirs = tiktoken.get_encoding("r50k_base")
    assert ours.name == "r50k_base"
    assert ours._report_shipped
    assert (ours.n_vocab, ours.special_tokens_set) == (theirs.n_vocab, theirs.special_tokens_set)
    for text in random_texts(300) + adversarial_texts()[:80]:
        assert ours.encode(text, allowed_special="all") == theirs.encode(text, allowed_special="all")
    assert pickle.loads(pickle.dumps(ours)).encode("pickled") == theirs.encode("pickled")


def test_the_model_table_is_tiktoken_s_restricted_to_the_encodings_served(vocabulary):
    import tiktoken.model

    from matisse_tokenizer import model

    served = set(matisse_tokenizer.list_encoding_names())
    assert model.MODEL_TO_ENCODING == {
        name: encoding for name, encoding in tiktoken.model.MODEL_TO_ENCODING.items() if encoding in served
    }
    assert list(model.MODEL_PREFIX_TO_ENCODING.items()) == [
        (prefix, encoding) for prefix, encoding in tiktoken.model.MODEL_PREFIX_TO_ENCODING.items() if encoding in served
    ]
    names = list(tiktoken.model.MODEL_TO_ENCODING)
    names += ["gpt-4o-2024-05-13", "gpt-4-0314", "gpt-oss-120b", "ft:gpt-4o:org", "o1-mini", "gpt-5-nano", "nope"]
    for name in names:
        try:
            expected = tiktoken.encoding_name_for_model(name)
        except KeyError:
            assert raised(lambda: matisse_tokenizer.encoding_name_for_model(name)) == raised(
                lambda: tiktoken.encoding_name_for_model(name)
            )
            continue
        if expected in served:
            assert matisse_tokenizer.encoding_name_for_model(name) == expected
            continue
        with pytest.raises(KeyError, match=expected):
            matisse_tokenizer.encoding_name_for_model(name)
    if "cl100k_base" in vocabulary:
        assert matisse_tokenizer.encoding_for_model("gpt-4") is matisse_tokenizer.get_encoding("cl100k_base")
