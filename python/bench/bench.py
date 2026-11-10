"""Times matisse-tokenizer beside tiktoken on matisse's corpora, in the row format of matisse_bench, so that the three,
matisse's C++ encoder, its Python package and tiktoken, set side by side compare like for like on one machine.

    python -I bench.py single VOCAB_DIR TOKENIZERS PASSES FILE          one long text, one call
    python -I bench.py prompts VOCAB_DIR TOKENIZERS PASSES FILE         NUL-separated prompts, one call each
    python -I bench.py parallel VOCAB_DIR TOKENIZERS PASSES FILE [N...] encode_parallel() at each chunk count
    python -I bench.py batch VOCAB_DIR TOKENIZERS PASSES FILE [N...]    NUL-separated documents at each thread count
    python -I bench.py threads VOCAB_DIR TOKENIZERS PASSES FILE [N...]  the prompts over a pool of Python threads
    python -I bench.py load VOCAB_DIR TOKENIZERS RUNS                   a fresh process building an encoding
    python -I bench.py short FILE OUTPUT BYTES                          the prompts cut to at most BYTES bytes each

TOKENIZERS is a comma-separated list of gpt2, cl100k, o200k and llama3, matisse_bench's names. VOCAB_DIR holds the
pinned files as matisse's tests read them; they are served to both packages through one cache directory named by
TIKTOKEN_CACHE_DIR, GPT-2's rank file written from encoder.json and vocab.bpe by tiktoken's own conversion, and Llama
3's encodings are read from llama3.tiktoken, tiktoken's built as Meta's tokenizer.py builds it. Every row times one
call after a first pass that gives the ids every timed pass must reproduce, and prints its ids line, the count and the
SHA-256 of the ids as little-endian 32-bit words, which matisse_bench's ids lines are compared by.

The rows of single separate three costs. "encode fresh" times encode() on a string Python has not yet been asked the
UTF-8 of, so the call converts it, and "encode cached" the same string again, whose UTF-8 CPython keeps beside it; the
two differ by the conversion, which an ASCII string does not need. "numpy cached" times encode_to_numpy(), which builds
no Python integer per id, so "encode cached" less it is the cost of the list. Nothing here is sent anywhere.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

NAMES = {"gpt2": "gpt2", "cl100k": "cl100k_base", "o200k": "o200k_base", "llama3": "llama3"}
"""Each tokenizer's encoding name, by matisse_bench's name."""

URLS = {
    "r50k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "gpt2-merges.txt": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
    "gpt2-vocab.json": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
}
"""The URL tiktoken reads each file from, whose SHA-1 names its cache entry."""

LLAMA3_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"""
    r"""\s+(?!\S)|\s+"""
)
"""Llama 3's pattern, as meta-llama/llama3's llama/tokenizer.py spells it."""


def serve(vocab: str) -> str:
    """Serves the pinned files to both packages through one cache directory, set as TIKTOKEN_CACHE_DIR.

    Args:
        vocab: The directory of the pinned files.

    Returns:
        The directory GPT-2's rank file is written to, beside the cache.
    """
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks

    root = tempfile.mkdtemp(prefix="matisse-bench-")
    cache = os.path.join(root, "cache")
    os.makedirs(cache)
    ranks = data_gym_to_mergeable_bpe_ranks(
        os.path.join(vocab, "gpt2-merges.txt"), os.path.join(vocab, "gpt2-vocab.json")
    )
    with open(os.path.join(root, "r50k_base.tiktoken"), "wb") as file:
        for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
            file.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")
    for file, url in URLS.items():
        source = os.path.join(root if file == "r50k_base.tiktoken" else vocab, file)
        shutil.copyfile(source, os.path.join(cache, hashlib.sha1(url.encode()).hexdigest()))
    os.environ["TIKTOKEN_CACHE_DIR"] = cache
    return root


def encodings(vocab: str, tokenizer: str) -> tuple[object, object]:
    """Returns a tokenizer's encoding under matisse-tokenizer and under tiktoken.

    Args:
        vocab: The directory of the pinned files.
        tokenizer: matisse_bench's name of the tokenizer.

    Returns:
        matisse-tokenizer's encoding and tiktoken's.
    """
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    import matisse_tokenizer

    if tokenizer != "llama3":
        return matisse_tokenizer.get_encoding(NAMES[tokenizer]), tiktoken.get_encoding(NAMES[tokenizer])
    path = os.path.join(vocab, "llama3.tiktoken")
    ranks = load_tiktoken_bpe(path)
    names = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
             "<|reserved_special_token_2|>", "<|reserved_special_token_3|>", "<|start_header_id|>",
             "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>"]
    names += [f"<|reserved_special_token_{index}|>" for index in range(5, 256 - 5)]
    specials = {name: len(ranks) + index for index, name in enumerate(names)}
    theirs = tiktoken.Encoding("llama3", pat_str=LLAMA3_PATTERN, mergeable_ranks=ranks, special_tokens=specials)
    return matisse_tokenizer.load_encoding("llama3", path), theirs


def digest_of(ids: object) -> str:
    """Returns the SHA-256 of ids as little-endian 32-bit words.

    Args:
        ids: The ids, a list or an array.

    Returns:
        The digest, in hex.
    """
    import numpy

    return hashlib.sha256(numpy.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def fresh(text: str) -> str:
    """Returns a new string equal to a text, whose UTF-8 CPython has not computed.

    Args:
        text: The text.

    Returns:
        The copy.
    """
    return (text + ".")[:-1]


def flat(results: list[object]) -> object:
    """Returns the ids of several calls concatenated, lists or arrays alike.

    Args:
        results: The ids of each call.

    Returns:
        The ids, one array.
    """
    import numpy

    return numpy.concatenate([numpy.asarray(ids, dtype="<u4") for ids in results]) if results else numpy.zeros(0)


def timed(passes: int, prepare: object, run: object, joined: object = None) -> tuple[object, list[float]]:
    """Runs a first pass, then the timed passes, each given what prepare() returns, every result held to the first
    outside the clock.

    Args:
        passes: The timed passes.
        prepare: What makes a pass's input, outside the clock.
        run: The pass.
        joined: What turns a pass's result into its ids, outside the clock; the result itself unless given.

    Returns:
        The first pass's ids and the seconds of every timed pass.
    """
    import numpy

    joined = joined or (lambda result: result)
    first = joined(run(prepare()))
    seconds = []
    for _ in range(passes):
        argument = prepare()
        start = time.perf_counter()
        result = run(argument)
        seconds.append(time.perf_counter() - start)
        assert numpy.array_equal(numpy.asarray(joined(result), dtype="<u4"), numpy.asarray(first, dtype="<u4"))
        del result
    return first, seconds


def row(vendor: str, tokenizer: str, label: str, name: str, size: int, seconds: list[float], count: int) -> None:
    """Prints a row in matisse_bench's format: the size, the tokens and the best, median and worst pass.

    Args:
        vendor: matisse-py or tiktoken.
        tokenizer: The tokenizer.
        label: The row's label.
        name: The input's name.
        size: The input's bytes.
        seconds: The seconds of every timed pass.
        count: The ids' count.
    """
    mib = size / (1024 * 1024)
    ordered = sorted(seconds)
    middle = statistics.median(ordered)
    millions = count / 1e6
    print(
        f"{vendor:<10} {tokenizer:<7} {label:<18} {name:<24} {mib:7.1f} MiB, {count:10d} tokens, "
        f"{len(seconds)} passes: best {mib / ordered[0]:7.1f}, median {mib / middle:7.1f}, "
        f"worst {mib / ordered[-1]:7.1f} MiB/s; best {millions / ordered[0]:6.2f}, median {millions / middle:6.2f} "
        f"M tokens/s",
        flush=True,
    )


def ids_line(vendor: str, tokenizer: str, name: str, ids: object) -> None:
    """Prints the ids line of a row.

    Args:
        vendor: matisse-py or tiktoken.
        tokenizer: The tokenizer.
        name: The input's name.
        ids: The ids.
    """
    print(f"ids {vendor} {tokenizer} {name} {len(ids)} {digest_of(ids)}", flush=True)


def read_documents(path: str) -> list[str]:
    """Returns the documents of a file, separated by NUL bytes.

    Args:
        path: The file.

    Returns:
        The documents.
    """
    return [document.decode("utf-8") for document in open(path, "rb").read().split(b"\0")]


def run_single(vocab: str, tokenizers: list[str], passes: int, path: str) -> None:
    """Times one long text in one call under each call that returns its ids.

    Args:
        vocab: The directory of the pinned files.
        tokenizers: The tokenizers.
        passes: The timed passes.
        path: The text's file.
    """
    text = open(path, "rb").read().decode("utf-8")
    name = os.path.basename(path)
    size = len(text.encode("utf-8"))
    for tokenizer in tokenizers:
        ours, theirs = encodings(vocab, tokenizer)
        ours_array = functools.partial(ours.encode_to_numpy, disallowed_special=())
        theirs_array = functools.partial(theirs.encode_to_numpy, disallowed_special=())
        calls = [
            ("matisse-py", "encode fresh", lambda: fresh(text), ours.encode_ordinary),
            ("matisse-py", "encode cached", lambda: text, ours.encode_ordinary),
            ("matisse-py", "numpy fresh", lambda: fresh(text), ours_array),
            ("matisse-py", "numpy cached", lambda: text, ours_array),
            ("tiktoken", "encode", lambda: fresh(text), theirs.encode_ordinary),
            ("tiktoken", "numpy", lambda: fresh(text), theirs_array),
        ]
        for vendor, label, prepare, call in calls:
            ids, seconds = timed(passes, prepare, call)
            ids_line(vendor, tokenizer, name, ids)
            row(vendor, tokenizer, label, name, size, seconds, len(ids))


def run_prompts(vocab: str, tokenizers: list[str], passes: int, path: str) -> None:
    """Times one call per prompt, reporting each pass's throughput and the mean latency of a call.

    Args:
        vocab: The directory of the pinned files.
        tokenizers: The tokenizers.
        passes: The timed passes.
        path: The prompts' file.
    """
    prompts = read_documents(path)
    name = os.path.basename(path)
    size = sum(len(prompt.encode("utf-8")) for prompt in prompts)
    for tokenizer in tokenizers:
        ours, theirs = encodings(vocab, tokenizer)
        calls = [
            ("matisse-py", "prompts", ours.encode_ordinary),
            ("matisse-py", "prompts numpy", lambda each: ours.encode_to_numpy(each, disallowed_special=())),
            ("tiktoken", "prompts", theirs.encode_ordinary),
            ("tiktoken", "prompts numpy", lambda each: theirs.encode_to_numpy(each, disallowed_special=())),
        ]
        for vendor, label, call in calls:

            def every(texts: list[str]) -> list[object]:
                return [call(prompt) for prompt in texts]

            ids, seconds = timed(passes, lambda: [fresh(prompt) for prompt in prompts], every, flat)
            ids_line(vendor, tokenizer, name, ids)
            row(vendor, tokenizer, label, name, size, seconds, len(ids))
            mean = statistics.median(seconds) / len(prompts) * 1e6
            print(f"{vendor:<10} {tokenizer:<7} {label + ' latency':<18} {name:<24} {len(prompts)} calls: "
                  f"per call median pass mean {mean:.2f} us", flush=True)


def run_parallel(vocab: str, tokenizers: list[str], passes: int, path: str, counts: list[int]) -> None:
    """Times encode_parallel() on one long text at each chunk count, beside encode_to_numpy() in the same run.

    Args:
        vocab: The directory of the pinned files.
        tokenizers: The tokenizers.
        passes: The timed passes.
        path: The text's file.
        counts: The chunk counts.
    """
    text = open(path, "rb").read().decode("utf-8")
    name = os.path.basename(path)
    size = len(text.encode("utf-8"))
    for tokenizer in tokenizers:
        ours, _ = encodings(vocab, tokenizer)
        serial, seconds = timed(passes, lambda: text, lambda each: ours.encode_to_numpy(each, disallowed_special=()))
        ids_line("matisse-py", tokenizer, name, serial)
        row("matisse-py", tokenizer, "numpy", name, size, seconds, len(serial))
        for count in counts:
            ids, seconds = timed(passes, lambda: text, lambda each: ours.encode_parallel_to_numpy(each, count))
            assert (ids == serial).all()
            row("matisse-py", tokenizer, f"parallel numpy x{count}", name, size, seconds, len(ids))
        for count in counts[-1:]:
            ids, seconds = timed(passes, lambda: text, lambda each: ours.encode_parallel(each, count))
            row("matisse-py", tokenizer, f"parallel list x{count}", name, size, seconds, len(ids))


def run_batch(vocab: str, tokenizers: list[str], passes: int, path: str, counts: list[int]) -> None:
    """Times encode_ordinary_batch() over the documents of a file at each thread count, under both packages.

    Args:
        vocab: The directory of the pinned files.
        tokenizers: The tokenizers.
        passes: The timed passes.
        path: The documents' file.
        counts: The thread counts.
    """
    documents = read_documents(path)
    name = os.path.basename(path)
    size = sum(len(document.encode("utf-8")) for document in documents)
    for tokenizer in tokenizers:
        ours, theirs = encodings(vocab, tokenizer)
        for vendor, encoding in (("matisse-py", ours), ("tiktoken", theirs)):
            for count in counts:

                def batch(texts: list[str]) -> list[list[int]]:
                    return encoding.encode_ordinary_batch(texts, num_threads=count)

                ids, seconds = timed(passes, lambda: [fresh(document) for document in documents], batch, flat)
                ids_line(vendor, tokenizer, name, ids)
                row(vendor, tokenizer, f"batch x{count}", name, size, seconds, len(ids))


def run_threads(vocab: str, tokenizers: list[str], passes: int, path: str, counts: list[int]) -> None:
    """Times one encode_ordinary() call per prompt spread over a pool of Python threads, at each thread count.

    Args:
        vocab: The directory of the pinned files.
        tokenizers: The tokenizers.
        passes: The timed passes.
        path: The prompts' file.
        counts: The thread counts.
    """
    prompts = read_documents(path)
    name = os.path.basename(path)
    size = sum(len(prompt.encode("utf-8")) for prompt in prompts)
    for tokenizer in tokenizers:
        ours, theirs = encodings(vocab, tokenizer)
        for vendor, encoding in (("matisse-py", ours), ("tiktoken", theirs)):
            for count in counts:
                with ThreadPoolExecutor(count) as pool:

                    def mapped(texts: list[str]) -> list[list[int]]:
                        return list(pool.map(encoding.encode_ordinary, texts))

                    ids, seconds = timed(passes, lambda: [fresh(prompt) for prompt in prompts], mapped, flat)
                ids_line(vendor, tokenizer, name, ids)
                row(vendor, tokenizer, f"threads x{count}", name, size, seconds, len(ids))


LOAD = """
import sys, time
start = time.perf_counter()
vendor, name, path = sys.argv[1], sys.argv[2], sys.argv[3]
if vendor == "tiktoken":
    import tiktoken
    encoding = tiktoken.get_encoding(name)
else:
    import matisse_tokenizer
    encoding = matisse_tokenizer.get_encoding(name) if path == "-" else matisse_tokenizer.load_encoding(name, path)
encoding.encode("warm")
print(time.perf_counter() - start)
"""
"""The program a fresh process runs to build one encoding and encode a word, printing the seconds from its start."""


def run_load(vocab: str, tokenizers: list[str], runs: int, root: str) -> None:
    """Times a fresh process building each encoding: matisse-tokenizer with the shipped report, with the report
    decided because the rank file is not the pinned one, and tiktoken.

    Args:
        vocab: The directory of the pinned files.
        tokenizers: The tokenizers.
        runs: The processes per row.
        root: The directory GPT-2's rank file was written to.
    """
    for tokenizer in tokenizers:
        name = NAMES[tokenizer]
        pinned = os.path.join(vocab, "llama3.tiktoken") if tokenizer == "llama3" else "-"
        source = pinned if tokenizer == "llama3" else os.path.join(root, "r50k_base.tiktoken")
        if tokenizer in ("cl100k", "o200k"):
            source = os.path.join(vocab, f"{name}.tiktoken")
        # The same table with an empty line more, which the loader skips: not the pinned file, so its report is
        # decided.
        other = os.path.join(root, f"{tokenizer}-other.tiktoken")
        shutil.copyfile(source, other)
        with open(other, "ab") as file:
            file.write(b"\n")
        rows = [("matisse-py", "load shipped", pinned), ("matisse-py", "load decided", other)]
        if tokenizer != "llama3":
            rows.append(("tiktoken", "load", "-"))
        for vendor, label, path in rows:
            seconds = []
            for _ in range(runs):
                output = subprocess.run(
                    [sys.executable, "-I", "-c", LOAD, vendor, name, path], capture_output=True, text=True, check=True
                )
                seconds.append(float(output.stdout))
            median = statistics.median(seconds)
            print(f"{vendor:<10} {tokenizer:<7} {label:<18} {runs} processes: median {median:.3f} s, best "
                  f"{min(seconds):.3f} s, worst {max(seconds):.3f} s", flush=True)


def write_short(path: str, output: str, limit: int) -> None:
    """Writes the prompts of a file cut to at most a number of bytes each, at a code point, separated by NUL bytes.

    Args:
        path: The prompts' file.
        output: The file written.
        limit: The most bytes a prompt keeps.
    """
    cut = []
    for prompt in read_documents(path):
        data = prompt.encode("utf-8")[:limit]
        cut.append(data.decode("utf-8", errors="ignore").encode("utf-8"))
    with open(output, "wb") as file:
        file.write(b"\0".join(cut))


def provenance() -> None:
    """Prints the versions, the machine and the processors the process may run on."""
    from importlib.metadata import version

    print("matisse-py bench")
    for package in ("matisse-tokenizer", "tiktoken", "numpy"):
        print(f"  {package:<18} {version(package)}")
    print(f"  python             {sys.version.split()[0]}")
    print(f"  system             {' '.join(os.uname()[i] for i in (0, 2, 4))}")
    print(f"  processors         {os.cpu_count()} visible, {len(os.sched_getaffinity(0))} allowed")
    print(f"  invocation         bench.py {' '.join(sys.argv[1:])}", flush=True)


def main() -> None:
    """Runs the mode the command line names."""
    mode = sys.argv[1]
    if mode == "short":
        write_short(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return
    provenance()
    vocab, tokenizers, number = sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
    root = serve(vocab)
    if mode == "load":
        run_load(vocab, tokenizers, number, root)
        return
    path = sys.argv[5]
    counts = [int(count) for count in sys.argv[6:]]
    if mode == "single":
        run_single(vocab, tokenizers, number, path)
    elif mode == "prompts":
        run_prompts(vocab, tokenizers, number, path)
    elif mode == "parallel":
        run_parallel(vocab, tokenizers, number, path, counts or [1, 2, 4, 8, 12, 16, 24])
    elif mode == "batch":
        run_batch(vocab, tokenizers, number, path, counts or [1, 24])
    elif mode == "threads":
        run_threads(vocab, tokenizers, number, path, counts or [1, 24])
    else:
        raise SystemExit(f"the modes are single, prompts, parallel, batch, threads, load and short, not {mode!r}")


main()
