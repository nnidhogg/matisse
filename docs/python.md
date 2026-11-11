# **Python Package**

matisse-tokenizer is matisse's Python package, imported as `matisse_tokenizer`, at version 0.1.0. It puts matisse's
encoder behind tiktoken's interface for GPT-2, cl100k_base, o200k_base and Llama 3, so that a program written against
tiktoken runs on matisse by its import, and its ids are tiktoken's, held to them by the tests below. matisse's class
tables are pinned to Unicode 16.0.0, the version of the tables tiktoken 0.14.0's regex engine reads, so the two class
every code point alike, as [limits.md](limits.md) describes; a tiktoken whose tables move past that version fails the
package's tests. Where matisse cannot do what tiktoken does, the package refuses with an error that says so, and this
page lists every such place. How the encoder works is in [design.md](design.md), what it guarantees in
[limits.md](limits.md), and the C++ interface in [usage.md](usage.md).

## **Installing and building**

The package needs CPython 3.12 or later. A wheel is built against the stable ABI of Python 3.12, so one wheel serves
every later CPython on its platform; free-threaded builds of CPython are not served, since the stable ABI does not cover
them. NumPy is needed only by the calls that return arrays, `encode_to_numpy()`, `encode_batch_to_numpy()` and
`encode_parallel_to_numpy()`, and is the extra `numpy`.

The build is scikit-build-core with nanobind, both pinned in `pyproject.toml`, over the CMake build of the libraries.
`pyproject.toml` stands at the root of the repository, not under `python/`, because the extension module builds from the
root project's libraries and the munch submodule, and an sdist made from a subdirectory could not carry them. The
package's own sources are under `python/`: `python/matisse_tokenizer/` the Python modules, `python/src/` the module's
C++, `python/tests/` the tests and `python/bench/` the bench. The root `CMakeLists.txt` adds `python/` under the option
`MATISSE_BUILD_PYTHON`, which `pyproject.toml` turns on and which builds every library as position-independent code;
nothing else in the libraries changes. A build from source needs what the libraries need, a C++23 compiler, GCC 13 or
Clang 19 or later, and CMake, and it fetches Boost's header-only modules as munch's build does:

```bash
git clone --recurse-submodules https://github.com/nnidhogg/matisse
cd matisse
pip install .
```

The wheels are built by cibuildwheel, configured in `pyproject.toml` and run by `.github/workflows/wheels.yml` when it
is started by hand or a tag `python-v*` is pushed. On Linux they are built for x86-64 and ARM64 in the manylinux_2_34
containers. On macOS they are built for ARM64 and x86-64 with Homebrew's LLVM and its libc++, since Apple's toolchain
lacks parts of the C++23 matisse uses, `std::jthread` among them. Neither the workflow nor the macOS configuration has
been run. There are no Windows wheels, since matisse builds with GCC and Clang on Linux alone. The workflow keeps the
wheels and the sdist as artifacts and publishes nothing.

## **In place of tiktoken**

```python
import matisse_tokenizer as tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

ids = encoding.encode("Hello, world!<|endoftext|>", allowed_special={"<|endoftext|>"})
assert ids == [9906, 11, 1917, 0, 100257]
assert encoding.decode(ids) == "Hello, world!<|endoftext|>"

batch = encoding.encode_ordinary_batch(["first text", "second text"], num_threads=8)
array = encoding.encode_to_numpy("a text whose ids need no Python list")

# matisse's own: one text encoded in up to 16 chunks on as many threads, with encode()'s ids.
parallel = encoding.encode_parallel(open("big.txt").read(), 16, disallowed_special=())
```

`get_encoding()` takes gpt2, cl100k_base and o200k_base, and `load_encoding(name, path)` reads a rank file from a path
under one of the four encodings' patterns and special tokens, Llama 3's among them. `list_encoding_names()` lists the
four. An `Encoding` offers tiktoken's calls with tiktoken's arguments and defaults:

- `encode()`, `encode_ordinary()`, `encode_to_numpy()`, `encode_batch()` and `encode_ordinary_batch()`, with
  `allowed_special`, `disallowed_special` and `num_threads` as tiktoken reads them;
- `decode()`, `decode_bytes()`, `decode_single_token_bytes()`, `decode_tokens_bytes()`, `decode_with_offsets()`,
  `decode_batch()` and `decode_bytes_batch()`;
- `encode_single_token()`, `token_byte_values()`, `is_special_token()`, `n_vocab`, `max_token_value`, `eot_token`,
  `special_tokens_set` and `name`;
- the attributes `_pat_str`, `_mergeable_ranks` and `_special_tokens`, from which tiktoken's README extends an encoding;
- pickling, by name for an encoding `get_encoding()` gives, by name and path for one `load_encoding()` read, and by its
  ranks and special tokens otherwise, as tiktoken pickles.

Three calls are matisse's own. `encode_parallel(text, chunks)` and `encode_parallel_to_numpy(text, chunks)` encode one
text in up to that many chunks on as many threads, cut only at matisse's certified cuts, with the ids `encode()` returns
and the same special-token arguments. `encode_batch_to_numpy()` takes `encode_batch()`'s arguments and returns each
text's ids as a NumPy array, so that a batch makes no Python integer per id.

The constructor `Encoding(name, pat_str=..., mergeable_ranks=..., special_tokens=...)` is tiktoken's, as Meta's
tokenizer.py and tiktoken's README call it. It takes a pattern only when it is one of the four, spelled exactly as
tiktoken spells gpt2's, cl100k_base's and o200k_base's and as Meta spells Llama 3's, and writes the ranks to a rank file
that matisse's loader reads, so any special tokens can stand beside them.

## **Where it refuses or differs**

Every call above returns tiktoken's ids, its exceptions' types and, for a disallowed special token, its message. These
are the places where matisse-tokenizer does something else:

- **Other encodings.** `get_encoding()` refuses with `ValueError` every name but gpt2, cl100k_base and o200k_base, and
  llama3, whose rank file Meta distributes under its own license and which `load_encoding()` reads from a path instead.
  Among tiktoken's other encodings, r50k_base is the rank file, pattern and special token gpt2 already reads, under
  another name the package does not register. p50k_base and p50k_edit read p50k_base.tiktoken under the same pattern,
  and the constructor refuses its ranks, as matisse's loader refuses them: they skip 50256, the id of `<|endoftext|>`,
  and so are not dense from zero. o200k_harmony the constructor refuses, since two of its special tokens,
  `<|endofprompt|>` and `<|reserved_200018|>`, share the id 200018.
- **Other patterns and special tokens.** The constructor refuses with `ValueError` a pattern that is not one of the four
  as spelled, since matisse encodes only under a pattern it has a verified description of. It refuses ranks that are not
  dense from zero or that spell one token twice, as matisse's loader does. It refuses a special token whose spelling is
  empty or whose id is an ordinary token's, two special tokens that share an id, as matisse's encoder refuses them, and
  a special token whose spelling begins another's, such as `<|a|>` and `<|a|>b`. tiktoken chooses between two such
  spellings at one place by the order of its table and steps past a disallowed one by a byte, so its ids there follow no
  rule matisse could hold to.
- **Argument types.** The arguments are read as tiktoken 0.14.0 reads them, in its order, with its exceptions.
  `allowed_special` is `"all"` or a `set` or `frozenset` of strings, and anything else, a string or a list among them,
  is refused with `TypeError`. A text that is not a string is refused with `TypeError`, before `allowed_special` is
  read, and so is a lone surrogate in a call that does not replace it. The ids given to the decoding calls are a
  sequence, and a set, a dict, an iterator or a string is refused with `TypeError`. `encode_single_token()` takes a
  string or `bytes` alone. The constructor refuses a special token's id that is negative or does not fit 32 bits with
  `OverflowError` and a spelling holding a lone surrogate with `UnicodeEncodeError`. `num_threads` takes what tiktoken's
  thread pool takes, `None`, a fraction and infinity among them, except a NaN, which tiktoken's pool waits on forever
  and which is refused here with `ValueError`. Every array returned is read-only, as tiktoken's are.
- **The load module.** `matisse_tokenizer.load` gives tiktoken's `load_tiktoken_bpe()`, `read_file_cached()` and
  `read_file()` with tiktoken's behavior, so that Meta's tokenizer.py reads Llama 3's ranks with it, but reads only
  local files and http and https URLs, refusing with `ValueError` the other schemes tiktoken reads through blobfile.
  tiktoken's `data_gym_to_mergeable_bpe_ranks()` and `dump_tiktoken_bpe()` do not exist.
- **Model names.** `encoding_for_model()` and `encoding_name_for_model()` refuse with `NotImplementedError`, since the
  package keeps no table of model names, which change with every model a vendor releases.
- **`encode_with_unstable()`** refuses with `NotImplementedError`. matisse lists no completions of a text's last piece;
  its streaming encoder, not yet in the package, returns the ids that are final whatever follows.
- **Private calls.** tiktoken's `_core_bpe`, `_encode_single_piece()`, `_encode_only_native_bpe()` and `_encode_bytes()`
  do not exist.
- **Messages.** The types of the exceptions are tiktoken's: `ValueError` for a disallowed special token, with tiktoken's
  message, `KeyError` for an id or a token the vocabulary does not hold, `OverflowError` for an id that is negative or
  does not fit 32 bits, `UnicodeEncodeError` where `encode_to_numpy()` meets a lone surrogate, which tiktoken's does not
  repair either, while `encode()` and the other calls replace it with U+FFFD first as tiktoken does. The other messages
  are matisse's own.

## **Vocabulary files and the cache**

`get_encoding()` fetches a rank file from the URL tiktoken reads it from and holds it to the SHA-256 tiktoken pins:
r50k_base.tiktoken for gpt2, which holds GPT-2's ranks as tiktoken builds its gpt2 encoding and has the digest
[benchmarks.md](benchmarks.md) pins for gpt2.tiktoken, and cl100k_base.tiktoken and o200k_base.tiktoken, with the
digests [design.md](design.md) pins. The cache is tiktoken's own: the directory `TIKTOKEN_CACHE_DIR` names, else
`DATA_GYM_CACHE_DIR`, else `data-gym-cache` under the system's temporary directory, each file under the SHA-1 of its
URL, so a file either package has fetched serves both. An empty `TIKTOKEN_CACHE_DIR` turns the cache off, as for
tiktoken. A cached file with another digest is fetched again and replaced, and a fetched file with another digest is
refused with `ValueError`. A request carries the User-Agent `matisse-tokenizer/0.1.0` and no other header of the
caller's.

## **Reports shipped with the package**

An encoder is built from its pattern's report, which `validate()` decides in 1.1 seconds for GPT-2 and 2.4 for cl100k at
the default width of five, as [limits.md](limits.md) gives. The package decides the reports of the four patterns when it
is built, GPT-2's, cl100k's and Llama 3's to width five and o200k's to width three, as matisse's bench decides them, and
ships each as the text `validate::serialize()` writes, under `matisse_tokenizer/reports/`. A process that reads a rank
file whose SHA-256 is the pinned file's reads its report back with `validate::deserialize()` instead of deciding it. Any
other rank file, read by `load_encoding()` or written by the constructor, is a table the package has not seen, and its
pattern's report is decided when it is read. The report depends on the pattern alone, so the digest is the stricter of
the two tests a shipped report could be held to. The tests hold each shipped report to the report `validate()` decides
at its width.

The encoder is built with the factorization inventory, the one [design.md](design.md) recommends, and with no special
token allowed. A call that allows special tokens cuts the text at their spellings in Python, as matisse's encoder cuts
at those it is built with, the earliest first and the longest among those that begin at one place, and encodes each
stretch between them as a complete input, which is the encoding matisse's encoder gives with those tokens allowed. One
encoder therefore serves every set of allowed special tokens, Llama 3's 256 among them.

## **Threads and the interpreter lock**

Every encoding call releases the interpreter lock while matisse encodes, so Python threads encode at once. Each thread
keeps a piece cache of its own across its calls, which never changes an id, as [design.md](design.md) argues. A batch is
encoded in C++ by a pool of `num_threads` threads, the calling thread one of them, that take the next text from a shared
counter, each with a cache of its own, as matisse's bench encodes a batch; `encode_parallel()` starts a thread per chunk
beyond the last, as matisse's `encode_parallel()` does. The text's UTF-8 is read from the Python string, which CPython
computes once and keeps beside the string, and an ASCII string needs no conversion. The ids are returned as a list of
Python integers, or as a NumPy array that owns matisse's buffer with no copy.

## **Performance**

Every figure below is a median of the passes of one collection, run on 2026-11-10 at commit f1a7825 by
`python/bench/run.sh` on the machine of [benchmarks.md](benchmarks.md), an i9-12900K whose 24 logical processors WSL2
shows, with Python 3.12.3, tiktoken 0.14.0 and NumPy 2.5.4. The outputs are kept in `python/bench/results/2026-11-10/`,
and `python/bench/tables.py` writes these tables from them. The C++ rows are matisse_bench's, the Python rows
`python/bench/bench.py`'s, which prints its rows and ids lines in matisse_bench's format; every row's ids are held to
the first pass's, and the package's ids lines equal matisse_bench's on every input. The single-thread rows ran under
`taskset -c 2`, as [benchmarks.md](benchmarks.md)'s do, and the threaded rows unpinned. The corpora are those of
[benchmarks.md](benchmarks.md). Under GPT-2 matisse_bench reads encoder.json and vocab.bpe and merges by the pair rule,
where the package reads r50k_base.tiktoken and merges by the rank rule, so the GPT-2 rows compare two definitions with
the same ids; the other three read the same rank file the same way.

One long text in one call on one thread, in MiB/s, the text a Python string made before the clock starts and never asked
for its UTF-8. "Python array" is `encode_to_numpy()` and "Python list" `encode_ordinary()`, and tiktoken's columns are
its calls of the same names:

| Tokenizer | Corpus      | C++ `encode()` | Python array | Python list | tiktoken array | tiktoken list |
|-----------|-------------|---------------:|-------------:|------------:|---------------:|--------------:|
| gpt2      | `gutenberg` |           72.1 |         79.7 |        66.0 |           26.8 |          24.8 |
| cl100k    | `gutenberg` |           80.6 |         80.3 |        67.3 |           26.5 |          23.6 |
| o200k     | `gutenberg` |           75.7 |         78.5 |        64.5 |           39.3 |          36.0 |
| llama3    | `gutenberg` |           76.4 |         84.3 |        67.5 |           37.3 |          33.8 |
| gpt2      | `wiki-ru`   |           88.6 |         73.6 |        48.7 |           26.0 |          22.3 |
| cl100k    | `wiki-ru`   |           62.5 |         60.5 |        51.2 |           19.9 |          18.9 |
| o200k     | `wiki-ru`   |           50.7 |         53.0 |        48.7 |           26.1 |          24.7 |
| llama3    | `wiki-ru`   |           51.8 |         55.3 |        49.1 |           25.6 |          24.3 |
| gpt2      | `wiki-zh`   |           82.0 |         73.5 |        43.9 |           24.7 |          23.2 |
| cl100k    | `wiki-zh`   |           65.3 |         67.1 |        46.1 |           22.0 |          19.8 |
| o200k     | `wiki-zh`   |           55.3 |         59.6 |        45.8 |           21.5 |          20.0 |
| llama3    | `wiki-zh`   |           60.6 |         64.8 |        52.2 |           23.5 |          21.9 |
| gpt2      | `enwik8`    |           53.5 |         59.0 |        46.3 |           24.4 |          22.2 |
| cl100k    | `enwik8`    |           62.2 |         60.4 |        46.5 |           22.5 |          21.1 |
| o200k     | `enwik8`    |           53.3 |         53.0 |        45.6 |           32.6 |          29.6 |
| llama3    | `enwik8`    |           56.5 |         55.2 |        45.2 |           32.6 |          28.2 |

Where a Python call's time goes on those texts, in milliseconds a call: the encoding is `encode_to_numpy()` on a string
whose UTF-8 CPython already keeps, the conversion is what the same call takes more on a string never converted, and the
list is what `encode_ordinary()` takes more than `encode_to_numpy()` on a converted string. The conversion is the
difference of two medians, which varies across the four tokenizers on one text by about as much as it measures, and it
stays at about a tenth of the encoding or less:

| Tokenizer | Corpus      | Encoding, ms | Conversion, ms | List, ms | List, ns per id |
|-----------|-------------|-------------:|---------------:|---------:|----------------:|
| gpt2      | `gutenberg` |         79.3 |            7.3 |     22.3 |            11.1 |
| cl100k    | `gutenberg` |         80.0 |            5.9 |     19.1 |            11.2 |
| o200k     | `gutenberg` |         85.0 |            2.9 |     18.5 |            10.9 |
| llama3    | `gutenberg` |         79.4 |            2.4 |     17.9 |            10.5 |
| gpt2      | `wiki-ru`   |         49.5 |            4.8 |     27.1 |            11.0 |
| cl100k    | `wiki-ru`   |         59.6 |            6.5 |     14.6 |            12.6 |
| o200k     | `wiki-ru`   |         68.6 |            6.9 |     10.0 |            14.1 |
| llama3    | `wiki-ru`   |         67.9 |            4.4 |      8.9 |            11.2 |
| gpt2      | `wiki-zh`   |         51.8 |            2.6 |     23.7 |             7.9 |
| cl100k    | `wiki-zh`   |         56.5 |            3.1 |     22.1 |            11.4 |
| o200k     | `wiki-zh`   |         64.3 |            2.8 |     20.3 |            15.6 |
| llama3    | `wiki-zh`   |         59.0 |            2.7 |     15.5 |            11.9 |
| gpt2      | `enwik8`    |       1502.4 |          114.6 |    428.8 |            14.8 |
| cl100k    | `enwik8`    |       1536.2 |           43.2 |    356.6 |            13.8 |
| o200k     | `enwik8`    |       1676.6 |          123.4 |    415.5 |            16.3 |
| llama3    | `enwik8`    |       1659.1 |           69.1 |    442.2 |            17.2 |

One call per prompt on one thread, the mean microseconds of a call, over the 20,000 prompts of `prompts` and over the
same prompts cut to their first 48 bytes at a code point. matisse_bench's calls make a piece cache for each call or keep
one across them, and the package keeps one per thread:

| Tokenizer | Prompts            | C++, own cache | C++, kept cache | Python list | Python array | tiktoken list | tiktoken array |
|-----------|--------------------|---------------:|----------------:|------------:|-------------:|--------------:|---------------:|
| gpt2      | 48 bytes at most   |           1.56 |            0.94 |        2.08 |         2.08 |          3.18 |           3.74 |
| cl100k    | 48 bytes at most   |           1.63 |            1.04 |        2.04 |         2.39 |          3.31 |           3.98 |
| o200k     | 48 bytes at most   |           1.77 |            1.17 |        1.99 |         2.20 |          2.63 |           3.33 |
| llama3    | 48 bytes at most   |           1.66 |            1.09 |        1.92 |         3.14 |          2.59 |           3.28 |
| gpt2      | 100 to 2,000 bytes |          24.23 |           17.39 |       30.31 |        21.77 |         59.02 |          49.70 |
| cl100k    | 100 to 2,000 bytes |          26.43 |           21.57 |       30.52 |        23.90 |         61.51 |          57.28 |
| o200k     | 100 to 2,000 bytes |          29.95 |           25.97 |       31.40 |        27.31 |         47.01 |          43.94 |
| llama3    | 100 to 2,000 bytes |          26.18 |           22.50 |       29.62 |        27.07 |         48.81 |          46.57 |

One text in parallel, `enwik8` in MiB/s at each number of chunks, `encode_parallel_to_numpy()` as the Python array and
`encode_parallel()` as the list. The interpreter lock is released while the chunks are encoded, so the array scales with
the chunks as the C++ call does, at 24 chunks to 84 to 99 percent of its rate, and building the list of 25 to 29 million
Python integers is most of the list's time:

| Tokenizer | Call                    |    1 |     2 |     4 |     8 |    12 |    16 |    24 |
|-----------|-------------------------|-----:|------:|------:|------:|------:|------:|------:|
| gpt2      | C++ `encode_parallel()` | 52.6 | 106.8 | 205.3 | 382.2 | 459.0 | 530.0 | 624.6 |
| gpt2      | Python array            | 63.3 | 119.8 | 227.4 | 369.6 | 445.1 | 519.3 | 619.5 |
| gpt2      | Python list             |    - |     - |     - |     - |     - |     - | 155.3 |
| cl100k    | C++ `encode_parallel()` | 56.3 | 108.2 | 228.1 | 402.5 | 494.7 | 579.7 | 714.4 |
| cl100k    | Python array            | 58.5 | 116.4 | 224.6 | 366.8 | 450.5 | 524.7 | 605.3 |
| cl100k    | Python list             |    - |     - |     - |     - |     - |     - | 164.5 |
| o200k     | C++ `encode_parallel()` | 56.5 | 110.7 | 204.7 | 363.7 | 450.3 | 526.0 | 631.1 |
| o200k     | Python array            | 52.0 | 102.4 | 204.5 | 331.8 | 412.0 | 480.1 | 570.7 |
| o200k     | Python list             |    - |     - |     - |     - |     - |     - | 161.8 |
| llama3    | C++ `encode_parallel()` | 60.8 | 116.5 | 225.0 | 393.8 | 484.2 | 567.1 | 685.1 |
| llama3    | Python array            | 56.9 |  99.9 | 201.9 | 342.8 | 409.0 | 494.1 | 572.7 |
| llama3    | Python list             |    - |     - |     - |     - |     - |     - | 152.9 |

The documents of `prompts` as a batch at one and at 24 threads, in MiB/s: matisse_bench's pool, the package's
`encode_ordinary_batch()` as the list and `encode_batch_to_numpy()` as the array, tiktoken's `encode_ordinary_batch()`,
and `encode_ordinary()` mapped over a `ThreadPoolExecutor` of that many threads under each package as the pool. The
package's batch encodes in C++ on its threads with the interpreter lock released, while the strings' conversion and the
lists or arrays it returns are made on the calling thread, which caps the batch below matisse_bench's; a pool of Python
threads calling one text at a time scales little under either package:

| Tokenizer | Threads | C++ batch | Python list | Python array | tiktoken | Python pool | tiktoken pool |
|-----------|--------:|----------:|------------:|-------------:|---------:|------------:|--------------:|
| gpt2      |       1 |      58.9 |        34.4 |         49.0 |     12.3 |        20.8 |          13.2 |
| cl100k    |       1 |      48.2 |        36.2 |         45.6 |     11.1 |        19.3 |          12.6 |
| o200k     |       1 |      39.0 |        32.9 |         36.5 |     13.2 |        18.8 |          14.3 |
| llama3    |       1 |      42.6 |        36.5 |         42.0 |     12.5 |        18.8 |          15.1 |
| gpt2      |      24 |     773.7 |       103.7 |        367.1 |     31.5 |        30.1 |          31.6 |
| cl100k    |      24 |     644.2 |       123.5 |        358.0 |     31.2 |        29.4 |          33.4 |
| o200k     |      24 |     539.9 |       152.9 |        321.5 |     34.3 |        33.9 |          35.3 |
| llama3    |      24 |     542.5 |       147.9 |        344.4 |     33.1 |        32.0 |          34.9 |

The seconds a fresh process takes from its start to an encoding that has encoded one word, the median of seven
processes: `get_encoding()`, or `load_encoding()` for Llama 3, with the shipped report, `load_encoding()` of the same
table with one empty line more, which is not the pinned file and so has its report decided, and tiktoken's
`get_encoding()`, the rank files in the cache. Beside the report, a process reads the rank file and builds matisse's
encoder, whose tables [limits.md](limits.md) gives the cost of:

| Tokenizer | Shipped report, s | Report decided, s | tiktoken, s |
|-----------|------------------:|------------------:|------------:|
| gpt2      |             0.190 |             0.989 |       0.102 |
| cl100k    |             0.384 |             2.158 |       0.075 |
| o200k     |             0.545 |             1.853 |       0.140 |
| llama3    |             0.524 |             2.236 |           - |

## **Tests**

The tests are a pytest suite under `python/tests`, which holds every id to tiktoken 0.14.0's, pip-installed beside the
package, under all four encodings:

```bash
pip install . --group test
MATISSE_VOCAB_DIR=VOCAB MATISSE_CORPUS_DIR=CORPORA python -m pytest python/tests
```

`MATISSE_VOCAB_DIR` names the directory of the pinned vocabulary files, as matisse's CMake option does, and the suite
writes GPT-2's rank file from its encoder.json and vocab.bpe with tiktoken's own conversion, held to r50k_base's digest,
and serves the files to both packages through one cache directory. With `MATISSE_TEST_DOWNLOAD=1` and no directory the
three public files are fetched from tiktoken's URLs instead, as the wheels' tests are, and Llama 3's tests skip. With
neither every test that needs a vocabulary skips. `MATISSE_CORPUS_DIR` names the corpora of
[benchmarks.md](benchmarks.md), and the tests that read them skip without them.

- **test_ids.py** holds matisse-tokenizer's ids to tiktoken's and to matisse's fixtures: the known answers of
  `known_answers_reference.txt`, 3,000 seeded random texts per encoding drawn from white space of every kind,
  contractions in both cases, letters of many scripts, marks, joiners, emoji, numbers, punctuation, special tokens'
  spellings and random code points of every plane, adversarial texts with runs of up to 100,000 spaces, letters, digits
  and CJK, newlines before runs of tabs, pieces either side of the merge's 128-byte threshold and special tokens'
  spellings against letters and each other, every special token allowed alone and as text, and the ten corpora of
  [benchmarks.md](benchmarks.md) below a gigabyte, each held to the vendors' digests of `vendors_reference.txt` and to
  tiktoken's ids, through every encoding call, both batch calls and `encode_parallel()`.
- **test_unicode.py** holds the Unicode version: matisse-tokenizer's ids are tiktoken's at every Unicode scalar value
  followed by `'s`, and at every code point of planes 0 to 3 and 14 in three more contexts; the class tables are of
  Unicode 16.0.0; and tiktoken reads every code point the tables leave out as the other class, by matisse's own
  tools/unicode/check_tiktoken.py, so that a tiktoken whose tables are of a later version fails the test.
- **test_api.py** holds the calls' arguments, defaults and refusals to tiktoken's: the special-token arguments in
  combination, the disallowed check's error and message, the vocabulary's figures, the single-token calls, decoding and
  its errors, lone surrogates, the batch calls and their thread counts, threads encoding at once, pickling, the
  constructor from ranks, a rank file that is not the pinned one, and the calls refused.
- **test_load.py** holds the fetch and the cache against a server on the loopback interface: tiktoken's cache keys and
  directories, the digest, a cached file that is not the pinned one, the cache turned off, and the request's headers.
- **test_reports.py** holds each shipped report to the report `validate()` decides at the registry's width, and its
  refusal under a pattern of other class symbols.

With every input given, the suite is 175 tests, all passing, in about six and a half minutes on the machine above, most
of them the Unicode tests; with no vocabulary, 13 pass and 162 skip.
