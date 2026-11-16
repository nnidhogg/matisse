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
root project's libraries and the munch and hopper submodules, and an sdist made from a subdirectory could not carry
them. The package's own sources are under `python/`: `python/matisse_tokenizer/` the Python modules, `python/src/` the
module's C++, `python/tests/` the tests and `python/bench/` the bench. The root `CMakeLists.txt` adds `python/` under
the option `MATISSE_BUILD_PYTHON`, which `pyproject.toml` turns on and which builds every library as
position-independent code; nothing else in the libraries changes. A build from source needs what the libraries need, a
C++23 compiler, GCC 13 or Clang 19 or later, and CMake, and it fetches Boost's header-only modules as munch's build
does:

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

`get_encoding()` takes gpt2, r50k_base, cl100k_base, o200k_base and o200k_harmony, and `load_encoding(name, path)` reads
a rank file from a path under one of the encodings' patterns and special tokens, Llama 3's among them.
`list_encoding_names()` lists them all, and `encoding_name_for_model()` and `encoding_for_model()` find a model's
encoding by tiktoken 0.14.0's table of models. r50k_base is gpt2's rank file, pattern and special token under the other
name tiktoken gives them. o200k_harmony is o200k_base's rank file and pattern with tiktoken's 1,091 special tokens of
it, two of which, `<|endofprompt|>` and `<|reserved_200018|>`, share the id 200018: each encodes to it, and it decodes
as `<|endofprompt|>`, as tiktoken's decoder gives it. An `Encoding` offers tiktoken's calls with tiktoken's arguments
and defaults:

- `encode()`, `encode_ordinary()`, `encode_to_numpy()`, `encode_batch()` and `encode_ordinary_batch()`, with
  `allowed_special`, `disallowed_special` and `num_threads` as tiktoken reads them;
- `decode()`, `decode_bytes()`, `decode_single_token_bytes()`, `decode_tokens_bytes()`, `decode_with_offsets()`,
  `decode_batch()` and `decode_bytes_batch()`;
- `encode_single_token()`, `token_byte_values()`, `is_special_token()`, `n_vocab`, `max_token_value`, `eot_token`,
  `special_tokens_set` and `name`;
- the attributes `_pat_str`, `_mergeable_ranks` and `_special_tokens`, from which tiktoken's README extends an encoding;
- pickling, by name for an encoding `get_encoding()` gives, by name and path for one `load_encoding()` read, and by its
  ranks and special tokens otherwise, as tiktoken pickles.

Six calls are matisse's own, not tiktoken's: `load_tokenizer_json()` and `validate_merges()`, below, and four of an
`Encoding`. `encode_parallel(text, chunks)` and `encode_parallel_to_numpy(text, chunks)` encode one text in up to that
many chunks on as many threads, cut only at matisse's certified cuts, with the ids `encode()` returns and the same
special-token arguments. `encode_batch_to_numpy()` takes `encode_batch()`'s arguments and returns each text's ids as a
NumPy array, so that a batch makes no Python integer per id. `stream()` returns matisse's streaming encoder over the
encoding, whose `push(part)` reads the next part of a text, a string or bytes that may end inside a code point's UTF-8
sequence, and returns the ids that have become final, which no continuation of the text can change, and whose `finish()`
returns the rest, after which the stream reads a new text. Every id a stream returns is one of `encode_ordinary()`'s of
the whole text and all of them are, so every special token's spelling is read as text. How far the ids lag behind the
parts is what [limits.md](limits.md) gives for matisse's `Stream_encoder`. A stream serves one thread at a time, its
calls release the interpreter lock, and once it has refused a text, ill-formed UTF-8 or a text that ends inside a
sequence, every further call is refused with `ValueError`.

The constructor `Encoding(name, pat_str=..., mergeable_ranks=..., special_tokens=...)` is tiktoken's, as Meta's
tokenizer.py and tiktoken's README call it. It takes a pattern only when it is one of the four, spelled exactly as
tiktoken spells gpt2's, cl100k_base's and o200k_base's and as Meta spells Llama 3's, and writes the ranks to a rank file
that matisse's loader reads, so any special tokens can stand beside them.

## **Reading a tokenizer.json**

```python
import matisse_tokenizer

encoding = matisse_tokenizer.load_tokenizer_json("Meta-Llama-3-8B/tokenizer.json")

assert encoding.name == "llama3"
ids = encoding.encode("Hello, world!<|eot_id|>", allowed_special="all")
```

`load_tokenizer_json(path)` returns an `Encoding` read from a Hugging Face tokenizer.json by matisse's
`read_tokenizer_json()`, under the one of the four patterns whose published spelling the file's pre-tokenizer runs, as
`tokenizer_json_spelling()` finds it, and with the file's added special tokens as its special tokens. The encoding is
named by that pattern's encoding, gpt2, o200k_base or llama3, and offers every call above. The cl100k conversions on the
Hugging Face Hub, Xenova/gpt-4 among them, carry cl100k's pattern as tiktoken shipped it before its possessive rewrite,
which is Llama 3's, so they read as llama3 with cl100k's vocabulary and special tokens, and on the corpora of the test
strategy their ids are Hugging Face's, which are tiktoken's cl100k_base ids there. No file reads as cl100k_base, since
Hugging Face's engine reads cl100k_base's published `\p{N}{1,3}+` as the interval repeated and so segments otherwise
than tiktoken.

Its ids are those Hugging Face tokenizers gives the file with `add_special_tokens=False`: a special token is cut out
where it is allowed, as Hugging Face cuts its added tokens, and read as text where it is not, as Hugging Face's
`encode_special_tokens` reads it. The tokens a post-processor adds under `add_special_tokens`, such as Llama 3's
`<|begin_of_text|>`, are the caller's to add. A file whose SHA-256 is one of the four pinned tokenizer.json files' takes
its pattern's shipped report, and any other file has the report decided when it is read. An encoding read so pickles by
its path.

## **Certifying a merge table without a pre-tokenizer**

```python
import collections

import matisse_tokenizer

text = open("corpus.txt", "rb").read()
pairs = collections.Counter(text[at : at + 2] for at in range(len(text) - 1))
candidates = [pair for pair, _ in pairs.most_common(1000)]

for window in matisse_tokenizer.validate_merges("tokenizer.json", candidates):
    if window.certified:
        print(window.window, "is cut at", window.origin)
```

`validate_merges(path, candidates)` is matisse's `validate_merges()` over a Hugging Face tokenizer.json whose
pre-tokenizer is a ByteLevel without its regex, which splits nothing, so that its merge table merges a whole text and
has no pattern an `Encoding` could be built under; matisse's loader reads the file under the empty spelling. Each
candidate is bytes, and each is returned as a `MergeWindow` of its `window`, its `origin`, where the window's own
encoding begins the token covering its last byte, the only origin that can be certified, and its `refutation`. That is
None when the window is certified, and `certified` is then true, and otherwise a `Refutation` of the `tokens` of a
string, the merge's segmentation of it, and the `occurrence` of the window in it, at which the token covering the
window's last byte begins elsewhere, so that the window certifies no cut at any origin. The verdicts are exact under the
pair rule, Hugging Face's merge, and of the merge alone, on a text with no special token cut out. The file is read and
the windows decided with the interpreter lock released and the table's token automaton built once for all of them, and
on GPT-2's table a window of three bytes takes about half a millisecond on average, as [limits.md](limits.md) gives.

## **Where it refuses or differs**

Every call above returns tiktoken's ids, its exceptions' types and, for a disallowed special token, its message. These
are the places where matisse-tokenizer does something else:

- **Other encodings.** `get_encoding()` refuses with `ValueError` every name but gpt2, r50k_base, cl100k_base,
  o200k_base and o200k_harmony, and llama3, whose rank file Meta distributes under its own license and which
  `load_encoding()` reads from a path instead. Among tiktoken's other encodings, p50k_base and p50k_edit read
  p50k_base.tiktoken under GPT-2's pattern, and the constructor refuses its ranks, as matisse's loader refuses them:
  they skip 50256, the id of `<|endoftext|>`, and so are not dense from zero.
- **Other patterns and special tokens.** The constructor refuses with `ValueError` a pattern that is not one of the four
  as spelled, since matisse encodes only under a pattern it has a verified description of. It refuses ranks that are not
  dense from zero or that spell one token twice, as matisse's loader does. It refuses a special token whose spelling is
  empty or whose id is an ordinary token's, as matisse's encoder refuses them, and a special token whose spelling begins
  another's, such as `<|a|>` and `<|a|>b`. tiktoken chooses between two such spellings at one place by the order of its
  table and steps past a disallowed one by a byte, so its ids there follow no rule matisse could hold to. It refuses two
  special tokens that share an id in any table but tiktoken's own of o200k_harmony, in whatever order: tiktoken decodes
  a shared id as one of its spellings by the hash order of its table, whichever order they are given in, and only
  o200k_harmony's has been checked against it.
- **tokenizer.json.** `load_tokenizer_json()` refuses with `ValueError`, with matisse's loader's message, which names
  the member of the file by its path from the root and what it holds, whatever the loader refuses: a model other than
  BPE, a normalizer, truncation or padding, a pre-tokenizer whose pattern is none of the four, a post-processor that
  adds tokens otherwise than under `add_special_tokens`, an added token that is not special or that stands at another id
  than the one Hugging Face gives it, and ordinary ids that are not dense from zero, among the others
  [design.md](design.md) lists. A file it cannot read is refused with `RuntimeError`. Its special tokens are refused as
  the constructor's are, an empty one, one with an ordinary token's id and one whose spelling begins another's among
  them.
- **Merge tables.** `validate_merges()` refuses with `ValueError`, with matisse's message, a file matisse's loader
  refuses under the empty spelling, a pre-tokenizer that splits a text among them, a model whose `ignore_merges` is
  true, which would look a whole text up, a table munch's token automaton refuses, with a merge whose part no earlier
  merge makes or that makes a token again, and an empty candidate, naming its index. A candidate that is not bytes is
  refused with `TypeError` and a file it cannot read with `RuntimeError`.
- **Argument types.** The arguments are read as tiktoken 0.14.0 reads them, in its order, with its exceptions.
  `allowed_special` is `"all"` or a `set` or `frozenset` of strings, and anything else, a string or a list among them,
  is refused with `TypeError`, and a set holding a string with a lone surrogate, a special token's or not, with
  `UnicodeEncodeError`. A text that is not a string is refused with `TypeError`, before `allowed_special` is read, and
  so is a lone surrogate in a call that does not replace it. The ids given to the decoding calls are a sequence, and a
  set, a dict, an iterator or a string is refused with `TypeError`. `encode_single_token()` takes a string or `bytes`
  alone. The constructor refuses a special token's id that is negative or does not fit 32 bits with `OverflowError` and
  a spelling holding a lone surrogate with `UnicodeEncodeError`. `num_threads` takes what tiktoken's thread pool takes,
  `None`, a fraction and infinity among them, except a NaN, which tiktoken's pool waits on forever and which is refused
  here with `ValueError`. Every array returned is read-only, as tiktoken's are.
- **The load module.** `matisse_tokenizer.load` gives tiktoken's `load_tiktoken_bpe()`, `read_file_cached()` and
  `read_file()` with tiktoken's behavior, so that Meta's tokenizer.py reads Llama 3's ranks with it, but reads only
  local files and http and https URLs, refusing with `ValueError` the other schemes tiktoken reads through blobfile.
  tiktoken's `data_gym_to_mergeable_bpe_ranks()` and `dump_tiktoken_bpe()` do not exist.
- **Model names.** `encoding_name_for_model()` and `encoding_for_model()` follow tiktoken 0.14.0's table of models and
  its prefixes, and refuse with `KeyError`, naming the encoding, a model tiktoken maps to p50k_base or p50k_edit, which
  the package does not serve. A model a later tiktoken knows and 0.14.0 does not is refused with tiktoken's `KeyError`
  until the package's table follows.
- **`encode_with_unstable()`** refuses with `NotImplementedError`. matisse lists no completions of a text's last piece;
  its streaming encoder, `stream()` above, returns the ids that are final whatever follows instead.
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
Python integers, or as a NumPy array that owns matisse's buffer with no copy. On many threads the arrays are the calls
to choose: a list holds one Python integer per id, made on the calling thread once the encoding is done, so a batch of
lists scales with its threads only as far as that one thread can build them, while an array, which owns matisse's
buffer, scales with the encoding itself, as the batch tables under Performance show.

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

## **Performance beside Hugging Face tokenizers**

The figures below set the package beside Hugging Face's tokenizers package, the two reading the same tokenizer.json
files. They are medians of the passes of one collection, begun on 2026-11-13 and timed on 2026-11-14 at commit 79126be
by `python/bench/run_tokenizer_json.sh` on the machine above, with Python 3.12.3, tokenizers 0.23.3, tiktoken 0.14.0 and
NumPy 2.5.4 installed from PyPI in a virtual environment of their own and matisse-tokenizer built from that commit;
`machine.txt` among the outputs lists every package of the environment and the SHA-256 of each file read. The outputs
are kept in `python/bench/results/2026-11-13/`, and `python/bench/tables.py` writes these tables from them. Both
packages read the four pinned tokenizer.json files of [design.md](design.md), each held to its SHA-256 before it is
read: GPT-2's from openai-community/gpt2 at commit 607a30d783dfa663caf39e06633721c8d4cfcd7e, Xenova's conversions of
cl100k and o200k from Xenova/gpt-4 at 1d9f1f1b1fae88c0e4df1dab0a397f8de6229075 and Xenova/gpt-4o at
7956d98f2a83b2751a98ea7136fdf7fe6cf54e69, and Llama 3's, held beside its rank file. The tables name each file by its
tokenizer, and Xenova/gpt-4's, the cl100k rows, reads under Llama 3's pattern, as the section on reading a
tokenizer.json says. The package reads each file with `load_tokenizer_json()` and Hugging Face with
`Tokenizer.from_file()` with `encode_special_tokens` set, so that both read a special token's spelling as text, and
every Hugging Face call passes `add_special_tokens=False`. The single-thread rows ran under `taskset -c 2` with
`RAYON_NUM_THREADS=1`, and the batch rows unpinned with `RAYON_NUM_THREADS` set to their thread count, the size of the
pool Hugging Face's batch calls run on. Each timed run is one tokenizer's, the machine otherwise idle through each, and
a run during which other work used a processor was repeated. Every process ran with its address space limited to
12,000,000 KiB, but those on enwik8, where the limit was raised to 40,000,000 KiB, since Hugging Face's `Encoding` of
enwik8 takes about 14 GiB resident and its process aborts under the lower limit.

Before any of it was timed, both packages' ids were compared document by document on every input, the ten corpora of
[benchmarks.md](benchmarks.md) below a gigabyte, the 20,000 prompts of `prompts` and the same prompts cut to their first
48 bytes at a code point, and each timed mode compares them again before its rows and holds every timed row's ids to
them; a tokenizer and an input whose ids differed would have been reported and left untimed. None differs. Every digest
below is also the one [benchmarks.md](benchmarks.md)'s exactness table gives for the corpus where it lists it, so that
under Xenova/gpt-4's file both packages give tiktoken's cl100k_base ids on every corpus that table lists:

| Corpus                      | tokenizer.json |        Ids | SHA-256 of the ids                                                 | Equal |
|-----------------------------|----------------|-----------:|--------------------------------------------------------------------|-------|
| `twitter.json`              | gpt2           |    332,556 | `44eabd1f3c08a75ea2181d5dc357ac3fb05d2c525b5fecf670f114990201cd67` | yes   |
| `twitter.json`              | cl100k         |    173,645 | `ea966e05d3246f335e1f360efeec236ea89682f8f7cb5c4a61c07340f0760c37` | yes   |
| `twitter.json`              | o200k          |    163,117 | `53f263ccd2c55d48da0ee7071860bea4769244e2cae750aac1429ad09d0980cc` | yes   |
| `twitter.json`              | llama3         |    160,933 | `097bdc5645cbdeca5822488fc84307633cad18d0b70290f0ba6c03e809f9d6d5` | yes   |
| `munch-python@15b2863c`     | gpt2           |    184,012 | `6eb94834cdca2d97450eb86d883cf3ddc447569944f5a44da0cc2b70216549b2` | yes   |
| `munch-python@15b2863c`     | cl100k         |    109,965 | `1453748b540ca6ffad8310025d7fc886182b2c05bbdc18c519eb5b2d9ee9fe4b` | yes   |
| `munch-python@15b2863c`     | o200k          |    110,390 | `be2c2699af3efa4f9f9a98cd78485682700e0b3a95c72d9e796fcd3d25e0c0f6` | yes   |
| `munch-python@15b2863c`     | llama3         |    109,965 | `1453748b540ca6ffad8310025d7fc886182b2c05bbdc18c519eb5b2d9ee9fe4b` | yes   |
| `munch-cpp@15b2863c`        | gpt2           |  1,639,282 | `8f34429ef998914438e243d129ae1ccdb0c26d97f29094f9a9cb1d93b18022fe` | yes   |
| `munch-cpp@15b2863c`        | cl100k         |  1,060,572 | `97ee02f4cb13d88775f8fe807d895b0810fecb718e4e53d2953017399e0da201` | yes   |
| `munch-cpp@15b2863c`        | o200k          |  1,063,148 | `bd829cb1281381b51aa3836f0e22ef90b3d60f9d891ed6939e3e4e8428ba5401` | yes   |
| `munch-cpp@15b2863c`        | llama3         |  1,060,516 | `c23dec7f0914366b8332666fc7bc098440c11b0e94de14379d9319a50830dfa3` | yes   |
| `gutenberg`                 | gpt2           |  2,002,145 | `f8c1359e9260f01b1a1bde0580480b4cac01d8f7c9ca20a8c2443718ed000434` | yes   |
| `gutenberg`                 | cl100k         |  1,708,273 | `9c925a5984d22eb7dbb8408b897f612f0a345931605a822aab52c480382c21ea` | yes   |
| `gutenberg`                 | o200k          |  1,695,829 | `f561e9d7ea3cf117591bc9b806e39bb094e378068728065e3c512eb992816e08` | yes   |
| `gutenberg`                 | llama3         |  1,706,691 | `96bf65b9a9a7e608664af5a1730be923e7350d71e4bcbfa8a26f92a051769183` | yes   |
| `wiki-zh`                   | gpt2           |  2,984,479 | `9e1315d4c8f265369c3891e10d08a5358c663db4368c6a8050f6e75759384017` | yes   |
| `wiki-zh`                   | cl100k         |  1,939,825 | `7a0ec4e167b68b621e6562e2191bdcc43b4d73a31dd8b20d7e1b5aac67b986fc` | yes   |
| `wiki-zh`                   | o200k          |  1,299,269 | `f3b08b0ba50957619314917ebc98cc3b26c9e82f4a09fab530aba5976d56d56e` | yes   |
| `wiki-zh`                   | llama3         |  1,298,383 | `9b3f7244ef292fb222cac56c62a7b68a86abbc95f2e8d7d7ce1d7a2fed7cf605` | yes   |
| `wiki-ar`                   | gpt2           |  2,385,003 | `b45551f291d84403bc9cbb0f3bb37ed0678e63788fba17560958570d0e11b17a` | yes   |
| `wiki-ar`                   | cl100k         |  1,663,728 | `ae59537c4ab51ccf984b5ac2e212f0cd1072612e00bce2711f2bca027fb9d109` | yes   |
| `wiki-ar`                   | o200k          |    806,639 | `8117c3677d2a42c469e8b94dd894bdf60351cee48e01e9b90b5ce6e7cc114be6` | yes   |
| `wiki-ar`                   | llama3         |    932,883 | `01c49be3bab1a0eb5d0406047d77e0b0ce0f9481dbc50d4b20e211aa731d365c` | yes   |
| `wiki-ru`                   | gpt2           |  2,470,902 | `1cff652bf33ba07b6e4899c9d997cc50495d115dda68bf252355a82397080b94` | yes   |
| `wiki-ru`                   | cl100k         |  1,162,181 | `9bebf2079fdff13a446ee5d02d723ebe16930e48af5a4c0b4e3e5f72ddd10be1` | yes   |
| `wiki-ru`                   | o200k          |    709,178 | `9b243bc017b9b6f62449eee4bde696ecb6d32ec2db90114b6ca4a0cd4d684713` | yes   |
| `wiki-ru`                   | llama3         |    791,648 | `87e884e6ac480da3f0cc024007f14ae2e69ec94fb847030c4d415f61bd779ee4` | yes   |
| `wiki-hi`                   | gpt2           |  2,502,380 | `019b5b9db2db6e35c198fa34d10e794952e4751976cd906bb26752edf123d38f` | yes   |
| `wiki-hi`                   | cl100k         |  1,606,330 | `a03e85f6eba4fb48e4deb13a881c1af981435a8dd514eb86ef0b47f38e1ebf82` | yes   |
| `wiki-hi`                   | o200k          |    549,062 | `7bfa0edf1d93ce07773a09f6169d930fc4fbe7120e05a74531e49f9fbb2b45ab` | yes   |
| `wiki-hi`                   | llama3         |    861,673 | `25012d454b9eded8553cb113d936ec27c2c3e03b4aa1fb8b4629ddb6a56974de` | yes   |
| `wiki-sv`                   | gpt2           |  1,585,688 | `c269cdbcf4f89d8cd35b2a4db25a5547609eee8b45c09ff041cbb7b79cdab544` | yes   |
| `wiki-sv`                   | cl100k         |  1,325,036 | `b3333acbec23d85191832bbcd87dd3947afc2762065414a54055187ef7cff56d` | yes   |
| `wiki-sv`                   | o200k          |  1,134,708 | `76bb07ee0c90962dabbbe607d7df4ae9226390ade251ad2297df7ec8d986dbf4` | yes   |
| `wiki-sv`                   | llama3         |  1,314,144 | `ab12bed671b83fd0991d19287fe99fd9e497cf77d61c03d2f9b10d3c295a3664` | yes   |
| `enwik8`                    | gpt2           | 29,013,572 | `a7e18c6d9ab250b10190b6e80c13942596b024e49552c63e699f056b38b9d53d` | yes   |
| `enwik8`                    | cl100k         | 25,793,085 | `fc905793ea8efed63d03122c453f1a0f48e062490201b416cb3eacacc1302311` | yes   |
| `enwik8`                    | o200k          | 25,426,300 | `7de481826895d0084a3bca6376494a2f70f6ff71224e7fb2bcdbee5439631ae1` | yes   |
| `enwik8`                    | llama3         | 25,706,310 | `3fc3bd5e699caaf8956c53b1e3dde1e1bfa9cea39d75454feb1e6c409c073a36` | yes   |
| `prompts`                   | gpt2           | 10,832,297 | `2e52723085d05b2a2e01e4e6356619f1d8d18870e9f80eb13f786f9548580bf1` | yes   |
| `prompts`                   | cl100k         |  7,202,788 | `88674a5751d4bea4470490c4561cf9ade97e13839097ca542196bcefb7400bca` | yes   |
| `prompts`                   | o200k          |  4,566,270 | `cba41e7a3fa25f4d46a664cd4c607d5921129872b108177725fdac0ce8e368c7` | yes   |
| `prompts`                   | llama3         |  5,142,480 | `2fc5ecf962ed0d2cc229aa7948c14a58427e270505fe018561ffc43f7872352c` | yes   |
| `prompts`, 48 bytes at most | gpt2           |    497,728 | `9a9f2f1bf62a9a98585d4a0f150d31bf40dabb6897d145876e69454208a16ef7` | yes   |
| `prompts`, 48 bytes at most | cl100k         |    335,374 | `775da66306b6e76ef048bbf2221813152dfe87213e79d9db2f22eae24f5b7d1b` | yes   |
| `prompts`, 48 bytes at most | o200k          |    221,969 | `0c4b79f1dd864df83a374f438d5202e18bd2cdb7adf60a44f7d00fa07bee75c5` | yes   |
| `prompts`, 48 bytes at most | llama3         |    245,906 | `a60be721afcc6606190f853c685a238d3cf17c9dcc966734ce5e487c2ac368a7` | yes   |

One long text in one call on one thread, in MiB/s, the text a Python string made before the clock starts and never asked
for its UTF-8. "Python array" is `encode_to_numpy()` and "Python list" `encode_ordinary()`, and Hugging Face's columns
are `encode(text).ids` and `encode_batch_fast([text])[0].ids`, the second of which computes no offsets, each with its
list of ids taken inside the clock, as `encode_ordinary()` returns one. The last column is the package's list over the
faster of Hugging Face's two:

| tokenizer.json | Corpus      | Python array | Python list | `encode()` | `encode_batch_fast()` | List over the faster |
|----------------|-------------|-------------:|------------:|-----------:|----------------------:|---------------------:|
| gpt2           | `gutenberg` |         68.7 |        55.9 |        2.7 |                   3.9 |                14.3x |
| cl100k         | `gutenberg` |         69.8 |        57.4 |        2.8 |                   3.7 |                15.5x |
| o200k          | `gutenberg` |         67.7 |        56.7 |        2.8 |                   3.7 |                15.3x |
| llama3         | `gutenberg` |         76.7 |        62.8 |        2.9 |                   4.1 |                15.3x |
| gpt2           | `wiki-ru`   |         81.3 |        51.8 |        2.7 |                   4.4 |                11.8x |
| cl100k         | `wiki-ru`   |         64.6 |        53.2 |        3.2 |                   4.8 |                11.1x |
| o200k          | `wiki-ru`   |         54.7 |        49.0 |        3.3 |                   4.7 |                10.4x |
| llama3         | `wiki-ru`   |         50.6 |        45.2 |        3.4 |                   5.5 |                 8.2x |
| gpt2           | `wiki-zh`   |         77.7 |        53.6 |        2.8 |                   4.4 |                12.2x |
| cl100k         | `wiki-zh`   |         71.6 |        50.5 |        3.0 |                   4.8 |                10.5x |
| o200k          | `wiki-zh`   |         62.1 |        49.9 |        3.1 |                   4.8 |                10.4x |
| llama3         | `wiki-zh`   |         59.8 |        48.8 |        3.3 |                   5.9 |                 8.3x |
| gpt2           | `enwik8`    |         57.1 |        47.4 |        2.3 |                   4.0 |                11.8x |
| cl100k         | `enwik8`    |         55.4 |        47.4 |        2.3 |                   3.8 |                12.5x |
| o200k          | `enwik8`    |         51.0 |        45.5 |        2.3 |                   3.7 |                12.3x |
| llama3         | `enwik8`    |         57.5 |        52.6 |        2.3 |                   4.2 |                12.5x |

One call per prompt on one thread, the mean microseconds of a call, over the 20,000 prompts of `prompts` and over the
same prompts cut to their first 48 bytes, through the same calls; the last column is the latency of Hugging Face's
faster call over the package's list:

| tokenizer.json | Prompts            | Python list | Python array | `encode()` | `encode_batch_fast()` | List over the faster |
|----------------|--------------------|------------:|-------------:|-----------:|----------------------:|---------------------:|
| gpt2           | 48 bytes at most   |        2.01 |         2.23 |      14.45 |                 13.77 |                 6.9x |
| cl100k         | 48 bytes at most   |        2.03 |         2.29 |      13.94 |                 12.85 |                 6.3x |
| o200k          | 48 bytes at most   |        2.32 |         2.72 |      14.15 |                 13.55 |                 5.8x |
| llama3         | 48 bytes at most   |        2.11 |         3.53 |      12.80 |                 11.87 |                 5.6x |
| gpt2           | 100 to 2,000 bytes |       30.03 |        21.42 |     370.31 |                330.60 |                11.0x |
| cl100k         | 100 to 2,000 bytes |       32.03 |        24.54 |     332.44 |                300.87 |                 9.4x |
| o200k          | 100 to 2,000 bytes |       34.91 |        30.30 |     317.99 |                286.07 |                 8.2x |
| llama3         | 100 to 2,000 bytes |       35.07 |        30.04 |     302.86 |                257.39 |                 7.3x |

The documents of `prompts` as one batch at one and at 24 threads, in MiB/s: the package's `encode_ordinary_batch()` as
the list and `encode_batch_to_numpy()` as the array, and Hugging Face's `encode_batch()` and `encode_batch_fast()`, each
with the lists of ids of every document taken inside the clock. The last column is the package's list over the faster of
Hugging Face's two:

| tokenizer.json | Threads | Python list | Python array | `encode_batch()` | `encode_batch_fast()` | List over the faster |
|----------------|--------:|------------:|-------------:|-----------------:|----------------------:|---------------------:|
| gpt2           |       1 |        37.3 |         50.6 |              3.9 |                   3.8 |                 9.6x |
| cl100k         |       1 |        36.6 |         46.1 |              4.0 |                   4.0 |                 9.2x |
| o200k          |       1 |        31.3 |         35.3 |              4.1 |                   4.0 |                 7.6x |
| llama3         |       1 |        32.6 |         37.8 |              4.4 |                   4.7 |                 6.9x |
| gpt2           |      24 |       103.4 |        362.1 |             23.3 |                  27.7 |                 3.7x |
| cl100k         |      24 |       121.9 |        344.2 |             26.7 |                  30.1 |                 4.0x |
| o200k          |      24 |       139.4 |        295.9 |             30.2 |                  33.4 |                 4.2x |
| llama3         |      24 |       138.7 |        316.5 |             31.6 |                  35.5 |                 3.9x |

On this machine and these files, the package's list of ids comes 8.2 to 15.5 times as fast as Hugging Face's faster call
on one long text, at 45.2 to 62.8 MiB/s against 3.7 to 5.9, and a prompt of 100 to 2,000 bytes costs it 30.03 to 35.07
microseconds against Hugging Face's 257.39 to 330.60, 7.3 to 11.0 times less; on the prompts cut to 48 bytes the lead
narrows to 5.6 to 6.9 times, 2.01 to 2.32 microseconds against 11.87 to 13.77. As one batch the list runs at 6.9 to 9.6
times Hugging Face's faster call on one thread and at 3.7 to 4.2 times it on 24, where the package's arrays reach 295.9
to 362.1 MiB/s against Hugging Face's 27.7 to 35.5, which is why the arrays are the calls to use on many threads. The
figures compare the calls a Python program makes to have the ids of these four files, and no more. Hugging Face's calls
return an `Encoding` that carries the tokens' strings, their offsets unless `encode_batch_fast()` drops them, and the
masks its other features use, none of which the package computes, so the tables measure what each call costs as it is,
not where Hugging Face's time goes or what its merge alone would cost; [benchmarks.md](benchmarks.md) sets the C++
encoder beside Hugging Face's Rust crate and the other engines. Nothing was measured of Hugging Face's normalizers,
post-processors, padding or truncation, of any tokenizer.json but the four pinned ones, or on any other machine, and the
single-thread figures carry the spread across WSL2's processors that [benchmarks.md](benchmarks.md) describes.

## **Tests**

The tests are a pytest suite under `python/tests`, which holds every id to tiktoken 0.14.0's, pip-installed beside the
package, under all four encodings, and the ids of every tokenizer.json read to Hugging Face tokenizers 0.23.3's,
pip-installed beside it too:

```bash
pip install . --group test
MATISSE_VOCAB_DIR=VOCAB MATISSE_CORPUS_DIR=CORPORA python -m pytest python/tests
```

`MATISSE_VOCAB_DIR` names the directory of the pinned vocabulary files, as matisse's CMake option does, and the suite
writes GPT-2's rank file from its encoder.json and vocab.bpe with tiktoken's own conversion, held to r50k_base's digest,
and serves the files to both packages through one cache directory. The directory also holds the pinned tokenizer.json
files that [design.md](design.md) lists, each held to its digest. With `MATISSE_TEST_DOWNLOAD=1` and no directory the
three public rank files are fetched from tiktoken's URLs instead, as the wheels' tests are, and the three public
tokenizer.json files from the Hugging Face Hub at their pinned commits, and Llama 3's tests skip. With neither every
test that needs a vocabulary skips. `MATISSE_CORPUS_DIR` names the corpora of [benchmarks.md](benchmarks.md), and the
tests that read them skip without them.

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
  constructor from ranks, a rank file that is not the pinned one, the calls refused, o200k_harmony and r50k_base against
  tiktoken's, the model table against tiktoken's, and streams in random parts and on threads, each id held to tiktoken's
  `encode_ordinary()`.
- **test_tokenizer_json.py** holds `load_tokenizer_json()` to Hugging Face tokenizers on each pinned tokenizer.json,
  GPT-2's, Xenova's conversions of cl100k and o200k and Llama 3's: the known answers, 1,000 seeded random texts, the
  adversarial texts, special tokens allowed and as text, and the three corpora of the test strategy, there held to the
  vendors' digests as well; and the encoding's name, special tokens, report and pickling, a file of the bytes alone, and
  the refusals, each the loader's message.
- **test_merges.py** holds `validate_merges()` to Hugging Face tokenizers: on six written tables over three letters at
  every window of up to three bytes, each verdict to the covering tokens Hugging Face gives every string up to the
  length past which no refutation is needed and each refutation to Hugging Face's tokens of it, and on GPT-2's pinned
  tokenizer.json with its pre-tokenizer a ByteLevel without its regex at the candidates drawn from twitter.json as
  `matisse-validate` draws them, every certified window's cut at every occurrence in Hugging Face's encoding of the
  whole corpus and every refutation that is UTF-8 to Hugging Face's tokens of it; and its refusals and what a window
  decided carries.
- **test_load.py** holds the fetch and the cache against a server on the loopback interface: tiktoken's cache keys and
  directories, the digest, a cached file that is not the pinned one, the cache turned off, and the request's headers.
- **test_reports.py** holds each shipped report to the report `validate()` decides at the registry's width, and its
  refusal under a pattern of other class symbols.

With every input given, the suite is 233 tests, all passing, in about nine minutes on the machine above, most of them
the Unicode tests; with no vocabulary, 25 pass and 208 skip.
