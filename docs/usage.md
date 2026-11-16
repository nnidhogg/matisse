# **Usage Overview**

How matisse is used: consuming it from CMake, loading a tokenizer's definition, choosing a tokenizer by its library or
by name, validating its pattern, encoding a text sequentially, with a kept cache and in parallel, streaming it,
describing a pattern of your own, and the `matisse-validate` command line. The [README](../README.md) shows one complete
program, [design.md](design.md) says why the libraries are shaped as they are, and [limits.md](limits.md) states what
every call guarantees.

## **Integrating with CMake**

```cmake
add_subdirectory(external/matisse)

target_link_libraries(your_target PRIVATE matisse::cl100k matisse::encode)   # one tokenizer and the encoder
```

A tokenizer library, `matisse::gpt2`, `matisse::cl100k`, `matisse::o200k` or `matisse::llama3`, brings load and pretok
with it, and `matisse::encode` brings validate. `matisse::lookup` links every tokenizer library, for a choice made at
run time, and `matisse::matisse` carries all of them. To install matisse, build it against an installed munch rather
than the submodule, `-DMATISSE_SYSTEM_MUNCH=ON -DMATISSE_INSTALL=ON`, so the exported targets refer to munch's own
installed package; the libraries, headers, `matisse-validate` and a package config then install under the usual prefix,
and a consumer writes `find_package(matisse)` and links the same `matisse::` names.

The build takes these options:

- `MATISSE_BUILD_TESTS` builds the tests, and `MATISSE_WERROR` treats warnings as errors; both are on only when matisse
  is the top-level project. `USE_SYSTEM_GTEST` finds an installed googletest, 1.12 or later, instead of fetching it.
- `MATISSE_VOCAB_DIR=DIR` names a directory holding the pinned vocabulary files, each held to the digest
  [design.md](design.md) gives, and the tests that read them skip when it is not given. `MATISSE_CORPUS_DIR=DIR` names a
  directory of the corpora of [benchmarks.md](benchmarks.md), whose ids the tokenizers' tests then hold to the vendors'.
- `MATISSE_BUILD_FUZZER` builds the libFuzzer harnesses, with Clang, in a build tree of their own.
- `MATISSE_SYSTEM_MUNCH` and `MATISSE_INSTALL`, above.

## **Loading a definition**

```cpp
#include <matisse/cl100k/tokenizer.hpp>
#include <matisse/llama3/tokenizer.hpp>
#include <matisse/load/definition.hpp>

// A rank file names neither its pattern nor its special tokens, so the tokenizer's library supplies them.
auto cl100k{matisse::load::read_tiktoken("cl100k_base.tiktoken", matisse::cl100k::specials())};

// GPT-2's two files, as openai/gpt-2's encoder.py reads them.
auto gpt2{matisse::load::read_gpt2("encoder.json", "vocab.bpe")};

// A tokenizer.json, its pre-tokenizer recognized by the published spelling of the pattern it is read under.
auto llama3{matisse::load::read_tokenizer_json("tokenizer.json", matisse::llama3::spelling())};
```

A `load::Definition` holds a tokenizer in raw bytes, and its ids are `load::Token_id_t`, 32 bits. Its members are:

- `merge_rule`, `Merge_rule::pairs` or `Merge_rule::ranks`;
- `whole_piece_lookup`, whether a piece that is itself a token is looked up before any merge;
- `spellings`, the bytes of the ordinary tokens by id, dense from zero;
- `merges`, the merge table in rank order under the pair rule, empty under the rank rule;
- `specials`, each a `Special_token` of a spelling and an id.

`read_tiktoken(path, specials)` reads a tiktoken rank file, one base64 token and its rank per line, with the special
tokens the caller gives, by id ascending. Its definition has the rank rule and the whole-piece lookup, as tiktoken has.
`read_gpt2(encoder, vocab)` reads GPT-2's encoder.json and vocab.bpe, undoing GPT-2's byte-to-character map, so `"Ġthe"`
is read as `" the"`. Its definition has the pair rule, no whole-piece lookup and no special tokens, since encoder.py
reads `<|endoftext|>` as text.

`read_tokenizer_json(path, published)` reads a Hugging Face tokenizer.json with a BPE model, its vocabulary and merges
with GPT-2's byte map undone and its special added tokens as the special tokens. It takes the published spelling of the
pattern the file is read under, as the tokenizer's library gives it with `spelling()`, and recognizes the file's
pre-tokenizer only by it: GPT-2's ByteLevel under `gpt2::spelling()`, a Split on the spelling followed by a ByteLevel
without its regex, which Llama 3's file and Xenova's conversions of cl100k and o200k carry, and a ByteLevel without its
regex alone under the empty spelling. Xenova's cl100k conversion carries Llama 3's pattern, so it is read under
`llama3::spelling()`, and `cl100k::spelling()` reads no file, since Hugging Face's engine reads cl100k's published
pattern otherwise. Its definition has the pair rule, Hugging Face's own, and the whole-piece lookup the model's
`ignore_merges` gives, and its ids are those Hugging Face's `encode()` gives with `add_special_tokens` false.
`tokenizer_json_spelling(path)` returns the spelling a file's pre-tokenizer runs, for a caller that chooses the
tokenizer by the file.

All three refuse content they cannot take exactly with `std::invalid_argument`, naming the file and the line, the byte
offset, the token, the merge or the member, and a file they cannot read with `std::runtime_error`. `read_tiktoken()`
refuses a line that is not a base64 token and a decimal rank, ranks that are not dense from zero, and two tokens with
one spelling. `read_gpt2()` refuses, among others, ids that are not dense from zero, an empty token, two tokens with one
spelling, a pair listed twice, and a table on which encoder.py's pass and one join at a time could differ, as
[limits.md](limits.md) describes. `read_tokenizer_json()` refuses every model, normalizer, pre-tokenizer, post-processor
and added token it does not take, as [design.md](design.md) lists them, naming the member by its path from the root,
such as `pre_tokenizer.pretokenizers[0].pattern.Regex`, what it holds and what the reader takes there.

## **Choosing a tokenizer**

Each tokenizer library declares three functions in `matisse/<name>/tokenizer.hpp`: `pattern()`, the description of the
vendor's pre-tokenizer as a `pretok::Pattern`, `spelling()`, the pattern as its vendor publishes it, which
`read_tokenizer_json()` takes, and `specials()`, the vendor's special tokens of a rank file read under it. A program
that knows its tokenizer calls them directly, as the README's program calls `cl100k::pattern()` and
`cl100k::specials()`. A program that chooses at run time finds the tokenizer by name:

```cpp
#include <matisse/lookup/tokenizers.hpp>

const auto tokenizer{matisse::lookup::tokenizer("o200k")};

if (!tokenizer)
{
    // Not one of gpt2, cl100k, o200k and llama3.
}

auto definition{matisse::load::read_tiktoken("o200k_base.tiktoken", tokenizer->specials)};
```

`lookup::tokenizer(name)` returns a `lookup::Tokenizer`, its `name`, `pattern`, `spelling` and `specials`, or
`std::nullopt` for a name it does not know, and `lookup::tokenizers()` lists all four.

## **Validating a pattern**

```cpp
#include <iostream>

#include <matisse/validate/report.hpp>
#include <matisse/validate/supply.hpp>

const auto report{matisse::validate::validate(matisse::cl100k::pattern())};

std::cout << matisse::validate::render(report);
```

`validate::validate(pattern, width)` decides every class window of width two to `width`, `validate::default_width`
(five) unless given, under both certificates, and returns a `validate::Report`. It takes about a second for GPT-2 and
two for cl100k at width five, and o200k's 30 classes take minutes there, so matisse's own tests and probes decide
o200k's report to width three. Decide a report once per pattern, since it serves every encoder built from it.

A report holds the `factorization` and the `policy` window tables, the boundary `gap` of the pattern's verifier, and the
`pattern` it was decided of. A `Window_table` holds its `inventory`, the `width` decided, the certified `windows`, each
a `Certified_window` of class symbols and the `origin` at which it places a cut, the `right_context` the inventory's
cuts need, in code points, and a `failing` truncation at every smaller context. A report is made only by `validate()`
and by `deserialize()`, which reads back the text `serialize()` wrote of one; its members are `const`, and it is copied
and moved whole and never assigned to. `render(report)` writes it as text and `json(report)` as one JSON object.

`validate::supply(report, inventory, input)` measures how many anchors one inventory finds on an input: a `Supply`
holding the input's `bytes`, the `anchors` with their count, the count per kibibyte and the `gaps` between them (median,
ninetieth and ninety-ninth percentile and longest, absent when fewer than two anchors leave none), the `boundary_share`
of the serial pre-token boundaries that are anchors, and the `longest_piece`. `supply_section()` and `supply_json()`
render the supply of both inventories on one input as the report's section and as JSON.

## **Validating a merge table without a pre-tokenizer**

```cpp
#include <iostream>
#include <string>
#include <vector>

#include <matisse/load/definition.hpp>
#include <matisse/validate/merges.hpp>

// A tokenizer.json whose pre-tokenizer is a ByteLevel without its regex is read under the empty spelling.
const auto definition{matisse::load::read_tokenizer_json("tokenizer.json", "")};

const std::vector<std::string> candidates{" ", "e,", ": ", "the"};

const auto windows{matisse::validate::validate_merges(definition, candidates)};

std::cout << matisse::validate::render_merges(windows);
```

A merge table with no pre-tokenizer merges a whole text as one piece, so it has no pattern, no report and no encoder,
and its cuts are the windows its merge table certifies. `validate::validate_merges(definition, candidates)` builds
munch's token automaton of the table once and decides each candidate window by `miscovering()` on it, under the pair
rule, and returns each as a `validate::Merge_window` in the order given: its `window`, the `origin` it was decided at,
where the window's own encoding begins the token covering its last byte, the only origin that can be certified, and its
`refutation`, `std::nullopt` when the window is certified and otherwise a `munch::verifier::Miscovering`, the encoder's
segmentation of a string with an occurrence of the window whose covering token begins elsewhere. On GPT-2's table the
automaton is built in 0.03 seconds and a decision takes about half a millisecond on average on a window of three bytes,
so decide a table's candidates in one call. It refuses with `std::invalid_argument` a definition under the rank rule or
with the whole-piece lookup, an empty candidate, and a table munch's `token_automaton()` refuses, with munch's message.
`render_merges(windows)` writes the windows decided as text, the certified and then the refuted ones, and
`merges_json(windows)` as one JSON object.

## **Encoding**

```cpp
#include <matisse/encode/encoder.hpp>
#include <matisse/encode/piece_cache.hpp>

const std::vector<std::string> allowed{"<|endoftext|>"};

const matisse::encode::Encoder encoder{std::move(cl100k), report, matisse::validate::Inventory::factorization, allowed};

const auto ids{encoder.encode("Hello, world!<|endoftext|>")};

// One cache per thread, kept across that thread's calls.
matisse::encode::Piece_cache cache{encoder, matisse::encode::piece_cache_capacity};

for (const auto& prompt : prompts)
{
    const auto prompt_ids{encoder.encode(prompt, cache)};
}

const auto parallel_ids{encoder.encode_parallel(text, 8)};

const auto bytes{encoder.decode(ids)};
```

The `encode::Encoder` constructor takes the definition by value, the report decided of its pattern, the inventory whose
cuts the parallel encoding takes, and the spellings of the special tokens to cut out of a text. Every other special
token is ordinary text. It refuses an allowed spelling that is not one of the definition's special tokens and a
definition the merge cannot take. No call changes an encoder, so one encoder serves any number of threads. An encoder is
copied and moved and never assigned to.

- `encode(text)` returns the ids of a text, as the vendor's tokenizer encodes it with those special tokens allowed,
  Llama 3's being tiktoken's encoding built from Meta's rank file, pattern and special tokens, as Meta's tokenizer.py
  builds it. It builds a piece cache for the call.
- `encode(text, cache)` encodes with a `Piece_cache` the caller keeps across its calls, made for this encoder with the
  number of pieces it holds before it stops filling, where `piece_cache_capacity` is the bound the encoder's own caches
  use. A cache has no lock, so keep one per thread. It pays on short texts, as [limits.md](limits.md) measures.
- `chunk_boundaries(text, chunks)` returns where `encode_parallel()` cuts a text, offsets from zero to the text's size,
  aiming for `chunks` chunks of equal bytes; fewer result when certified cuts are scarce, and zero behaves as one.
- `encode_parallel(text, chunks)` encodes the chunks of `chunk_boundaries()` on as many threads, the last on the calling
  thread, and returns the ids `encode()` returns.
- `decode(ids)` returns the bytes a run of ids spells, raw and not checked to be UTF-8, and refuses an id that is
  neither an ordinary nor a special token with `std::out_of_range`.
- `definition()` and `allowed()` return the definition and the special tokens allowed.

A text must be well-formed UTF-8, and an ill-formed sequence is refused with `std::invalid_argument` naming its byte
offset. A piece of 2^32 bytes or longer is refused with `std::length_error`.

## **Streaming**

```cpp
#include <matisse/encode/stream_encoder.hpp>

matisse::encode::Stream_encoder stream{encoder};

for (const auto& part : parts)
{
    for (const auto id : stream.push(part))
    {
        // Every id returned is final: no later byte can change it.
    }
}

const auto rest{stream.finish()};
```

A `Stream_encoder` borrows an encoder, which must outlive it and must not be moved while the stream reads it, and allows
the encoder's special tokens. `push(bytes)` reads the next bytes and returns the ids that have become final, and
`finish()` ends the text and returns the rest, after which the stream reads a new text. The ids returned between two
`finish()` calls, concatenated, are `encode()` of the bytes pushed between them. A push may end inside a UTF-8 sequence
or inside a special token's spelling, and those bytes are held until a later push completes them. A stream serves one
thread at a time, and any number of streams may read over one encoder at once. [limits.md](limits.md) gives how far a
stream's ids lag and what it keeps.

## **Describing a pattern**

A tokenizer library's `pattern()` builds a `pretok::Pattern`, and a new tokenizer, added as [design.md](design.md)
describes, builds one the same way. A pattern holds an `Alphabet`, the class `symbols` ascending, the code point `sets`,
each a `Code_point_set` of a `symbol` and its `ranges`, the `other` symbol of every code point in no set and the
`unicode_version` of the tables, and its `alternatives` in the order of the published pattern, each an `Alternative` of
a `language`, a `munch::regex::Regex` over the class symbols, and a `gate`, the class symbols that may follow an accept.

pretok also gives the pieces the stages are built from, for a caller that wants them alone:

- `Class_table`, built from an alphabet, whose `symbols(text)` is a text's class string and `classes(text)` its classes;
  `classes_of(alphabet, text)` is `classes()` on a table built for the one text.
- `Segmenter`, built from a total pattern, whose `boundaries(text)` returns where the pattern ends each piece, as byte
  offsets.
- `split_verifier(pattern)`, the pattern's verifier as a `munch::verifier::Verifier`, which refuses a pattern that is
  not total and names a shortest residual no alternative matches, and `factorization_lexer(pattern)`, the gate-erased
  union of the alternatives as a `munch::core::Lexer`.

## **The command line**

`matisse-validate` reports a tokenizer.json or a rank file read under a known pattern, or the windows of a merge table
without a pre-tokenizer:

```
matisse-validate [--tiktoken PATTERN] [--windows N] [--input FILE] [--candidates N] [--json] FILE
```

`--tiktoken` names the pattern of the rank file, gpt2, cl100k, o200k or llama3, since the file names none, and the
pattern and the vendor's special tokens are found through `lookup::tokenizer()`. A file named without it is a
tokenizer.json, read under the tokenizer whose `spelling` its pre-tokenizer runs, as `tokenizer_json_spelling()` finds
it; a pre-tokenizer whose pattern no tokenizer has is a refusal, as is any content the reader refuses, and the JSON
document's tokenizer is then null. GPT-2 reaches the command line as its tokenizer.json or as a rank file under
`--tiktoken gpt2`, since encoder.json and vocab.bpe are two files. `--windows` is the widest window decided, 2 to 5, 5
unless given. `--input` adds the anchor supply of both inventories on a text. The text report is headed by the file and
the tokenizer it was read as, with its token counts. `--json` writes one document instead, holding the path, the
tokenizer, the refusal or null, the token counts, the report and the supply. The exit status is 0 when the definition
was read and reported, 1 when the loader refused it, the refusal written in place of the report, and 2 on a command-line
error, with the usage, or on a file or an input the command cannot read, an input that is not UTF-8 among them.

A tokenizer.json whose pre-tokenizer is a ByteLevel without its regex has no pattern, and the command decides the
windows of its merge table by `validate_merges()` instead, on candidates drawn from `--input`, which is then required:
every byte the input holds and its most frequent pairs and triples, `--candidates` of each, 1,000 unless given, the most
frequent first and a tie in the order of the bytes. The text is headed by the file, the table's token counts and the
candidates drawn, and `--json` writes the path, a null tokenizer, the refusal or null, the token counts, a null report,
the candidates drawn and the windows decided. A table `validate_merges()` refuses, one with the whole-piece lookup or
one munch's `token_automaton()` refuses, is a refusal, exit status 1, and a merge table named without `--input` a
command-line error. `--windows` is not read for a merge table, and `--candidates` is read for nothing else.
