# **Matisse**

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-23-blue.svg" alt="C++23">
  <img src="https://github.com/nnidhogg/matisse/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://github.com/nnidhogg/matisse/actions/workflows/codeql.yml/badge.svg" alt="CodeQL">
  <img src="https://codecov.io/gh/nnidhogg/matisse/branch/master/graph/badge.svg" alt="Coverage">
  <img src="https://img.shields.io/github/license/nnidhogg/matisse" alt="License">
</p>

`matisse` is a **C++23 library** that reads the definition of a **real tokenizer**, decides where its output can be
**cut with certainty**, and **encodes text in parallel** by cutting only there, with token ids byte-identical to the
sequential encoding. The first version does this for **GPT-2**, **cl100k_base**, **o200k_base** and **Llama 3**, whose
deployed pre-tokenizer patterns have exact verifiers. It is not a general tokenizer library: it trains nothing, applies
no normalizer but the identity, and does not try to match Hugging Face's feature surface. Its product is exactness
against the reference tokenizer, shown on pinned vocabularies and corpora under the Unicode 15.0.0 tables it pins: a
text holding a code point assigned a letter or a number since then can be segmented differently by an engine with newer
tables, as [docs/limits.md](docs/limits.md) describes.

matisse is built on **[`munch`](https://github.com/nnidhogg/munch)**, and the two divide the work by one test: whether a
piece would make sense if no LLM existed. munch keeps the general constructions, the verifier framework and the
decisions on abstract inputs, merge tables and vocabularies among them. Everything about a specific tokenizer is here:
reading its files and special tokens, the vendor pre-tokenizer verifiers tied to a regex and to an engine's Unicode
tables, and the runtime with its token ids, normalizers and special-token cuts.

## **Status**

Not yet released, the version is 0.1.0, pre-1.0 and nothing is stable yet, and matisse depends on munch 3.0.0, the
release its submodule is pinned to. It loads GPT-2's encoder.json and vocab.bpe and tiktoken rank files, and it encodes
text under GPT-2, cl100k_base, o200k_base and Llama 3 sequentially, in parallel at certified cuts and as a stream, with
the vendors' special tokens cut out where they are allowed, and decodes ids back to bytes; validate decides each
pattern's certified windows, and `matisse-validate` reports them for a rank file. The ids are held to the vendors' on
pinned vocabularies, fixtures and corpora, with the classes of code points pinned to Unicode 15.0.0, while a Hugging
Face tokenizer.json reader and the certification of merge tables without a pre-tokenizer are planned and not yet
written. What each library will hold, the proved results the loaders rest on and the order of work are in
[docs/design.md](docs/design.md), and what every call guarantees is in [docs/limits.md](docs/limits.md).

## **Libraries**

Four stage libraries hold nothing of a specific tokenizer and take its description as data:

- **load** reads a tokenizer's files, GPT-2's encoder.json and vocab.bpe or a tiktoken rank file, into one definition
  held in raw bytes, refusing whatever it cannot take exactly.
- **pretok** holds the pre-tokenizers over a pattern description, the ordered alternatives with their languages and
  gates and the class alphabet: the classes of a text, the runtime segmenter, and the construction of the pattern's
  verifier for munch.
- **validate** decides which class windows certify a cut, at which origin and with which right context, and measures how
  many anchors they supply on an input.
- **encode** is the runtime, a sequential, a parallel and a streaming encoder, the parallel one's ids identical at every
  chunk count, and the piece cache a caller keeps across its calls.

Each tokenizer is a small library of its own beside them, holding its pattern description, its Unicode class tables, the
vendor's special tokens and its tests, and a consumer that encodes under one tokenizer links its library, which brings
load and pretok, beside encode, which brings validate, as `matisse::cl100k matisse::encode` for cl100k:

- **gpt2** describes GPT-2's pattern, whose files load reads from encoder.json and vocab.bpe or from a tiktoken rank
  file.
- **cl100k** describes cl100k_base's pattern, whose tiktoken rank file load reads with the special tokens it supplies.
- **o200k** describes o200k_base's pattern over its 30 classes, whose tiktoken rank file load reads with the special
  tokens it supplies.
- **llama3** describes Llama 3's pattern, which is cl100k's as tiktoken shipped it before the possessive rewrite; load
  reads its rank file with the 256 special tokens it supplies.

The **lookup** library finds the tokenizer libraries by name, for the command line and any choice made at run time.

A further tokenizer is added as a new directory under `libs/` with no change to the stage libraries, as
[docs/design.md](docs/design.md) describes.

## **Getting Started**

### **Requirements**

- A C++23 compiler; GCC and Clang on Linux are the toolchains built and tested (GCC 13.3 and Clang 19 in CI), on x86-64
  and 64-bit ARM. Clang 18 and older cannot compile munch's tokenizer, which matisse builds against.
- CMake 3.20.6 or later.
- munch as the git submodule under `external/munch`, checked out at the release matisse builds against; googletest
  beside it for the tests, and Boost's header-only container_hash, which munch fetches and the encoder builds with; an
  installed matisse does not depend on it. Everything else is the standard library.

### **Building the Project**

```bash
git clone --recurse-submodules https://github.com/nnidhogg/matisse
cd matisse
cmake -S . -B build
cmake --build build -j 8
```

The default `CMAKE_BUILD_TYPE` is `Release` when unset.

## **A First Encoding**

A complete program, from a rank file to its ids and back, takes the path of cl100k_base.tiktoken, decides the pattern's
report once, allows `<|endoftext|>` and encodes a text sequentially and in four chunks:

```cpp
#include <iostream>
#include <string>
#include <vector>

#include <matisse/cl100k/tokenizer.hpp>
#include <matisse/encode/encoder.hpp>
#include <matisse/load/definition.hpp>
#include <matisse/validate/report.hpp>

using namespace matisse;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: first cl100k_base.tiktoken\n";

        return 1;
    }

    // The rank file names neither its pattern nor its special tokens, so the cl100k library supplies both.
    auto definition{load::read_tiktoken(argv[1], cl100k::specials())};

    // Decided once per pattern, in about two seconds for cl100k at the default width of five; the report holds the
    // pattern, which the encoder segments by.
    const auto report{validate::validate(cl100k::pattern())};

    const std::vector<std::string> allowed{"<|endoftext|>"};

    const encode::Encoder encoder{std::move(definition), report, validate::Inventory::factorization, allowed};

    const std::string text{"Hello, world!<|endoftext|>"};

    for (const auto id : encoder.encode(text))
    {
        std::cout << id << ' ';
    }

    std::cout << '\n' << encoder.decode(encoder.encode_parallel(text, 4)) << '\n';

    return 0;
}
```

It prints `9906 11 1917 0 100257`, the special token's id last, and then the text again, decoded from the parallel ids,
and a CMake project builds it by linking the cl100k library and encode:

```cmake
add_subdirectory(matisse)
add_executable(first first.cpp)
target_link_libraries(first PRIVATE matisse::cl100k matisse::encode)
```

## **Documentation**

- [docs/design.md](docs/design.md): what matisse is and is not, the repository and its build, the sketched interfaces of
  the libraries, how a tokenizer is added, the loaders' proved results, how the vendor verifiers move from their
  reference into C++, the test strategy against the real tokenizers, what matisse needs from munch 3.0.0, the open
  choices, and the order of work.
- [docs/limits.md](docs/limits.md): the library's scope, what every call guarantees, and the escape hatches.
- [docs/benchmarks.md](docs/benchmarks.md): the ids matisse gives on fourteen corpora beside tiktoken, Hugging Face
  tokenizers and the bpe crate, how fast each encodes on one core, on short prompts and in parallel, the profile of
  matisse's encoding, and the commands that reproduce them.

The theory matisse builds on is in munch's technical reports, listed in the [munch
README](https://github.com/nnidhogg/munch) by their arXiv identifiers:
[arXiv:2608.03473](https://arxiv.org/abs/2608.03473), [arXiv:2608.09761](https://arxiv.org/abs/2608.09761),
[arXiv:2609.10600](https://arxiv.org/abs/2609.10600), [arXiv:2610.08854](https://arxiv.org/abs/2610.08854), which holds
the verifiers matisse's certificates are decided on, and [arXiv:2610.30168](https://arxiv.org/abs/2610.30168), which
brings every tokenizer with a finite verifier, byte-pair dictionaries among them, under the same decisions.

## **Testing**

Each library under `libs/` has a GoogleTest suite in a `tests/` subdirectory, registered with CTest:

```bash
ctest --test-dir build --timeout 300 --output-on-failure
```

Tests and warnings-as-errors are enabled by default only when matisse is the top-level project; a build consuming
matisse through `add_subdirectory` opts in with `-DMATISSE_BUILD_TESTS=ON` or `-DMATISSE_WERROR=ON`.

The vocabulary files the loaders' and the encoder's tests read, GPT-2's gpt2-vocab.json and gpt2-merges.txt,
cl100k_base.tiktoken, o200k_base.tiktoken and llama3.tiktoken, are not in the tree. A directory holding them is named
with `-DMATISSE_VOCAB_DIR=DIR`, each file is held to the SHA-256 docs/design.md gives, and the tests that need them skip
when no directory is named; nothing is fetched at test time.

The corpus probe and the bench in `tools/probes` register with CTest when the vocabulary directory is named, the probe
on generated texts and the bench for one pass on a sample. `-DMATISSE_CORPUS_DIR=DIR` names a directory of the corpora
of [docs/benchmarks.md](docs/benchmarks.md), which are not in the tree either and which the page says how to build: it
adds the probe on the three corpora of the test strategy, and each tokenizer's tests hold matisse's ids of every corpus
the directory holds to the vendors', whose digests libs/load/tests/data/vendors_reference.txt records, skipping by name
the corpora the directory lacks.

## **Versioning and Stability**

matisse follows semantic versioning from 1.0.0 on, munch's rule: a minor release adds and never removes or renames on
the stable surface, and a major release is the only place a name disappears. The stable surface is the following, under
the contracts their headers and [docs/limits.md](docs/limits.md) state. It holds load's `Token_id_t`, `Merge_rule`,
`Special_token`, `Definition`, `read_gpt2()` and `read_tiktoken()`; pretok's `Code_point_set`, `Alphabet`,
`Alternative`, `Pattern`, `Class_t`, `Class_table` with its constructor, `symbols()` and `classes()`, `classes_of()`,
`Segmenter` with its constructor and `boundaries()`, `split_verifier()` and `factorization_lexer()`; validate's
`Inventory`, `default_width`, `Certified_window`, `Window_table`, `Report`, `validate()`, `render()`, `json()`, `Gaps`,
`Anchors`, `Supply`, `supply()`, `supply_json()` and `supply_section()`; encode's `Encoder` with its constructor, both
`encode()`, `chunk_boundaries()`, `encode_parallel()`, `decode()`, `definition()` and `allowed()`, `Piece_cache` with
its constructor, the bounds `piece_cache_capacity` and `longest_cached_piece`, and `Stream_encoder` with its
constructor, `push()` and `finish()`; each tokenizer library's `pattern()` and `specials()`; lookup's `Tokenizer`,
`tokenizers()` and `tokenizer()`; and the `matisse-validate` command line with its JSON document, as munch puts
`munch-audit` on its surface. The class tables are pinned to Unicode 15.0.0, as [docs/limits.md](docs/limits.md)
describes, and moving the pin changes the ids of some texts, those holding a code point whose class differs between the
two versions, so the pin moves only in a major release.

The classes the encoders are built from, `Merger`, `Backtracker`, `Join_table`, `Spelling_table`, `Token_trie`,
`Window_index`, `Verifier_table` and `Live_runs`, and pretok's `automata` kit, which builds the verifier, are installed
only because every header of a library lives under include, as munch keeps none under src; they serve the pipeline and
may change in minor releases. So may the five accessors through which an `Encoder` serves the streaming encoder,
`merger()`, `class_table()`, `verifier()`, `spellings_sorted()` and `unformed_sorted()`, the `serves()`, `find()` and
`store()` through which a `Piece_cache` serves the encoders, and validate's `json_string()` and `plural()`, which serve
the renderings. The corpus probe and the bench under `tools/probes` and the prose in `docs/` carry no compatibility
promise, and performance numbers are measurements, not contracts.

**Targets and headers.** The surface includes how a consumer reaches it: the CMake targets `matisse::load`,
`matisse::pretok`, `matisse::validate`, `matisse::encode`, `matisse::gpt2`, `matisse::cl100k`, `matisse::o200k`,
`matisse::llama3` and `matisse::lookup`, with the umbrella `matisse::matisse` that links them all, the same names under
`add_subdirectory` and in an installed package, and the include paths of the headers that declare the surface,
`matisse/load/definition.hpp`, `matisse/pretok/pattern.hpp`, `classes.hpp`, `segmenter.hpp` and `verifier.hpp`,
`matisse/validate/report.hpp` and `supply.hpp`, `matisse/encode/encoder.hpp`, `piece_cache.hpp` and
`stream_encoder.hpp`, each tokenizer's `matisse/<name>/tokenizer.hpp` and `matisse/lookup/tokenizers.hpp`. A target is
renamed or a header moved only in a major release, as munch's 3.0.0 moved its verifier to a library and an include path
of its own.

**Copies and moves.** An `Encoder` is copied and moved and never assigned to, both assignments deleted, since nothing
changes it after construction: a copy is an encoder of its own, which refuses a piece cache made for the original. A
`Report` is copied and moved whole, a move copying its `const` members, and never assigned to, so it keeps the pattern
its window tables were decided of. A `Piece_cache` and a `Stream_encoder` borrow their encoder, are copied and moved and
are not assigned to: a copy borrows the same encoder, a copied cache holds the same pieces, and a copied stream reads on
from where the original stood, apart from it. A cache or a stream reads the encoder object it was made for, so that
encoder must outlive it and must not be moved while it is in use. An `Encoder`, a `Piece_cache` or a `Stream_encoder`
that has been moved from may only be destroyed: none of the three can be assigned a fresh value, and no other call on
one is defined.

**Errors.** A function refuses content it cannot take with `std::invalid_argument`, the message naming what it refuses,
among them ill-formed UTF-8 at its byte offset, a definition the merge cannot take, a special token the definition does
not hold, whose spelling is empty or whose id is an ordinary token's, two special tokens that share a spelling or an id,
a special token whose spelling is not UTF-8, a pattern that is not total and a piece cache made for another encoder.
`decode()` refuses an id that names no token with `std::out_of_range`, naming the id and its index, as munch's
`Mode_lexer` refuses an index that names no mode. A file that cannot be read is refused with `std::runtime_error`, and a
piece of 2^32 bytes or longer with `std::length_error` naming its length, a stream counting only the bytes of a piece
after the ids it has returned, together with the rest of a token that the test of a boundary merges after them. Nothing
is repaired or guessed, as [docs/limits.md](docs/limits.md) describes.

**Platforms.** The promise is munch's: 64-bit Linux with GCC 13 or later and Clang 19 or later, source compatibility
within a major version and no ABI promise. CI builds and tests on x86-64 and ARM64 and runs the address,
undefined-behavior and thread sanitizers on x86-64. The munch submodule is pinned to a release, 3.0.0 at this version,
and a matisse release names the munch release it was built and tested against; the `Segmenter` runs a
`munch::dfa::Simulator`, from the layer munch keeps outside its stable surface, so matisse's promise holds with the
munch release it names.

## **License**

MIT, see [LICENSE](LICENSE). The generated Unicode class tables derive from the Unicode Character Database and are used
under the Unicode License v3; the complete notice is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), installed
alongside the package.

## **Author**

Developed and maintained by **Nicklas Nidhögg**, [nnidhogg](https://github.com/nnidhogg) on GitHub.
