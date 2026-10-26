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

Not yet released; the version is 0.1.0, pre-1.0, and nothing is stable yet. matisse depends on munch 3.0.0, the release
its submodule is pinned to. It loads GPT-2's encoder.json and vocab.bpe and tiktoken rank files, and it encodes text
under GPT-2, cl100k_base, o200k_base and Llama 3 sequentially, in parallel at certified cuts and as a stream, with the
vendors' special tokens cut out where they are allowed, and decodes ids back to bytes; validate decides each pattern's
certified windows, and `matisse-validate` reports them for a rank file. The ids are held to the vendors' on pinned
vocabularies, fixtures and corpora, with the classes of code points pinned to Unicode 15.0.0. A Hugging Face
tokenizer.json reader and the certification of merge tables without a pre-tokenizer are planned and not written. What
each library will hold, the proved results the loaders rest on and the order of work are in
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
- **encode** is the runtime, a sequential and a parallel encoder whose ids are identical at every chunk count.

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

A complete program, from a rank file to its ids and back. It takes the path of cl100k_base.tiktoken, decides the
pattern's report once, allows `<|endoftext|>` and encodes a text sequentially and in four chunks:

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

    const auto pattern{cl100k::pattern()};

    // Decided once per pattern, in about two seconds for cl100k at the default width of five.
    const auto report{validate::validate(pattern)};

    const std::vector<std::string> allowed{"<|endoftext|>"};

    const encode::Encoder encoder{std::move(definition), report, validate::Inventory::factorization, pattern, allowed};

    const std::string text{"Hello, world!<|endoftext|>"};

    for (const auto id : encoder.encode(text))
    {
        std::cout << id << ' ';
    }

    std::cout << '\n' << encoder.decode(encoder.encode_parallel(text, 4)) << '\n';

    return 0;
}
```

It prints `9906 11 1917 0 100257`, the special token's id last, and then the text again, decoded from the parallel ids.
A CMake project builds it with the cl100k library and encode:

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
[arXiv:2609.10600](https://arxiv.org/abs/2609.10600) and [arXiv:2610.08854](https://arxiv.org/abs/2610.08854), the last
of which holds the verifiers matisse's certificates are decided on.

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
on generated texts and the bench for one pass on a sample, and `-DMATISSE_CORPUS_DIR=DIR` adds the probe on the three
corpora of the test strategy, which are not in the tree either; [docs/benchmarks.md](docs/benchmarks.md) says how they
are built.

## **License**

MIT, see [LICENSE](LICENSE). The generated Unicode class tables derive from the Unicode Character Database and are used
under the Unicode License v3; the complete notice is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), installed
alongside the package.

## **Author**

Developed and maintained by **Nicklas Nidhögg**, [nnidhogg](https://github.com/nnidhogg) on GitHub.
