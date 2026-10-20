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
sequential encoding. The first version does this for **GPT-2** and **cl100k_base**, whose deployed pre-tokenizer
patterns have exact verifiers, and for a byte-level tokenizer without a pre-tokenizer it reports the certified windows
of the merge table through the lazy certification. It is not a general tokenizer library: it trains nothing, applies no
normalizer but the identity, and does not try to match Hugging Face's feature surface. Its product is exactness against
the reference tokenizer, shown on pinned vocabularies and corpora.

matisse is built on **[`munch`](https://github.com/nnidhogg/munch)**, and the two divide the work by one test: whether a
piece would make sense if no LLM existed. munch keeps the general constructions, the verifier framework and the
decisions on abstract inputs, merge tables and vocabularies among them. Everything about a specific tokenizer is here:
reading its files and special tokens, the vendor pre-tokenizer verifiers tied to a regex and to an engine's Unicode
tables, and the runtime with its token ids, normalizers and special-token cuts.

## **Status**

Not yet released; matisse depends on munch 3.0.0, the release its submodule is pinned to. load reads GPT-2's
encoder.json and vocab.bpe and tiktoken rank files, and pretok maps a text to each pattern's classes, from tables pinned
to Unicode 15.0.0, the version of the reference the fixtures are written by, segments it as the pattern's engine does,
and builds the pattern's split verifier and factorization lexer, the automata the policy and factorization certificates
are decided on; validate decides each pattern's certified windows to width five under both certificates, with the right
context each inventory needs, measures each inventory's anchor supply on an input and renders its report as text and as
JSON; gpt2 and cl100k describe their patterns and lookup finds them by name; `matisse-validate` reports a tiktoken rank
file read under a pattern, with the supply on an input when one is given; encode encodes a text and decodes its ids
under both merge rules, with the whole-piece lookup, the special-token cuts and a bounded cache of merged pieces, its
merge held to the plain references on every multi-byte token of four vocabularies and its ids to the fixtures' known
answers, and encodes it in parallel, cut only at the certified cuts of either inventory, its ids held to the sequential
encoding's on seeded texts at one to sixteen chunks, at sixty-four and at every cut, while the tokenizer.json reader and
the lazy certification of merge tables are still to be written. The version is 0.1.0, pre-1.0, and nothing is stable
yet. What each library will hold, the proved results the loaders rest on and the order of work are in
[docs/design.md](docs/design.md).

## **Libraries**

Four stage libraries hold nothing of a specific tokenizer and take its description as data:

- **load** reads a tokenizer's files, a Hugging Face tokenizer.json, GPT-2's encoder.json and vocab.bpe, or a tiktoken
  rank file, into one definition held in raw bytes, refusing whatever it cannot take exactly.
- **pretok** holds the pre-tokenizers over a pattern description, the ordered alternatives with their languages and
  gates and the class alphabet: the classes of a text, the runtime segmenter, and the construction of the pattern's
  verifier for munch.
- **validate** decides which class windows certify a cut, at which origin and with which right context, and measures how
  many anchors they supply on an input.
- **encode** is the runtime, a sequential and a parallel encoder whose ids are identical at every chunk count.

Each tokenizer is a small library of its own beside them, holding its pattern description, its Unicode class tables, the
vendor's special tokens and its tests, and a consumer of one tokenizer links its library alone:

- **gpt2** describes GPT-2's pattern, whose files load reads from encoder.json and vocab.bpe or from a tiktoken rank
  file.
- **cl100k** describes cl100k_base's pattern, whose tiktoken rank file load reads with the special tokens it supplies.

The **lookup** library finds the tokenizer libraries by name, for the command line and any choice made at run time.

A further tokenizer, o200k or Llama 3, is added as a new directory under `libs/` with no change to the stage libraries,
as [docs/design.md](docs/design.md) describes.

## **Getting Started**

### **Requirements**

- A C++23 compiler; GCC and Clang on Linux are the toolchains built and tested (GCC 13.3 and Clang 19 in CI), on x86-64
  and 64-bit ARM. Clang 18 and older cannot compile munch's tokenizer, which matisse builds against.
- CMake 3.20.6 or later.
- munch as the git submodule under `external/munch`, checked out at the release matisse builds against; googletest
  beside it for the tests. Everything else is the standard library.

### **Building the Project**

```bash
git clone --recurse-submodules https://github.com/nnidhogg/matisse
cd matisse
cmake -S . -B build
cmake --build build -j 8
```

The default `CMAKE_BUILD_TYPE` is `Release` when unset.

## **Documentation**

- [docs/design.md](docs/design.md): what matisse is and is not, the repository and its build, the sketched interfaces of
  the libraries, how a tokenizer is added, the loaders' proved results, how the vendor verifiers move from their
  reference into C++, the test strategy against the real tokenizers, what matisse needs from munch 3.0.0, the open
  choices, and the order of work.

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
cl100k_base.tiktoken, and for the merge's tests also o200k_base.tiktoken and llama3.tiktoken, are not in the tree. A
directory holding them is named with `-DMATISSE_VOCAB_DIR=DIR`, each file is held to the SHA-256 docs/design.md gives,
and the tests that need them skip when no directory is named; nothing is fetched at test time.

## **License**

MIT, see [LICENSE](LICENSE). The generated Unicode class tables derive from the Unicode Character Database and are used
under the Unicode License v3; the complete notice is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), installed
alongside the package.

## **Author**

Developed and maintained by **Nicklas Nidhögg**, [nnidhogg](https://github.com/nnidhogg) on GitHub.
