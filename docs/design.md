# **Design**

The decisions behind matisse, in the order a reader meets them: what the library is and is not, the repository and its
build, the libraries' interfaces, how the vendor verifiers move from their Python reference into C++, how the result is
tested against the real tokenizers, what matisse needs from munch, the choices still open, and the order of work. Steps
1 to 13 of the order of work are built, `read_tokenizer_json()` waiting on open choice 5: load reads GPT-2's files and
tiktoken rank files, pretok classes and segments a text and builds a pattern's split verifier and factorization lexer,
validate decides a pattern's window tables under both certificates with their right contexts, measures their anchor
supply on an input and renders the report as text and as JSON, the gpt2 and cl100k libraries describe their patterns,
lookup finds them, `matisse-validate` reports a rank file read under a pattern, and encode encodes a text and decodes
its ids under both merge rules, with the whole-piece lookup, the special-token cuts, the fast merge and the piece cache,
and encodes it in parallel, cut only at the certified cuts of either inventory, while `validate_merges()` is still to be
written; on the fourteen corpora of [benchmarks.md](benchmarks.md) matisse's ids equal tiktoken's, Hugging Face
tokenizers' and the bpe crate's by digest wherever each of them ran, and the page holds what the corpus probe, the bench
and the other encoders' harnesses measured. The sketches below give names and types, and they change as the code is
written.

## **What matisse is and is not**

matisse reads the definition of a real tokenizer, decides where its output can be cut with certainty, and encodes text
in parallel by cutting only there, with token ids byte-identical to the sequential encoding. The first version does this
for GPT-2 and cl100k_base, whose deployed pre-tokenizer patterns have exact verifiers, and for a byte-level tokenizer
without a pre-tokenizer it reports the certified windows of the merge table through the lazy certification. It is not a
general tokenizer library. It trains nothing, applies no normalizer but the identity, and does not try to match Hugging
Face's feature surface; its product is exactness against the reference tokenizer, shown on pinned vocabularies and
corpora.

munch stays the decision library, with the verifier framework and the general constructions on abstract inputs, merge
tables and vocabularies among them. matisse takes everything about specific real tokenizers, which is reading their
files and special tokens, the vendor pre-tokenizer verifiers tied to a regex and to an engine's Unicode tables, and the
runtime with its token ids, normalizers and special-token cuts. The test for where a piece goes is whether it would make
sense if no LLM existed; if it would, it belongs in munch. matisse is also the intended home of the tokenizer pieces of
an execution-contract compiler (the full-dictionary verifier, the two-mode WordPiece construction, a labeler carrying
token ids, certified-prefix streaming), and none of them is in the first version.

The name is after Matisse's paper cut-outs, cutting with certainty, beside munch, the lexer library, and hopper, the
parser toolkit.

## **Repository and build**

The repository is laid out as hopper's is, one library per subject under libs/, each with include/, src/ and tests/, the
tools beside them, munch as a git submodule and googletest beside it:

```
matisse/
  CMakeLists.txt  CITATION.cff  LICENSE  README.md  THIRD_PARTY_NOTICES.md  .clang-format  .gitignore  .gitmodules
  .github/workflows/  ci.yml  codeql.yml
  cmake/matisse-config.cmake
  docs/  usage.md  how_it_works.md  design.md  benchmarks.md
  external/  CMakeLists.txt  googletest/  munch/          (submodule, pinned to the munch release it builds against)
  libs/
    CMakeLists.txt
    common/     include/matisse/common/                   (the tests' SHA-256, neither installed nor exported)
    load/       include/matisse/load/      src/  tests/  tests/data/
    pretok/     include/matisse/pretok/    src/  tests/  tests/data/
    validate/   include/matisse/validate/  src/  tests/  tests/data/
    encode/     include/matisse/encode/    src/  tests/  tests/data/
    gpt2/       include/matisse/gpt2/      src/  tests/
    cl100k/     include/matisse/cl100k/    src/  tests/
    lookup/     include/matisse/lookup/    src/  tests/
  tools/
    CMakeLists.txt
    validate/   the matisse-validate command line
    probes/     the corpus probe and the bench
    fuzz/       libFuzzer harnesses
    unicode/    generate_classes.py
```

The libraries are of two kinds, as hopper's generic parse library sits beside its json and clike grammars. The four
stage libraries are the four subjects of the first version, and none of them holds a tokenizer's pattern or special
tokens; load's `read_gpt2()` reads the file format GPT-2 published. load reads tokenizer files into one definition type.
pretok holds the pre-tokenizer machinery over a pattern description: the classes of a text, the runtime segmenter, and
the construction of a pattern's verifier. validate decides the certified windows and their right context and measures
anchor supply. encode is the runtime. Beside them every tokenizer is a small library of its own, gpt2 and cl100k in the
first version, holding its pattern description, its generated class tables, the vendor's special tokens for a file that
names none, and its tests, and lookup finds the tokenizer libraries by name. Each is a target `matisse_<name>` with the
alias `matisse::<name>`, in namespace `matisse::<name>`, and the umbrella `matisse::matisse` links all of them, as
`hopper::hopper` links hopper's; a consumer of one tokenizer links its library alone, `matisse::cl100k`, which brings
load and pretok with it. common holds the tests' and the probes' SHA-256 alone, which the tokenizer tests hold the
pinned files and the class tables to, and the probes the corpora and their ids; its target `matisse_common` lives in the
build tree only, with no alias, no install rule and no place in the umbrella.

The top-level CMakeLists.txt is hopper's with the prefix changed: `project(matisse VERSION 0.1.0 LANGUAGES CXX)`, C++23
required, `MATISSE_IS_TOP_LEVEL`, Release by default at the top level only, `MATISSE_SOURCE_DIR` for test data paths,
`-Wall -Wunused -Wextra` with `MATISSE_WERROR` on at the top level, `MATISSE_BUILD_TESTS`, `USE_SYSTEM_GTEST`, and, once
the harnesses exist, `MATISSE_BUILD_FUZZER` instrumenting the whole tree with the submodule included. munch is consumed
exactly as hopper consumes it, as the submodule by default, added by external/CMakeLists.txt, or with
`MATISSE_SYSTEM_MUNCH` an installed munch found by `find_package(munch REQUIRED)` at the top level so that every library
sees its imported targets; `MATISSE_INSTALL` requires the system munch, exports `matisse_targets` under
`lib/cmake/matisse`, and cmake/matisse-config.cmake calls `find_dependency(munch)`. Every library whose headers name
munch types links munch publicly, as hopper_json does: load (its definition holds `munch::verifier::Merge`), pretok (a
pattern's sets are `munch::regex::utf8::Code_point_range` tables, its verifier a `munch::verifier::Verifier`) and
validate (`munch::verifier::Verifier`); a tokenizer library reaches munch through pretok. The version starts at 0.1.0
with "pre-1.0" in the README, as hopper's did.

CI copies hopper's workflow with munch's sanitizer matrix: format-check with clang-format-19, build and test with GCC
13.3 (with coverage) and with Clang 19, an ARM64 job where plain char is unsigned, the address and undefined-behavior
sanitizers and, because the encoder spawns threads, munch's thread-sanitizer job and its `-fprofile-update=atomic`
coverage flag, the fuzz job once the harnesses exist, the installed-consumer job, and CodeQL. The README has hopper's
sections (Status, Features, a first example, Getting Started, Documentation, Testing, Versioning and Stability, License,
Author), docs/ has hopper's three pages, and CITATION.cff names the munch release with its version DOI under
`references`, as hopper's does. Tests run with `ctest --test-dir build --timeout 300`, as munch's do.

## **The API**

Every declaration gets munch's full doc block when written. Aggregates are built with designated initializers, results
are returned rather than written through out-parameters, and there are no raw pointers. Refusals follow munch. Content a
function cannot take throws `std::invalid_argument` with a message naming the item and where it is, as `merge_tower()`
names a merge and its rank, and a file that cannot be read throws `std::runtime_error{"cannot read <path>"}`, as
munch-audit's main does.

### **Loaders, matisse::load**

```cpp
namespace matisse::load
{
using Token_id_t = std::uint32_t;

enum class Merge_rule : std::uint8_t
{
    pairs,
    ranks
};

struct Special_token
{
    std::string spelling{};

    Token_id_t id{};
};

struct Definition
{
    Merge_rule merge_rule{};

    bool whole_piece_lookup{};

    std::vector<std::string> spellings{};

    std::vector<munch::verifier::Merge> merges{};

    std::vector<Special_token> specials{};
};

[[nodiscard]] Definition read_tokenizer_json(const std::filesystem::path& path, std::string_view published);

[[nodiscard]] Definition read_gpt2(const std::filesystem::path& encoder, const std::filesystem::path& vocab);

[[nodiscard]] Definition read_tiktoken(const std::filesystem::path& path, std::span<const Special_token> specials);

} // namespace matisse::load
```

A `Definition` is held in raw bytes throughout, and it names no pattern: the pattern is the tokenizer library's
description, which the stages take beside the definition. `spellings` gives the bytes of every ordinary token by id,
dense from zero; GPT-2's byte-to-character map is undone by the loader, so `"Ġthe"` is read as `" the"`. `merges` is the
merge table in rank order under `Merge_rule::pairs` and empty under `Merge_rule::ranks` (tiktoken's rank merge, which
joins the adjacent pair whose concatenation has the least rank, leftmost first, and needs only the ranks). Settled:
`Merge_rule::pairs` is Hugging Face's rule, one join at a time, the adjacent pair of least rank at its leftmost
occurrence, which is its BPE model's `merge_all`. encoder.py's loop joins every occurrence of the least pair in one pass
instead, and the two give the same tokens on every piece of a table in which no part of a pair is formed by a pair of a
later rank, condition C, each pass then being a run of single joins; without C they can differ, (ab, a) then (a, b)
giving ab ab on abab under the pass and aba b one join at a time, and a tokenizers model with that list loads. The
agreement under C is not a new result: it is Lemma 1 of Berglund and van der Merwe's Formalizing BPE Tokenization (NCMA
2023), the equivalence of its Definition 3, one merge at a time at the leftmost occurrence, and its Definition 4, the
pass, which holds under the ordering condition that clause (b) of their revised Definition 5 (2024) states, which is C.
GPT-2's table satisfies C. `read_gpt2()` checks it once at load and refuses a table outside it, naming the merge, its
rank and its line; it also refuses a pair listed twice, which both vendors would read at its last rank. So on every file
`read_gpt2()` accepts, the pair rule is encoder.py's by that proof. On a tokenizer.json it is Hugging Face's rule
itself, Llama 3's list included, which violates C in 149,956 places. `whole_piece_lookup` says whether a piece that is
itself an ordinary token is that one token, looked up before any merge: tiktoken looks every piece up, so
`read_tiktoken()` sets it; encoder.py never does, so `read_gpt2()` leaves it false; Hugging Face's BPE model does when
its `ignore_merges` is true, so `read_tokenizer_json()` sets it to the file's value, true for Llama 3's and false for
GPT-2's tokenizer.json, where it is absent. It changes nothing on cl100k, o200k or GPT-2, where every multi-byte token
is formed by the merge rule from its own bytes, GPT-2's `<|endoftext|>` aside, which its pattern never leaves as one
piece; it is load-bearing on Llama 3, where 588 of 127,744 multi-byte tokens are not formed from their own bytes.
`Token_id_t` is 32 bits because ids are what a model reads, and every vocabulary in scope fits.

`read_tokenizer_json()` takes a Hugging Face tokenizer.json with a BPE model and the published spelling of the pattern
it is read under, which the tokenizer library's description carries once the reader is written. It recognizes a
pre-tokenizer only by that exact spelling: a ByteLevel pre-tokenizer with `use_regex` true and `add_prefix_space` false
is GPT-2's, and a Split on cl100k's published pattern, isolated and not inverted, followed by ByteLevel without its
regex is cl100k's. An empty spelling reads a null pre-tokenizer, for the lazy route. The `added_tokens` marked special
become `specials`. It refuses, naming the member: a model other than BPE, a normalizer other than null, any other
pre-tokenizer or pattern (the cl100k pattern as tiktoken shipped it before its possessive rewrite is a different
pattern, refused until it has its own verifier), `byte_fallback`, a post-processor that adds tokens, ordinary ids that
are not dense from zero, and an empty token. `read_gpt2()` takes encoder.json and vocab.bpe as encoder.py reads them,
with no special tokens, since encoder.py treats `<|endoftext|>` as text. `read_tiktoken()` takes a rank file, one base64
token and its rank per line; the file names neither its pattern nor its special tokens, so the caller gives the vendor's
special tokens, which the tokenizer library supplies from its own table, quoted from tiktoken's openai_public.py (for
cl100k_base `<|endoftext|>` 100257, `<|fim_prefix|>` 100258, `<|fim_middle|>` 100259, `<|fim_suffix|>` 100260 and
`<|endofprompt|>` 100276, to be held to the pinned source).

The JSON files need a JSON reader; which one is open choice 5.

#### **The SentencePiece loader and its tie-breaking**

The table a SentencePiece BPE model gives munch is the self-split table, not the Hugging Face converter's. For every
piece P, the loader encodes P's own characters with SentencePiece's loop; when the result is P alone, the last pair
merged is P's split, and that split is P's one merge. The merges are ranked by score, highest first, products of equal
score shortest first. On Mistral 7B v0.1 and Yi 6B every product encodes to itself, no product is left out, every part
precedes its product, and munch accepts the table. The converter's table, every split of every piece, is refused by
munch's constructions and must be, since most of its products have several splits and some parts come after their
products.

Proved (the self-split proposition): in SentencePiece's loop on any input, whenever a pair (L, R) is merged into a piece
P, the loop run on P's spelling alone ends with P as one token and merges (L, R) last. Tokens only grow, so the span of
P's characters is a union of tokens throughout the run, and each merge inside it was the loop's choice at its time, the
highest-scored pair and leftmost on a tie, hence also the highest and leftmost among the span's own pairs; by induction
the loop on the span alone makes the same merges in the same order, ties included, since merges outside the span keep
the relative order of the symbols inside it. So SentencePiece only ever forms a piece by its self split, and dropping
every other split changes nothing in its run.

Proved, no ties: when no two products share a score, SentencePiece's rule on the self-split pairs is munch's encoder on
the self-split table, since the highest-scored pair present is the lowest-ranked merge present, and SentencePiece's
leftmost tie between occurrences of one merge is munch's leftmost occurrence. Yi's scores are all distinct, and its
table agrees with the library on 525,864 inputs with no difference.

Mistral's one tie group, a separate compilation result. Mistral scores U+2581 and its runs of 2 to 14 and 16 all at
-1e9, below every other piece, and no other piece holds two U+2581 in a row, so these merges act last and only inside
runs of U+2581, where SentencePiece's position tie grows the leftmost token one character at a time: its self splits are
(k - 1, 1) for the runs of 2 to 14 and (14, 2) for 16. No rank order of those self splits reproduces it, since every
admissible order applies (1, 1) at both occurrences of four spaces and gets stuck at 2 and 2. A search over the
admissible tables of the group, splits and ranks both, found one in 50,017 nodes, the runs joined being 1+1, 2+2, 4+4,
4+1, 4+2, 2+1, 8+8, 8+1, 8+2, 8+3, 8+4, 8+5, 8+6 and 6+1: a split-and-rank compilation, not the self-split table (four
spaces form as 2+2, eight as 4+4, sixteen as 8+8). What is proved of it is output equivalence on every run, by this
argument: after the first six rules a run of length 8q + r consists of q tokens of length 8 followed by a tail of r for
r up to 6 and of 6 then 1 for r = 7; (8, 8) then joins pairs into blocks of 16 from the left, an unmatched 8 absorbs a
tail of 1 to 6 by the (8, k) rules, giving (14, 1) on the tail (6, 1), and the last rule (6, 1) joins a tail of 7 with
no unmatched 8 before it; so a run of 16q + r ends as q sixteens then nothing, [r] for r from 1 to 14, or [14, 1] for r
= 15, which is SentencePiece's own output, since it grows the leftmost token to 14, merges the next two into 2 and then
(14, 2) into 16, a 16 being a barrier to every further merge, and ends with (14, 1) when one character remains.
Validated on every run of 1 to 4,000 as well. The group acts in isolation only because of the two model facts stated
first, which the loader checks when it compiles a group. With that group compiled and placed at the end of the table,
Mistral agrees with the library on 526,292 inputs with no difference, and certificates decided on the compiled table
hold for SentencePiece's segmentations by the equivalence.

The general tie, refuted. Two equal-scored pieces ab and ba over {a, b} need opposite rule orders: on aba
SentencePiece's position tie gives ab then a, on bab it gives ba then b, and a rank order that puts ab first turns bab
into b then ab, while one that puts ba first turns aba into a then ba. So no rank order reproduces a position tie in
general, and alternative splits cannot help, since two-symbol products have none; three equal-scored products ab, bc and
ca likewise force a cyclic priority on abc, bca and cab, so all six orders fail. What the loader does: it runs the
search over the group's admissible tables, which may change splits, bounded by the group's runs, and refuses a model
whose group admits no output-equivalent table, naming the group. Mistral's group is the proved positive case, ab and ba
the proved negative one. The realizability of a candidate table is decidable exactly: canonical adjacency holds for
every deterministic merge encoder by interval projection, so two encoders are equal exactly when their canonical token
sets and adjacency relations agree, every disagreement having a witness of at most 2L - 1, attained; the loader's search
over admissible tables therefore ends in a complete finite test, at most |V|^2 pairs, and runs to 31 suffice for the
whitespace group because L = 16. What remains is the cost of synthesis, not decidability. And conversion is not a
prerequisite for certification: munch's lazy certification needs only the canonical tokens and the adjacency relation,
which SentencePiece's own encoder supplies as a pair oracle, so a model whose tie group admits no table can still be
certified through that relation once munch takes it directly, a planned munch addition.

The class the loader may claim, proved: canonical adjacency holds for every projective merge process, every key rule
among them (least priority among mergeable pairs, priority any function of the pair, ties by position), SentencePiece's
score-first leftmost loop with score ties included, and the source paper's pass order for every dictionary; it fails for
the count rule and for a parity tie, so "deterministic and memoryless" is not the class. Two encoders of the class are
equal exactly when their canonical tokens and pairs agree, with a shortest witness of at most 2L - 1, so the loader's
acceptance test is the pair test over the straddling candidates (Mistral: 184,745,026 candidate pairs, 18 seconds on 20
threads, 0 pairs in one encoder only; Yi: 495,683,098, 58 seconds, 0 differences on the core, 516 skipped for a
user-defined symbol across the join), and the compiled tables equal SentencePiece's core on every string by that
theorem, no longer by sampling. Certification needs no table at all: the chain search with a pair oracle (encode a
token, encode a pair) decides window certificates for every projective encoder, 15,436 decisions against brute force
with 0 disagreements.

What the loader does beyond the table, each outside munch's construction by the division: the dummy prefix (input
normalization), byte fallback (a wrapper, since munch's character alphabet refuses boundaries inside a character by
design; it contradicted no certificate on Mistral), and user-defined symbols, a longest-match pre-tokenizer composed
with the BPE certificate by the closure calculus (on Yi they override certificates on 1,028,924 corpus occurrences,
every one under a user-defined token). The Python reference implementation of the self-split table, the tie-group
compilation and its search is the loader's model. SentencePiece's source also holds an unused-piece resegmentation
stage, so the loader states its treatment of `UNUSED` pieces before it claims to cover a model, and the result statement
reads: SentencePiece's deterministic BPE core forms every piece by a unique self split; with distinct product scores the
score-ranked self-split table reproduces the core encoder, which is Yi's case; Mistral's tied separator group instead
admits a split-and-rank compilation, proved equivalent by its 16-block structure; not every equal-score group admits
one, ab and ba being a two-product counterexample; representability is decidable by finite enumeration of candidates and
canonical-piece and pair equivalence, and efficient synthesis remains open.

### **Pre-tokenizers, matisse::pretok**

```cpp
namespace matisse::pretok
{
struct Code_point_set
{
    char symbol{};

    std::span<const munch::regex::utf8::Code_point_range> ranges{};
};

struct Alphabet
{
    std::string_view symbols{};

    std::span<const Code_point_set> sets{};

    char other{};

    std::string_view unicode_version{};
};

struct Alternative
{
    std::string_view language{};

    std::string_view gate{};
};

struct Pattern
{
    Alphabet alphabet{};

    std::span<const Alternative> alternatives{};
};

using Class_t = unsigned char;

class Class_table
{
public:
    explicit Class_table(const Alphabet& alphabet);

    [[nodiscard]] std::string symbols(std::string_view text) const;

    [[nodiscard]] std::vector<Class_t> classes(std::string_view text) const;
};

[[nodiscard]] std::vector<Class_t> classes_of(const Alphabet& alphabet, std::string_view text);

class Segmenter
{
public:
    explicit Segmenter(const Pattern& pattern);

    [[nodiscard]] std::vector<std::size_t> boundaries(std::string_view text) const;
};

[[nodiscard]] munch::verifier::Verifier split_verifier(const Pattern& pattern);

[[nodiscard]] munch::core::Lexer factorization_lexer(const Pattern& pattern);

} // namespace matisse::pretok
```

A `Pattern` is the contract of the class theorem of the report on pre-tokenizer certification named below, as data: an
ordered gated pattern over a class alphabet, a list of regular languages of nonempty class strings, each with its gate,
the class symbols that may follow its accepts, the end of the input always passing. Its discipline commits the longest
prefix in an alternative's language whose next symbol passes that alternative's gate, for the first alternative that has
one. The theorem gives the route, the verifier M, both certificates and the splice, for every such pattern that is
total, and carries them to an engine that realizes it: no alternative reads left of where its match begins, each
alternative matches the longest gate-passing prefix of its language and fails only where there is none, and the engine
reads code points only through the classes. Those conditions are facts about the published regex and its engine, which
no description can check; the fixtures, written by the engine, hold each description to them. The alphabet is the class
partition: the class symbols ascending, the code point sets in the order a code point is classed by them, each with its
symbol and its ranges, and the symbol of everything else. A literal's class is a set listed before the Unicode sets,
holding the code points the literal matches, so a case-sensitive letter holds itself, cl100k's caseless `s` holds `S`,
`s` and U+017F, and o200k's classes, which read Lu and Ll apart and put the long s with the lower s alone, are stated
the same way. An alternative is its language written over the class symbols in munch's pattern syntax and its gate as a
string of class symbols: every symbol for an ungated alternative, the white space for `(?!\S)`, and none for an
alternative anchored at the end of the input. Everything the verifier construction reads, the alphabet, the languages
and the gates, is in the description, so split_verifier() and factorization_lexer() take the pattern and nothing else.

`classes_of()` decodes UTF-8 and maps every codepoint to its class under an alphabet, numbered from zero, read from the
Unicode tables the tokenizer library pins (open choice 6). Malformed UTF-8 is refused with its byte offset, since the
model is defined on codepoints and invalid text is outside it. `Class_table` holds the class symbol of every codepoint
under an alphabet, built once from its sets: a table of the ASCII bytes and a table of blocks of 256 codepoints, each
distinct block kept once and every block's index among them, so that a codepoint's symbol is two loads and an ASCII
byte's one; GPT-2's and cl100k's alphabets have 127 and 128 distinct blocks, 37 KB a table. Its `symbols()` is a text's
class string and its `classes()` the classes, and `classes_of()` is `classes()` on a table built for the one text; the
`Segmenter`, the encoder's `Window_index` and `supply()` each build a table of their own. The tokenizer libraries' tests
hold the table to the first set holding every one of the 1,112,064 scalar values. `Segmenter` is the runtime
pre-tokenizer, the gated commitment order of the gate-sufficiency proposition read over codepoints. At each boundary it
commits the longest gate-passing accept of the first alternative that has one, each alternative a small DFA over the
classes, an accept passing when the class symbol after it is in its gate or the input ends there. `boundaries()` returns
the piece ends as byte offsets. `split_verifier()` is the verifier M built from the one-step relation Q of each pattern,
minimized, and handed to munch as a `Verifier` table over the class symbols; the table munch's constructor makes of M
has 21 states for GPT-2 and 32 for cl100k, 22 and 33 counting the dead state. As it builds Q it decides that the pattern
is total, every nonempty residual matched by some alternative, which is the verifier fixture's `covered` decision, and
refuses a pattern that is not, naming a shortest residual no alternative matches; the `Segmenter` takes a total pattern
and does not check it, so the check runs once, where the validator builds the verifier, and never in the segmenter's
loop. `factorization_lexer()` compiles the gate-erased union U of the alternatives over the class symbols as a munch
token set, on which `is_split_window()` decides the factorization certificate.

The construction behind these, the gate-sufficiency proposition, the transfer of a cut from class strings to text and
the splice theorem the encoder rests on are the results of a report on pre-tokenizer certification that is not yet
published.

### **Tokenizers, matisse::gpt2 and matisse::cl100k, and matisse::lookup**

```cpp
namespace matisse::cl100k
{
[[nodiscard]] pretok::Pattern pattern() noexcept;

[[nodiscard]] std::vector<load::Special_token> specials();

} // namespace matisse::cl100k

namespace matisse::lookup
{
struct Tokenizer
{
    std::string_view name{};

    pretok::Pattern pattern{};

    std::vector<load::Special_token> specials{};
};

[[nodiscard]] std::vector<Tokenizer> tokenizers();

[[nodiscard]] std::optional<Tokenizer> tokenizer(std::string_view name);

} // namespace matisse::lookup
```

Each tokenizer library declares the same two functions in `matisse/<name>/tokenizer.hpp`, matisse::gpt2 as
matisse::cl100k does. `pattern()` is the description of the published pattern over its classes, 14 for GPT-2 and 16 for
cl100k, its views and spans pointing into constants of the library's source, among them the Unicode tables generated
into the library's src/class_ranges.inc. `specials()` is the vendor's special tokens of a rank file read under the
pattern, which `read_tiktoken()` takes. A file format of its own would be a reader binding in the library beside them;
GPT-2's encoder.json and vocab.bpe are read by `read_gpt2()`, which needs nothing from the description. The library's
tests hold the description to the classes, segmentation and known-answer fixtures, the pinned vocabulary to its digest,
and the vendor's special tokens to their quoted values.

`tokenizers()` lists every tokenizer library by name, gpt2 and cl100k, and `tokenizer()` finds one by its name, for the
command line and any caller choosing a tokenizer at run time. The list is one function naming the libraries explicitly,
the simplest form that needs no registry: a library registering itself when it is loaded would rest on the order in
which static objects are initialized and on the linker keeping an object nothing names, and munch's style has neither.
The lookup links every tokenizer library, so a consumer of one tokenizer links that library alone and calls it directly.

#### **Adding a tokenizer**

A tokenizer is added as a new directory under libs/, as o200k and Llama 3 will be, and the stage libraries do not
change. libs/o200k/ holds its CMakeLists.txt, `matisse/o200k/tokenizer.hpp` declaring `pattern()` and `specials()`,
src/tokenizer.cpp with the description, src/class_ranges.inc generated by tools/unicode/generate_classes.py with the
sets its pattern names, and its tests, which check the lines of its universe in the shared fixtures once the fixture
script writes one. Outside the directory it is named in four places: its `add_subdirectory()` in libs/CMakeLists.txt,
its entry in `tokenizers()` with the link in libs/lookup/CMakeLists.txt, its alias and umbrella link in the top-level
CMakeLists.txt, and its target and include directory in the install rules. Its pattern must meet the class theorem's
conditions above, which the fixtures, written by the engine, hold it to; where the pattern needs a reader of its own,
the binding goes into the library beside `specials()`.

### **The validator, matisse::validate**

```cpp
namespace matisse::validate
{
enum class Inventory : std::uint8_t
{
    factorization,
    policy
};

inline constexpr std::size_t default_width{5};

struct Certified_window
{
    std::string window{};

    std::size_t origin{};
};

struct Window_table
{
    Inventory inventory{};

    std::size_t width{};

    std::vector<Certified_window> windows{};

    std::size_t right_context{};

    std::vector<munch::verifier::Truncation> failing{};
};

struct Report
{
    Window_table factorization{};

    Window_table policy{};

    munch::verifier::Boundary_gap gap{};
};

[[nodiscard]] Report validate(const pretok::Pattern& pattern, std::size_t width = default_width);

struct Supply
{
    std::size_t bytes{};

    Anchors anchors{};

    double boundary_share{};

    std::size_t longest_piece{};
};

[[nodiscard]] Supply supply(
        const Report& report, Inventory inventory, const pretok::Pattern& pattern, std::string_view input);

struct Merge_window
{
    std::string window{};

    std::size_t origin{};

    std::optional<munch::verifier::Miscovering> refutation{};
};

[[nodiscard]] std::vector<Merge_window> validate_merges(
        const load::Definition& definition, std::span<const munch::verifier::Certified_pair> candidates);

[[nodiscard]] std::string render(const Report& report);

[[nodiscard]] std::string json(const Report& report);

[[nodiscard]] std::string supply_section(const Supply& factorization, const Supply& policy, std::string_view path);

[[nodiscard]] std::string supply_json(const Supply& factorization, const Supply& policy, std::string_view path);

} // namespace matisse::validate
```

`validate()` decides every class window of a pattern of width two to `width` (five by default) under both certificates:
the policy certificate of `munch::verifier::miscovering()` on the split verifier, the factorization certificate by
`is_split_window()` on the factorization lexer. A table holds its certified windows as munch-audit's report holds them,
`Certified_window` duplicated rather than linked, widths ascending and each width's windows in the order the class
symbols give, which is the windows fixture's order; an index over the class strings is the encoder's to build, since
windows of several widths share no single numbering. The policy certificate is decided for every window at once, as the
Python reference decides it and as munch's `modulo_cuts()` decides its cuts: one walk over the windows, each carrying
its cloud, the states of the split verifier an occurrence can leave it in with where the piece holding the occurrence's
last symbol begins and whether a piece ends after it, extended by one symbol from its prefix's cloud, the window
certified at an origin when every entry places the piece there. That is `miscovering()`'s certificate, and it takes
`validate()` to width four in 0.19 s for GPT-2 and 0.45 s for cl100k, and to width five in 1.1 s and 2.4 s, where a
`miscovering()` search per window would take 3.8 s and 9.3 s, and 92 s and 213 s. The default is five for that reason
and because GPT-2's policy anchors are complete at width five, every cut a wider window certifies being one of them, as
the report on pre-tokenizer certification proves. Each table carries the right context its inventory needs, decided on
the split verifier (section "What matisse needs from munch 3.0.0"), with the shortest truncation at every context that
fails, asked of the windows that hold no shorter certified window whose origin lands on the same symbol, since every cut
of one that does is a cut of the shorter one; to width four, for GPT-2 that is 87 and 540 windows of 18,357 and 32,218,
for cl100k 112 and 678 of 30,576 and 58,580. The report carries `boundary_gap()` of the split verifier, which is
unbounded for both patterns since a letter run of any length is one piece. A definition without a pre-tokenizer goes to
`validate_merges()` instead, which builds `munch::verifier::token_automaton()` once and decides each candidate window by
`miscovering()` on it, the lazy certification. Its candidates are given by the caller; the command line draws them from
an input, every byte and the most frequent byte pairs and triples, since all byte windows of width three are too many to
decide at full scale.

`supply()` measures one inventory's anchors on an input, the interior positions where an occurrence of a certified
window places its origin, counted at the byte offsets of their code points. It takes the pattern the report was decided
of, since the report holds no alphabet to class the input with and no segmenter for its pre-token boundaries. `Anchors`
and `Gaps` are munch-audit's, duplicated rather than linked (count, per kibibyte, median, ninetieth and ninety-ninth
percentile, longest): the gaps are the distances between consecutive anchors, the runs before the first and after the
last left out, and a percentile is the order statistic at floor(q n) of the n gaps ascending. The report on
pre-tokenizer certification measures its gaps otherwise, the two ends of the corpus counted as cuts, the percentiles at
round(q (n - 1)) and the density per 1,000 bytes, so matisse's gap figures and density are not that report's and are not
compared with its. `Supply` adds two figures beside them: the share of the serial pre-token boundaries that are anchors,
and the longest piece, which is the floor no cut rule can go below. `validate_merges()` returns each candidate with its
refutation or none. `render()` and `json()` follow munch-audit's report renderings: the text is rows of a label and a
value, the verdict first, then each inventory's windows, counted by width with six examples, and its right context with
the shortest truncation of every context that fails, then the boundary gap; the JSON is one object with a member per
figure, class strings holding each byte as the code point of its value. `supply_section()` and `supply_json()` render
the supply of both inventories on one input as munch-audit's renderings of its supply do, in supply.hpp beside the
report, as munch-audit keeps its supply in a unit of its own.

`matisse-validate` in tools/validate is the command line, modelled on munch-audit's (a `matisse_validate_cli` target
with that output name): `matisse-validate FILE [--tiktoken PATTERN] [--windows N] [--input FILE] [--json]`, with
`--tiktoken` naming the pattern of a rank file since the file names none, a tokenizer.json read under the first
tokenizer of the lookup whose published spelling it holds, the report always, the supply section only with `--input`, so
that runs without it are unchanged by it, as munch-audit's `--input` is. The pattern names are the lookup's, and the
command line finds the pattern and the vendor's special tokens through `lookup::tokenizer()`. Until
`read_tokenizer_json()` is written a FILE without `--tiktoken` is a command-line error, and GPT-2 reaches the command
line only as a rank file under `--tiktoken gpt2`, since FILE is one file and encoder.json and vocab.bpe are two.
`--windows` takes 2 to 5. The text heads the report with the file and the tokenizer it was read as, with its token
counts, and `--json` writes one document holding the path, the tokenizer, the refusal or null, the token counts, the
report and the supply. The exit status is munch-audit's: 0 when the definition was read and reported, 1 when the loader
refused it, the refusal written in place of the report, and 2 on a command-line error, with the usage, or on a file or
an input the command cannot read, an input that is not UTF-8 among them. Its tests run it on a rank file of the 256
single bytes, which every pattern takes, and a short sample beside them, and hold its text at the default width and its
JSON at width two to fixtures it wrote, each headed by the line that says how, as munch-audit's tests hold its own; the
anchors and the share of the supply on the sample agree with the Python reference's measurement of the same text, and
the gap figures and the density are munch-audit's, as `supply()` defines them.

### **The encoder, matisse::encode**

```cpp
namespace matisse::encode
{
class Merger
{
public:
    explicit Merger(const load::Definition& definition);

    [[nodiscard]] std::optional<load::Token_id_t> whole_piece(std::string_view piece) const;

    [[nodiscard]] std::vector<load::Token_id_t> merged(std::string_view piece) const;
};

inline constexpr std::size_t piece_cache_capacity{16384};

inline constexpr std::size_t longest_cached_piece{255};

class Piece_cache
{
public:
    explicit Piece_cache(std::size_t capacity);

    [[nodiscard]] std::optional<std::span<const load::Token_id_t>> find(std::string_view piece) const;

    void store(std::string_view piece, std::vector<load::Token_id_t> ids);
};

class Window_index
{
public:
    Window_index(const validate::Window_table& table, const pretok::Alphabet& alphabet);

    [[nodiscard]] std::size_t width() const noexcept;

    [[nodiscard]] std::optional<std::size_t> first_anchor(std::string_view stretch, std::size_t from) const;
};

class Encoder
{
public:
    Encoder(load::Definition definition, const validate::Report& report, validate::Inventory inventory,
            const pretok::Pattern& pattern, std::span<const std::string> allowed);

    [[nodiscard]] std::vector<load::Token_id_t> encode(std::string_view text) const;

    [[nodiscard]] std::vector<std::size_t> chunk_boundaries(std::string_view text, std::size_t chunks) const;

    [[nodiscard]] std::vector<load::Token_id_t> encode_parallel(std::string_view text, std::size_t chunks) const;

    [[nodiscard]] std::string decode(std::span<const load::Token_id_t> ids) const;
};

} // namespace matisse::encode
```

The names follow `core::Lexer`'s `tokenize_all()`, `chunk_boundaries()` and `tokenize_all_parallel()`. The constructor
takes the report `validate()` decided of the pattern, as `supply()` takes it, and keeps the inventory's window table and
right context, so that a pattern is validated once however many encoders are built, a fuzz harness's and the
thread-sanitizer job's among them; it takes a pattern, so a definition without a pre-tokenizer has no encoder in the
first version, and it refuses an `allowed` spelling that is not one of the definition's special tokens. `encode()` is
the sequential encoding: the text is split at every occurrence of an allowed special token, the earliest first and the
longest of those that begin at one byte, which emits its id, and each stretch between them is segmented by the
`Segmenter` as a complete input, every piece then looked up whole where the definition's whole-piece lookup says so and
otherwise merged alone by its merge rule, the pair rule or tiktoken's rank merge. Special tokens not allowed are
ordinary text, which is tiktoken's `encode()` with that `allowed_special` and an empty `disallowed_special`,
encoder.py's behavior when none is allowed; tiktoken leaves the choice between allowed spellings that begin at one byte
to the order of its table, and no allowed spelling begins another on any definition in scope. `decode()` spells each
ordinary id's bytes and each special id's spelling, and refuses an id that is neither, naming it and its index.
`chunk_boundaries()` aims for `chunks` pieces of equal size, as `core::Lexer`'s does: each interior boundary is the
first cut at or after the later of its equal-division target and one past the previous boundary, and a target with no
cut after it ends the walk. A cut is an anchor, a position `t + o` for an occurrence at `t` inside one stretch of a
window the table certifies at origin `o`, at least the width less one codepoints after the previous boundary, so that
the occurrence certifying it begins in the chunk the cut ends, as the splice needs, and as far after the stretch's
start, as the Python reference spaces its first cut from the text's start; or it is where an allowed special token
begins or ends, a cut of its own. The spacing is the Python reference's, and it leaves out an anchor that stands closer
than that to the previous boundary even when its occurrence begins after it. `Window_index` holds the table's windows by
their class strings and finds the least anchor of a stretch at or after a byte, classing the text in blocks of 256 bytes
from there, so that a walk reads the text near its target and no more. `encode_parallel()` finds the allowed special
tokens once and gives each chunk to a worker, one `std::jthread` per chunk beyond the last, which runs on the calling
thread, as `tokenize_all_parallel()` does (open choice 4). A worker segments its chunk's stretches as `encode()` does,
the last one followed by `right_context` further codepoints of the real text, read as a complete input there unless a
special token or the text's end comes first, keeps the pieces that end at or before its cut, and merges them with a
piece cache of its own. The ids of the chunks, concatenated, are the result, and an exception a worker throws is
rethrown on the calling thread once every worker has joined.

`Merger` holds a definition's whole-piece lookup and its merge, built once: the id of every spelling and of every byte,
and under the pair rule the rank and product of every pair of the merge table keyed by the ids of its parts, a pair
listed twice taking its last rank as both vendors read it. It refuses a definition in which a byte is not a token and,
under the pair rule, a merge whose part or product is not one, naming the byte, or the merge, its rank and the spelling.
The merge keeps the parts of a piece as a doubly linked list over its bytes, as Hugging Face's `Word` keeps its symbols,
and the candidate joins in a binary heap ordered by rank and then by where the left part starts, each entry recording
that start and the ends of both parts; an entry popped whose parts no longer stand as recorded is discarded. The rule
enters only in the lookup of a pair: the rank of the parts' concatenation under `Merge_rule::ranks`, the rank of the
pair of parts in the merge table under `Merge_rule::pairs`. It is exact on every table and every piece. A part's end
only grows and a part joined into the one before it never stands again, so an entry once stale stays stale, and every
adjacent pair the rule can join has a live entry, since the initial pairs are pushed and each join pushes the two pairs
it makes. So each join is of the pair of least rank present, the leftmost on a tie: under the rank rule that is
tiktoken's `_byte_pair_merge`, a join that makes a pair of lower rank than its own, such as `.:.:` on Llama 3's rank
file, needing no case of its own, and under the pair rule it is Hugging Face's `merge_all`. A piece of n bytes takes at
most n - 1 joins and fewer than 3n entries. The tests hold the merge to the plain pair loop and to tiktoken's scan,
written out in the tests, on every multi-byte token of GPT-2's, cl100k's, o200k's and Llama 3's files (50,001, 100,000,
199,742 and 127,744 tokens) and on 20,000 seeded random pieces of each, GPT-2's to encoder.py's pass as well, and on
3,000 random tables over three symbols, which make ties, overlaps, joins that make a pair of lower rank, and tables on
which the pass and one join at a time differ. The merge forms every multi-byte token from its own bytes on cl100k and
o200k, all but `<|endoftext|>` on GPT-2, and all but 588 on Llama 3, which Llama 3's whole-piece lookup gives whole.

`encode()` looks each piece up whole first where the definition says so, then in a `Piece_cache`, and only then merges
it, storing its ids. The ids of a piece are a function of its bytes and the definition alone, the merge building its
parts afresh for every piece, so a cache keyed by the whole of a piece's bytes and filled under one definition never
changes an id. A cache serves one call of `encode()` or one worker of `encode_parallel()`, with no lock and no lifetime
beyond the call; it holds 16,384 pieces of up to 255 bytes and then stops filling, with no eviction, as Hugging Face's
BPE model's cache stops. On the three corpora of the test strategy GPT-2's pattern makes 3,684 to 14,032 distinct pieces
and cl100k's leaves 1,527 to 9,218 distinct pieces that are not looked up whole, so the bound holds every piece the
merge makes on them. The tests hold `encode()` to the known answers and to every piece looked up or merged with no
cache, on seeded random texts and on a text whose distinct pieces fill the cache past its bound, and `decode()` of
`encode()` to the text, special tokens allowed and not.

The encoder claims one thing. For every text the definition accepts, `encode_parallel()` returns the ids `encode()`
returns, at every chunk count. That is the splice theorem with the proposition that the cut transfers and the decided
right context, given the four hypotheses (N), (S), (P) and (B), which the loader's refusals enforce, and given that the
`Segmenter` is the engine's segmentation. The tests hold the claim on seeded texts under both inventories, at every
chunk count from one to sixteen, at sixty-four and at every cut the rule takes, on a pattern of the tests' own and on
GPT-2's and cl100k's with their pinned vocabularies, special tokens allowed and not on the tests' pattern and cl100k's;
they hold `chunk_boundaries()` on the tests' pattern to brute force over every certified cut of the report, and a cut in
white space, at the right-context counterexamples, to the right context each inventory decided. Its speed is what
[benchmarks.md](benchmarks.md) measures, and nothing is claimed about the vendors' own implementations beyond the ids
and the speeds measured there.

## **The vendor verifiers, from Python into C++**

The Python reference implementation holds everything the port needs: the class alphabets, the per-alternative languages
and the one-step relation Q, the verifier M with its functionality and totality decisions, the one-codepoint gate, the
certificates, the right-context decisions and the references and end-to-end run, over a small DFA kit. Each moves as one
C++ unit of pretok or validate, written in munch's style and modelled on its nearest munch neighbour: the verifier
construction on `merge_tower()` and `trie_chain()`, the window tables on munch-audit's report, the supply on
munch-audit's supply. The DFA kit the construction of M needs (product, complement, concatenation, minimization over a
small alphabet) moves as one file of pretok; where it lives is open choice 7.

The Unicode tables are generated, not ported by hand. tools/unicode/generate_classes.py is munch's generator of the same
name duplicated and changed to emit the sets named on its command line, each a general category or a property of
PropList.txt under the name of its table, with the source digests and the `--version` guard munch's has, into each
tokenizer library's src/class_ranges.inc, checked in, so the build never touches the network; GPT-2's and cl100k's
tables are the same file generated twice from L, N and White_Space, duplicated rather than shared, so that a tokenizer
whose pattern names other sets brings its own, o200k's Lu, Ll, Lt, Lm, Lo, M, N and White_Space. This is munch-audit's
precedent for logos, whose reader keeps tables of the Unicode version logos is locked to rather than the library's. The
tables are pinned to Unicode 15.0.0, the version of the Python reference, so that the fixtures replay (open choice 6).
The classes and therefore the verifiers, the certified windows and the right context do not depend on the version; only
the map from codepoints to classes does, and with it the ids of text holding a codepoint whose class moved between
versions.

The ported units stay exact through fixtures written by the reference and held byte for byte, as munch's verifier tests
hold `token_automaton_reference.txt`. A fixture script beside the reference code writes each fixture with a header line
naming the script and saying to regenerate with it, never by hand, one fact per line, bytes and class strings in hex,
and the C++ tests read every line, check the answer, and count the lines of each kind. Each fixture file holds every
tokenizer's universe and stays as the script writes it, under the tests/data of the stage its subject belongs to, and
each tokenizer library's tests check the lines of its own universe and skip the rest, so that every universe's lines are
checked by its tokenizer's tests. The fixtures, one per subject:

- **Classes**: the Unicode version, the class numbering by symbol, the range count and SHA-256 of each set the tables
  hold, L, N and White_Space, and the class of every code point on either side of a change of class.
- **Segmentation**: the engine's segmentation of every class string of lengths zero to four (41,371 strings for GPT-2
  and 69,905 for cl100k, 2.4 MB, the script's default length; to five, the gate check's own population, they would be
  579,195 and 1,118,481), against which the `Segmenter` is held, and the boundaries of the three corpora's class
  projections as digests.
- **Verifier**: the table munch's constructor makes of the minimized M (trimmed, start first, breadth-first over the
  marked symbols ascending), the script applying the same renumbering, so the C++ construction must produce the
  identical table state for state; and the reference's decisions that M is functional and total and that the pattern
  covers every nonempty residual.
- **Windows**: every certified window to width four under both certificates with its origin (18,357 and 32,218 for
  GPT-2, 30,576 and 58,580 for cl100k), and every refused one by count; the tokenizers' report tests decide theirs at
  width four to hold them to it, the default being five.
- **Right context**: the decided R of each inventory, and the counterexample of each failing R with its replay
  (`"\t\t!"` at cut two for R = 0, `"\t !"` and `"\t\tA"` at cut one for the policy inventory at R = 1).
- **Known answers**: the ids of short strings under each reference, among them the published `"Hello world"`, 15496 995
  for GPT-2, and `"tiktoken is great!"`, 83 1609 5963 374 2294 0 for cl100k.
- **Lazy certification**: the verdicts and refutations of the lazy certification's validation set at the depths it
  records.

Where a C++ answer and a fixture line disagree, the test fails with both; the fix goes into whichever side is wrong, and
a fixture changes only by rerunning its script.

## **Test strategy against the real tokenizers**

Each library has a GoogleTest suite registered with CTest, as hopper's do. The reference vocabularies are pinned by
SHA-256:

| File | SHA-256 |
|---|---|
| gpt2-merges.txt (vocab.bpe) | 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5 |
| gpt2-vocab.json (encoder.json) | 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783 |
| cl100k_base.tiktoken | 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7 |
| o200k_base.tiktoken | 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d |
| llama3.tiktoken | 82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55 |
| twitter.json | 30721e496a8d73cfc50658923c34eb2c0fbe15ee6835005e43ee624d8dedf200 |

No GPT-2 or cl100k tokenizer.json is pinned yet; the loader's tests need one of each, taken once and pinned the same
way. Nothing is fetched at test time: the test data is in the tree, or given to the probe as paths. Whether the
vocabulary files themselves go into the tree is open choice 8.

Three layers, each against something independent of the code under test:

1. **The fixtures** of the previous section, in CTest: classes, segmentation, M, windows, right context, known answers,
   lazy verdicts.
2. **Self-consistency**, in CTest, on texts generated with a fixed seed as hopper's probe generates its corpus:
   `encode_parallel()` against `encode()` under both inventories at every chunk count from one to sixteen, at
   sixty-four, and at every cut the rule takes; the cuts of `chunk_boundaries()` against brute force over the report;
   `decode(encode(x)) == x`; M accepts the `Segmenter`'s marking of every text; every anchor is a `Segmenter` boundary.
   The texts mix the known counterexample shapes, whitespace runs before letters and punctuation, newlines followed by
   tab runs, long letter runs, contractions in both cases, and allowed and disallowed special tokens.
3. **The corpora**, by the probe in tools/probes: `matisse_corpus` takes the vocabulary directory and corpus files,
   holds GPT-2's and cl100k's pinned files to their digests, a corpus the fixtures name to the digest of its text, and
   the ids of `encode()` to the known-answer fixture's digest of the reference ids (the SHA-256 of the ids as
   little-endian 32-bit words, with their count), written by the fixture script. On every corpus it checks the ids of
   every piece merged with no cache and `decode()`, and for both inventories `encode_parallel()` at the worker counts
   past one that the bench times, at the 4,096- and 65,536-codepoint chunkings and at every anchor, skipping a chunk
   count past 32,768 since every chunk is a thread. At every anchor it also holds the pieces a worker keeps to the
   serial pieces, with no thread per chunk. Given the vocabulary directory alone it runs the generated texts of layer 2,
   which is the form registered with CTest, as hopper_edit_relex does, and `-DMATISSE_CORPUS_DIR` registers the three
   corpora as well. The corpora are twitter.json, and munch's 10 tracked Python files and its 306 tracked C++ sources
   and headers at 15b2863c, concatenated in path order by the fixture script and pinned by digest;
   [benchmarks.md](benchmarks.md) adds eleven more, and the probe's ids lines are what the vendors' ids are compared by.
   The probe prints the supply table, which [benchmarks.md](benchmarks.md) quotes from its output.

The reference ids in the fixtures are the Python references, encoder.py's loop and tiktoken's rank merge restated over
Python's engine with Unicode 15.0, which are held to the published known answers and not run against the vendors' code
on the corpora. One step of the order of work closes that: tiktoken and Hugging Face tokenizers are run on the three
corpora outside the tree, and the fixture script records their id digests beside the restated references', so the probe
holds matisse to the vendors themselves. A difference there would be the Unicode version or a vendor's engine, and it
would name the codepoint. On the three corpora and the eleven of [benchmarks.md](benchmarks.md) there is none:
tiktoken's, Hugging Face's and the bpe crate's ids are matisse's by digest on every corpus each of them ran on, and
recording them into the fixture is step 15.

The fuzz harnesses in tools/fuzz follow hopper's contract (a refusal is the answer to bad input, a crash, a sanitizer
report or a hang is a finding): one feeds arbitrary bytes to the tokenizer.json and tiktoken readers, and one feeds
arbitrary well-formed UTF-8 to the encoder and asserts that `encode_parallel()` equals `encode()` at a chunk count drawn
from the input, which turns the encoder's one claim into a fuzz property.

## **What matisse needs from munch 3.0.0**

Used as munch 3.0.0 has it:

- `munch::verifier::Verifier`, built from a table, for M over the class symbols, which fit the byte alphabet (16 of 256
  for cl100k).
- `Verifier::step()` and `state_count()`, which validate's walk over the windows reads to decide the certificate of
  `miscovering(const Verifier&, window, origin)` for every window at once, and `boundary_gap()` for the gap verdict.
- `core::Builder` and `Lexer::is_split_window()` for the factorization certificate on U, the procedure cross-checked
  against munch-audit exactly.
- `Merge`, `token_automaton()` and `miscovering(const Token_automaton&, window, origin)` for the lazy certification,
  with their refusal of a table that recreates a token passed through as the loader's.
- `right_context(const Verifier&, window, origin)` for the right context of each certified pair, an inventory's being
  the largest of its pairs', and `dependence()` for the suffix half beside the case R = 0; the Python reference's
  decision is the reference and its counterexamples the fixture, as context_reference.txt holds `right_context()`.
  `dependence()` returns the shortest refutation of either side, so at a window whose prefix already fails with no
  symbol past the cut it leaves the suffix half open: at 12 of the 87 and 12 of the 540 windows GPT-2's right contexts
  are asked of, under the factorization and the policy certificate, and at 4 of the 112 and 24 of the 678 of cl100k's,
  every other one refuting neither side.

Missing, each with where it would go by the division:

- **Every window's origin at once.** `miscovering()` decides one window at one origin, and validate's walk decides the
  certified origin of every window to a width in one pass, the shape of `modulo_cuts()`; a decision beside it returning
  each window's certified origin would hold the walk. General, so munch, later.
- **The suffix half alone.** The Python reference decides it for every boundary of M, and `dependence()` decides it only
  together with the prefix at R = 0, so the fixture's suffix line is held at every certified cut except at those open
  windows, and not at the boundaries of M that no certified window places. A decision of the suffix side alone, the
  restarted copy of `dependence()`'s product without the prefix flag, would close it. General, so munch, later.
- **Functionality and totality of a verifier.** The Python reference decides both for M, and `divergence()` presumes the
  first. The first version does not need them, since M's table is held state for state to the fixture whose M was
  decided functional and total; they would let a new pattern's Q be checked by the validator itself. General, so munch,
  later.
- **Alphabets beyond bytes.** `Marked` carries an `unsigned char`, which holds every class alphabet in sight, o200k's
  included. It does not hold a character-level merge table: SentencePiece BPE tokenizers without a pre-tokenizer, Llama
  2 among them, merge Unicode characters, and `token_automaton()` refuses a part that is not a byte or an earlier
  product. They also carry a normalizer and byte fallback. So the lazy route of the first version covers byte-level
  merge tables, and character-level ones wait for a decision about munch's alphabet.

Not needed from munch: the Unicode tables (munch's own are 17.0.0 and have no L or N; matisse keeps its own, as the
logos reader does), a UTF-8 decoder (a small one in pretok, duplicated rather than shared), and the runtime, which is
matisse's by the division.

## **Open choices**

1. **License.** munch and hopper are MIT, with a THIRD_PARTY_NOTICES.md for what they carry under other terms.
   Recommendation: MIT, with notices for the Unicode data (Unicode License v3, munch's notice), for the published
   patterns and special tokens quoted from openai/gpt-2 and openai/tiktoken, and for any vendored vocabulary (choice 8).
2. **Python bindings.** Neither munch nor hopper has any, and the PyPI name matisse is taken, so a binding could be
   called matisse-tok. Recommendation: later, once the validator has been used and the C++ surface has settled, so that
   a binding does not fix names before the stability boundary is stated.
3. **Tokenizers after GPT-2 and cl100k.** Recommendation, in this order. First the cl100k pattern as tiktoken shipped it
   before the possessive rewrite. The construction already runs it, it differs from the current one only on text ending
   in whitespace that holds a newline, and as far as is known Llama 3's published pattern and the tokenizer.json
   conversions of cl100k carry it, which the pinned files will show. Then o200k, whose classes are more (Lu, Ll, Lt, Lm,
   Lo, M) but whose pattern is still a regex without lookbehind; the conjecture that the same construction holds is
   built and decided first. Then WordPiece through munch's `trie_chain()` with bert-vocab.txt, which brings BERT's
   normalizer and so the first normalizer. Character-level SentencePiece waits on munch's alphabet.
4. **Threading model.** munch's `tokenize_all_parallel()` spawns one `std::jthread` per chunk beyond the last, runs the
   last on the calling thread, joins at scope end, and rethrows the first exception on the calling thread. Settled: the
   same, with no pool and no executor, the caller choosing the chunk count; a pool is a question for later, settled by
   measurement. Measured ([benchmarks.md](benchmarks.md)): 24 workers encode enwik9 11.2 times faster than `encode()`,
   and a chunking at every anchor of munch's C++ sources, 559,642 chunks, fails with `std::system_error` where the
   system cannot start that many threads, so the corpus probe asks for at most 32,768 chunks.
5. **The JSON reader.** tokenizer.json and encoder.json are JSON, with `\u` escapes and surrogate pairs in GPT-2's
   vocabulary. hopper has a complete RFC 8259 parser (`hopper::json::Parser`, held to the JSONTestSuite), munch has
   none, and a third-party library would be a new vendored dependency. Taking hopper means a second submodule, and
   hopper's external/CMakeLists.txt adds its own munch submodule unless hopper is built against a system munch, which
   would define munch's targets twice beside matisse's own submodule. Recommendation: hopper's parser, with matisse
   keeping its own munch submodule as hopper does and hopper skipping its submodule when `munch::munch` already exists,
   a one-line change in hopper's next release, which has to build against munch 3.0.0 anyway (as 1.0.1 lifted hopper to
   2.1.0). A third JSON parser in the family would duplicate a tested one.
6. **The Unicode version of the class tables.** The fixtures come from Python 3.12, Unicode 15.0.0; tiktoken's and
   Hugging Face's engines carry their own versions. Recommendation: 15.0.0 for the first version so that the fixtures
   replay, then the version of the engine the vendor cross-check (layer 3) shows to matter, regenerating the fixtures
   with it.
7. **Where the DFA kit lives.** The construction of M needs product, complement and concatenation of DFAs, which munch
   does not offer publicly. By the division test a DFA kit is general and belongs in munch; by the rule of duplicating
   rather than sharing unless really needed it stays with the one construction that uses it. Settled: in matisse's
   pretok, as automata.hpp and its source, because nothing in munch calls it and a munch addition would wait for a munch
   release; the header is installed, as munch keeps no private headers under src, and the kit moves if a second user
   appears. So automata.hpp is not part of the stable surface. Its file comment says so, and the README's Versioning and
   Stability section names it outside the surface when it is written at step 16.
8. **Vocabulary files in the tree.** CTest without network needs the pinned files on disk: about 1.5 MB for GPT-2's pair
   and 1.7 MB for cl100k, plus the two tokenizer.json files. hopper vendors the JSONTestSuite cases with their licence
   and provenance. The vocabulary files' own terms need checking first. As far as is known, GPT-2's are published with
   the model under MIT on its Hugging Face card, and cl100k_base.tiktoken is served by OpenAI without a licence file of
   its own. Recommendation: vendor them under libs/load/tests/data with provenance, as hopper vendors its cases, if the
   terms allow; otherwise a `MATISSE_VOCAB_DIR` option naming a directory of the pinned files, the tests that need them
   registered only when it is given, as the vendored-or-system pattern of `USE_SYSTEM_GTEST` already works.
9. **The encoder's default inventory.** The factorization inventory needs one codepoint of right context and rests on
   less, its cut holding for every complete factorization into the gate-erased languages whatever the engine does inside
   its matches; the policy inventory needs two and rests on Q, validated and not machine-checked, but lands on more
   boundaries (93.91 to 100.00 per cent against 57.30 to 92.88 at width four, and 96.09 to 100.00 against 57.30 to 92.88
   at the default width of five, as [benchmarks.md](benchmarks.md) quotes the probe). Both reach a longest uncut stretch
   within 24 bytes of the longest piece on all six runs, at width four. Recommendation: the factorization inventory by
   default, the policy inventory by choice.

## **Order of work**

Each step lands green on its own, its tests first where a fixture exists.

1. munch 3.0.0 released (no matisse work before it).
2. The right-context decision in munch, with its fixture from the Python reference (3.0.0).
3. The fixture script beside the Python reference, writing the fixtures of the vendor section and the corpus digests.
4. The repository skeleton: hopper's CMake, CI, README and docs pages with the names changed, the munch submodule at the
   release, an empty library, a green CI.
5. load: the tiktoken reader and `read_gpt2()`, with the known-answer spellings; then `read_tokenizer_json()` after
   choice 5, with its refusals.
6. pretok: the generated class tables and `classes_of()`, held to the classes fixture.
7. pretok: the `Segmenter`, held to the segmentation fixture.
8. pretok: the DFA kit and `split_verifier()`, held to the verifier fixture state for state; `factorization_lexer()`.
9. validate: the window tables under both certificates, held to the windows fixture; the right context through munch,
   held to its fixture.
10. validate: `supply()`, `render()`, `json()`, and `matisse-validate` with `--input`.
11. encode: the `Encoder` constructed from a definition, a report decided once and its pattern; `encode()` and
    `decode()` with both merge rules, the whole-piece lookup and special-token cuts, held to the known answers; a fast
    BPE merge, linear or near-linear per piece in the manner of the GitHub bpe crate and tiktoken's rank merge, held
    byte-identical to the simple pair loop and to the known answers, with its exactness argued before it lands; and a
    bounded piece cache keyed by pre-token bytes, one per call and per worker, which never changes an id.
12. encode: `chunk_boundaries()` and `encode_parallel()`, with the self-consistency suite of layer 2 and the
    thread-sanitizer job.
13. The corpus probe on the three corpora, its figures into [benchmarks.md](benchmarks.md). The timing compares against
    tiktoken, Hugging Face tokenizers and the bpe crate. Built: `matisse_corpus` and `matisse_bench` in tools/probes,
    the figures in [benchmarks.md](benchmarks.md). On fourteen corpora, from twitter.json to a mixed file of 1.98 GB,
    matisse's ids under GPT-2 and cl100k equal tiktoken's, Hugging Face's and the bpe crate's by digest wherever each
    ran. At the default width of five the supply on the three corpora reaches 57.30 to 92.88 per cent of the pre-token
    boundaries under the factorization inventory and 96.09 to 100.00 under the policy inventory, the longest gap between
    anchors within 24 bytes of the longest piece. On one core `encode()` runs at 9.0 to 37.1 MiB/s, 0.42 to 1.74 times
    tiktoken's rate, and the bpe crate runs 1.8 to 5.1 times faster than matisse. A call on a prompt of about a kilobyte
    costs matisse a mean of 94 microseconds under GPT-2 and 116 under cl100k, against tiktoken's 58 and 63 and the bpe
    crate's 26 and 25. `encode_parallel()` encodes enwik9 at 162.7 and 177.9 MiB/s with 24 workers, 11.2 times
    `encode()`. The profile puts the class of a code point first on prose and code, 35 to 58 per cent of the time, and
    the merge first on Chinese under cl100k, 76 per cent.
14. validate: `validate_merges()`, the lazy route for byte-level tables without a pre-tokenizer, held to the lazy
    fixture.
15. The vendor cross-check (tiktoken, Hugging Face tokenizers), its digests into the fixture, the Unicode version
    settled (choice 6).
16. The fuzz harnesses and their CI job; the README's Versioning and Stability section naming the stable surface; then
    the first release.
