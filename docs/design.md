# **Design**

The decisions behind matisse as it ships, each with its reason: what the library is and is not, how its libraries fit
together, how a tokenizer is described, validated and encoded, how exactness is guaranteed and tested against the real
tokenizers, what matisse needs from munch, and the questions that are still open. [usage.md](usage.md) shows how each
library is called, [limits.md](limits.md) states what every call guarantees, and [benchmarks.md](benchmarks.md) gives
the measured numbers.

## **What matisse is and is not**

matisse reads the definition of a real tokenizer, decides where its output can be cut with certainty, and encodes text
in parallel by cutting only there, with token ids byte-identical to the sequential encoding. It does this for GPT-2,
cl100k_base, o200k_base and Llama 3, whose deployed pre-tokenizer patterns have exact verifiers. It is not a general
tokenizer library. It trains nothing, applies no normalizer but the identity, and does not try to match Hugging Face's
feature surface. Its product is exactness against the reference tokenizer, shown on pinned vocabularies and corpora.

munch stays the decision library, with the verifier framework and the general constructions on abstract inputs, merge
tables and vocabularies among them. matisse takes everything about specific real tokenizers, which is reading their
files and special tokens, the vendor pre-tokenizer verifiers tied to a regex and to an engine's Unicode tables, and the
runtime with its token ids, normalizers and special-token cuts. The test for where a piece goes is whether it would make
sense if no LLM existed, and if it would, it belongs in munch.

matisse is also where the tokenizer pieces of an execution-contract compiler would live: the full-dictionary verifier,
the two-mode WordPiece construction, a labeler carrying token ids, and certified-prefix streaming. Only the streaming is
built, as the `Stream_encoder` described below, and it claims that every id it returns is a prefix of `encode()` of the
whole text and that what it returns is the longest such prefix but for three exceptions.

The name is after Matisse's paper cut-outs, cutting with certainty, beside munch, the lexer library, and hopper, the
parser toolkit.

## **How the libraries fit together**

A tokenizer reaches matisse as two things read apart. Its files are read by load into a `load::Definition`, which holds
the merge rule, whether a piece is looked up whole, the bytes of every ordinary token, the merge table and the special
tokens. Its pre-tokenizer is a `pretok::Pattern`, the published regex described as data, which the tokenizer's own
library supplies with `pattern()`, beside `spelling()`, the published regex as its vendor spells it, by which a
tokenizer.json's pre-tokenizer is recognized, and `specials()`, the vendor's special tokens for a rank file that names
none. validate decides a report of the pattern once: the windows of classes that certify a cut under each of two
certificates, and the right context each certificate needs. encode builds an `Encoder` from the definition, the report,
the inventory of certified windows chosen and the special tokens allowed, and the encoder encodes a text sequentially,
in parallel at certified cuts, and as a stream through a `Stream_encoder` over it. For cl100k_base:

```
cl100k_base.tiktoken ──▶ load::read_tiktoken() ──▶ load::Definition ────┐
                                                                        │
cl100k::pattern() ──▶ validate::validate() ──▶ validate::Report ────────┤
                                                                        ▼
                                                                 encode::Encoder
                                                                        │
                                                                        ├──▶ encode(), encode_parallel(), decode()
                                                                        └──▶ encode::Stream_encoder: push(), finish()
```

The libraries are of two kinds, as hopper's generic parse library sits beside its json and clike grammars. The four
stage libraries hold nothing of a specific tokenizer and take its description as data. load reads tokenizer files into
one definition type; its `read_gpt2()` reads the file format GPT-2 published, `read_tiktoken()` the rank files tiktoken
publishes, and `read_tokenizer_json()` Hugging Face's tokenizer.json with a BPE model, its JSON read by hopper's parser.
pretok holds the pre-tokenizer machinery over a pattern description: the classes of a text, the runtime segmenter, and
the construction of a pattern's verifier. validate decides the certified windows and their right context and measures
anchor supply. encode is the runtime. Beside them every tokenizer is a small library of its own, gpt2, cl100k, o200k and
llama3, holding its pattern description, the pattern's published spelling, its generated class tables, the vendor's
special tokens for a file that names none, and its tests, and lookup finds the tokenizer libraries by name, for
`matisse-validate` and any caller choosing a tokenizer at run time.

Each library is a target `matisse_<name>` with the alias `matisse::<name>`, in namespace `matisse::<name>`, and the
umbrella `matisse::matisse` links all of them, as `hopper::hopper` links hopper's. A consumer that encodes under one
tokenizer links its library, which brings load and pretok with it, beside encode, which brings validate, as
`matisse::cl100k matisse::encode`. common holds the SHA-256 that the tests and the probes use, the tests holding the
pinned files and the class tables to their digests and the probes the corpora and their ids; its target `matisse_common`
lives in the build tree only, with no alias, no install rule and no place in the umbrella. Every library whose headers
name munch types links munch publicly, as hopper_json does: load, whose definition holds `munch::verifier::Merge`,
pretok, whose pattern's sets are `munch::regex::utf8::Code_point_range` tables and whose verifier is a
`munch::verifier::Verifier`, and validate, which decides on that verifier. A tokenizer library reaches munch through
pretok. load links hopper_json privately, since none of its headers names a hopper type.

The repository is laid out as hopper's is, one library per subject under libs/, each with include/, src/ and tests/, the
tools beside them, munch and hopper as git submodules and googletest beside them:

```
matisse/
  CMakeLists.txt  CITATION.cff  LICENSE  README.md  THIRD_PARTY_NOTICES.md
  .clang-format  .gitattributes  .gitignore  .gitmodules
  .github/workflows/  ci.yml  codeql.yml
  cmake/matisse-config.cmake
  docs/  design.md  usage.md  limits.md  benchmarks.md
  external/  CMakeLists.txt  googletest/  hopper/  munch/  (submodules, pinned to the releases it builds against)
  libs/
    CMakeLists.txt
    common/     include/matisse/common/                   (the tests' SHA-256, neither installed nor exported)
    load/       include/matisse/load/      src/  tests/  tests/data/
    pretok/     include/matisse/pretok/    src/  tests/  tests/data/
    validate/   include/matisse/validate/  src/  tests/  tests/data/
    encode/     include/matisse/encode/    src/  tests/
    gpt2/       include/matisse/gpt2/      src/  tests/
    cl100k/     include/matisse/cl100k/    src/  tests/
    o200k/      include/matisse/o200k/     src/  tests/
    llama3/     include/matisse/llama3/    src/  tests/
    lookup/     include/matisse/lookup/    src/  tests/
  tools/
    CMakeLists.txt
    validate/   src/  tests/  tests/data/              (the matisse-validate command line)
    probes/     src/                                    (the corpus probe and the bench)
    fuzz/       include/  src/  seeds/                  (the libFuzzer harnesses)
    unicode/    generate_classes.py
```

The top-level CMakeLists.txt is hopper's with the prefix changed, and munch is consumed exactly as hopper consumes it:
the submodule by default, or with `MATISSE_SYSTEM_MUNCH` an installed munch found by `find_package(munch REQUIRED)` at
the top level, so that every library sees its imported targets. Installing requires the installed munch, because an
exported target can refer to munch only as an installed package. [usage.md](usage.md) lists the options.

hopper is the second submodule, pinned to hopper 1.0.0, for `hopper::json::Parser`, its RFC 8259 parser held to the
JSONTestSuite, which reads tokenizer.json, so that the family has one JSON parser and matisse vendors no new dependency.
matisse needs from it the parser and its `Value` tree, objects keeping their members in the file's order, strings with
every escape resolved and surrogate pairs combined, numbers as spelled. hopper 1.0.0's top level adds its own munch
submodule unless hopper is built against an installed munch, which would define munch's targets twice beside matisse's
own submodule, so external/CMakeLists.txt adds hopper's parse and json libraries directly, and both link `munch::munch`,
the submodule's or the installed one; hopper 1.0.0's sources build unchanged against munch 3.1.0. A hopper release whose
external/CMakeLists.txt adds its munch submodule only when the target `munch::munch` does not exist yet, one more
condition on the line that adds it, would let matisse add hopper whole. Installing installs hopper's two libraries with
matisse's targets, since load links hopper_json privately and a static library carries it to the consumer's link.

## **A tokenizer is data**

Nothing in the stage libraries names a tokenizer. A tokenizer's pre-tokenizer is a `pretok::Pattern`, the contract of
the class theorem of the report on pre-tokenizer certification written as data, and its files are read by a reader that
knows the format and nothing of the vendor. Every construction therefore lives in one place, and a new tokenizer brings
a description and its tests while the stage libraries stay as they are.

A pattern is an ordered gated pattern over a class alphabet: a list of regular languages of nonempty class strings, each
with its gate, the class symbols that may follow its accepts, the end of the input always passing. Its discipline
commits the longest prefix in an alternative's language whose next symbol passes that alternative's gate, for the first
alternative that has one. For every such pattern that is total the theorem gives the route, the verifier M, both
certificates and the splice, and it carries them to an engine that realizes the pattern. The engine realizes it when no
alternative reads left of where its match begins, each alternative matches the longest gate-passing prefix of its
language and fails only where there is none, and the engine reads code points only through the classes. Those conditions
are facts about the published regex and its engine, which no description can check, so the fixtures, written by the
engine, hold each description to them.

The alphabet is the class partition: the class symbols ascending, the code point sets in the order a code point is
classed by them, each with its symbol and its ranges, and the symbol of everything else. A literal's class is a set
listed before the Unicode sets, holding the code points the literal matches, so a case-sensitive letter holds itself and
cl100k's caseless `s` holds `S`, `s` and U+017F. o200k's classes read Lu and Ll apart and put the long s with the lower
s alone, and they are stated the same way. Each class symbol is one ASCII byte, so a class string is a text over bytes
and munch's verifier alphabet holds it. A class with an ASCII member is written by one of its members: a literal's class
by the literal, the class of everything else by `!` and any other by its least member. In GPT-2's, cl100k's and Llama
3's alphabets a class string is therefore a text of the pattern again. A class with no ASCII member is written by the
next of `#`, `$`, `%` and `&`, bytes of the class of everything else, in the order of the classes' least code points,
which makes o200k's other letters, title case letters, modifier letters and marks `#`, `$`, `%` and `&`. Such a symbol
is no member of its class, so an o200k class string is no text of its pattern, and the fixtures and the tests read one
with each of those symbols replaced by its class's least code point, U+00AA, U+01C5, U+02B0 and U+0300.

An alternative is its language over the class symbols, a value of munch's regex nodes built with munch's combinators,
and its gate as a string of class symbols: every symbol for an ungated alternative, the white space for `(?!\S)`, and
none for an alternative anchored at the end of the input. A spelling of the language in munch's pattern syntax, written
as a vendor's published alternative reads over the classes, is read into the same nodes by `munch::pattern::parse()`.
The pattern holds all of it by value, the alphabet with its sets and ranges, the languages and the gates, so a copy of
it, such as the one a report keeps, stands apart from the strings and tables it was built from. Everything the verifier
construction reads is in the description, so `split_verifier()` and `factorization_lexer()` take the pattern and nothing
else.

Each tokenizer library declares the same three functions in `matisse/<name>/tokenizer.hpp`. `pattern()` is the
description of the published pattern over its classes, 14 for GPT-2, 16 for cl100k and Llama 3 and 30 for o200k. Each
alternative is built with munch's combinators, below a comment quoting the vendor's alternative it describes, from small
documented pieces of the library's source, each one symbol of a set or literal the pattern names, such as `letter()` for
`\p{L}` and `white_space()` for `\s`, or the contractions, so that what the alternatives share is written once. Its sets
are copied from constants of the library's source, among them the Unicode tables generated into the library's
src/class_ranges.inc. `spelling()` is the published pattern as its vendor spells it, the regex the description
describes, which `read_tokenizer_json()` recognizes a tokenizer.json's pre-tokenizer by. `specials()` is the vendor's
special tokens of a rank file read under the pattern, which `read_tiktoken()` takes, quoted from tiktoken's
openai_public.py for the OpenAI encodings (for cl100k_base `<|endoftext|>` 100257, `<|fim_prefix|>` 100258,
`<|fim_middle|>` 100259, `<|fim_suffix|>` 100260 and `<|endofprompt|>` 100276) and from Meta's tokenizer.py for Llama
3's 256 special tokens. GPT-2's encoder.json and vocab.bpe are read by `read_gpt2()`, which needs nothing from the
description, and a file format of its own would be a reader binding in the tokenizer's library beside `specials()`. The
library's tests hold the description to the classes, segmentation and known-answer fixtures, each alternative's
combinators to its spelling over the class symbols in munch's pattern syntax, both compiled to the same minimal DFA
state for state, the pinned vocabulary to its digest, and the vendor's special tokens to their quoted values.

`lookup::tokenizers()` lists every tokenizer library by name, gpt2, cl100k, o200k and llama3, and `lookup::tokenizer()`
finds one by its name. The list is one function naming the libraries explicitly, the simplest form that needs no
registry. A library registering itself when it is loaded would rest on the order in which static objects are initialized
and on the linker keeping an object nothing names, and munch's style has neither. The lookup links every tokenizer
library, so a consumer of one tokenizer links that library alone and calls it directly.

### **Adding a tokenizer**

A tokenizer is added as a new directory under libs/, as o200k and Llama 3 were, and the stage libraries do not change.
libs/o200k/ holds its CMakeLists.txt, `matisse/o200k/tokenizer.hpp` declaring `pattern()`, `spelling()` and
`specials()`, src/tokenizer.cpp with the description, its alternatives built with munch's combinators below the vendor's
alternatives they describe, src/class_ranges.inc generated by tools/unicode/generate_classes.py with the sets its
pattern names, and its tests, which check the lines of its universe in the shared fixtures and hold each alternative's
combinators to its spelling in munch's pattern syntax. Outside the directory it is named in five places: its
`add_subdirectory()` in libs/CMakeLists.txt, its entry in `tokenizers()` with the link in libs/lookup/CMakeLists.txt,
its alias and umbrella link in the top-level CMakeLists.txt, its target and include directory in the install rules, and
its name in `matisse-validate`'s usage text. Its pattern must meet the conditions of the class theorem above, which the
fixtures hold it to. `split_verifier()` refuses a pattern that is not total and names a shortest residual no alternative
matches, which is where a wrong description shows first.

## **Classes, not code points**

A pattern reads a code point only through the sets and literals it names, so two code points of one class are
interchangeable: the pattern segments a text exactly as it segments the text with every code point replaced by another
member of its class. matisse therefore reads a text as its class string, and every automaton runs over a few class
symbols instead of the code points.

`pretok::Class_table` holds the class symbol of every code point under an alphabet, built once from its sets: a table of
the ASCII bytes, and a table of blocks of 256 code points with each distinct block kept once and every block's index
among them. A code point's symbol is then two loads and an ASCII byte's one. GPT-2's, cl100k's and Llama 3's alphabets
have 133, 134 and 134 distinct blocks, 39 KB a table, and o200k's 141, 41 KB. `symbols()` returns a text's class string
and `classes()` its classes, and `classes_of()` is `classes()` on a table built for the one text; the `Segmenter`, the
encoder and `supply()` each build a table of their own. Malformed UTF-8 is refused with its byte offset, since the model
is defined on code points and invalid text is outside it. The UTF-8 decoder is a small one of pretok's own, duplicated
rather than shared. The tokenizer libraries' tests hold the table to the first set holding each of the 1,112,064 scalar
values.

The Unicode tables are generated, not ported by hand. tools/unicode/generate_classes.py is munch's generator of the same
name, duplicated and changed to emit the sets named on its command line, each a general category or a property of
PropList.txt under the name of its table, with the source digests and the `--version` guard munch's has. It writes into
each tokenizer library's src/class_ranges.inc, which is checked in, so the build never touches the network. GPT-2's,
cl100k's and Llama 3's tables are the same file generated three times from L, N and White_Space, duplicated rather than
shared, so that a tokenizer whose pattern names other sets brings its own, as o200k brings Lu, Ll, Lt, Lm, Lo, M, N and
White_Space. This follows munch-audit's precedent for logos, whose reader keeps tables of the Unicode version logos is
locked to rather than the library's. munch's own tables are of Unicode 17.0.0 and have no L or N, and matisse does not
use them.

The tables are pinned to Unicode 16.0.0, the version of the tables tiktoken 0.14.0's regex engine carries, so that
matisse classes every code point as tiktoken does, and the version of Python 3.14's unicodedata, which the Python
reference the fixtures are written by reads, so that the fixtures replay. The classes, and therefore the verifiers, the
certified windows and the right context, do not depend on the version. Only the map from code points to classes does,
and with it the ids of a text holding a code point whose class moved between versions. Hugging Face's engine carries a
version of its own, and the vendor cross-check below shows no difference on any corpus. Moving the pin changes the ids
of some texts, so it is a major release, as the README's Versioning and Stability section says, with the fixtures
regenerated under the new version.

tools/unicode/check_tiktoken.py holds the pin to tiktoken, run with the newest tiktoken and the pinned rank files. Its
first check needs nothing else: every code point a tokenizer's tables leave out of their sets must be read by tiktoken
as of the other class, the code point and an apostrophe one piece before `s`, where a letter, a number or, under o200k,
a mark is a piece before the contraction, so a tiktoken release whose tables are of a later version fails it at the code
points that version assigns. Its second check, given a build's corpus probe, encodes every Unicode scalar value in five
contexts, one text per context, and every corpus named on its command line, with both matisse and tiktoken, and holds
the two to the same count and digest of ids. CI runs the first check with the newest tiktoken on every push, beside the
regeneration of the tables from the pinned database, which must leave them as checked in.

## **One automaton segments a text**

`pretok::Segmenter` is the runtime pre-tokenizer, the gated commitment order of the gate-sufficiency proposition read
over code points. At each boundary it commits the longest gate-passing accept of the first alternative that has one, an
accept passing when the class symbol after it is in its gate or the input ends there. A gate reads one class symbol, so
each alternative is compiled by munch as the DFA of its language followed by one symbol of its gate or a sentinel byte,
0xFF, which no class symbol is and which `boundaries()` appends to the class string. munch's longest-match `run()` of
that DFA at a boundary, less the one symbol, is the alternative's longest gate-passing accept, the end of the input
passing through the sentinel.

The alternatives' DFAs are walked at once as one automaton, their product, so that one `run()` at each boundary commits
the piece. A state of the product is each alternative's state and the first alternative that has accepted so far, every
later alternative stopped, and the product accepts where that alternative accepts. When the walk stops, the first
alternative that has accepted is the first alternative with an accept anywhere. From that alternative's first accept on,
the product accepts exactly where it does, and every accept before it is shorter, so the product's longest accept is
that alternative's longest. The product has 36 states for GPT-2's twelve alternatives and 33 for cl100k's eight.
`boundaries()` returns the piece ends as byte offsets.

The segmenter takes a total pattern and does not check it. Totality is decided once, where the validator builds the
verifier, and never in the segmenter's loop.

## **The verifier and the two certificates**

`pretok::split_verifier()` builds the verifier M of a pattern from the one-step relation Q of its alternatives,
minimizes it and hands it to munch as a `Verifier` table over the class symbols, each marked with whether a piece ends
after it. M accepts a marked class string exactly when its marking is the pattern's segmentation, so the marking the
`Segmenter` produces is the one M accepts. The table munch's constructor makes of M has 21 states for GPT-2, 32 for
cl100k, 46 for o200k and 33 for Llama 3, or 22, 33, 47 and 34 counting the dead state. As it builds Q, the construction
decides that the pattern is total, every nonempty residual matched by some alternative, which is the verifier fixture's
`covered` decision. It refuses a pattern that is not total, naming a shortest residual no alternative matches.

`pretok::factorization_lexer()` compiles the gate-erased union U of the alternatives over the class symbols as a munch
token set, each alternative's language one token ranked in the order of the pattern. Every piece the pattern commits is
a word of U, a gate only withholding accepts, so `is_split_window()` on this lexer decides the factorization
certificate.

The two certificates give the two inventories a caller chooses between. The factorization inventory's cut holds for
every complete factorization into the gate-erased languages, whatever the engine does inside its matches, and it needs
one code point of right context. The policy inventory's cut holds for the pattern's own segmentation, decided on M, and
so rests on Q, which is validated against the reference and not machine-checked; it needs two code points of right
context and lands on more boundaries. At the default width of five, on the three corpora of the test strategy under
GPT-2 and cl100k, the factorization anchors reach 57.30 to 92.88 percent of the pre-token boundaries and the policy
anchors 96.09 to 100.00, and under both the longest gap between anchors is within 24 bytes of the longest piece. The
encoder takes the inventory as an argument and has no default of its own. The recommended choice is the factorization
inventory, which rests on less, with the policy inventory taken by choice where cuts are scarce, and the probes and the
bench time the factorization inventory.

The construction of M needs the product, the complement and the concatenation of DFAs, which munch does not offer
publicly. By the division test a DFA kit is general and belongs in munch, and by the rule of duplicating rather than
sharing unless really needed it stays with the one construction that uses it. It lives in pretok, as automata.hpp and
its source, because nothing in munch calls it and a munch addition would wait for a munch release. The header is
installed, since munch keeps no private headers under src, but it is not part of the stable surface: its file comment
says so, the README's Versioning and Stability section names it outside the surface, and the kit moves if a second user
appears.

## **A pattern is validated once**

`validate::validate()` decides every class window of a pattern of width two to `width`, five by default, under both
certificates. The policy certificate is that of `munch::verifier::miscovering()` on the split verifier, and the
factorization certificate is decided by `is_split_window()` on the factorization lexer. A table holds its certified
windows as munch-audit's report holds them, `Certified_window` duplicated rather than linked, widths ascending and each
width's windows in the order the class symbols give, which is the windows fixture's order. An index over the class
strings is the encoder's to build, since windows of several widths share no single numbering.

The policy certificate is decided for every window at once, as the Python reference decides it and as munch's
`modulo_cuts()` decides its cuts. One walk over the windows carries each window's cloud: the states of the split
verifier an occurrence can leave it in, each with where the piece holding the occurrence's last symbol begins and
whether a piece ends after it. A window's cloud extends its prefix's by one symbol, and the window is certified at an
origin when every entry places the piece there. That is `miscovering()`'s certificate, and the walk takes `validate()`
to width four in 0.19 s for GPT-2 and 0.45 s for cl100k, and to width five in 1.1 s and 2.4 s, where a `miscovering()`
search per window would take 3.8 s and 9.3 s, and 92 s and 213 s.

The default width is five for that reason, and because GPT-2's policy anchors are complete at width five, every cut a
wider window certifies being one of them, as the report on pre-tokenizer certification proves. o200k's 30 classes make
each further symbol thirty times the windows. `validate()` takes o200k to width three in 1.5 s, to four in 16 s and to
five in 245 s and 2.9 GB. Width four adds to width three's factorization anchors 23 on munch's C++ sources and none on
twitter.json or wiki-zh, while the policy inventory gains 20,484 there, from 97.18 to 99.28 percent of the boundaries.
o200k's tests, the corpus probe and the bench, which time the factorization inventory, therefore decide its report to
width three, the width of its windows fixture.

Each table carries the right context its inventory needs, decided by munch's `right_context()` on the split verifier,
with the shortest truncation at every context that fails. It is asked of the windows that hold no shorter certified
window whose origin lands on the same symbol, since every cut of one that does is a cut of the shorter one. To width
four, for GPT-2 that is 87 and 540 windows of 18,357 and 32,218, and for cl100k 112 and 678 of 30,576 and 58,580. The
report carries munch's `boundary_gap()` of the split verifier, which is unbounded for every pattern in scope, since a
run of letters of any length is one piece, of lower case letters under o200k.

A report also holds a copy of the pattern it was decided of. Its constructor is private with `validate()` its friend, as
munch's `Lexer` exists only over what its `Builder` compiled, and its members are `const`. A report therefore holds its
window tables beside the pattern they were decided of for as long as it lives, it is copied and moved whole and never
assigned to, and the encoder and `supply()` read the pattern from the report. The encoder takes the report rather than
deciding it, so a pattern is validated once however many encoders are built, a fuzz harness's and the thread-sanitizer
job's among them.

`deserialize()` is the constructor's other friend. It reads back the text `serialize()` writes of a report, plain ASCII
with one fact per line as the fixtures are written, every window of a width written as one character in the order the
class symbols give, the digit of its certified origin or `-`. The Python package decides the reports of its four pinned
tokenizers when it is built and ships their texts, so that a process reading a pinned vocabulary reads its report back
instead of deciding it, as [python.md](python.md) describes. `deserialize()` checks the text's form and its class
symbols against the pattern's and not its windows, which only deciding them would check, so the report it returns is as
certified as the text it read. Both functions stay outside the stable surface.

`validate::supply()` measures one inventory's anchors on an input: the interior positions where an occurrence of a
certified window places its origin, counted at the byte offsets of their code points. It classes the input with the
alphabet of the pattern the report holds and segments it with that pattern's segmenter for its pre-token boundaries.
`Anchors` and `Gaps` are munch-audit's, duplicated rather than linked (count, per kibibyte, median, ninetieth and
ninety-ninth percentile, longest). The gaps are the distances between consecutive anchors, the runs before the first and
after the last left out, and a percentile is the order statistic at floor(q n) of the n gaps ascending. The report on
pre-tokenizer certification measures its gaps otherwise, with the two ends of the corpus counted as cuts, the
percentiles at round(q (n - 1)) and the density per 1,000 bytes, so matisse's gap figures and density are not that
report's and are not compared with it. `Supply` adds two figures beside them: the share of the serial pre-token
boundaries that are anchors, and the longest piece, which is the floor no cut rule can go below.

`render()` and `json()` follow munch-audit's report renderings. The text is rows of a label and a value, the verdict
first, then each inventory's windows, counted by width with six examples, and its right context with the shortest
truncation of every context that fails, then the boundary gap. The JSON is one object with a member per figure, class
strings holding each byte as the code point of its value. `supply_section()` and `supply_json()` render the supply of
both inventories on one input as munch-audit's renderings of its supply do, in supply.hpp beside the report, as
munch-audit keeps its supply in a unit of its own. `matisse-validate` in tools/validate is the command line, modeled on
munch-audit's, with munch-audit's exit statuses and with the supply section only under `--input`, so that runs without
it are unchanged by it, as munch-audit's `--input` is. [usage.md](usage.md) gives its options.

## **Two merge rules, each a vendor's**

A definition merges each piece by one of two rules. `Merge_rule::ranks` is tiktoken's rank merge, which joins the
adjacent pair whose concatenation has the least rank, leftmost first, and needs only the ranks, so a definition
`read_tiktoken()` reads has an empty merge table. `Merge_rule::pairs` is Hugging Face's rule, one join at a time, the
adjacent pair of least rank at its leftmost occurrence, which is its BPE model's `merge_all`.

GPT-2's encoder.py joins every occurrence of the least pair in one pass instead. Call C the condition that no part of a
pair is formed by a pair of a later rank. On every piece of a table satisfying C the pass and one join at a time give
the same tokens, each pass then being a run of single joins. Without C they can differ: under (ab, a) then (a, b), abab
gives ab ab under the pass and aba b one join at a time, and a tokenizers model with that list loads. The agreement
under C is not a new result. It is Lemma 1 of Berglund and van der Merwe's Formalizing BPE Tokenization (NCMA 2023), the
equivalence of its Definition 3, one merge at a time at the leftmost occurrence, and its Definition 4, the pass, which
holds under the ordering condition that clause (b) of their revised Definition 5 (2024) states, which is C. GPT-2's
table satisfies C. `read_gpt2()` checks it once at load and refuses a table outside it, naming the merge, its rank and
its line, so on every file `read_gpt2()` accepts the pair rule is encoder.py's. It also refuses a pair listed twice,
which both vendors would read at its last rank. On a tokenizer.json the pair rule is Hugging Face's rule itself, Llama
3's list included, which violates C in 149,956 places, so `read_tokenizer_json()` reads the merges as the file lists
them and checks no C; it refuses a pair listed twice, as `read_gpt2()` does.

`whole_piece_lookup` says whether a piece that is itself an ordinary token is that one token, looked up before any
merge. tiktoken looks every piece up, so `read_tiktoken()` sets it, encoder.py never does, so `read_gpt2()` leaves it
false, and `read_tokenizer_json()` sets it to the BPE model's `ignore_merges`, which is true in Llama 3's file and
absent, so false, in GPT-2's and in Xenova's conversions of cl100k and o200k. It changes nothing on cl100k, o200k or
GPT-2, where every multi-byte token is formed by the merge rule from its own bytes, GPT-2's `<|endoftext|>` aside, which
its pattern never leaves as one piece. It is load-bearing on Llama 3, where 588 of 127,744 multi-byte tokens are not
formed from their own bytes.

A definition is held in raw bytes throughout, and it names no pattern, since the pattern is the tokenizer library's
description, which the stages take beside the definition. GPT-2's byte-to-character map is undone by the loader, so
`"Ġthe"` is read as `" the"`. `Token_id_t` is 32 bits because ids are what a model reads, and every vocabulary in scope
fits.

## **Reading a tokenizer.json**

`read_tokenizer_json(path, published)` reads a Hugging Face tokenizer.json with a BPE model. It takes the published
spelling of the pattern the file is read under, which the tokenizer library's `spelling()` gives, and recognizes the
file's pre-tokenizer only by that exact spelling. Three pre-tokenizers are recognized, each with `add_prefix_space`
false. A ByteLevel with its regex, `use_regex` true or absent, runs GPT-2's pattern as encoder.py spells it, so it is
recognized under `gpt2::spelling()`. A Sequence of a Split on the published spelling followed by a ByteLevel without its
regex runs that pattern, the Split isolated and not inverted, as Llama 3's file writes it, or removed and inverted, as
Xenova's conversions write it, which keeps the regex's matches and drops what lies between them, nothing under a total
pattern, so that both keep the same pieces. The empty spelling reads a ByteLevel without its regex alone, which maps the
bytes and splits nothing, for a merge table without a pre-tokenizer, which the next section certifies. A null
pre-tokenizer is none of them, since Hugging Face's BPE then merges a text's characters and not its bytes: under GPT-2's
vocabulary, whose space is the character `Ġ`, it drops every space.

A spelling is the vendor's regex, and the description describes it under the vendor's engine, so a spelling is read only
where Hugging Face's engine reads it as published. Oniguruma, the engine of Hugging Face's tokenizers, reads the
possessive `?+`, `*+` and `++` as the vendors' engines do, but its Ruby syntax has no possessive interval and reads
`{1,3}+` as the interval repeated. cl100k's published pattern holds `\p{N}{1,3}+`, which tokenizers 0.23.3 reads as
`(?:\p{N}{1,3})+`, so that a1234b is the pieces a, 1234 and b where tiktoken's are a, 123, 4 and b. A Split whose
pattern holds an interval followed by a plus is therefore refused, and no tokenizer.json is read under
`cl100k::spelling()`. The cl100k conversions on the Hugging Face Hub, Xenova/gpt-4 among them, carry cl100k's pattern as
tiktoken shipped it before its possessive rewrite, which is Llama 3's pattern, so they are read under
`llama3::spelling()`: cl100k's vocabulary under Llama 3's description and verifier. On every corpus of the vendor
fixture that Hugging Face ran on, its ids of Xenova's file are tiktoken's cl100k ids.

The vocabulary's spellings and the merges' parts are read with GPT-2's byte map undone, as the ByteLevel maps a text's
bytes before the model reads them. The `added_tokens` marked special become the special tokens, by id ascending, and the
ordinary tokens are the vocabulary's other entries, their ids dense from zero: GPT-2's `<|endoftext|>` stands in its
vocabulary at 50256 and leaves the ordinary tokens as a special token, the special tokens of Xenova's conversions stand
in their vocabularies above the ordinary ids, and Llama 3's stand in its added tokens alone. Hugging Face reads no added
token's own id but gives it one, in the file's order: its id in the vocabulary when its spelling is there, and otherwise
the vocabulary's size, or one past the greatest id given before it when that is no less. The reader refuses a special
token whose id is another, naming the id Hugging Face gives it, so that every special token stands at Hugging Face's id.
The ids are those Hugging Face's `encode()` gives with `add_special_tokens` false. A post-processor is taken when it
adds tokens only under `add_special_tokens`: ByteLevel, which moves offsets alone, TemplateProcessing, as Llama 3's file
adds its `<|begin_of_text|>` with one, or a Sequence of the two, and a caller that wants such a token adds its id, as
[limits.md](limits.md) says of every beginning-of-sequence token.

The reader refuses, naming the member by its path from the root, such as `pre_tokenizer.pretokenizers[1].use_regex`,
what it holds and what the reader takes there: a model other than BPE, dropout, a continuing-subword prefix, an
end-of-word suffix or byte fallback; a normalizer, truncation or padding other than null; any other pre-tokenizer or
pattern, the empty pattern among them, which splits a text into its characters, and any other post-processor; an added
token that is not special, since Hugging Face matches every added token in the text before its pre-tokenizer, or that
strips white space or matches single words; ordinary ids that are not dense from zero, an empty token, two tokens with
one spelling, a character the byte map does not carry, a merge that is not two parts, and a pair listed twice. The
model's `unk_token` is not read, since the encoder refuses a definition in which a byte is no token.
`tokenizer_json_spelling(path)` returns the spelling a file's pre-tokenizer runs, by the same recognition, for a caller
that finds the tokenizer library by the file, as `matisse-validate` does through lookup.

load's tests hold every recognition and refusal on small files the tests write. The tokenizer libraries' tests read the
pinned tokenizer.json files: GPT-2's under `gpt2::spelling()` holds encoder.json's tokens but `<|endoftext|>`,
vocab.bpe's merges and the special token; o200k's and Llama 3's hold their rank files' tokens and the vendor's special
tokens; Xenova/gpt-4's holds cl100k_base.tiktoken's tokens under `llama3::spelling()` and is refused under
`cl100k::spelling()`, naming the pattern it carries. The encoder tests of gpt2, o200k and llama3 hold the ids of the
definitions read, Xenova/gpt-4's among them under Llama 3's pattern, to the vendor fixture's Hugging Face ids on the
three corpora of the test strategy, and `matisse_tokenizer_json_fuzzer` feeds the reader arbitrary bytes.

## **A merge table without a pre-tokenizer**

A merge table published with no pre-tokenizer, as a tokenizer.json whose pre-tokenizer is a ByteLevel without its regex,
which `read_tokenizer_json()` reads under the empty spelling, merges a whole text as one piece. It has no pattern and so
no report, and the `Encoder`, which segments by the report's pattern, has none for it: its cuts are the windows its
merge table certifies itself. `validate::validate_merges(definition, candidates)` decides them by munch's lazy
certification. It builds munch's `token_automaton()` of the table once, decides each candidate by `miscovering()` on it,
and returns each candidate as a `Merge_window` of its bytes, the origin it was decided at and its refutation or none.
The decision rests on the canonical adjacency theorem of the report The Left Ripple of Byte-Pair Encoding: a list of
tokens is the encoder's segmentation of its spelling exactly when every token and every adjacent pair is, so a
miscovering, when there is one, is spelled by the tokens over one occurrence of the window, at most 2L + |window| - 2
bytes, L the longest token, and the search over those is exact. A refutation is the encoder's segmentation of such a
string, the first the search finds. A certified window's cut is a boundary of every text's encoding, and by the same
theorem the bytes on either side of a boundary are encoded alone as the whole text encodes them, so a caller that cuts
there reads no byte past the cut.

The encoder decided is canonical byte-pair encoding, the merge of least rank applied at its leftmost occurrence until
none applies, which is the pair rule. A definition under the rank rule carries no merge table and is refused, and so is
one with the whole-piece lookup, which with no pre-tokenizer would take a whole text that is a token as that one token,
outside the encoder decided. munch's `token_automaton()` refuses a table in which a merge names a part that neither a
byte nor an earlier merge carries, or recreates a token that a byte or an earlier merge already carries, as (c, c), (c,
cc), (ccc, a), (cc, c) recreates ccc, and that refusal passes through with munch's message naming the merge and its
rank. `read_tokenizer_json()` takes such tables, since Hugging Face's BPE model reads them, but the automaton is built
merge by merge over the parts made before, and it certifies nothing of them.

A candidate is a window, not a window with an origin. Every byte string is encoded, the window among them, so at its
occurrence in itself the token covering its last byte begins where the window's own encoding begins it, and that is the
only origin that can be certified. `validate_merges()` encodes each candidate alone by the table and decides it at that
origin, as munch's `chunk_plan()` decides a window of a byte table, so a refuted window certifies no cut at any origin
and a caller has no origin to choose. The candidates are the caller's, since every window of a width is too many to
decide at full scale. On GPT-2's table of 50,000 merges the automaton has 50,001 states, one past the base state for
every merge since every product is read, is built in 0.03 seconds and holds about 8 MB, so its memory is no concern at
that size; a decision takes on average about 0.1 ms on a byte, 0.2 ms on a pair and 0.5 ms on a triple, so that the
16,777,216 windows of three bytes would take over two hours.

`matisse-validate` draws the candidates from `--input`: every byte the input holds and its most frequent pairs and
triples, `--candidates` of each, 1,000 unless given, the most frequent first and a tie in the order of the bytes. On
twitter.json under GPT-2's table that is 177 bytes, 1,000 pairs and 1,000 triples, of which 4, 488 and 527 certify a
cut, decided in 0.7 seconds; of every pair and triple twitter.json holds, 4,874 and 14,383, which with its bytes are
19,434 windows decided in 7.4 seconds, 13,597 certify. The four bytes that certify alone are the UTF-8 lead bytes 0xCB,
0xDB, 0xE4 and 0xEA, each beginning every token that holds it, and every other byte of the input lies in some token that
begins before it, as the e of he does. Among the pairs and triples that certify are `"e,"` and `": "` at 1 and two
spaces at 1, which no token the encoder emits holds, the table having been learned on text GPT-2's pattern had cut.
`render_merges()` and `merges_json()` write the windows decided, as text the certified and then the refuted ones counted
by width with examples, and as JSON each in the order decided with its refutation or null.

validate's tests hold every window of one to three bytes over a, b and c, under six written tables and 300 seeded ones,
to a scan of every string over the three of up to 2L + 1 bytes, past which no refutation is needed, and on the written
tables and 40 of the seeded ones to `miscovering()` on munch's merge tower of the same table, and they replay every
refutation as the encoder's segmentation of its text with its window covered elsewhere at the occurrence. gpt2's tests
decide twitter.json's candidates on GPT-2's pinned table and hold the cut of every certified window at every occurrence
in twitter.json encoded whole by `Merger`, and every refutation to `Merger`'s segmentation of its text. The command
line's tests hold its text and JSON on a written table and its refusals.

## **One merge, three routes, the same ids**

`encode::Merger` holds a definition's whole-piece lookup and its merge, built once when the encoder is built. It refuses
a definition in which a byte is not a token, since every piece starts as its bytes, and under the pair rule a merge
whose part or product is not a token, naming the byte, or the merge, its rank and the spelling.

The id of every spelling is a `Spelling_table`, a flat table with open addressing over the spellings' bytes kept once in
a string, each slot holding a spelling's hash, where its bytes stand, their length and its id, with the least power of
two of slots at least twice the spellings. A lookup hashes the piece by `std::hash<std::string_view>` once and compares
bytes only in a slot of the same hash and length. The whole-piece lookup reads it, which makes `encode()` on every
corpus 1.08 to 1.28 times faster under cl100k. The tests hold it to a map keeping the first id of each spelling on
20,000 seeded random spellings.

The merge keeps the parts of a piece as a doubly linked list over its bytes, as Hugging Face's `Word` keeps its symbols,
and the candidate joins in a binary heap ordered by rank and then by where the left part starts. Each entry records that
start and the ends of both parts, and an entry popped whose parts no longer stand as recorded is discarded. The merge
looks every pair up by its parts' ids, and the rule enters only in that table. Under `Merge_rule::ranks` a pair is in it
exactly when the parts' concatenation is a spelling, at that spelling's rank, which is its id, 233,378 pairs on cl100k's
file. Under `Merge_rule::pairs` a pair is in it exactly when the merge table lists it, a pair listed twice taking its
last rank as both vendors read it.

That table is a `Join_table`, a flat table with open addressing built from the map of the joins, with the least power of
two of slots at least twice the joins, 524,288 on cl100k's file. A key's walk starts at its hash, by `hash_combine` from
Boost's container_hash, the header-only module munch fetches, and runs to the next slot until it meets the key or a free
one. Every key is placed at the first free slot of its walk, and the slots that placement passed over were full then and
no slot is ever emptied, so a lookup meets every key the map holds before a free slot and stops at a free slot for any
other. The tests hold the table to its map on every pair of 300 ids, a third of the pairs drawn into the map.

The heap merge is exact on every table and every piece. A part's end only grows and a part joined into the one before it
never stands again, so an entry once stale stays stale. Every adjacent pair the rule can join has a live entry, since
the initial pairs are pushed and each join pushes the two pairs it makes. So each join is of the pair of least rank
present, the leftmost on a tie. Under the rank rule that is tiktoken's `_byte_pair_merge`, with a join that makes a pair
of lower rank than its own, such as `.:.:` on Llama 3's rank file, needing no case of its own, and under the pair rule
it is Hugging Face's `merge_all`. A piece of n bytes takes at most n - 1 joins and fewer than 3n entries, each entry's
and each node's offsets held in 32 bits, so the heap merges a piece shorter than 2^32 bytes and `Merger::merged()`
refuses a longer one.

A piece of at most 128 bytes is merged instead by a scan of its parts, held in an array on the stack, each part with the
join it makes with the next. The scan finds the least rank from the left, so the leftmost on a tie, makes that join and
looks up the joined part's pairs with its two neighbors. That is the join the heap makes and the two pairs it pushes, so
both make the same joins in the same order. With the scan and the heap alone, the encoding of Chinese with no cache
gains as the threshold rises to 128 and no further. The heap and the scan read the first pairs, every part a byte to
begin with, from a table of the joins of all 65,536 pairs of bytes, filled from the join table at construction, which
makes `encode()` on Chinese 1.20 and 1.16 times faster under GPT-2 and cl100k with the scan and the heap alone.

The tests hold the merge to the plain pair loop and to tiktoken's scan, written out in the tests, on every multi-byte
token of GPT-2's, cl100k's, o200k's and Llama 3's files (50,001, 100,000, 199,742 and 127,744 tokens) and on 20,000
seeded random pieces of each, one in 32 of them cut at 100 to 200 bytes, either side of the scan's threshold, GPT-2's to
encoder.py's pass as well. They also hold it on 3,000 random tables over three symbols, every tenth with one piece of
100 to 160 symbols beside its short ones, which make ties, overlaps, joins that make a pair of lower rank, and tables on
which the pass and one join at a time differ. The merge forms every multi-byte token from its own bytes on cl100k and
o200k, all but `<|endoftext|>` on GPT-2, and all but 588 on Llama 3, which Llama 3's whole-piece lookup gives whole.

### **The backtracking**

The `Merger` also builds a `Backtracker`, and a piece longer than 128 bytes, or one at least half of whose bytes lie
outside ASCII, is merged by backtracking instead. A token encodes to itself when the merge of its spelling ends with it
alone. Its split is the pair of parts that merge joins last, and its order is the rank of that join plus one, zero for a
byte. A token's own merge joins in order of rank when the parts of every split in it, its own among them, have lower
orders than the token they make. The constructor decides that once for every token, reading each token's split from the
merge of its own spelling run until two parts are left and taking the tokens shortest first, so that both parts of a
split are decided before the token they make. On GPT-2's, cl100k's and o200k's files every token's merge joins in order.
On Llama 3's 107 do not: 81 of its tokens have a split with a part of higher rank, `.:.:` at 100,421 the first, joined
from `.:.` at 105,051 and `:`, and 26 more are built on them.

The backtracking lays along the piece, from the left, the longest token that encodes to itself and begins the rest of
the piece, read from a `Token_trie`, the trie of those tokens in a double array. It holds each token to the one before
it by a pair test and tries the longest token that is a proper prefix of it when the test fails. When no prefix is left
it takes the last token back and marks the position it went back from, at which no token is laid again. Where the pair
test would be asked about a token whose own merge does not join in order, the search gives way, and the piece is merged
by the scan or the heap as any piece the backtracking does not take is.

The pieces it takes are where it pays. Timed piece by piece on nine corpora under GPT-2 and cl100k, it takes 0.28 to
0.90 times the scan's time on a piece of up to 128 bytes at least half outside ASCII, and 0.10 to 0.34 times the heap's
on the pieces longer than 128 bytes, which only `twitter.json`, `wiki-zh` and enwik8 hold. On the other pieces it takes
1.16 to 2.36 times the time under cl100k, whose whole-piece lookup leaves the merge only the pieces that are no token,
and 0.60 to 1.38 times under GPT-2.

### **Why the backtracking gives the merge's ids**

The backtracking's ids are the heap merge's on every piece. The argument has four steps.

1. **Interval projection.** While no join of a run crosses either end of an interval, the joins inside it are a prefix
   of the run on the interval alone, the least pair inside being the least among the interval's own. So a run that forms
   a token forms it by its split, and no run forms a token that does not encode to itself.
2. **Order.** A token's own merge joins in order of rank, the leftmost first among equal ranks, exactly when the parts
   of every split in it have lower orders than the token they make. If a join of rank r makes a part p and the next join
   has a lower rank, that join holds p, since every other pair stood when the join of rank r was the least, and it forms
   a token whose split has the part p of higher order than the token. A pair holding p at rank r would be the join that
   made p, which no pair holding p is, being longer than p. Conversely, a split whose part has a higher order than its
   token is a join of lower rank made after the join that made the part.
3. **The pair test.** Take two tokens whose own merges join in order. Until a join crosses between them, the run on
   their spellings is the joins of their own merges in order of rank and then of place, the left token's first on a tie,
   since the least pair on each side is the next join of that side's own merge. The parts beside the boundary are then
   one on the left token's right spine and one on the right token's left spine. Going back from the two tokens, each
   step undoing the later of the joins that made the two parts, lists every pair of them that stands beside the
   boundary, each until the join undone. The pair's join, if it has one, comes first exactly when its rank is below the
   undone join's on the left, which precedes it on a tie, or at most the undone join's on the right, which it precedes.
   So the test answers whether the merge of the two spellings ends with the two tokens, their compatibility. The first
   join of the replayed run that crosses is a join the real run makes, which is the split of the token it makes,
   whatever that token's own merge. The test therefore looks up the splits' pairs of every token that encodes to itself,
   and only where each part's 32 bits of hashed partners hold the other.
4. **The local characterization.** Tokens t_1 to t_k that encode to themselves, every adjacent pair compatible, are the
   merge of their concatenation x, under the rank rule and under the pair rule, with no condition on the table. Take the
   first join of the run on x that crosses a boundary between two of them, t_i and t_(i+1). Before it every part lies
   inside one token, so the join lies inside t_i t_(i+1) and neither end of that interval has been crossed, and by
   interval projection the run on t_i t_(i+1) alone crosses its middle, against their compatibility. So no boundary is
   crossed, and the run ends with each token's own merge, the token itself. The local characterization of byte-pair
   encoding is not new: Vieira et al. prove it for merge lists in Language Models over Canonical Byte-Pair Encodings
   (ICML 2025), and it follows from the gluing corollary of Berglund and van der Merwe's Formalizing BPE Tokenization.
   The proof here is a short one that also covers the rank rule.

The search ends with such a sequence unless it gives way. Until then every answer of the pair test it acted on was about
two tokens whose own merges join in order, so it was their compatibility, and the search has made the moves of a search
whose test is exact on every pair. The tokens that search has laid before a position are compatible and encode to
themselves, so they are the merge of the bytes before it, one sequence however the search reaches the position, and a
position from which no token fits once fits from none. Every token that encodes to itself and begins at a position is
tried there, longest first, and the merge of the piece is such a sequence, so the search reaches the piece's end with
the merge. A search that does not give way is that search to its end, and a piece on which it gives way is merged by the
scan or the heap, so every piece has the heap merge's ids.

### **Where the backtracking gives way**

A split with a part of higher order breaks the pair test alone. Interval projection and the local characterization hold
on every table, and the search asks of the test only that it answer compatibility. On a token whose own merge does not
join in order a later join can rank below an earlier one, so going back by orders undoes the joins beside the boundary
in the wrong order. A pair beside the boundary can then be joined while it stands though it ranks above the join that
ends its stand, once the next joins of both sides have risen above it. Under aab and then aa after the bytes, aab, of
order 257, is joined from aa, of order 258, and b, and the test reads the pair of aa and aab back as if the left aa were
the later join and refuses it, though aa aab is the merge of aaaab.

Three other ways were weighed. A pair test that read the run in its true order would need, for every pair beside the
boundary, the greatest rank the lesser of the two sides' next joins reaches while the pair stands. That turns on how the
two tokens' merges interleave and not on their spines alone, and the four steps would need a fifth for it. A routing
test could send to the scan or the heap every piece holding the spelling of a token whose split has a part of higher
order, one of 81 on Llama 3's file. It keeps the proof as it stood on the pieces it lets through, but it reads every
piece for those spellings before merging it and sends on at least every piece on which the search gives way, since a
token the search asks about lies in the piece and holds such a spelling. A trie without the tokens whose own merges do
not join in order never asks about one and fails exactly on the pieces whose merge holds one, but only after searching
every position it reaches, and the heap then merges the piece afresh. Giving way costs at most the search up to the
first test it would ask about such a token, reads one flag a token beside the split the test reads anyway, and leaves
the four steps as they stood but for the order they ask of the two tokens, so it is the way taken.

The method is GitHub's bpe crate's, bpe 0.2.3 at github/rust-gems 31331e8: the Corollaries III and IIIa of its README,
its backtracking encoder with the table of each token's next shorter prefix and the marks of the positions given up, and
its pair test that undoes joins. Its README assumes every multi-byte token is made from two tokens before it, and its
constructor checks only that every token encodes to itself. Its pair test reads a token's split as the longest prefix of
lower rank whose rest is a token of lower rank and whose pair its own test accepts. On a table whose tokens all encode
to themselves but where a split has a part of higher rank it returns other ids than the rule's: [aaa, b] on aaab for the
ranks a, b, ab, aaa, aa, where the rule gives [aa, ab], and [aba, a] on abaa for aa, aba and ab after the bytes, where
it gives [ab, aa]. matisse adds the order of each token's own merge decided at construction, the search giving way
beside a token whose merge does not join in order, which sends those pieces to the scan or the heap, and the splits read
from each token's own merge.

The tests hold the backtracking to the plain loop over the joins, wherever it does not give way, on every multi-byte
token of GPT-2's, cl100k's, o200k's and Llama 3's files and on 20,000 seeded random pieces of each. On those it gives
way only under Llama 3, on 20 of its tokens and 50 of its pieces. They hold it as well on 6,000 random tables over three
symbols, where it never gives way on the 4,446 whose every token's merge joins in order and gives way on 6,402 of the
pieces of the other 1,554. The tests read from the plain loop which tokens' merges join in order, 107 of Llama 3's not.
On those two tables and a third, aab and then aa after the bytes, on which the pair test would refuse the rule's own aa
aab on aaaab, they hold the search to give way where the test would be asked about the token out of order and the merge
to the rule's ids. The merge's references hold the routed merge on every multi-byte token of the four files and on their
random pieces.

## **A cache that never changes an id**

`encode()` looks each piece up whole first where the definition says so, then in a `Piece_cache`, and only then merges
it, storing its ids. The ids of a piece are a function of its bytes and the definition alone, the merge building its
parts afresh for every piece, so a cache keyed by the whole of a piece's bytes and filled under one definition never
changes an id.

A cache serves one caller and has no lock. `encode(text)` makes one for the call, `encode(text, cache)` reads and fills
one the caller keeps across its calls, one per thread, and every worker of `encode_parallel()` makes its own. A cache is
made for the encoder it serves, which it borrows, and `encode(text, cache)` refuses one made for another encoder with
`std::invalid_argument`, since its ids would be another definition's. It holds 16,384 pieces of up to 255 bytes and then
stops filling, with no eviction, as Hugging Face's BPE model's cache stops. On the three corpora of the test strategy
GPT-2's pattern makes 3,684 to 14,032 distinct pieces and cl100k's leaves 1,527 to 9,218 distinct pieces that are not
looked up whole, so the bound holds every piece the merge makes on them.

The cache is a flat table with open addressing over the bytes and the ids of its pieces, each kept once in a string and
a vector of its own. A slot holds a piece's hash, where its bytes and ids stand and their lengths. The slots are a power
of two in number from 64 on, doubled when the pieces fill half of them, so a call that stores nothing takes no memory,
and a lookup hashes the piece once and compares its bytes only in a slot of the same hash and length. A piece's ids are
valid until the next store.

The tests hold `encode()` to the known answers and to every piece looked up or merged with no cache, on seeded random
texts and on a text whose distinct pieces fill the cache past its bound, encoded twice more with one kept cache, and
`decode()` of `encode()` to the text, special tokens allowed and not. They hold `encode()` with a cache kept across 500
seeded texts, and on four threads each keeping a cache of its own, to `encode()` with a cache of its own.

## **Cut only where a cut is certified**

The constructor of `encode::Encoder` takes the report `validate()` decided of the pattern, as `supply()` takes it, and
keeps the inventory's window table and right context. It segments by the pattern the report holds, so a definition
without a pre-tokenizer has no encoder, and its cuts are the windows `validate_merges()` certifies of its merge table.
It refuses an `allowed` spelling that is not one of the definition's special tokens. The names follow munch's
`core::Lexer`, whose `tokenize_all()`, `chunk_boundaries()` and `tokenize_all_parallel()` are `encode()`,
`chunk_boundaries()` and `encode_parallel()` here.

`encode()` is the sequential encoding. The text is split at every occurrence of an allowed special token, the earliest
first and the longest of those that begin at one byte, which emits its id. Each stretch between them is segmented by the
`Segmenter` as a complete input, and every piece is then looked up whole where the definition's whole-piece lookup says
so and otherwise merged alone by its merge rule. Special tokens not allowed are ordinary text, which is tiktoken's
`encode()` with that `allowed_special` and an empty `disallowed_special`, and encoder.py's behavior when none is
allowed. tiktoken leaves the choice between allowed spellings that begin at one byte to the order of its table, and no
allowed spelling begins another on any definition in scope. `decode()` spells each ordinary id's bytes and each special
id's spelling as raw bytes, not checked to be UTF-8, since a run of ids cut inside a code point spells bytes that are
not.

`chunk_boundaries()` aims for `chunks` pieces of equal size, as `core::Lexer`'s does. Each interior boundary is the
first cut at or after the later of its equal-division target and one past the previous boundary, and a target with no
cut after it ends the walk. A cut is an anchor, a position `t + o` for an occurrence at `t`, inside one stretch, of a
window the table certifies at origin `o`, standing at least the width less one code points after the previous boundary.
The occurrence certifying it then begins in the chunk the cut ends, as the splice needs. An anchor stands as far after
its stretch's start as well, as the Python reference spaces its first cut from the text's start. A cut is also where an
allowed special token begins or ends, a cut of its own. The spacing is the Python reference's, and it leaves out an
anchor that stands closer than that to the previous boundary even when its occurrence begins after it. `Window_index`
holds the table's windows by their class strings and finds the least anchor of a stretch at or after a byte, classing
the text in blocks of 256 bytes from there, so that a walk reads the text near its target and no more.

`encode_parallel()` finds the allowed special tokens once and gives each chunk to a worker, one `std::jthread` per chunk
beyond the last, which runs on the calling thread, as munch's `tokenize_all_parallel()` does. A worker segments its
chunk's stretches as `encode()` does, the last one followed by `right_context` further code points of the real text,
read as a complete input there unless a special token or the text's end comes first. It keeps the pieces that end at or
before its cut and merges them with a piece cache of its own. The ids of the chunks, concatenated, are the result, and
an exception a worker throws is rethrown on the calling thread once every worker has joined.

There is no pool and no executor, and the caller chooses the chunk count. A pool is a question for later, to be settled
by measurement. 24 workers encode enwik9 at 557.5 and 626.1 MiB/s under GPT-2 and cl100k in the fairness section of
[benchmarks.md](benchmarks.md), where `encode()` on one thread runs at 54.6 and 57.1. A chunking at every anchor of
munch's C++ sources, 559,642 chunks, fails with `std::system_error` where the system cannot start that many threads, so
the corpus probe asks for at most 32,768 chunks.

## **Every streamed id is final**

`encode::Stream_encoder` encodes a text that arrives in parts. It is made from an `Encoder` it borrows, which must
outlive it and not be moved while the stream reads it, so the constructor taking a temporary encoder is deleted, and it
allows the encoder's special tokens. `push()` takes the next bytes and returns the ids that have become final, and
`finish()` ends the text and returns the rest, after which the stream reads a new text. A push may end inside a UTF-8
sequence or inside a special token's spelling, and those bytes are held until a later push completes them. A sequence
that cannot be completed is refused as `classes_of()` refuses it.

The stream claims one thing. After every push the ids it has returned are a prefix of `encode()` of the whole text,
whatever the rest of the text turns out to be, and after `finish()` they are `encode()` of it. What a push returns is
the longest such prefix, the greatest stable prefix of Dalirrooyfard and Ye's online tokenization, with three
exceptions.

- While the bytes read end in a proper prefix of an allowed spelling, those bytes are held, fewer than the longest
  allowed spelling, and not classed, as their online special-token segmentation holds them. A push then returns the
  longest prefix final for the bytes before them, and the ids can lag by more than the held bytes: a word before a held
  `<` stays an open piece when `<|endoftext|>` is allowed.
- While the bytes read end inside a UTF-8 sequence, its bytes are held and not classed until the sequence completes. A
  push then returns the longest prefix final for the code points before it, even where an id of those bytes is final
  under every completion, as GPT-2's byte token for `\xF4` is, since no token of two or more bytes begins with it.
- Inside a piece an id is held against every byte continuation of the piece, where the pattern allows fewer.

One `Encoder` serves any number of streams on as many threads with no lock, since a stream keeps its runs, its held
bytes and its piece cache to itself and reads the encoder through `const` accessors alone.

### **What the encoder builds for a stream**

The encoder builds what a stream reads once, beside the `Segmenter`: the class table of the pattern's alphabet, the
split verifier M of `split_verifier()` as a `Verifier_table`, the ids of the ordinary tokens in the order of their
spellings, and, where the definition looks pieces up whole, the tokens of two or more bytes that the merge does not form
from their own bytes, Llama 3's 588. The `Verifier_table` is a flat array of M's steps by state, class and mark, with
the states that accept and the dead ends, the accepting states with no move. Building these adds 9, 36, 69 and 49
milliseconds to building GPT-2's, cl100k's, o200k's and Llama 3's encoders, and a stream is then made in about ten
nanoseconds. A step of the table returns a state, `no_state` where M has no move, rather than an empty `std::optional`.
GCC 13 returns an optional through a byte and a word stored apart and read back as one word, which stalls every step.

### **The three layers**

The stream has three layers, each exact at its level.

The first cuts the special tokens. An occurrence of an allowed spelling, the earliest and the longest at its byte as
`encode()` finds it, is committed once no proper prefix of an allowed spelling begins at or before it in the bytes read.
The stretches between special tokens are then `encode()`'s on every completion of the text, and only the whole code
points before the earliest such prefix reach the second layer.

The second keeps the live runs of M over the stretch, a `Live_runs`. A run is a path of M over a marking of the code
points read, and since every state of M is reachable and reaches acceptance, the runs are exactly the restrictions of
the segmentations of every continuation, at most one per state of M. They are kept as a tree, one layer per code point
since the last stable one, each node a state, the mark that led to it and the number of its live children, held in two
vectors dropped from the front and erased in bulk. The front layer is stable once only one of its nodes is live. At the
last code point read every run must say the same of a piece ending there. A marked run says it ends, and an unmarked one
in a state that does not accept says it does not. An unmarked one in an accepting state says both unless the state is a
dead end, as after cl100k's three digits, where it ends. A piece that ends at a stable code point is closed and its ids
returned.

The third layer works inside each run's open piece. The ids returned of it are those of its merge up to the last
boundary that every byte continuation keeps. By interval projection that is decided by the continuations a token can
straddle the piece's end with, which are the empty one and the rest of every token that begins with a nonempty suffix of
the piece: a continuation's merge is the merge of the piece followed by one of them, up to its first boundary at or
after the piece's end. The test runs from the right and stops at the first boundary that holds, since the boundaries
every continuation keeps are closed downward. A boundary holds with no merge when no token runs across it from inside
the token before it, that is, when no token is a suffix of the token before the boundary followed by a prefix of the
bytes after it, or by all of them and more. The first join that crosses a boundary joins such a suffix, so such a
boundary holds, and it is tested first. Where pieces are looked up whole, no id is stable while a token the merge does
not form begins with the piece. A push returns the common prefix over the runs of each run's closed pieces followed by
its open piece's stable ids, less what has been returned.

### **The cursor**

The stream keeps one cursor, the byte that the ids returned since the stretch began reach, and works from it alone.
Those ids are a prefix of `encode()` of every continuation, so every continuation keeps the boundary at the cursor and
nothing before it is read again. A piece of a run, open or closed by the run, adds nothing once the cursor reaches its
end, and where the cursor stands inside it only the rest after the cursor is merged, with no whole-piece lookup. Each
node of the live runs keeps the byte its code point ends at and a link to the last node before it on its run after which
a piece ends, and a push reads each run's piece ends back from the last one only as far as the last at or before the
cursor.

The bytes before the cursor are erased in bulk, after each code point fed and after each push, once they are half of the
stretch's bytes held and at least 64, so each byte is moved a constant number of times on average. A piece closed behind
the cursor costs nothing but its end, even where one code point closes a long run of pieces. On a pattern whose
alternatives are `a+` anchored at the end of the text, then `a`, then any other code point, a run of `a` is held as
one-byte pieces until a code point other than `a` closes them all. What a push reads of the runs and merges therefore
grows with the bytes not yet returned. It does not grow with the part of a piece already returned or with the code
points the runs have read since they parted, however long they stay apart, as after a line feed and a run of tabs under
cl100k's pattern.

### **Where the construction comes from**

The third layer rests on results stated for one join at a time, the pair rule. They hold for the rank rule too, which
makes on every piece the joins that one join at a time makes on the dictionary of the tokens' self splits, the last join
of the merge of each token's own bytes, in the same order. By interval projection a join the rank rule makes is the self
split of the token it makes, and among the pairs of self splits standing in the piece the rank rule's choice is the
least, the leftmost on a tie.

The problem and its frame are Dalirrooyfard and Ye's, in Online Tokenization with Optimal Delay in Linear Time
(arXiv:2610.19139). They define the stable prefix of an online tokenizer and its delay, treat the special-token
segmentation, the pre-tokenization and the byte-pair encoding as three stages, and observe that across hard segment
boundaries the stable output is the output of every segment but the last and the stable prefix of the last. They show
zero delay for the composed pipeline NP-hard, give an algorithm of delay O(mt) in near-linear time, and leave open
whether byte-pair encoding alone has zero delay. The report *The Left Ripple of Byte-Pair Encoding* answers that for the
byte-pair stage, for every dictionary and with no tree. On the merge lists of Hayase et al. (arXiv:2506.14123), reading
the whole common prefix of their covering tree gives the same output at the cost of keeping the tree. The stable prefix
of a piece is the common prefix of its merge followed by the empty string and by each proper suffix of a token, a finite
set, and a zero-delay streamer holds at most the right span radius of the dictionary past what it has returned, 129
bytes for GPT-2's table.

matisse adds the combination and three steps it needs. The split verifier M, built for the report on pre-tokenizer
certification, decides the pre-token boundaries every continuation keeps exactly and with no delay, where an online
pre-tokenizer that commits a match once it cannot extend pays a delay. The rank rule is carried to one join at a time on
the self splits. And the continuations are restricted to those that can straddle the piece's end, from every proper
suffix of every token to the tokens that extend the piece's last bytes, with the whole-piece lookup folded in, which
makes the exact rule affordable at every push.

### **How the stream is tested**

The tests hold the stream to `encode()` on seeded texts pushed in random parts of up to 9, 24 and 64 bytes, a byte at a
time and whole, every push's ids a prefix of `encode()` of the whole text and all of them `encode()` of it. They run on
the tests' pattern and definition, special tokens allowed and not, and on GPT-2's, cl100k's, o200k's and Llama 3's
pinned files, for cl100k, o200k and Llama 3 with the vendors' special tokens allowed and not. On letter runs of the
tests' pattern pushed in parts they hold the ids returned to the common prefix of the merges of the run followed by the
empty string and by every proper suffix of every token, the stable prefix of the report on the left ripple over its
whole finite set. On the tests' pattern they hold every push to brute force over every continuation of up to four
fragments: its ids are a prefix of the encoding of each, and unless the bytes read end in a proper prefix of an allowed
spelling they reach every boundary the encodings of all of them keep. They hold the live runs over seeded class strings
to the segmentations of every continuation of up to five code points, the pieces closed to the boundaries those share,
the verifier table to munch's verifier, and streams on four threads over one encoder to `encode()`. The bench holds
every streamed corpus to `encode()`'s ids.

## **How exactness is guaranteed**

The encoder claims one thing: for every text the definition accepts, `encode_parallel()` returns the ids `encode()`
returns, at every chunk count. That is the splice theorem of the report on pre-tokenizer certification, with the
proposition that a certified cut transfers from class strings to the text and with the decided right context. It holds
given the theorem's four hypotheses (N), (S), (P) and (B), which the loaders' refusals enforce, and given that the
`Segmenter` is the engine's segmentation. The construction behind the verifier, the gate-sufficiency proposition, the
transfer of a cut and the splice theorem are results of that report, which is not yet published.

The rest of the chain is held in the code and its tests. The loaders refuse what they cannot take exactly. The segmenter
is held to the engine's own segmentation of every short class string. The merge is exact on every table and piece by the
arguments above and is held to the plain loop and to tiktoken's scan. The stream's claim is held to `encode()` and to
brute force over continuations. And `encode()` is held to the vendors' own ids on fourteen corpora, as the next section
describes.

The parallel claim is tested on seeded texts under both inventories, at every chunk count from one to sixteen, at
sixty-four and at every cut the rule takes, on a pattern of the tests' own and on GPT-2's, cl100k's, o200k's and Llama
3's with their pinned vocabularies, special tokens allowed and not on the tests' pattern and on cl100k's, o200k's and
Llama 3's. The tests hold `chunk_boundaries()` on the tests' pattern to brute force over every certified cut of the
report, and a cut in white space, at the right-context counterexamples, to the right context each inventory decided. Its
speed is what [benchmarks.md](benchmarks.md) measures, and nothing is claimed about the vendors' own implementations
beyond the ids and the speeds measured there.

## **Testing against the real tokenizers**

Each library has a GoogleTest suite registered with CTest, as hopper's do. The reference vocabularies are pinned by
SHA-256:

| File                                        | SHA-256                                                          |
|---------------------------------------------|------------------------------------------------------------------|
| gpt2-merges.txt (vocab.bpe)                 | 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5 |
| gpt2-vocab.json (encoder.json)              | 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783 |
| cl100k_base.tiktoken                        | 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7 |
| o200k_base.tiktoken                         | 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d |
| llama3.tiktoken                             | 82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55 |
| gpt2-tokenizer.json (openai-community/gpt2) | 8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6 |
| cl100k-tokenizer.json (Xenova/gpt-4)        | 239eb2359f79c38497476671aaa835e01fb43d42743c612a8514a0dfa2ac93a2 |
| o200k-tokenizer.json (Xenova/gpt-4o)        | 43a3ad4618a6a938f8c2614154ea10c31ad53a62d5683ae0b0e6133575cef07e |
| llama3-tokenizer.json                       | e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7 |
| twitter.json                                | 30721e496a8d73cfc50658923c34eb2c0fbe15ee6835005e43ee624d8dedf200 |

The vocabulary files are not in the tree. CTest without network needs them on disk, and their own terms would have to be
checked before they were vendored, so `MATISSE_VOCAB_DIR` names a directory holding them, each held to its digest, and
nothing is fetched at test time. Every test program of the libraries and the validator registers with CTest whether or
not a directory is named, and the tests in it that read the files skip when none is. The corpus probe and the bench in
tools/probes read all four tokenizers, so they register only when the directory holds every pinned vocabulary file but
the tokenizer.json files, which they do not read, Llama 3's rank file included. The three public tokenizer.json files
were taken once from the Hugging Face Hub, GPT-2's from openai-community/gpt2 at commit
607a30d783dfa663caf39e06633721c8d4cfcd7e, cl100k's from Xenova/gpt-4 at 1d9f1f1b1fae88c0e4df1dab0a397f8de6229075 and
o200k's from Xenova/gpt-4o at 7956d98f2a83b2751a98ea7136fdf7fe6cf54e69, the files [benchmarks.md](benchmarks.md) runs
Hugging Face tokenizers on, and Llama 3's, Meta's under the Llama 3 Community License as its rank file is, is held
beside the rank file.

### **The fixtures**

The ported units stay exact through fixtures written by the Python reference and held byte for byte, as munch's verifier
tests hold `token_automaton_reference.txt`. The reference, kept outside this repository, holds everything the port
needed: the class alphabets, the per-alternative languages and the one-step relation Q, the verifier M with its
functionality and totality decisions, the one-code-point gate, the certificates, the right-context decisions, and the
references and end-to-end run, over a small DFA kit. Each moved as one C++ unit of pretok or validate, written in
munch's style and modeled on its nearest munch neighbor: the verifier construction on `merge_tower()` and
`trie_chain()`, the window tables on munch-audit's report, the supply on munch-audit's supply.

A fixture script beside the reference code writes each fixture with a header line naming the script and saying to
regenerate with it, never by hand, one fact per line, bytes and class strings in hex. The C++ tests read every line,
check the answer, and count the lines of each kind. Each fixture file holds every tokenizer's universe and stays as the
script writes it, under the tests/data of the stage its subject belongs to, and each tokenizer library's tests check the
lines of its own universe and skip the rest. The fixtures, one per subject:

- **Classes**: the Unicode version, the class numbering by symbol, the range count and SHA-256 of each set the tables
  hold, L, N and White_Space, for o200k Lu, Ll, Lt, Lm, Lo and M in place of L, and the class of every code point on
  either side of a change of class.
- **Segmentation**: the engine's segmentation of every class string of lengths zero to four (41,371 strings for GPT-2
  and 69,905 for cl100k and for Llama 3, the script's default length; to five, the gate check's own population, they
  would be 579,195 and 1,118,481), and of lengths zero to three for o200k, 27,931 strings, 4.4 MB in all, against which
  the `Segmenter` is held, and the boundaries of the three corpora's class projections as digests.
- **Verifier**: the table munch's constructor makes of the minimized M (trimmed, start first, breadth-first over the
  marked symbols ascending), the script applying the same renumbering, so the C++ construction must produce the
  identical table state for state; and the reference's decisions that M is functional and total and that the pattern
  covers every nonempty residual.
- **Windows**: every certified window to width four under both certificates with its origin (18,357 and 32,218 for
  GPT-2, 30,576 and 58,580 for cl100k and for Llama 3), and to width three for o200k (10,365 and 17,888), and every
  refused one by count. The tokenizers' report tests decide theirs at the fixture's width to hold them to it, the
  default being five.
- **Right context**: the decided R of each inventory, and the counterexample of each failing R with its replay
  (`"\t\t!"` at cut two for R = 0, and `"\t !"`, `"\t\tA"` and, for o200k, the tabs before the other letter `#` at cut
  one for the policy inventory at R = 1).
- **Known answers**: the ids of short strings under each reference, among them the published `"Hello world"`, 15496 995
  for GPT-2, and `"tiktoken is great!"`, 83 1609 5963 374 2294 0 for cl100k and 83 8251 2488 382 2212 0 for o200k, with
  three more of OpenAI's examples from its openai-cookbook's How_to_count_tokens_with_tiktoken.ipynb at db31449, and
  Meta's `"This is a test sentence."`, 2028 374 264 1296 11914 13 for Llama 3, from meta-llama/llama3's
  llama/test_tokenizer.py at a0940f9, less the beginning and end of text ids it adds.
- **Vendors**: the fourteen corpora of [benchmarks.md](benchmarks.md) by file name, each with its bytes, the documents
  its NUL bytes separate and its SHA-256, held to the corpus directory's manifest, and under each tokenizer the count
  and digest of the ids tiktoken, Hugging Face tokenizers and the bpe crate gave on each corpus, its documents' ids
  concatenated, with the vendors that ran on it. They are read from the ids lines of the sessions the page's exactness
  table comes from, the vendors agreeing on every corpus and, on the three corpora of the test strategy, with the
  reference.

Where a C++ answer and a fixture line disagree, the test fails with both, the fix goes into whichever side is wrong, and
a fixture changes only by rerunning its script.

### **Three layers of tests**

The tests run in three layers, each against something independent of the code under test.

1. **The fixtures** of the previous section, in CTest: classes, segmentation, M, windows, right context and known
   answers.
2. **Self-consistency**, in CTest, on texts generated with a fixed seed as hopper's probe generates its corpus:
   `encode_parallel()` against `encode()` under both inventories at every chunk count from one to sixteen, at
   sixty-four, and at every cut the rule takes; the cuts of `chunk_boundaries()` against brute force over the report;
   `decode(encode(x)) == x`; M accepts the `Segmenter`'s marking of every text; every anchor is a `Segmenter` boundary;
   the streaming encoder's pushes against `encode()` under random parts, and against brute force over the continuations
   on the tests' pattern. The texts mix the known counterexample shapes, white space runs before letters and
   punctuation, newlines followed by tab runs, long letter runs, contractions in both cases, and allowed and disallowed
   special tokens.
3. **The corpora**, by the probe in tools/probes. `matisse_corpus` takes the vocabulary directory and corpus files,
   holds GPT-2's, cl100k's, o200k's and Llama 3's pinned files to their digests, a corpus the fixtures name to the
   digest of its text, and the ids of `encode()` to the known-answer fixture's digest of the reference ids (the SHA-256
   of the ids as little-endian 32-bit words, with their count). On every corpus it checks the ids of every piece merged
   with no cache and `decode()`, and for both inventories `encode_parallel()` at the worker counts past one that the
   bench times, at the 4,096- and 65,536-code-point chunkings and at every anchor, skipping a chunk count past 32,768
   since every chunk is a thread. At every anchor it also holds the pieces a worker keeps to the serial pieces, with no
   thread per chunk. Given the vocabulary directory alone it runs the generated texts of layer 2, which is the form
   registered with CTest, as hopper_edit_relex does, and `-DMATISSE_CORPUS_DIR` registers the three corpora as well. The
   corpora are twitter.json, and munch's 10 tracked Python files and its 306 tracked C++ sources and headers at
   15b2863c, concatenated in path order by the fixture script and pinned by digest. [benchmarks.md](benchmarks.md) adds
   eleven more, and the probe's ids lines are what the vendors' ids are compared by. The probe prints the supply table,
   which [benchmarks.md](benchmarks.md) quotes from its output.

### **The vendors themselves**

The reference ids in the known-answer fixture are the Python references, encoder.py's loop and tiktoken's rank merge
restated over Python's engine with Unicode 16.0, which are held to the published known answers. The vendor fixture holds
matisse to the vendors' own code as well. tiktoken, Hugging Face tokenizers and the bpe crate were run on the fourteen
corpora of [benchmarks.md](benchmarks.md) outside the tree, and the fixture script records the digests of their ids in
libs/load/tests/data/vendors_reference.txt, beside the known-answer fixture, holding them to the restated references' on
the three corpora of the test strategy. The fixture holds the fourteen corpora with their digests and the 56 counts and
digests of ids under the four tokenizers.

Given the corpus directory with `-DMATISSE_CORPUS_DIR`, each tokenizer's tests encode every corpus of the fixture the
directory holds and hold the corpus to its digest and the ids to the vendors'. Each document is pushed to a stream in
parts of a mebibyte, whose ids between two `finish()` calls are `encode()`'s of the bytes pushed between them, so each
push and the ids it returns stay bounded whatever the corpus's size, though the test reads the whole corpus first. A
corpus the directory lacks is skipped by name, and every one is skipped when no directory is named. On all fourteen,
4.14 GB, every digest holds, in 228 seconds under GPT-2, 235 under cl100k, 243 under o200k and 254 under Llama 3, each
test within 2.2 GB resident. A difference would be the Unicode version or a vendor's engine, and on every corpus each
vendor ran on there is none.

### **Fuzzing**

The fuzz harnesses in tools/fuzz follow hopper's contract: a refusal is the answer to bad input, and a crash, a
sanitizer report or a hang is a finding. Each is its own target with its own seeds under tools/fuzz/seeds, so that each
corpus holds inputs of one format. `matisse_tiktoken_fuzzer` writes arbitrary bytes to the rank file `read_tiktoken()`
reads, `matisse_gpt2_fuzzer` to the encoder.json and the vocab.bpe `read_gpt2()` reads, the input cut at its first NUL
byte, which no encoder.json holds, and `matisse_tokenizer_json_fuzzer` to the tokenizer.json `read_tokenizer_json()`
reads under the spelling `tokenizer_json_spelling()` finds. A `std::invalid_argument` is the answer, and a definition
read is held to what its reader promises: the merge rule and the lookup, tokens none empty and none spelled twice, under
GPT-2 and a tokenizer.json no pair listed twice, and a tokenizer.json's special tokens past every ordinary id and
refused under any other spelling.

`matisse_encoder_fuzzer` feeds arbitrary well-formed UTF-8 to the encoder, each maximal ill-formed subpart of the input
replaced by U+FFFD. It asserts that `encode_parallel()` equals `encode()` at a chunk count drawn from the input, from
zero to 255, and that `decode()` spells the text back, which turns the encoder's one claim into a fuzz property. Its
sixteen encoders are GPT-2's, cl100k's, o200k's and Llama 3's patterns, each report decided to width three before the
first input, under both inventories and a synthetic definition of each merge rule built in the harness. The synthetic
definition holds the 256 bytes, every string of two and three bytes over a, b, the space, the line feed and the two
bytes of é in a seeded order of rank, three tokens that only the whole-piece lookup gives, and the vendor's special
tokens, all of them allowed. No harness reads a vocabulary file, so the fuzz job runs without one.

### **Continuous integration**

CI follows hopper's workflow with munch's sanitizer matrix. It checks the format with clang-format-19, and builds and
tests with GCC 13.3 with coverage, with GCC in strict C++23, extensions off and warnings as errors, as munch's strict
job builds, and with Clang 19. An ARM64 job builds where plain char is unsigned. The address and undefined-behavior
sanitizers run, and so does munch's thread-sanitizer job, because the encoder spawns threads, which is also why the
coverage build takes munch's `-fprofile-update=atomic` flag. The fuzz job runs each harness for a minute, and an
installed-consumer job builds a consumer against the installed package. Every job runs on each push to master and each
pull request against it, and CodeQL runs on both and weekly. The coverage job fetches the seven public vocabulary files
the tests pin, GPT-2's two files and the two rank files from OpenAI and the three tokenizer.json files from the Hugging
Face Hub at their commits, held to their digests, so the real-vocabulary tests of the loaders, the encoders and the
command line run in the measured build. Llama 3's rank file and tokenizer.json, Meta's under the Llama 3 Community
License, are not fetched, so their tests skip there and the two probes' tests are not registered. Under the instrumented
build each tokenizer's suite runs for many minutes, so that job runs the suites four at a time, each given an hour.

## **What matisse needs from munch**

matisse builds against munch 3.1.0 and uses:

- `munch::regex::Regex` and its combinators for the alternatives' languages, and, in the tests,
  `munch::pattern::parse()` reading a spelling of one into the same nodes.
- `core::Builder` and `Lexer::simulator()`, which compile each alternative and its gate into a DFA, and `dfa::Builder`
  and `dfa::Simulator`, which build and run the `Segmenter`'s product automaton. The `Simulator` is from the layer munch
  keeps outside its stable surface, so matisse's promise holds with the munch release it names.
- `munch::verifier::Verifier`, built from a table, for M over the class symbols, which fit the byte alphabet (16 of 256
  for cl100k).
- `Verifier::step()` and `state_count()`, which validate's walk over the windows reads to decide the certificate of
  `miscovering(const Verifier&, window, origin)` for every window at once, with `miscovering()` itself in the tests, and
  `boundary_gap()` for the gap verdict.
- `Lexer::is_split_window()` for the factorization certificate on U, the procedure cross-checked against munch-audit
  exactly.
- `right_context(const Verifier&, window, origin)` for the right context of each certified pair, an inventory's being
  the largest of its pairs', and, in the tests, `dependence()` for the suffix half beside the case R = 0. The Python
  reference's decision is the reference and its counterexamples the fixture, as munch's context_reference.txt holds
  `right_context()`. `dependence()` returns the shortest refutation of either side, so at a window whose prefix already
  fails with no symbol past the cut it leaves the suffix half open. That happens at 12 of the 87 and 12 of the 540
  windows GPT-2's right contexts are asked of, under the factorization and the policy certificate, and at 4 of the 112
  and 24 of the 678 of cl100k's, every other one refuting neither side.
- `munch::verifier::Merge` as the entry of a definition's merge table.
- `token_automaton()` and `miscovering(const Token_automaton&, window, origin)` for the certification of a merge table
  without a pre-tokenizer, and, in the tests, `merge_tower()` and `miscovering()` on it.

matisse does not need munch's Unicode tables, which are of version 17.0.0 and have no L or N, a UTF-8 decoder, of which
pretok has a small one of its own, or a runtime, which is matisse's by the division.

munch does not yet offer four things matisse could use. Each is general, so it would go into munch by the division.

- **Every window's origin at once.** `miscovering()` decides one window at one origin, and validate's walk decides the
  certified origin of every window to a width in one pass, the shape of `modulo_cuts()`. A decision beside it returning
  each window's certified origin would hold the walk.
- **The suffix half alone.** The Python reference decides it for every boundary of M, and `dependence()` decides it only
  together with the prefix at R = 0. The fixture's suffix line is therefore held at every certified cut except at those
  open windows, and not at the boundaries of M that no certified window places. A decision of the suffix side alone, the
  restarted copy of `dependence()`'s product without the prefix flag, would close it.
- **Functionality and totality of a verifier.** The Python reference decides both for M, and `divergence()` presumes the
  first. matisse does not need them, since M's table is held state for state to the fixture whose M was decided
  functional and total, but they would let a new pattern's Q be checked by the validator itself.
- **Alphabets beyond bytes.** `Marked` carries an `unsigned char`, which holds every class alphabet in sight, o200k's
  included. It does not hold a character-level merge table, which the open questions below come back to.

## **Open questions**

**Character-level merge tables.** SentencePiece BPE tokenizers without a pre-tokenizer, Llama 2 among them, merge
Unicode characters, which munch 3.1.0's `token_automaton(merges, alphabet)` reads over an alphabet of characters,
deciding windows at character boundaries. They also carry a normalizer and byte fallback, so the lazy route covers
byte-level merge tables now and character-level ones once a loader supplies the alphabet and handles those two. A
SentencePiece loader would give munch the self-split table: for every piece, the pair its own characters merge last
under SentencePiece's loop, ranked by score, highest first. SentencePiece forms every piece by that split, so with
distinct product scores the score-ranked self-split table reproduces its encoder, which is Yi's case. Equal scores are
harder. Mistral's tied group of whitespace runs admits a split-and-rank compilation proved equivalent on every run,
while the two equal-scored pieces ab and ba admit no rank order at all, so a loader would search each tie group for an
output-equivalent table and refuse a model whose group admits none, naming the group. Whether a candidate table is
equivalent is decidable by a finite test over canonical tokens and adjacent pairs, so what remains open is the cost of
synthesis. Certification needs no table at all, since munch's lazy certification needs only the canonical tokens and the
adjacency relation, which SentencePiece's own encoder supplies as a pair oracle, once munch takes that relation
directly.

**Further tokenizers.** After GPT-2, cl100k, o200k and Llama 3, the recommendation is WordPiece through munch's
`trie_chain()` with bert-vocab.txt, which brings BERT's normalizer and so the first normalizer.

**Python bindings.** The PyPI name matisse is taken, and the package is matisse-tokenizer, imported as
`matisse_tokenizer`, which [python.md](python.md) describes: tiktoken's interface over the encoder, built from the
libraries by scikit-build-core and nanobind, with the reports of the pinned tokenizers shipped, and the streaming
encoder beside it, with no special token allowed, and `validate_merges()` over a tokenizer.json whose pre-tokenizer
splits nothing. The policy inventory and other widths are not reached from Python yet, and neither munch nor hopper has
bindings.
