# **Scope, Guarantees, and Limits**

matisse is exact because it refuses to guess, and fast because it refuses everything exactness does not need.
[benchmarks.md](benchmarks.md) measures what those two refusals buy. This note is the inventory: the model the library
commits to, the guarantees that commitment buys, the boundaries you can hit, and the escape hatch for each. Read it when
deciding whether matisse fits a tokenizer, not after.

## **The Model: The Vendor's Ids, or a Refusal**

matisse reproduces the token ids of a vendor's tokenizer exactly, or it refuses. A tokenizer reaches it as two things
read apart: a `load::Definition`, which holds the merge rule, the whole-piece lookup, the bytes of every ordinary token,
the merge table and the special tokens, and a `pretok::Pattern` from the tokenizer's own library, which describes the
published pre-tokenizer as data. A pattern is the contract of the class theorem of the report on pre-tokenizer
certification: an ordered gated pattern over a class alphabet, each alternative a regular language of class strings with
a gate, the class symbols that may follow its accepts. At each boundary it commits the longest gate-passing prefix of
the first alternative that has one. The theorem carries a cut to the vendor's engine only when the pattern is total, and
`split_verifier()` decides that as it builds the verifier, refusing with `std::invalid_argument` a pattern on which some
nonempty residual matches no alternative and naming a shortest such residual. `validate()` builds the verifier before
anything else, so a pattern that is not total gets no window table and therefore no `Encoder`. The theorem's other
conditions, that no alternative reads left of where its match begins, that each matches the longest gate-passing prefix
of its language and fails only where there is none, and that the engine reads code points only through the classes, are
facts about the published regex and its engine that no description can check. The fixtures, written by the reference
over Python's regex engine, hold each description to them, and on the fourteen corpora of [benchmarks.md](benchmarks.md)
matisse's ids equal tiktoken's, Hugging Face tokenizers' and the bpe crate's by digest wherever each of them ran.

That model excludes, some of it permanently and some only in the first version:

- **Approximate tokenization.** There is no mode that is merely close, no fallback when a piece cannot be merged and no
  token count estimated from bytes. Content a loader cannot take exactly is refused with `std::invalid_argument` naming
  the item and where it is, a definition the merge cannot run on is refused when the encoder is built, and `decode()`
  refuses an id that is neither an ordinary nor a special token, naming it and its index. This one is permanent.
- **Guessed cuts.** The parallel encoder cuts only where a window table certifies a cut, as the section on parallel
  encoding below describes, never at white space or at a fixed size with a repair afterwards. There is nothing to
  repair. This one is permanent too.
- **Normalizers and post-processors.** The first version applies no normalizer but the identity, adds no token of its
  own around the encoding, such as a beginning-of-sequence token, and has no byte fallback. WordPiece, which brings
  BERT's normalizer and so the first normalizer, comes after o200k in the order of tokenizers that open choice 3 of
  [design.md](design.md) recommends.
- **Training.** matisse reads the files a vendor published and trains nothing.
- **A tokenizer.json reader.** `read_tokenizer_json()` is designed in [design.md](design.md) and not yet declared,
  waiting on open choice 5, which JSON reader it uses. Until then load reads GPT-2's encoder.json and vocab.bpe through
  `read_gpt2()` and tiktoken rank files through `read_tiktoken()`, and `matisse-validate` treats a file named without
  `--tiktoken` as a command-line error.
- **Tokenizers beyond GPT-2 and cl100k_base.** Those two have pattern descriptions, in the gpt2 and cl100k libraries,
  which lookup finds by name. A further tokenizer, o200k or Llama 3, comes as a library of its own under `libs/`, named
  in the lookup's list, with no change to the stage libraries, once its pattern has a description that meets the class
  theorem's conditions; until then it has no encoder.
- **Definitions without a pre-tokenizer.** The `Encoder` takes a pattern, so a definition without one has no encoder in
  the first version.

Special tokens follow the same rule. The spellings the caller allows are cut out of the text, the earliest first and the
longest among those that begin at one byte, each emitting its id, and the constructor refuses an allowed spelling that
is not one of the definition's special tokens. Every special token that is not allowed is ordinary text, which is
tiktoken's `encode()` with that `allowed_special` and an empty `disallowed_special` where no allowed spelling begins
another, as on every definition in scope, and encoder.py's behavior when none is allowed. There is no form that refuses
a text for holding one.

## **Merges: Two Rules, Each a Vendor's**

A definition merges each piece by one of two rules. `Merge_rule::ranks` is tiktoken's rank merge, which joins the
adjacent pair whose concatenation has the least rank, leftmost first, and needs the ranks alone, so a definition
`read_tiktoken()` reads carries no merge table. `Merge_rule::pairs` is Hugging Face's rule, one join at a time, the
adjacent pair of least rank at its leftmost occurrence, which is its BPE model's `merge_all`. GPT-2's encoder.py joins
every occurrence of the least pair in one pass instead. The two give the same tokens on every piece of a table in which
no part of a merge is formed by a merge of a later rank, and without that condition they can differ, (ab, a) then (a, b)
giving ab ab on abab under the pass and aba b one join at a time. The agreement under the condition is Lemma 1 of
Berglund and van der Merwe's Formalizing BPE Tokenization (NCMA 2023). `read_gpt2()` checks the condition once at load
and refuses a table outside it, naming the merge, its rank and its line, so on every file it accepts the pair rule is
encoder.py's; GPT-2's table satisfies it. It also refuses a pair listed twice, naming both ranks, which both vendors
would read at its last rank.

`Merger` refuses, when the encoder is built, a definition in which one of the 256 bytes is not a token, since every
piece starts as its bytes, and under the pair rule a merge whose part or product is not a token; the message names the
byte, or the merge, its rank and the spelling. Both merges are exact on every table and every piece. The tests hold them
to the plain pair loop and to tiktoken's scan, written out in the tests, on every multi-byte token of GPT-2's, cl100k's,
o200k's and Llama 3's files and on 3,000 random tables over three symbols, among them tables on which the pass and one
join at a time differ.

The whole-piece lookup is the definition's `whole_piece_lookup` flag: whether a piece that is itself an ordinary token
is that one token, looked up before any merge. `read_tiktoken()` sets it, since tiktoken looks every piece up, and
`read_gpt2()` leaves it false, since encoder.py never does. It changes no id on cl100k or GPT-2, where every multi-byte
token is formed by the merge rule from its own bytes, GPT-2's `<|endoftext|>` aside, which its pattern never leaves as
one piece. It is load-bearing on Llama 3's rank file, where 588 of 127,744 multi-byte tokens are not formed from their
own bytes and the lookup is what gives each of them whole.

## **Parallel: Certified Cuts, or One Chunk**

`encode_parallel()` cuts a text only at certified anchors. An anchor is where an occurrence of a window the inventory's
table certifies places its origin, inside one stretch between allowed special tokens, and the edges of an allowed
special token are cuts of their own. Each worker segments its chunk with the inventory's right context, that many
further code points of the real text, and keeps the pieces that end at or before its cut. The encoder claims one thing:
for every text the definition accepts, `encode_parallel()` returns the ids `encode()` returns, at every chunk count.
That rests on the splice theorem of the report on pre-tokenizer certification, which is not yet published, and on the
`Segmenter` being the engine's segmentation. The tests hold the claim on seeded texts under both inventories at one to
sixteen chunks, at sixty-four and at every cut, and on the corpora of [benchmarks.md](benchmarks.md) the corpus probe
holds it on eleven and the bench on the mixed file.

`chunk_boundaries()` aims for chunks of equal bytes. Each interior boundary is the first cut at or after the later of
its equal-division target and one past the previous boundary, and an anchor counts only when it stands at least the
width less one code points after the previous boundary, so that the occurrence certifying it begins in the chunk the cut
ends, and as far after its stretch's start. A target with no cut after it ends the walk, the last chunk running to the
text's end. So a text with no certified cut is one chunk, encoded serially, and honestly, on the calling thread, and a
text whose cuts are scarce gets fewer chunks than it asked for. The longest piece of a text is a floor no cut rule can
go below, and the boundary gap of GPT-2's and cl100k's verifiers is unbounded, since a letter run of any length is one
piece. The chunks are of equal bytes, not of equal work, which shows on a text that mixes scripts, as the mixed file of
[benchmarks.md](benchmarks.md) does under cl100k.

There are two inventories, and the encoder keeps the one it is built with. The factorization inventory's cut holds for
every complete factorization into the gate-erased languages of the alternatives, whatever the engine does inside its
matches; it needs one code point of right context on GPT-2 and cl100k. The policy inventory's cut holds for the
pattern's own segmentation, decided on its verifier, which rests on the one-step relation the verifier is built from,
validated against the reference and not machine-checked; it needs two code points and lands on more boundaries. At the
default width of five, on the three corpora of the test strategy, the factorization inventory's anchors reach 57.30 to
92.88 per cent of the pre-token boundaries and the policy inventory's 96.09 to 100.00. The factorization inventory,
which rests on less, is the one [design.md](design.md) recommends by default and the one [benchmarks.md](benchmarks.md)
times.

The width is five by default, `validate::default_width`, for two reasons. The walk over the windows decides width five
in 1.1 seconds for GPT-2 and 2.4 for cl100k, and at width five GPT-2's policy anchors are complete, every cut a wider
window certifies being one of them, as the report on pre-tokenizer certification proves. The windows of a width are
every string of that length over the class symbols, 14 for GPT-2 and 16 for cl100k, so each further symbol multiplies
them; `matisse-validate` takes widths two to five.

Each chunk is a thread. `encode_parallel()` starts one `std::jthread` per chunk beyond the last, which runs on the
calling thread, with no pool and no executor, so the chunk count, at most the count the caller asks for, is also the
thread count. Nothing caps it but the text's bytes and its cuts, and the system's limit is the practical one: a chunking
at every cut of munch's C++ sources under cl100k's factorization inventory, 559,642 chunks, fails with
`std::system_error` on the machine of [benchmarks.md](benchmarks.md), which is why the corpus probe asks for at most
32,768. The useful range is near the machine's logical processors; on the 24 of that machine, in the session of its
fairness section, 24 workers encode enwik9 at 557.5 and 626.1 MiB/s under GPT-2 and cl100k, where `encode()` on one
thread runs at 54.6 and 57.1.

## **Unicode: Tables Pinned to 15.0.0**

The model is defined on code points. Every entry point that reads a text decodes it as UTF-8 and refuses an ill-formed
sequence it reads with `std::invalid_argument`, naming its byte offset within the stretch it reads, and every cut falls
at a code point. A code point's class is the class of the first set of the pattern's alphabet that holds it, the sets
GPT-2's and cl100k's patterns name being L, N, White_Space and the literals' own, and everything else is of one other
class. A literal matched caselessly holds its case partners, as cl100k's caseless `s` holds `S`, `s` and U+017F; beyond
that there is no case folding and no normalization.

The tables are generated by tools/unicode/generate_classes.py into each tokenizer library's src/class_ranges.inc,
checked in so the build never touches the network, and pinned to Unicode 15.0.0, the version of the reference the
fixtures are written by, which the alphabet's `unicode_version` reports. munch's own tables are 17.0.0 and are not used.
The classes, and with them the verifiers, the certified windows and the right contexts, do not depend on the version;
only the map from code points to classes does, and with it the ids of a text holding a code point whose class moved
between versions. A code point assigned a letter or a number after 15.0.0 is in none of the tables' sets, so matisse
puts it in the other class where an engine with newer tables reads it as a letter or a number, and the two can segment
it differently. Each vendor's engine carries tables of its own, and on the fourteen corpora the difference changes no
id. Moving the pin is open choice 6 of [design.md](design.md).

## **Hard Bounds**

- Token ids are `load::Token_id_t`, 32 bits, because ids are what a model reads and every vocabulary in scope fits.
- The merge holds every offset in 32 bits, so it merges a piece shorter than 2^32 bytes, and `Merger::merged()` refuses
  a longer one with `std::length_error` naming its length, which `encode()` and `encode_parallel()` pass on.
- A piece of at most 128 bytes is merged by a scan of its parts held in an array on the stack, a longer one by a binary
  heap. Both make the same joins in the same order, so the threshold changes the speed and never an id; on Chinese the
  encoding with no cache gains as the threshold rises to 128 and no further.
- The encoder's own piece caches hold `piece_cache_capacity` pieces, 16,384, each of at most `longest_cached_piece`
  bytes, 255, and then stops filling, with no eviction, as Hugging Face's BPE model's cache stops. A longer piece, and
  every piece past the bound, is merged each time it occurs. A cached piece's ids are a function of its bytes and the
  definition alone, so the cache never changes an id; on the three corpora of the test strategy the bound holds every
  piece the merge makes.
- A `pretok::Class_table` keeps one table per distinct block of 256 code points and refuses an alphabet with more
  distinct blocks than a byte indexes; GPT-2's and cl100k's have 127 and 128. Class symbols are bytes, as munch's
  verifier alphabet is, which holds every class alphabet in sight and no character-level merge table, so character-level
  SentencePiece waits on a decision about munch's alphabet.
- `validate()` refuses a pattern that is not total, as `split_verifier()` refuses it, a certified window whose right
  context is unbounded, and a factorization window the split verifier does not certify.
- `encode()` takes the whole text and returns every id in one vector. There is no streaming or incremental feeding, and
  certified-prefix streaming is among the pieces [design.md](design.md) leaves out of the first version, so a text is
  bounded by memory; `encode_parallel()` holds every chunk's ids before it joins them into one vector.

## **Concurrency and Lifetimes**

- An `Encoder` is not changed by any call. `encode()`, `chunk_boundaries()`, `encode_parallel()` and `decode()` are
  `const` and keep no state between calls, so one encoder serves any number of threads; the tests run four threads on
  one encoder, each keeping a cache of its own, and the thread sanitizer runs in CI.
- A `Piece_cache` has no lock and serves one thread at a time. `encode(text)` makes one for the call, and `encode(text,
  cache)` reads and fills one the caller keeps across its calls, one per thread. A cache holds the ids of one
  definition, so a cache filled by one encoder is not passed to another, and the ids `find()` returns are valid until
  the next store.
- `encode_parallel()` gives every worker a piece cache of its own and joins every worker before returning. An exception
  a worker throws is kept, and the first one is rethrown on the calling thread once every worker has joined.
- The encoder takes the definition by value and keeps its own copies of what it reads from the report and the pattern,
  the window table and its right context, the class table and the segmentation's automaton, so neither has to outlive
  it. The report must be the one `validate()` decided of the same pattern, which the encoder does not check. GPT-2's and
  cl100k's patterns point their views and spans into constants of their libraries' sources.
- The text is read during the call only, and the ids are returned by value.

## **Construction Cost**

`validate()` is the expensive step, and it runs once per pattern. It decides every window of width two to four in 0.19
seconds for GPT-2 and 0.45 for cl100k, and to width five in 1.1 and 2.4, where a `miscovering()` search per window would
take 3.8 and 9.3 seconds, and 92 and 213. The `Encoder` takes the report rather than deciding it, so a pattern is
validated once however many encoders are built. Building an encoder builds the merge's tables once: the id of every
spelling in a `Spelling_table`, the rank and product of every pair the merge may join in a `Join_table`, 233,378 pairs
on cl100k's file in 524,288 slots, and the joins of all 65,536 pairs of bytes; beside them the `Segmenter`'s product
automaton, 36 states for GPT-2's twelve alternatives and 33 for cl100k's eight, and the window index. Build the encoder
once and reuse it across texts and threads; building one per request is a design mistake the library does not try to
make cheap.

A call has a cost of its own. `encode(text)` builds its classes, its segmentation and its piece cache afresh, so on a
short text a cache kept across the calls is what pays. In the fairness section of [benchmarks.md](benchmarks.md) a call
on a prompt of about a kilobyte costs a mean of 29.39 microseconds under GPT-2 and 38.57 under cl100k with a cache of
its own, and 21.67 and 31.53 with one kept across the calls.

## **The Escape Hatches**

Real workloads exceed the model in places. The library carries the primitives to handle them beside the certificates
instead of weakening them:

- **Many documents.** Documents are independent inputs, so encoding each with `encode(text, cache)` from a pool of
  threads, each thread keeping its own cache, needs no certificate at all. The bench's batch of enwik9's pages does
  exactly that, at 795.3 and 778.8 MiB/s on 24 threads under GPT-2 and cl100k in the fairness section of
  [benchmarks.md](benchmarks.md).
- **Short calls.** `encode(text, cache)` with a cache kept across the calls amortizes the piece cache `encode(text)`
  builds for each, though every call still classes and segments its own text.
- **Scarce cuts.** The policy inventory lands on more boundaries than the factorization inventory, at the cost of
  resting on the verifier's one-step relation, and an allowed special token is a cut wherever it stands.
- **A new pattern.** A tokenizer is added as a library of its own, its pattern described as data; `split_verifier()`
  refuses a pattern that is not total with a shortest residual no alternative matches, which is where a wrong
  description shows first, and `validate()` decides its windows.
- **A rank file under a known pattern.** `matisse-validate FILE --tiktoken PATTERN` reports a rank file read under gpt2
  or cl100k, its windows and right context, and with `--input` the anchor supply on a text.
- **No pre-tokenizer.** The lazy certification of byte-level merge tables, `validate_merges()`, is step 14 of the order
  of work in [design.md](design.md) and not written yet.

If a tokenizer needs a normalizer, a post-processor, a pattern matisse has no description of, a character-level merge
table or a reader matisse does not have yet, that is the signal matisse is the wrong tool for it today, and the honest
answer is the vendor's own library, which defines the ids matisse would be held to anyway. If it needs approximate
counts or training, matisse is the wrong tool for good.
