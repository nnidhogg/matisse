# **Scope, Guarantees, and Limits**

matisse is exact because it refuses to guess, and fast because it refuses everything exactness does not need.
[benchmarks.md](benchmarks.md) measures what those two refusals buy. This note is the inventory: the model the library
commits to, the guarantees that commitment buys, the boundaries you can hit, and the escape hatch for each. Read it when
deciding whether matisse fits a tokenizer, not after.

## **The Model: The Vendor's Ids, or a Refusal**

matisse reproduces the token ids of a vendor's tokenizer exactly, or it refuses, under the Unicode version its class
tables are pinned to: a text holding a code point whose class moved after Unicode 15.0.0 can be segmented differently by
an engine with newer tables, as the section on Unicode below describes. A tokenizer reaches it as two things read apart:
a `load::Definition`, which holds the merge rule, the whole-piece lookup, the bytes of every ordinary token, the merge
table and the special tokens, and a `pretok::Pattern` from the tokenizer's own library, which describes the published
pre-tokenizer as data. A pattern is the contract of the class theorem of the report on pre-tokenizer certification: an
ordered gated pattern over a class alphabet, each alternative a regular language of class strings with a gate, the class
symbols that may follow its accepts. At each boundary it commits the longest gate-passing prefix of the first
alternative that has one. The theorem carries a cut to the vendor's engine only when the pattern is total, and
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
  refuses an id that is neither an ordinary nor a special token with `std::out_of_range`, naming it and its index. This
  one is permanent.
- **Guessed cuts.** The parallel encoder cuts only where a window table certifies a cut, as the section on parallel
  encoding below describes, never at white space or at a fixed size with a repair afterwards. There is nothing to
  repair. This one is permanent too.
- **Normalizers and post-processors.** The first version applies no normalizer but the identity, adds no token of its
  own around the encoding, such as a beginning-of-sequence token, and has no byte fallback. WordPiece, which brings
  BERT's normalizer and so the first normalizer, comes next in the order of tokenizers that open choice 3 of
  [design.md](design.md) recommends.
- **Training.** matisse reads the files a vendor published and trains nothing.
- **A tokenizer.json reader.** `read_tokenizer_json()` is designed in [design.md](design.md) and not yet declared,
  waiting on open choice 5, which JSON reader it uses. Until then load reads GPT-2's encoder.json and vocab.bpe through
  `read_gpt2()` and tiktoken rank files through `read_tiktoken()`, and `matisse-validate` treats a file named without
  `--tiktoken` as a command-line error.
- **Tokenizers beyond GPT-2, cl100k_base, o200k_base and Llama 3.** Those four have pattern descriptions, in the gpt2,
  cl100k, o200k and llama3 libraries, which lookup finds by name. A further tokenizer comes as a library of its own
  under `libs/`, named in the lookup's list, with no change to the stage libraries, once its pattern has a description
  that meets the class theorem's conditions; until then it has no encoder.
- **Definitions without a pre-tokenizer.** The `Encoder` takes a report, which holds a pattern, so a definition without
  one has no encoder in the first version.

Special tokens follow the same rule. The spellings the caller allows are cut out of the text, the earliest first and the
longest among those that begin at one byte, each emitting its id, and the constructor refuses an allowed spelling that
is not one of the definition's special tokens, one that is empty, since an empty spelling would occur at every byte, and
one whose id is an ordinary token's, which `decode()` would read as that token; it refuses as well two special tokens
that share a spelling, allowed or not, which `encode()` would cut out as the first of them, two that share an id,
allowed or not, which `decode()` would read as the first of them, and one whose spelling is not UTF-8, allowed or not,
which no vendor's special token is and which a stream would hold back from inside a code point's sequence. Every special
token that is not allowed is ordinary text, which is tiktoken's `encode()` with that `allowed_special` and an empty
`disallowed_special` where no allowed spelling begins another, as on every definition in scope, and encoder.py's
behavior when none is allowed. There is no form that refuses a text for holding one.

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
join at a time differ. On a definition in which the last join of every token that encodes to itself, in the merge of its
own spelling, ranks above the last joins of its two parts, which `Merger` decides when the encoder is built, a piece
longer than 128 bytes or at least half outside ASCII is merged by backtracking, a `Backtracker`, whose ids are the same
by the proof in [design.md](design.md). GPT-2's, cl100k's and o200k's files meet that condition and Llama 3's does not;
a definition that does not is merged by the scan and the heap. The tests hold the backtracking to the plain loop on
every multi-byte token of the three files that meet it, on 20,000 seeded random pieces of each and on the random tables
that meet it, and hold three small tables on which a pair test that assumed the condition goes wrong to be refused the
backtracking and merged to the rule's ids.

The whole-piece lookup is the definition's `whole_piece_lookup` flag: whether a piece that is itself an ordinary token
is that one token, looked up before any merge. `read_tiktoken()` sets it, since tiktoken looks every piece up, and
`read_gpt2()` leaves it false, since encoder.py never does. It changes no id on cl100k, o200k or GPT-2, where every
multi-byte token is formed by the merge rule from its own bytes, GPT-2's `<|endoftext|>` aside, which its pattern never
leaves as one piece. It is load-bearing on Llama 3's rank file, where 588 of 127,744 multi-byte tokens are not formed
from their own bytes and the lookup is what gives each of them whole.

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
go below, and the boundary gap of GPT-2's, cl100k's, o200k's and Llama 3's verifiers is unbounded, since a letter run of
any length, of lower case letters under o200k, is one piece. The chunks are of equal bytes, not of equal work, which
shows on a text that mixes scripts, as the mixed file of [benchmarks.md](benchmarks.md) does under cl100k.

There are two inventories, and the encoder keeps the one it is built with. The factorization inventory's cut holds for
every complete factorization into the gate-erased languages of the alternatives, whatever the engine does inside its
matches; it needs one code point of right context on all four patterns in scope. The policy inventory's cut holds for
the pattern's own segmentation, decided on its verifier, which rests on the one-step relation the verifier is built
from, validated against the reference and not machine-checked; it needs two code points and lands on more boundaries. At
the default width of five, on the three corpora of the test strategy under GPT-2 and cl100k, the factorization
inventory's anchors reach 57.30 to 92.88 per cent of the pre-token boundaries and the policy inventory's 96.09 to
100.00. The factorization inventory, which rests on less, is the one [design.md](design.md) recommends by default and
the one [benchmarks.md](benchmarks.md) times.

The width is five by default, `validate::default_width`, for two reasons. The walk over the windows decides width five
in 1.1 seconds for GPT-2 and 2.4 for cl100k, and at width five GPT-2's policy anchors are complete, every cut a wider
window certifies being one of them, as the report on pre-tokenizer certification proves. The windows of a width are
every string of that length over the class symbols, 14 for GPT-2, 16 for cl100k and Llama 3 and 30 for o200k, so each
further symbol multiplies them; `matisse-validate` takes widths two to five. The walk decides o200k to width three in
1.5 seconds, to four in 16 and to five in 245, holding 2.9 GB, and width four adds almost no factorization anchor to
three's, 23 on munch's C++ sources and none on twitter.json or wiki-zh, where the policy inventory gains 20,484 on
munch's C++ sources, from 97.18 to 99.28 per cent of the boundaries, so o200k's tests, the corpus probe and the bench,
which time the factorization inventory, decide its report to width three.

Each chunk is a thread. `encode_parallel()` starts one `std::jthread` per chunk beyond the last, which runs on the
calling thread, with no pool and no executor, so the chunk count, at most the count the caller asks for, is also the
thread count. Nothing caps it but the text's bytes and its cuts, and the system's limit is the practical one: a chunking
at every cut of munch's C++ sources under cl100k's factorization inventory, 559,642 chunks, fails with
`std::system_error` on the machine of [benchmarks.md](benchmarks.md), which is why the corpus probe asks for at most
32,768. The useful range is near the machine's logical processors; on the 24 of that machine, in the session of its
fairness section, 24 workers encode enwik9 at 557.5 and 626.1 MiB/s under GPT-2 and cl100k, where `encode()` on one
thread runs at 54.6 and 57.1.

## **Streaming: Every Id Final When Returned**

A `Stream_encoder` reads a text in pushes and returns each id once no continuation of the text can change it. It claims
one thing: after every push the ids it has returned are a prefix of `encode()` of the whole text, whatever the rest
turns out to be, and after `finish()` they are `encode()` of it. What a push returns is the longest such prefix but for
three exceptions: while the bytes read end in a proper prefix of an allowed special token's spelling those bytes are
held, fewer than the longest allowed spelling, and a push returns the longest prefix final for the bytes before them;
while they end inside a UTF-8 sequence its bytes are held until the sequence completes, and a push returns the longest
prefix final for the code points before it, even where an id of those bytes is final under every completion, as GPT-2's
byte token for `\xF4` is, since no token of two or more bytes begins with it; and inside a piece an id is held against
every byte continuation where the pattern allows fewer. The tests hold the claim on seeded texts pushed in random parts,
a byte at a time and whole, under all four tokenizers and the tests' own pattern, and against brute force over the
continuations on the tests' pattern, as [design.md](design.md) describes.

How far a stream's ids lag after a push, the bytes it has read and not yet spelled by the ids it has returned, its
delay, comes from four things. A UTF-8 sequence the push ended inside, at most three bytes. A proper prefix of an
allowed spelling. The code points after the last one every live run of the pattern's verifier agrees on: on GPT-2's
pattern the runs agree on every pre-token boundary but the last two code points', by the bounded backoff of the report
on pre-tokenizer certification, and on cl100k's no bound exists, since a newline followed by tabs is one piece when the
text ends there and a newline and pieces of tabs when a letter follows, so the boundary after the newline stays open
until something other than white space or the end arrives, and where the two segmentations' ids part no streamer can
return them earlier, as Dalirrooyfard and Ye show for o200k's as well. On cl100k's pattern a vocabulary whose merge
joins the newline to the first white-space byte parts the two runs' ids at a fixed id, `\n\t` against `\n`, after which
the bytes not yet returned and the cost of each push grow without bound for as long as the white space lasts; cl100k's
own ranks are not such a vocabulary, and under them a newline followed by tabs or by spaces streams in time linear in
its length. And inside each run's open piece what the merge cannot yet decide, at most the right span radius of the
dictionary by the report *The Left Ripple of Byte-Pair Encoding*, finite for every dictionary and 129 bytes for GPT-2's
table, not computed for the others. On the corpora of [benchmarks.md](benchmarks.md), which gives the figures for every
tokenizer and corpus, a stream in pushes of 64 bytes or 4 KiB lags 1.01 to 6.71 bytes on average after a push and at
most 132. That is the delay of its ids, and not the memory the stream keeps, which the next paragraph gives.

A stream keeps the stretch's bytes that the ids it has returned do not yet spell, and before them fewer bytes than the
larger of their number and 64, since it erases those in bulk once they are as many as the rest and at least 64. Beside
them it keeps the live runs' tree, a layer per code point since the last stable one, at or after the end of the last
closed piece, which the ids returned do not move: each layer takes 40 bytes and up to one node of 40 bytes per state of
the verifier, and the layers and nodes dropped from the front are kept until they are as many as the rest and at
least 64. The tree is the larger part when the runs stay apart: on a newline followed by tabs under cl100k's pattern and
ranks the ids lag a few bytes while the tree keeps a layer for every tab since the newline, about 250 bytes per code
point as measured on that case. None of these containers gives its capacity back, so a stream holds the most it has
needed for any stretch until it is destroyed.

Each code point costs a class lookup and at most two steps of the verifier's table per live run, at most one run per
state of the verifier; each piece closed costs at most its merge, as `encode()` merges it, with a piece cache of its
own, and its bytes are moved a constant number of times on average, however many pieces one code point closes; each push
costs, for every live run, the reading of its piece ends back to the last one the ids returned reach, the merge of what
its open piece has left after the ids returned and the tests of its boundaries from the right, which grow with the bytes
not yet returned and not with how long the runs have been apart. Ill-formed UTF-8 is refused with
`std::invalid_argument` by the push that classes it, and a text that ends inside a sequence by `finish()`; a stream that
has refused is not pushed to or finished again, and a new `Stream_encoder` over the same encoder reads the next text.

## **Unicode: Tables Pinned to 15.0.0**

The model is defined on code points. Every entry point that reads a text decodes it as UTF-8 and refuses an ill-formed
sequence it reads with `std::invalid_argument`, naming its byte offset within the stretch it reads, and every cut falls
at a code point. A code point's class is the class of the first set of the pattern's alphabet that holds it, the sets
GPT-2's, cl100k's and Llama 3's patterns name being L, N, White_Space and the literals' own, and o200k's Lu, Ll, Lt, Lm,
Lo, M, N, White_Space and its literals', and everything else is of one other class. A literal matched caselessly holds
its case partners, as cl100k's caseless `s` holds `S`, `s` and U+017F, and where the pattern reads upper and lower case
apart, as o200k's does, the partners are split by case, `S` in a class of its own and U+017F with `s`; beyond that there
is no case folding and no normalization.

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
  a longer one with `std::length_error` naming its length, which `encode()` and `encode_parallel()` pass on. A
  `Stream_encoder` merges only the bytes of a piece after the ids it has returned, when the piece closes and at the end
  of every push before then, so it refuses a piece once 2^32 or more bytes are merged at once, counting with those bytes
  the rest of a token that the test of a boundary merges after them, and a piece whose ids it returns as it reads them
  can grow past 2^32 bytes without a refusal.
- A piece of at most 128 bytes is merged by a scan of its parts held in an array on the stack, a longer one by a binary
  heap, and on a definition that meets the backtracking's condition every piece longer than 128 bytes and every one at
  least half outside ASCII by backtracking. All three give the same ids, so the routing changes the speed and never an
  id; with the scan and the heap alone, the encoding of Chinese with no cache gains as the scan's threshold rises to 128
  and no further.
- The encoder's own piece caches hold `piece_cache_capacity` pieces, 16,384, each of at most `longest_cached_piece`
  bytes, 255, and then stops filling, with no eviction, as Hugging Face's BPE model's cache stops. A longer piece, and
  every piece past the bound, is merged each time it occurs. A cached piece's ids are a function of its bytes and the
  definition alone, so the cache never changes an id; on the three corpora of the test strategy the bound holds every
  piece the merge makes.
- A `pretok::Class_table` keeps one table per distinct block of 256 code points and refuses an alphabet with more
  distinct blocks than a byte indexes; GPT-2's, cl100k's, o200k's and Llama 3's have 127, 128, 135 and 128. Class
  symbols are bytes, as munch's verifier alphabet is, which holds every class alphabet in sight and no character-level
  merge table, so character-level SentencePiece waits on a decision about munch's alphabet.
- `validate()` refuses a pattern that is not total, as `split_verifier()` refuses it, a certified window whose right
  context is unbounded, and a factorization window the split verifier does not certify.
- `encode()` takes the whole text and returns every id in one vector, so a text it reads is bounded by memory, and
  `encode_parallel()` holds every chunk's ids before it joins them into one vector. A text that arrives in parts is read
  by a `Stream_encoder`, which holds what the section on streaming bounds.

## **Concurrency and Lifetimes**

- An `Encoder` is not changed by any call. `encode()`, `chunk_boundaries()`, `encode_parallel()` and `decode()` are
  `const` and keep no state between calls, so one encoder serves any number of threads; the tests run four threads on
  one encoder, each keeping a cache of its own, and the thread sanitizer runs in CI.
- A `Piece_cache` has no lock and serves one thread at a time. `encode(text)` makes one for the call, and `encode(text,
  cache)` reads and fills one the caller keeps across its calls, one per thread. A cache holds the ids of one
  definition, so it is made for the encoder it serves, which must outlive it and must not be moved while the cache is in
  use, and `encode(text, cache)` refuses a cache made for another encoder with `std::invalid_argument`; the ids `find()`
  returns are valid until the next store.
- `encode_parallel()` gives every worker a piece cache of its own and joins every worker before returning. An exception
  a worker throws is kept, and the first one is rethrown on the calling thread once every worker has joined.
- The encoder takes the definition by value and keeps its own copies of what it reads from the report and the pattern it
  holds, the window table and its right context, the class table and the segmentation's automaton, so the report does
  not have to outlive it. Only `validate()` makes a report, holding the pattern its window tables were decided of, and
  an encoder reads the pattern from the report it is given. A pattern holds its alphabet, its languages and its gates by
  value, so a report stays usable after every string and table its pattern was built from is gone.
- The text is read during the call only, and the ids are returned by value.
- A `Stream_encoder` reads one text at a time and serves one thread at a time; the encoder it is made from must outlive
  it and must not be moved while the stream reads it, and it serves any number of streams on as many threads, each
  stream keeping its runs, its held bytes and its piece cache to itself with no lock. The tests run four streams on four
  threads over one encoder.
- A copy of an `Encoder` encodes as the original does and is an encoder of its own, so `encode(text, cache)` on the copy
  refuses a cache made for the original, and an encoder is never assigned to, both assignments deleted, so a cache made
  for it never meets another definition at its address.
- An `Encoder`, a `Piece_cache` or a `Stream_encoder` that has been moved from may only be destroyed. None of the three
  can be assigned a fresh value, the encoder's assignments being deleted and the cache and the stream holding a
  reference to their encoder, and no other call on one is defined.

## **Construction Cost**

`validate()` is the expensive step, and it runs once per pattern. It decides every window of width two to four in 0.19
seconds for GPT-2 and 0.45 for cl100k, and to width five in 1.1 and 2.4, where a `miscovering()` search per window would
take 3.8 and 9.3 seconds, and 92 and 213. The `Encoder` takes the report rather than deciding it, so a pattern is
validated once however many encoders are built. Building an encoder builds the merge's tables once: the id of every
spelling in a `Spelling_table`, the rank and product of every pair the merge may join in a `Join_table`, 233,378 pairs
on cl100k's file in 524,288 slots, and the joins of all 65,536 pairs of bytes, and where the definition meets the
backtracking's condition the split of every token, read from the merge of its own spelling, the trie of the tokens that
encode to themselves and the split of each by its parts, which take the construction of the merge's tables from 0.01,
0.05 and 0.15 seconds to 0.04, 0.14 and 0.33 on GPT-2's, cl100k's and o200k's files on the machine of
[benchmarks.md](benchmarks.md); beside them the `Segmenter`'s product automaton, 36 states for GPT-2's twelve
alternatives and 33 for cl100k's eight, and the window index. What the streaming encoder reads is built with them, the
class table, the split verifier's table, the ordinary tokens in the order of their spellings and, where pieces are
looked up whole, the tokens the merge does not form, which adds 9, 36, 69 and 49 milliseconds on GPT-2's, cl100k's,
o200k's and Llama 3's files, so that a stream is made in about ten nanoseconds. Build the encoder once and reuse it
across texts and threads; building one per request is a design mistake the library does not try to make cheap.

A call has a cost of its own. `encode(text)` builds its classes, its segmentation and its piece cache afresh, so on a
short text a cache kept across the calls is what pays. In the backtracking section of [benchmarks.md](benchmarks.md) a
call on a prompt of about a kilobyte costs a mean of 24.52 microseconds under GPT-2 and 24.77 under cl100k with a cache
of its own, and 16.44 and 20.53 with one kept across the calls.

## **The Escape Hatches**

Real workloads exceed the model in places. The library carries the primitives to handle them beside the certificates
instead of weakening them:

- **Many documents.** Documents are independent inputs, so encoding each with `encode(text, cache)` from a pool of
  threads, each thread keeping its own cache, needs no certificate at all. The bench's batch of enwik9's pages does
  exactly that, at 795.3 and 778.8 MiB/s on 24 threads under GPT-2 and cl100k in the fairness section of
  [benchmarks.md](benchmarks.md).
- **Short calls.** `encode(text, cache)` with a cache kept across the calls amortizes the piece cache `encode(text)`
  builds for each, though every call still classes and segments its own text.
- **A text that arrives in parts.** A `Stream_encoder` returns each id as soon as it is final, so a caller never
  re-encodes a growing text, and its ids lag the bytes pushed by no more than the section on streaming describes.
- **Scarce cuts.** The policy inventory lands on more boundaries than the factorization inventory, at the cost of
  resting on the verifier's one-step relation, and an allowed special token is a cut wherever it stands.
- **A new pattern.** A tokenizer is added as a library of its own, its pattern described as data; `split_verifier()`
  refuses a pattern that is not total with a shortest residual no alternative matches, which is where a wrong
  description shows first, and `validate()` decides its windows.
- **A rank file under a known pattern.** `matisse-validate FILE --tiktoken PATTERN` reports a rank file read under gpt2,
  cl100k, o200k or llama3, its windows and right context, and with `--input` the anchor supply on a text.
- **No pre-tokenizer.** The lazy certification of byte-level merge tables, `validate_merges()`, is step 14 of the order
  of work in [design.md](design.md) and not written yet.

If a tokenizer needs a normalizer, a post-processor, a pattern matisse has no description of, a character-level merge
table or a reader matisse does not have yet, that is the signal matisse is the wrong tool for it today, and the honest
answer is the vendor's own library, which defines the ids matisse would be held to anyway. If it needs approximate
counts or training, matisse is the wrong tool for good.
