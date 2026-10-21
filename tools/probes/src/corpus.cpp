// Holds the ids of encode() and of encode_parallel() on GPT-2's and cl100k's pinned vocabularies to their references on
// real text. This is the third layer of design.md's test strategy run as a program.
//
// What it checks on every text, for both tokenizers: that encode() gives the ids of every piece looked up whole where
// the definition says so and otherwise merged with no cache, and that decode() spells the text back; and for both
// inventories that encode_parallel() gives encode()'s ids at the chunk counts the bench times, at the chunkings of
// 4,096 and 65,536 code points and at one chunk per byte asked for, which takes every cut the rule offers; and that at
// every cut the pieces a worker keeps, its chunk and the inventory's right context segmented as a complete input, are
// the serial pieces of that chunk. encode_parallel() runs one thread per chunk, so a chunk count past most_threads is
// skipped, with a line saying so, and the cut check covers that chunking without a thread per chunk; every cut of a
// corpus is far more chunks than threads.
//
// What runs as a test. With the vocabulary directory alone the probe encodes seeded texts of the shapes the patterns
// meet, as the encoder tests generate them, and requires all of the above, encode_parallel() at every cut included, and
// at least least_generated_cuts cuts in every text, so that the cut checks cannot pass on texts that take none.
//
// Corpus mode. Every further argument is a corpus file, read whole into memory. A corpus named, after a trailing .txt
// is dropped, as the fixtures name one (twitter.json, munch-python@15b2863c, munch-cpp@15b2863c) is held to the
// segmentation fixture's digest of its text and to the known-answer fixture's count and digest of its ids; any other
// corpus is measured with no reference. For each tokenizer the probe prints the ids line, `ids TOKENIZER NAME COUNT
// DIGEST`, the digest being the SHA-256 of the ids as little-endian 32-bit words, as the fixtures write it, which a
// vendor's ids are compared by, and the supply section of both inventories, which docs/benchmarks.md quotes. The
// fixtures must hold the three corpora's texts and both tokenizers' ids of them. Any failure exits 1.
//
// Usage: matisse_corpus VOCAB_DIR [CORPUS...]

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/common/sha256.hpp"
#include "matisse/encode/encoder.hpp"
#include "matisse/encode/merger.hpp"
#include "matisse/gpt2/tokenizer.hpp"
#include "matisse/load/definition.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/validate/report.hpp"
#include "matisse/validate/supply.hpp"

namespace
{
using matisse::encode::Encoder;
using matisse::load::Definition;
using matisse::load::Token_id_t;
using matisse::validate::Inventory;

/**
 * @brief The SHA-256 of GPT-2's encoder.json as pinned, gpt2-vocab.json in the vocabulary directory.
 */
constexpr std::string_view gpt2_encoder_digest{"196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"};

/**
 * @brief The SHA-256 of GPT-2's vocab.bpe as pinned, gpt2-merges.txt in the vocabulary directory.
 */
constexpr std::string_view gpt2_vocab_digest{"1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"};

/**
 * @brief The SHA-256 of cl100k_base.tiktoken as pinned.
 */
constexpr std::string_view cl100k_digest{"223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"};

/**
 * @brief The corpora the fixtures name, each with its text's digest and both tokenizers' ids of it.
 */
constexpr std::size_t fixture_corpora{3};

/**
 * @brief The chunk counts encode_parallel() is held to on a corpus, the worker counts the bench times.
 */
constexpr std::array<std::size_t, 6> worker_counts{2, 4, 8, 12, 16, 24};

/**
 * @brief The chunk lengths in code points of the chunkings encode_parallel() is held to on a corpus.
 */
constexpr std::array<std::size_t, 2> chunk_points{4096, 65536};

/**
 * @brief The most chunks encode_parallel() is asked for, each a thread of its own; a finer chunking is skipped.
 */
constexpr std::size_t most_threads{32768};

/**
 * @brief The seeded texts of the test form.
 */
constexpr std::size_t generated_texts{100};

/**
 * @brief The fragments each seeded text of the test form joins.
 */
constexpr std::size_t generated_fragments{200};

/**
 * @brief The fewest cuts a seeded text of the test form must take, each text taking eighty or more.
 */
constexpr std::size_t least_generated_cuts{50};

/**
 * @brief Expectations that did not hold, which decide the exit status.
 */
std::size_t failures{0};

/**
 * @brief Records an expectation, printing the failed ones.
 * @param condition Whether it held.
 * @param what What was expected, printed when it did not.
 */
void expect(const bool condition, const std::string_view what)
{
    if (condition)
    {
        return;
    }

    ++failures;

    std::printf("FAIL: %.*s\n", static_cast<int>(what.size()), what.data());
}

/**
 * @brief A tokenizer under measure: its name as the fixtures write it, its pattern and its definition.
 */
struct Tokenizer
{
    /**
     * @brief The universe name of the fixtures, gpt2 or cl100k.
     */
    std::string_view name{};

    /**
     * @brief The pattern, whose views point into the tokenizer library's constants.
     */
    matisse::pretok::Pattern pattern{};

    /**
     * @brief The definition read from the pinned vocabulary.
     */
    Definition definition{};
};

/**
 * @brief The fixtures' references of the corpora they name.
 */
struct References
{
    /**
     * @brief The SHA-256 of each corpus's text, by name.
     */
    std::map<std::string, std::string, std::less<>> texts{};

    /**
     * @brief The count and digest of each corpus's ids, by universe and name.
     */
    std::map<std::pair<std::string, std::string>, std::pair<std::size_t, std::string>> ids{};
};

/**
 * @brief Returns the whole of a file.
 * @param path The file's path.
 * @return Its bytes.
 * @throws std::runtime_error If the file cannot be read.
 */
std::string contents(const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};

    if (!stream)
    {
        throw std::runtime_error{std::format("cannot read {}", path.string())};
    }

    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

/**
 * @brief Returns the path of a pinned vocabulary file after holding its bytes to the SHA-256 it is pinned by.
 * @param directory The vocabulary directory.
 * @param name The file's name.
 * @param digest The pinned SHA-256.
 * @return The path.
 * @throws std::runtime_error If the file cannot be read or its digest is another.
 */
std::filesystem::path pinned(
        const std::filesystem::path& directory, const std::string_view name, const std::string_view digest)
{
    const auto path{directory / name};

    const auto bytes{contents(path)};

    const auto actual{matisse::common::sha256(bytes)};

    if (actual != digest)
    {
        throw std::runtime_error{
                std::format("{} has the SHA-256 {}, not the pinned {}", path.string(), actual, digest)};
    }

    return path;
}

/**
 * @brief Returns GPT-2 and cl100k read from the pinned files of a vocabulary directory, each held to its digest.
 * @param directory The vocabulary directory, holding gpt2-vocab.json, gpt2-merges.txt and cl100k_base.tiktoken.
 * @return The two tokenizers.
 * @throws std::runtime_error If a file cannot be read or is not the pinned one.
 */
std::vector<Tokenizer> tokenizers(const std::filesystem::path& directory)
{
    const auto encoder{pinned(directory, "gpt2-vocab.json", gpt2_encoder_digest)};

    const auto vocab{pinned(directory, "gpt2-merges.txt", gpt2_vocab_digest)};

    const auto ranks{pinned(directory, "cl100k_base.tiktoken", cl100k_digest)};

    std::vector<Tokenizer> read{};

    read.push_back(Tokenizer{
            .name = "gpt2",
            .pattern = matisse::gpt2::pattern(),
            .definition = matisse::load::read_gpt2(encoder, vocab)});

    read.push_back(Tokenizer{
            .name = "cl100k",
            .pattern = matisse::cl100k::pattern(),
            .definition = matisse::load::read_tiktoken(ranks, matisse::cl100k::specials())});

    return read;
}

/**
 * @brief Returns the corpus references of the segmentation and known-answer fixtures.
 * @return The references.
 * @throws std::runtime_error If a fixture cannot be read.
 */
References references()
{
    References found{};

    const std::filesystem::path source{SOURCE_DIR};

    std::istringstream segmentation{contents(source / "libs/pretok/tests/data/segmentation_reference.txt")};

    for (std::string line{}; std::getline(segmentation, line);)
    {
        std::istringstream fields{line};

        std::string kind{};
        std::string name{};
        std::string digest{};

        fields >> kind >> name >> digest;

        if (kind == "corpus")
        {
            found.texts[name] = digest;
        }
    }

    std::istringstream known{contents(source / "libs/load/tests/data/known_answers_reference.txt")};

    std::string universe{};

    for (std::string line{}; std::getline(known, line);)
    {
        std::istringstream fields{line};

        std::string kind{};

        fields >> kind;

        if (kind == "universe")
        {
            fields >> universe;

            continue;
        }

        if (kind != "corpus")
        {
            continue;
        }

        std::string name{};
        std::size_t count{};
        std::string digest{};

        fields >> name >> count >> digest;

        found.ids[{universe, name}] = {count, digest};
    }

    return found;
}

/**
 * @brief Returns the name a corpus is held to the fixtures by: its file name, a trailing .txt dropped.
 * @param path The corpus's path.
 * @return The name.
 */
std::string corpus_name(const std::filesystem::path& path)
{
    auto name{path.filename().string()};

    if (name.ends_with(".txt"))
    {
        name.resize(name.size() - 4);
    }

    return name;
}

/**
 * @brief Returns the UTF-8 encoding of a code point.
 * @param point The code point, a Unicode scalar value.
 * @return Its one to four bytes.
 */
std::string utf8_of(const char32_t point)
{
    std::string bytes{};

    if (point < 0x80)
    {
        bytes += static_cast<char>(point);
    }
    else if (point < 0x800)
    {
        bytes += static_cast<char>(0xC0U | (point >> 6U));
        bytes += static_cast<char>(0x80U | (point & 0x3FU));
    }
    else if (point < 0x10000)
    {
        bytes += static_cast<char>(0xE0U | (point >> 12U));
        bytes += static_cast<char>(0x80U | ((point >> 6U) & 0x3FU));
        bytes += static_cast<char>(0x80U | (point & 0x3FU));
    }
    else
    {
        bytes += static_cast<char>(0xF0U | (point >> 18U));
        bytes += static_cast<char>(0x80U | ((point >> 12U) & 0x3FU));
        bytes += static_cast<char>(0x80U | ((point >> 6U) & 0x3FU));
        bytes += static_cast<char>(0x80U | (point & 0x3FU));
    }

    return bytes;
}

/**
 * @brief Returns a seeded random text of the shapes the patterns meet: white space runs before letters and
 *        punctuation, newlines followed by tab runs, long letter runs, contractions in both cases, digits, the special
 *        tokens' spellings whole and in parts, and code points drawn from several scripts and the emoji.
 * @param random The random sequence.
 * @param fragments The number of fragments joined.
 * @return The text, well-formed UTF-8.
 */
std::string random_text(std::mt19937& random, const std::size_t fragments)
{
    constexpr std::array<std::string_view, 22> shapes{
            " ",
            "  ",
            "   a",
            "\t\t!",
            "\t !",
            "\n\n\t\t",
            "\r\n",
            "Hello",
            " world",
            "don't",
            "DON'T",
            "'s",
            "'LL",
            " 1234567",
            "!!",
            "<|endoftext|>",
            "<|endof",
            "text|>",
            "<|fim_prefix|>",
            "abcdefghijklmnopqrstuvwxyz",
            " ?!",
            "\n \n"};

    // Latin-1, Latin Extended-A, Greek, Cyrillic, Arabic, Devanagari, Hangul, kana, CJK and the emoji.
    constexpr std::array<std::pair<char32_t, char32_t>, 10> scripts{{
            {0xC0, 0xFF},
            {0x100, 0x17F},
            {0x391, 0x3C9},
            {0x410, 0x44F},
            {0x621, 0x64A},
            {0x905, 0x939},
            {0xAC00, 0xD7A3},
            {0x3041, 0x30FF},
            {0x4E00, 0x9FFF},
            {0x1F600, 0x1F64F},
    }};

    std::uniform_int_distribution<std::size_t> shape{0, shapes.size() + scripts.size() - 1};

    std::string text{};

    for (std::size_t count{0}; count < fragments; ++count)
    {
        const auto drawn{shape(random)};

        if (drawn < shapes.size())
        {
            text += shapes[drawn];

            continue;
        }

        const auto [first, last]{scripts[drawn - shapes.size()]};

        std::uniform_int_distribution<std::uint32_t> point{first, last};

        const auto drawn_point{static_cast<char32_t>(point(random))};

        text += utf8_of(drawn_point);
    }

    return text;
}

/**
 * @brief Returns the ids of a text with no cache: every piece the segmenter cuts looked up whole where the definition
 *        says so, and otherwise merged alone.
 * @param merger The definition's merger.
 * @param segmenter The pattern's segmenter.
 * @param text The text.
 * @return The ids.
 */
std::vector<Token_id_t> uncached_ids(
        const matisse::encode::Merger& merger, const matisse::pretok::Segmenter& segmenter, const std::string_view text)
{
    std::vector<Token_id_t> ids{};

    std::size_t start{0};

    for (const auto end : segmenter.boundaries(text))
    {
        const auto piece{text.substr(start, end - start)};

        start = end;

        if (const auto whole{merger.whole_piece(piece)})
        {
            ids.push_back(*whole);

            continue;
        }

        const auto merged{merger.merged(piece)};

        ids.insert(ids.end(), merged.begin(), merged.end());
    }

    return ids;
}

/**
 * @brief Expects encode_parallel() to give encode()'s ids on a text at each chunk count, skipping a count past
 *        most_threads with a line saying so.
 * @param encoder The encoder.
 * @param text The text.
 * @param ids encode()'s ids of the text.
 * @param counts The chunk counts.
 * @param what What the text and the encoder are, printed beside each skip and failure.
 */
void expect_parallel_ids(
        const Encoder& encoder, const std::string_view text, const std::span<const Token_id_t> ids,
        const std::span<const std::size_t> counts, const std::string_view what)
{
    for (const auto chunks : counts)
    {
        if (chunks > most_threads)
        {
            std::printf(
                    "%.*s: %zu chunks skipped, more threads than %zu\n", static_cast<int>(what.size()), what.data(),
                    chunks, most_threads);

            continue;
        }

        const auto parallel{encoder.encode_parallel(text, chunks)};

        expect(std::ranges::equal(parallel, ids),
               std::format("{}: encode_parallel() in {} chunks gives encode()'s ids", what, chunks));
    }
}

/**
 * @brief Returns whether a byte continues a UTF-8 sequence rather than beginning one.
 * @param byte The byte.
 * @return True for a continuation byte, `10xxxxxx`.
 */
constexpr bool is_continuation(const char byte) noexcept
{
    return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

/**
 * @brief Returns the byte offset a number of code points after an offset of a UTF-8 text, or the text's end.
 * @param text The text.
 * @param from The offset, at the start of a code point.
 * @param points The number of code points.
 * @return The offset.
 */
std::size_t byte_after(const std::string_view text, const std::size_t from, const std::size_t points)
{
    auto byte{from};

    for (std::size_t point{0}; point < points && byte < text.size(); ++point)
    {
        ++byte;

        while (byte < text.size() && is_continuation(text[byte]))
        {
            ++byte;
        }
    }

    return byte;
}

/**
 * @brief Expects every cut of a text to splice: the pieces of each chunk between consecutive cuts, segmented with the
 *        right context as a complete input and kept while they end in the chunk, are the serial pieces of the chunk.
 *
 * This is what a worker of encode_parallel() keeps, held at every cut without a thread per chunk; the ids follow, since
 * every piece is encoded alone.
 * @param segmenter The pattern's segmenter.
 * @param right_context The inventory's right context in code points.
 * @param text The text, with no special token allowed.
 * @param cuts The cuts, from 0 to the text's size inclusive.
 * @param what What the cuts are of, printed on a failure.
 */
void expect_every_cut_splices(
        const matisse::pretok::Segmenter& segmenter, const std::size_t right_context, const std::string_view text,
        const std::span<const std::size_t> cuts, const std::string_view what)
{
    const auto serial{segmenter.boundaries(text)};

    auto next{serial.begin()};

    for (std::size_t chunk{0}; chunk + 1 < cuts.size(); ++chunk)
    {
        const auto begin{cuts[chunk]};
        const auto end{cuts[chunk + 1]};

        const auto context_end{byte_after(text, end, right_context)};

        const auto ends{segmenter.boundaries(text.substr(begin, context_end - begin))};

        std::vector<std::size_t> kept{};

        for (const auto piece_end : ends)
        {
            if (begin + piece_end > end)
            {
                break;
            }

            kept.push_back(begin + piece_end);
        }

        const auto chunk_end{std::ranges::upper_bound(next, serial.end(), end)};

        const std::vector<std::size_t> expected{next, chunk_end};

        next = chunk_end;

        const auto ends_at_cut{kept.empty() ? begin == end : kept.back() == end};

        if (kept != expected || !ends_at_cut)
        {
            expect(false,
                   std::format("{}: the chunk {} to {} keeps the serial pieces and ends at its cut", what, begin, end));

            return;
        }
    }
}

/**
 * @brief Checks the seeded texts under one tokenizer: the ids with no cache, decode(), and for both inventories
 *        encode_parallel() at one to sixteen chunks, at sixty-four and at every cut, and every cut's splice.
 * @param tokenizer The tokenizer.
 * @param report The report validate() decided of the tokenizer's pattern.
 */
void check_generated(const Tokenizer& tokenizer, const matisse::validate::Report& report)
{
    const auto& [tokenizer_name, pattern, definition]{tokenizer};

    const matisse::encode::Merger merger{definition};

    const matisse::pretok::Segmenter segmenter{pattern};

    std::vector<std::size_t> counts{};

    for (std::size_t chunks{1}; chunks <= 16; ++chunks)
    {
        counts.push_back(chunks);
    }

    counts.push_back(64);

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const Encoder encoder{definition, report, inventory, pattern, {}};

        const auto& table{inventory == Inventory::factorization ? report.factorization : report.policy};

        std::mt19937 random{20261020};

        for (std::size_t text_index{0}; text_index < generated_texts; ++text_index)
        {
            const auto text{random_text(random, generated_fragments)};

            const auto ids{encoder.encode(text)};

            const auto what{std::format("{} generated text {}", tokenizer_name, text_index)};

            const auto uncached{uncached_ids(merger, segmenter, text)};

            expect(uncached == ids, what + ": encode() gives the ids with no cache");

            const auto decoded{encoder.decode(ids)};

            expect(decoded == text, what + ": decode() spells the text back");

            auto every{counts};

            every.push_back(text.size() + 1);

            expect_parallel_ids(encoder, text, ids, every, what);

            const auto cuts{encoder.chunk_boundaries(text, text.size() + 1)};

            expect(cuts.size() >= least_generated_cuts + 2,
                   std::format("{}: the text takes at least {} cuts", what, least_generated_cuts));

            expect_every_cut_splices(segmenter, table.right_context, text, cuts, what);
        }
    }

    std::printf(
            "%.*s: %zu generated texts under both inventories\n", static_cast<int>(tokenizer_name.size()),
            tokenizer_name.data(), generated_texts);
}

/**
 * @brief Returns the SHA-256 of a run of ids written as little-endian 32-bit words, the fixtures' digest of ids.
 * @param ids The ids.
 * @return The 64 hexadecimal digits.
 */
std::string digest_of(const std::span<const Token_id_t> ids)
{
    std::string words{};

    words.reserve(ids.size() * 4);

    for (const auto id : ids)
    {
        for (unsigned shift{0}; shift < 32; shift += 8)
        {
            words += static_cast<char>((id >> shift) & 0xFFU);
        }
    }

    return matisse::common::sha256(words);
}

/**
 * @brief Returns the number of code points of a UTF-8 text.
 * @param text The text.
 * @return The count.
 */
std::size_t code_points(const std::string_view text)
{
    const auto starts{std::ranges::count_if(text, [](const char byte) { return !is_continuation(byte); })};

    return static_cast<std::size_t>(starts);
}

/**
 * @brief Returns the chunk counts encode_parallel() is held to on a corpus: the worker counts and the chunkings of
 *        chunk_points code points.
 * @param text The corpus.
 * @return The counts, ascending, none repeated.
 */
std::vector<std::size_t> corpus_chunk_counts(const std::string_view text)
{
    std::vector<std::size_t> counts{worker_counts.begin(), worker_counts.end()};

    const auto points{code_points(text)};

    for (const auto length : chunk_points)
    {
        const auto chunks{std::max(1UZ, points / length)};

        counts.push_back(chunks);
    }

    std::ranges::sort(counts);

    const auto repeated{std::ranges::unique(counts)};

    counts.erase(repeated.begin(), repeated.end());

    return counts;
}

/**
 * @brief Checks one corpus under one tokenizer and prints its ids line and supply section.
 * @param tokenizer The tokenizer.
 * @param report The report validate() decided of the tokenizer's pattern.
 * @param found The fixtures' references.
 * @param name The corpus's name.
 * @param text The corpus.
 */
void check_corpus(
        const Tokenizer& tokenizer, const matisse::validate::Report& report, const References& found,
        const std::string& name, const std::string_view text)
{
    const auto& [tokenizer_name, pattern, definition]{tokenizer};

    const Encoder factorization{definition, report, Inventory::factorization, pattern, {}};

    const auto ids{factorization.encode(text)};

    const auto digest{digest_of(ids)};

    std::printf(
            "ids %.*s %s %zu %s\n", static_cast<int>(tokenizer_name.size()), tokenizer_name.data(), name.c_str(),
            ids.size(), digest.c_str());

    const auto reference{found.ids.find({std::string{tokenizer_name}, name})};

    if (reference != found.ids.end())
    {
        const auto& [count, reference_digest]{reference->second};

        expect(ids.size() == count && digest == reference_digest,
               std::format(
                       "{} {}: the ids are the known-answer fixture's {} {}", tokenizer_name, name, count,
                       reference_digest));
    }

    const matisse::encode::Merger merger{definition};

    const matisse::pretok::Segmenter segmenter{pattern};

    const auto uncached{uncached_ids(merger, segmenter, text)};

    expect(uncached == ids,
           std::format("{} {}: encode() gives the ids of every piece merged with no cache", tokenizer_name, name));

    const auto decoded{factorization.decode(ids)};

    expect(decoded == text, std::format("{} {}: decode() spells the text back", tokenizer_name, name));

    auto counts{corpus_chunk_counts(text)};

    counts.push_back(text.size() + 1);

    const auto check_inventory{[&](const Inventory inventory, const Encoder& encoder) {
        const std::string_view label{inventory == Inventory::factorization ? "factorization" : "policy"};

        const auto what{std::format("{} {} under {}", tokenizer_name, name, label)};

        expect_parallel_ids(encoder, text, ids, counts, what);

        const auto cuts{encoder.chunk_boundaries(text, text.size() + 1)};

        const auto& table{inventory == Inventory::factorization ? report.factorization : report.policy};

        expect_every_cut_splices(segmenter, table.right_context, text, cuts, what);

        std::printf("%s: %zu cuts checked\n", what.c_str(), cuts.size() - 2);
    }};

    check_inventory(Inventory::factorization, factorization);

    const Encoder policy{definition, report, Inventory::policy, pattern, {}};

    check_inventory(Inventory::policy, policy);

    const auto factorization_supply{matisse::validate::supply(report, Inventory::factorization, pattern, text)};

    const auto policy_supply{matisse::validate::supply(report, Inventory::policy, pattern, text)};

    const auto section{matisse::validate::supply_section(factorization_supply, policy_supply, name)};

    std::printf("%s", section.c_str());
}

} // namespace

/**
 * @brief Runs the probe.
 * @param argc The argument count.
 * @param argv The arguments: the vocabulary directory, then the corpus files, none for the seeded texts.
 * @return EXIT_SUCCESS when every expectation held, EXIT_FAILURE otherwise.
 */
int main(const int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: matisse_corpus VOCAB_DIR [CORPUS...]\n");

        return EXIT_FAILURE;
    }

    try
    {
        const auto measured{tokenizers(argv[1])};

        const auto found{references()};

        expect(found.texts.size() == fixture_corpora && found.ids.size() == 2 * fixture_corpora,
               "the fixtures hold the three corpora's texts and both tokenizers' ids of them");

        std::vector<std::pair<std::string, std::string>> corpora{};

        for (int argument{2}; argument < argc; ++argument)
        {
            const std::filesystem::path path{argv[argument]};

            corpora.emplace_back(corpus_name(path), contents(path));
        }

        for (const auto& [name, text] : corpora)
        {
            const auto digest{matisse::common::sha256(text)};

            std::printf("corpus %s %zu bytes %s\n", name.c_str(), text.size(), digest.c_str());

            const auto reference{found.texts.find(name)};

            if (reference != found.texts.end())
            {
                expect(digest == reference->second, std::format("{} is the text the fixtures pin", name));
            }
        }

        for (const auto& tokenizer : measured)
        {
            const auto& [tokenizer_name, pattern, definition]{tokenizer};

            const auto report{matisse::validate::validate(pattern)};

            if (corpora.empty())
            {
                check_generated(tokenizer, report);
            }

            for (const auto& [name, text] : corpora)
            {
                check_corpus(tokenizer, report, found, name, text);
            }
        }
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "matisse_corpus: %s\n", error.what());

        return EXIT_FAILURE;
    }

    std::printf("%s\n", failures == 0 ? "OK" : "FAILED");

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
