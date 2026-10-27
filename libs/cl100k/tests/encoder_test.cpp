#include "matisse/encode/encoder.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/common/sha256.hpp"
#include "matisse/encode/merger.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/encode/stream_encoder.hpp"
#include "matisse/encode/window_index.hpp"
#include "matisse/pretok/classes.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::encode;
using matisse::load::Definition;
using matisse::load::Token_id_t;
using matisse::validate::Inventory;

namespace
{
/**
 * @brief The SHA-256 of cl100k_base.tiktoken as pinned.
 */
constexpr std::string_view cl100k_digest{"223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"};

/**
 * @brief What the known-answer fixture holds, counted as it is checked.
 */
struct Fixture_counts
{
    /**
     * @brief The universe lines.
     */
    std::size_t universes{0};

    /**
     * @brief The encode lines.
     */
    std::size_t encodings{0};

    /**
     * @brief The corpus lines, which the corpus probe checks.
     */
    std::size_t corpora{0};
};

/**
 * @brief The known-answer fixture under check: whether the current universe is cl100k's, and the counts of its lines so
 *        far.
 */
struct Fixture_check
{
    /**
     * @brief Whether the current universe is cl100k's, false before the first universe line.
     */
    bool inside{};

    /**
     * @brief The lines checked so far, by kind.
     */
    Fixture_counts counts{};
};

/**
 * @brief A corpus the vendor fixture names: its file name in the corpus directory, its bytes, its documents, the parts
 *        its NUL bytes separate, and the SHA-256 of its bytes.
 */
struct Vendor_corpus
{
    /**
     * @brief The file name.
     */
    std::string name{};

    /**
     * @brief The bytes.
     */
    std::size_t bytes{};

    /**
     * @brief The documents.
     */
    std::size_t documents{};

    /**
     * @brief The SHA-256 of the bytes.
     */
    std::string digest{};
};

/**
 * @brief The vendors' ids of a corpus under cl100k's definition: their count and their SHA-256 as little-endian 32-bit
 *        words, the documents' ids concatenated.
 */
struct Vendor_ids
{
    /**
     * @brief The count of the ids.
     */
    std::size_t count{};

    /**
     * @brief The SHA-256 of the ids.
     */
    std::string digest{};
};

/**
 * @brief The vendor fixture as the tests of cl100k read it: its universe lines naming cl100k, every corpus, and the
 *        vendors' ids of each under cl100k's definition, by the corpus's file name.
 */
struct Vendor_fixture
{
    /**
     * @brief The universe lines naming cl100k.
     */
    std::size_t universes{0};

    /**
     * @brief The corpora, in the fixture's order.
     */
    std::vector<Vendor_corpus> corpora{};

    /**
     * @brief The vendors' ids, by the corpus's file name.
     */
    std::map<std::string, Vendor_ids> ids{};
};

/**
 * @brief Returns the whole of a file.
 * @param path The file's path.
 * @return Its bytes.
 */
std::string contents(const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};

    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

/**
 * @brief Returns the path of a pinned vocabulary file in the directory MATISSE_VOCAB_DIR names.
 * @param name The file's name.
 * @return The path, std::nullopt when no directory is named or it does not hold the file.
 */
std::optional<std::filesystem::path> pinned(const std::string_view name)
{
    const std::filesystem::path directory{VOCAB_DIR};

    if (directory.empty() || !std::filesystem::exists(directory / name))
    {
        return std::nullopt;
    }

    return directory / name;
}

/**
 * @brief Returns cl100k_base's pinned rank file read with the vendor's special tokens, after checking it against its
 *        digest.
 * @return The definition, std::nullopt when the vocabulary directory does not hold the file.
 */
std::optional<Definition> pinned_cl100k()
{
    const auto path{pinned("cl100k_base.tiktoken")};

    if (!path)
    {
        return std::nullopt;
    }

    EXPECT_EQ(matisse::common::sha256(contents(*path)), cl100k_digest);

    const auto specials{matisse::cl100k::specials()};

    return matisse::load::read_tiktoken(*path, specials);
}

/**
 * @brief Returns cl100k's report at the default width, decided once for every test of this file.
 * @return The report.
 */
const matisse::validate::Report& decided_report()
{
    static const auto report{matisse::validate::validate(matisse::cl100k::pattern())};

    return report;
}

/**
 * @brief Returns cl100k_base's encoder, the report decided once and its factorization inventory kept.
 * @param definition cl100k_base's definition.
 * @param allowed The special tokens allowed.
 * @return The encoder.
 */
Encoder cl100k_encoder(Definition definition, const std::vector<std::string>& allowed = {})
{
    const auto pattern{matisse::cl100k::pattern()};

    return {std::move(definition), decided_report(), matisse::validate::Inventory::factorization, allowed};
}

/**
 * @brief Returns the ids of a text with no allowed special token, each piece cl100k's pattern segments looked up whole
 *        or merged, with no cache.
 * @param merger cl100k_base's merger.
 * @param text The text.
 * @return The ids.
 */
std::vector<Token_id_t> uncached_ids(const Merger& merger, const std::string_view text)
{
    const matisse::pretok::Segmenter segmenter{matisse::cl100k::pattern()};

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
 * @brief Returns the spellings of the vendor's special tokens of cl100k_base.
 * @return The spellings.
 */
std::vector<std::string> special_spellings()
{
    std::vector<std::string> spellings{};

    for (const auto& [spelling, id] : matisse::cl100k::specials())
    {
        spellings.push_back(spelling);
    }

    return spellings;
}

/**
 * @brief Returns the bytes a lowercase hex field spells, two digits per byte, "-" spelling the empty text.
 * @param field The hex digits.
 * @return The bytes.
 */
std::string bytes_of(const std::string_view field)
{
    std::string bytes{};

    if (field == "-")
    {
        return bytes;
    }

    for (std::size_t at{0}; at + 1 < field.size(); at += 2)
    {
        unsigned value{};

        std::ignore = std::from_chars(field.data() + at, field.data() + at + 2, value, 16);

        bytes += static_cast<char>(value);
    }

    return bytes;
}

/**
 * @brief Starts a universe line's universe, cl100k's when it names cl100k.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.inside = name == "cl100k";

    if (check.inside)
    {
        ++check.counts.universes;
    }
}

/**
 * @brief Checks an encode line: cl100k_base's encoder gives its text, written in hex, its ids, and decodes them back.
 * @param check The fixture under check.
 * @param encoder cl100k_base's encoder.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_encoding(
        Fixture_check& check, const Encoder& encoder, std::istringstream& fields, const std::string& context)
{
    std::string text{};

    fields >> text;

    std::vector<Token_id_t> ids{};

    for (Token_id_t id{}; fields >> id;)
    {
        ids.push_back(id);
    }

    const auto bytes{bytes_of(text)};

    EXPECT_EQ(encoder.encode(bytes), ids) << context;
    EXPECT_EQ(encoder.decode(ids), bytes) << context;

    ++check.counts.encodings;
}

/**
 * @brief Checks one fixture line with cl100k_base's encoder, a comment or blank line checking nothing, a line of
 *        another universe skipped and a corpus line only counted.
 * @param check The fixture under check.
 * @param encoder cl100k_base's encoder.
 * @param line The line.
 */
void expect_fixture_line(Fixture_check& check, const Encoder& encoder, const std::string& line)
{
    std::istringstream fields{line};

    std::string kind{};

    if (!(fields >> kind) || kind.starts_with('#'))
    {
        return;
    }

    if (kind == "universe")
    {
        start_universe(check, fields);
    }
    else if (!check.inside)
    {
        return;
    }
    else if (kind == "corpus")
    {
        ++check.counts.corpora;
    }
    else if (kind == "encode")
    {
        expect_encoding(check, encoder, fields, line);
    }
    else
    {
        ADD_FAILURE() << "unknown fixture line: " << line;
    }
}

/**
 * @brief Returns the vendor fixture read for cl100k: the corpus lines before the first universe line, and the ids lines
 *        of cl100k's universe, a line of another universe skipped and a line of another kind a failure.
 * @return The fixture.
 */
Vendor_fixture vendor_fixture()
{
    std::ifstream file{std::string{SOURCE_DIR} + "/libs/load/tests/data/vendors_reference.txt"};

    EXPECT_TRUE(file.is_open());

    Vendor_fixture fixture{};

    std::string universe{};

    for (std::string line{}; std::getline(file, line);)
    {
        std::istringstream fields{line};

        std::string kind{};

        if (!(fields >> kind) || kind.starts_with('#'))
        {
            continue;
        }

        if (kind == "universe")
        {
            fields >> universe;

            if (universe == "cl100k")
            {
                ++fixture.universes;
            }

            continue;
        }

        std::string name{};

        if (kind == "corpus" && universe.empty())
        {
            std::size_t bytes{};
            std::size_t documents{};
            std::string digest{};

            fields >> name >> bytes >> documents >> digest;

            fixture.corpora.push_back(
                    Vendor_corpus{.name = name, .bytes = bytes, .documents = documents, .digest = digest});

            continue;
        }

        if (universe != "cl100k")
        {
            continue;
        }

        if (kind != "ids")
        {
            ADD_FAILURE() << "unknown fixture line: " << line;

            continue;
        }

        std::size_t count{};
        std::string digest{};

        fields >> name >> count >> digest;

        fixture.ids[name] = Vendor_ids{.count = count, .digest = digest};
    }

    return fixture;
}

/**
 * @brief Returns the path of a corpus in the directory MATISSE_CORPUS_DIR names.
 * @param name The corpus's file name.
 * @return The path, std::nullopt when no directory is named or it does not hold the corpus.
 */
std::optional<std::filesystem::path> corpus_path(const std::string_view name)
{
    const std::filesystem::path directory{CORPUS_DIR};

    if (directory.empty() || !std::filesystem::exists(directory / name))
    {
        return std::nullopt;
    }

    return directory / name;
}

/**
 * @brief Adds a run of ids to a digest as little-endian 32-bit words, the fixtures' digest of ids.
 * @param hash The digest.
 * @param ids The ids.
 */
void add_ids(matisse::common::Sha256& hash, const std::span<const Token_id_t> ids)
{
    for (const auto id : ids)
    {
        const std::array word{
                static_cast<char>(id & 0xFFU), static_cast<char>((id >> 8U) & 0xFFU),
                static_cast<char>((id >> 16U) & 0xFFU), static_cast<char>(id >> 24U)};

        hash.add({word.data(), word.size()});
    }
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
 * @brief Returns a seeded random text of the shapes the pattern meets: white space runs before letters and punctuation,
 *        newlines followed by tab runs, long letter runs, contractions in both cases, digits, the special tokens'
 *        spellings whole and broken, and code points drawn from several scripts and the emoji.
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
            "<|endofprompt|>",
            "abcdefghijklmnopqrstuvwxyz",
            " ?!"};

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

        std::uniform_int_distribution<char32_t> point{first, last};

        text += utf8_of(point(random));
    }

    return text;
}

/**
 * @brief Returns the chunk counts the self-consistency checks ask for: one to sixteen, sixty-four, and one more than
 *        the text's bytes, which takes every cut the rule offers.
 * @param text The text.
 * @return The counts.
 */
std::vector<std::size_t> chunk_counts(const std::string_view text)
{
    std::vector<std::size_t> counts{};

    for (std::size_t chunks{1}; chunks <= 16; ++chunks)
    {
        counts.push_back(chunks);
    }

    counts.push_back(64);

    counts.push_back(text.size() + 1);

    return counts;
}

/**
 * @brief Expects every anchor of a text, read as one stretch, to be a boundary of the pattern's segmentation of it.
 * @param index The window index of an inventory.
 * @param class_table The class table of the pattern's alphabet.
 * @param segmenter The pattern's segmenter.
 * @param text The text, well-formed UTF-8.
 */
void expect_anchors_are_boundaries(
        const Window_index& index, const matisse::pretok::Class_table& class_table,
        const matisse::pretok::Segmenter& segmenter, const std::string_view text)
{
    const auto pieces{segmenter.boundaries(text)};

    for (std::size_t at{0}; at < text.size(); ++at)
    {
        const auto begins_code_point{(static_cast<unsigned char>(text[at]) & 0xC0U) != 0x80U};

        if (begins_code_point && index.first_anchor(class_table, text, at) == at)
        {
            EXPECT_TRUE(std::ranges::contains(pieces, at)) << text << " anchor " << at;
        }
    }
}

/**
 * @brief Returns seeded random ends of the pushes of a text, each push 0 to a bound of bytes long, so that pushes end
 *        inside UTF-8 sequences and inside spellings and some push nothing.
 * @param random The random sequence.
 * @param size The text's size.
 * @param widest The longest push.
 * @return The byte after each push, ascending to the text's size.
 */
std::vector<std::size_t> random_cuts(std::mt19937& random, const std::size_t size, const std::size_t widest)
{
    std::uniform_int_distribution<std::size_t> width{0, widest};

    std::vector<std::size_t> cuts{};

    for (std::size_t at{0}; at < size;)
    {
        at = std::min(size, at + width(random));

        cuts.push_back(at);
    }

    return cuts;
}

/**
 * @brief Pushes a text to a stream in parts and expects every push to return ids that extend a prefix of encode() of
 *        the whole text, and the ids of all pushes and finish() to be encode() of it.
 * @param encoder The encoder.
 * @param text The text.
 * @param cuts The byte after each push.
 */
void expect_streamed(const Encoder& encoder, const std::string_view text, const std::span<const std::size_t> cuts)
{
    const auto expected{encoder.encode(text)};

    Stream_encoder stream{encoder};

    std::vector<Token_id_t> returned{};

    std::size_t start{0};

    for (const auto cut : cuts)
    {
        const auto ids{stream.push(text.substr(start, cut - start))};

        returned.insert(returned.end(), ids.begin(), ids.end());

        start = cut;

        ASSERT_LE(returned.size(), expected.size()) << text << " after " << cut;
        ASSERT_TRUE(std::ranges::equal(returned, std::span{expected}.first(returned.size())))
                << text << " after " << cut;
    }

    const auto rest{stream.finish()};

    returned.insert(returned.end(), rest.begin(), rest.end());

    EXPECT_EQ(returned, expected) << text;
}

} // namespace

TEST(Encoder_test, The_published_known_answer_tiktoken_is_great_is_83_1609_5963_374_2294_0)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto encoder{cl100k_encoder(std::move(*definition))};

    EXPECT_EQ(encoder.encode("tiktoken is great!"), (std::vector<Token_id_t>{83, 1609, 5963, 374, 2294, 0}));
}

TEST(Encoder_test, Every_known_answer_of_the_fixture_is_the_encoding_of_its_text_and_decodes_back_to_it)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto encoder{cl100k_encoder(std::move(*definition))};

    std::ifstream file{std::string{SOURCE_DIR} + "/libs/load/tests/data/known_answers_reference.txt"};

    ASSERT_TRUE(file.is_open());

    Fixture_check check{};

    for (std::string line{}; std::getline(file, line);)
    {
        expect_fixture_line(check, encoder, line);
    }

    EXPECT_EQ(check.counts.universes, 1U);
    EXPECT_EQ(check.counts.encodings, 15U);
    EXPECT_EQ(check.counts.corpora, 3U);
}

TEST(Encoder_test, Every_corpus_of_the_vendor_fixture_the_corpus_directory_holds_encodes_to_the_vendors_ids)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto [universes, corpora, vendor_ids]{vendor_fixture()};

    EXPECT_EQ(universes, 1U);
    EXPECT_EQ(corpora.size(), 14U);
    EXPECT_EQ(vendor_ids.size(), 14U);

    const auto encoder{cl100k_encoder(std::move(*definition))};

    std::string absent{};

    for (const auto& [name, bytes, documents, digest] : corpora)
    {
        const auto path{corpus_path(name)};

        if (!path)
        {
            absent += absent.empty() ? name : ", " + name;

            continue;
        }

        const auto text{contents(*path)};

        EXPECT_EQ(text.size(), bytes) << name;
        EXPECT_EQ(matisse::common::sha256(text), digest) << name;

        matisse::common::Sha256 hash{};

        std::size_t count{0};
        std::size_t parts{0};

        // Each document is pushed to the stream in parts of a mebibyte, so that a corpus of any size encodes in bounded
        // memory, and finish() ends it, the ids returned since the last finish() being encode()'s of the document.
        constexpr std::size_t push_bytes{1UZ << 20U};

        Stream_encoder stream{encoder};

        for (const auto part : std::views::split(text, '\0'))
        {
            const std::string_view document{part.begin(), part.end()};

            for (std::size_t at{0}; at < document.size(); at += push_bytes)
            {
                const auto ids{stream.push(document.substr(at, push_bytes))};

                add_ids(hash, ids);

                count += ids.size();
            }

            const auto rest{stream.finish()};

            add_ids(hash, rest);

            count += rest.size();
            ++parts;
        }

        const auto vendor{vendor_ids.find(name)};

        ASSERT_NE(vendor, vendor_ids.end()) << name;

        const auto& [vendor_count, vendor_digest]{vendor->second};

        EXPECT_EQ(parts, documents) << name;
        EXPECT_EQ(count, vendor_count) << name;
        EXPECT_EQ(hash.digest(), vendor_digest) << name;
    }

    if (!absent.empty())
    {
        GTEST_SKIP() << "MATISSE_CORPUS_DIR names no directory holding " << absent;
    }
}

TEST(Encoder_test, An_allowed_special_token_is_its_id_and_one_not_allowed_is_ordinary_text)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto allowing{cl100k_encoder(*definition, {"<|endoftext|>"})};

    const auto refusing{cl100k_encoder(std::move(*definition))};

    // The fixture's ids of "<|endoftext|>" as text, which an encoder allowing no special token gives too.
    const std::vector<Token_id_t> as_text{27, 91, 8862, 728, 428, 91, 29};

    EXPECT_EQ(allowing.encode("a<|endoftext|>b"), (std::vector<Token_id_t>{64, 100257, 65}));
    EXPECT_EQ(allowing.encode("<|endoftext|>"), std::vector<Token_id_t>{100257});
    EXPECT_EQ(refusing.encode("<|endoftext|>"), as_text);
    EXPECT_EQ(allowing.decode(std::vector<Token_id_t>{100257, 100276}), "<|endoftext|><|endofprompt|>");
}

TEST(Encoder_test, The_ids_are_those_of_no_cache_and_decode_spells_the_text_back_on_seeded_random_texts)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto allowing{cl100k_encoder(*definition, special_spellings())};

    const auto refusing{cl100k_encoder(*definition)};

    const Merger merger{*definition};

    std::mt19937 random{7};

    for (std::size_t count{0}; count < 300; ++count)
    {
        const auto text{random_text(random, 60)};

        const auto ids{refusing.encode(text)};

        EXPECT_EQ(ids, uncached_ids(merger, text)) << text;
        EXPECT_EQ(refusing.decode(ids), text);
        EXPECT_EQ(allowing.decode(allowing.encode(text)), text);
    }
}

TEST(Encoder_test, The_ids_are_those_of_no_cache_on_a_text_whose_distinct_pieces_fill_the_cache_past_its_bound)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    // Twenty thousand words of eight to fourteen letters, almost all distinct and few of them tokens, so that most are
    // merged and the cache stops filling at its bound.
    constexpr std::string_view letters{"abcdefghijklmnopqrstuvwxyz"};

    std::mt19937 random{9};

    std::uniform_int_distribution<std::size_t> length{8, 14};

    std::uniform_int_distribution<std::size_t> letter{0, letters.size() - 1};

    std::string text{};

    std::set<std::string> merged_words{};

    const Merger merger{*definition};

    for (std::size_t count{0}; count < 20000; ++count)
    {
        std::string word{" "};

        for (auto left{length(random)}; left > 0; --left)
        {
            word += letters[letter(random)];
        }

        text += word;

        if (!merger.whole_piece(word))
        {
            merged_words.insert(std::move(word));
        }
    }

    ASSERT_GT(merged_words.size(), piece_cache_capacity);

    const auto encoder{cl100k_encoder(*definition)};

    EXPECT_EQ(encoder.encode(text), uncached_ids(merger, text));
}

TEST(Encoder_test, Encode_parallel_gives_encode_s_ids_under_both_inventories_and_every_anchor_ends_a_piece)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto pattern{matisse::cl100k::pattern()};

    const auto& report{decided_report()};

    const matisse::pretok::Segmenter segmenter{pattern};

    const matisse::pretok::Class_table class_table{pattern.alphabet};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        for (const auto& allowed : {std::vector<std::string>{}, special_spellings()})
        {
            const Encoder encoder{*definition, report, inventory, allowed};

            const Window_index index{inventory == Inventory::factorization ? report.factorization : report.policy};

            std::mt19937 random{23};

            for (std::size_t count{0}; count < 60; ++count)
            {
                const auto text{random_text(random, 60)};

                const auto ids{encoder.encode(text)};

                for (const auto chunks : chunk_counts(text))
                {
                    EXPECT_EQ(encoder.encode_parallel(text, chunks), ids) << text << " in " << chunks;
                }

                expect_anchors_are_boundaries(index, class_table, segmenter, text);
            }
        }
    }
}

TEST(Encoder_test, A_cut_in_white_space_holds_with_the_right_context_each_inventory_decided)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto pattern{matisse::cl100k::pattern()};

    const auto& report{decided_report()};

    const Encoder factorization{*definition, report, Inventory::factorization, {}};

    const Encoder policy{*definition, report, Inventory::policy, {}};

    // The right-context fixture's counterexamples: a worker reading one code point less past the cut before the
    // punctuation, or the one between the tabs, reads the tabs as one piece.
    const std::string before_punctuation{"aaaa\t\t!aaaaa"};

    const std::string between_tabs{"aaaa\t\tAaaaa"};

    EXPECT_EQ(report.factorization.right_context, 1U);
    EXPECT_EQ(report.policy.right_context, 2U);
    EXPECT_EQ(factorization.chunk_boundaries(before_punctuation, 2), (std::vector<std::size_t>{0, 6, 12}));
    EXPECT_EQ(factorization.encode_parallel(before_punctuation, 2), factorization.encode(before_punctuation));
    EXPECT_EQ(policy.chunk_boundaries(between_tabs, 2), (std::vector<std::size_t>{0, 5, 11}));
    EXPECT_EQ(policy.encode_parallel(between_tabs, 2), policy.encode(between_tabs));
}

TEST(Encoder_test, A_stream_pushed_in_parts_of_every_size_returns_prefixes_of_encode_and_concatenates_to_it)
{
    auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const std::array encoders{cl100k_encoder(*definition, special_spellings()), cl100k_encoder(*definition)};

    std::mt19937 random{17};

    for (std::size_t count{0}; count < 40; ++count)
    {
        const auto text{random_text(random, 40)};

        const auto one_byte{std::views::iota(1UZ, text.size() + 1)};

        const std::vector<std::size_t> one_byte_cuts{one_byte.begin(), one_byte.end()};

        for (const auto& encoder : encoders)
        {
            expect_streamed(encoder, text, random_cuts(random, text.size(), 24));
            expect_streamed(encoder, text, one_byte_cuts);
            expect_streamed(encoder, text, std::array{text.size()});
        }
    }
}
