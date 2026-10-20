#include "matisse/encode/encoder.hpp"

#include <array>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <optional>
#include <random>
#include <set>
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
#include "matisse/pretok/segmenter.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::encode;
using matisse::load::Definition;
using matisse::load::Token_id_t;

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
 * @brief Returns cl100k_base's encoder, its report decided once and its factorization inventory kept.
 * @param definition cl100k_base's definition.
 * @param allowed The special tokens allowed.
 * @return The encoder.
 */
Encoder cl100k_encoder(Definition definition, const std::vector<std::string>& allowed = {})
{
    const auto pattern{matisse::cl100k::pattern()};

    const auto report{matisse::validate::validate(pattern)};

    return {std::move(definition), report, matisse::validate::Inventory::factorization, pattern, allowed};
}

/**
 * @brief Returns the ids of a text with no allowed special token, each piece cl100k's pattern segments looked up whole
 *        or merged, with no cache.
 * @param definition cl100k_base's definition.
 * @param text The text.
 * @return The ids.
 */
std::vector<Token_id_t> uncached_ids(const Definition& definition, const std::string_view text)
{
    const Merger merger{definition};

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

    std::mt19937 random{7};

    for (std::size_t count{0}; count < 300; ++count)
    {
        const auto text{random_text(random, 60)};

        const auto ids{refusing.encode(text)};

        EXPECT_EQ(ids, uncached_ids(*definition, text)) << text;
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

    EXPECT_EQ(encoder.encode(text), uncached_ids(*definition, text));
}
