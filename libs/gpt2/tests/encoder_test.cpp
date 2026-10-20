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

#include "matisse/common/sha256.hpp"
#include "matisse/encode/merger.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/gpt2/tokenizer.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::encode;
using matisse::load::Definition;
using matisse::load::Token_id_t;

namespace
{
/**
 * @brief The SHA-256 of GPT-2's encoder.json as pinned, gpt2-vocab.json in the vocabulary directory.
 */
constexpr std::string_view gpt2_encoder_digest{"196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"};

/**
 * @brief The SHA-256 of GPT-2's vocab.bpe as pinned, gpt2-merges.txt in the vocabulary directory.
 */
constexpr std::string_view gpt2_vocab_digest{"1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"};

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
 * @brief The known-answer fixture under check: whether the current universe is GPT-2's, and the counts of its lines so
 *        far.
 */
struct Fixture_check
{
    /**
     * @brief Whether the current universe is GPT-2's, false before the first universe line.
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
 * @brief Returns GPT-2's pinned files read, after checking each against its digest.
 * @return The definition, std::nullopt when the vocabulary directory does not hold the files.
 */
std::optional<Definition> pinned_gpt2()
{
    const auto encoder{pinned("gpt2-vocab.json")};
    const auto vocab{pinned("gpt2-merges.txt")};

    if (!encoder || !vocab)
    {
        return std::nullopt;
    }

    EXPECT_EQ(matisse::common::sha256(contents(*encoder)), gpt2_encoder_digest);
    EXPECT_EQ(matisse::common::sha256(contents(*vocab)), gpt2_vocab_digest);

    return matisse::load::read_gpt2(*encoder, *vocab);
}

/**
 * @brief Returns GPT-2's encoder, its report decided once and its factorization inventory kept, with no special token,
 *        since encoder.py reads `<|endoftext|>` as text.
 * @param definition GPT-2's definition.
 * @return The encoder.
 */
Encoder gpt2_encoder(Definition definition)
{
    const auto pattern{matisse::gpt2::pattern()};

    const auto report{matisse::validate::validate(pattern)};

    return {std::move(definition), report, matisse::validate::Inventory::factorization, pattern, {}};
}

/**
 * @brief Returns the ids of a text, each piece GPT-2's pattern segments merged with no cache.
 * @param definition GPT-2's definition.
 * @param text The text.
 * @return The ids.
 */
std::vector<Token_id_t> uncached_ids(const Definition& definition, const std::string_view text)
{
    const Merger merger{definition};

    const matisse::pretok::Segmenter segmenter{matisse::gpt2::pattern()};

    std::vector<Token_id_t> ids{};

    std::size_t start{0};

    for (const auto end : segmenter.boundaries(text))
    {
        const auto piece{text.substr(start, end - start)};

        start = end;

        const auto merged{merger.merged(piece)};

        ids.insert(ids.end(), merged.begin(), merged.end());
    }

    return ids;
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
 * @brief Starts a universe line's universe, GPT-2's when it names gpt2.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.inside = name == "gpt2";

    if (check.inside)
    {
        ++check.counts.universes;
    }
}

/**
 * @brief Checks an encode line: GPT-2's encoder gives its text, written in hex, its ids, and decodes them back.
 * @param check The fixture under check.
 * @param encoder GPT-2's encoder.
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
 * @brief Checks one fixture line with GPT-2's encoder, a comment or blank line checking nothing, a line of another
 *        universe skipped and a corpus line only counted.
 * @param check The fixture under check.
 * @param encoder GPT-2's encoder.
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
 *        newlines followed by tab runs, long letter runs, contractions in both cases, digits, the special token's
 *        spelling, and code points drawn from several scripts and the emoji.
 * @param random The random sequence.
 * @param fragments The number of fragments joined.
 * @return The text, well-formed UTF-8.
 */
std::string random_text(std::mt19937& random, const std::size_t fragments)
{
    constexpr std::array<std::string_view, 20> shapes{
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

TEST(Encoder_test, The_published_known_answer_hello_world_is_15496_995)
{
    auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    const auto encoder{gpt2_encoder(std::move(*definition))};

    EXPECT_EQ(encoder.encode("Hello world"), (std::vector<Token_id_t>{15496, 995}));
}

TEST(Encoder_test, Every_known_answer_of_the_fixture_is_the_encoding_of_its_text_and_decodes_back_to_it)
{
    auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    const auto encoder{gpt2_encoder(std::move(*definition))};

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

TEST(Encoder_test, The_ids_are_those_of_no_cache_and_decode_spells_the_text_back_on_seeded_random_texts)
{
    auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    const auto encoder{gpt2_encoder(*definition)};

    std::mt19937 random{7};

    for (std::size_t count{0}; count < 300; ++count)
    {
        const auto text{random_text(random, 60)};

        const auto ids{encoder.encode(text)};

        EXPECT_EQ(ids, uncached_ids(*definition, text)) << text;
        EXPECT_EQ(encoder.decode(ids), text);
    }
}

TEST(Encoder_test, The_ids_are_those_of_no_cache_on_a_text_whose_distinct_pieces_fill_the_cache_past_its_bound)
{
    auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    // Twenty thousand words of five to twelve letters, almost all distinct, each merged since GPT-2 looks no piece up
    // whole, so the cache stops filling at its bound.
    constexpr std::string_view letters{"abcdefghijklmnopqrstuvwxyz"};

    std::mt19937 random{9};

    std::uniform_int_distribution<std::size_t> length{5, 12};

    std::uniform_int_distribution<std::size_t> letter{0, letters.size() - 1};

    std::string text{};

    std::set<std::string> words{};

    for (std::size_t count{0}; count < 20000; ++count)
    {
        std::string word{" "};

        for (auto left{length(random)}; left > 0; --left)
        {
            word += letters[letter(random)];
        }

        text += word;

        words.insert(std::move(word));
    }

    ASSERT_GT(words.size(), piece_cache_capacity);

    const auto encoder{gpt2_encoder(*definition)};

    EXPECT_EQ(encoder.encode(text), uncached_ids(*definition, text));
}
