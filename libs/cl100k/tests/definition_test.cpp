#include "matisse/load/definition.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/common/sha256.hpp"

using namespace matisse::load;

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
 * @brief The known-answer fixture under check: cl100k_base's pinned vocabulary, whether the current universe is
 *        cl100k's, and the counts of its lines so far.
 */
struct Fixture_check
{
    /**
     * @brief cl100k_base's pinned vocabulary.
     */
    Definition cl100k{};

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
 * @brief Returns the text a run of ordinary token ids spells under a definition.
 * @param definition The definition.
 * @param ids The ids.
 * @return The concatenated spellings.
 */
std::string spelled(const Definition& definition, const std::vector<Token_id_t>& ids)
{
    std::string text{};

    for (const auto id : ids)
    {
        text += definition.spellings.at(id);
    }

    return text;
}

/**
 * @brief Returns cl100k_base's pinned rank file read, after checking it against its digest.
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

    return read_tiktoken(*path, specials);
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
 * @brief Checks an encode line: its ids spell its text, written in hex, under cl100k_base's vocabulary.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_encoding(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string text{};

    fields >> text;

    std::vector<Token_id_t> ids{};

    for (Token_id_t id{}; fields >> id;)
    {
        ids.push_back(id);
    }

    EXPECT_EQ(spelled(check.cl100k, ids), bytes_of(text)) << context;

    ++check.counts.encodings;
}

/**
 * @brief Checks one fixture line against cl100k_base's pinned vocabulary, a comment or blank line checking nothing, a
 *        line of another universe skipped and a corpus line only counted.
 * @param check The fixture under check.
 * @param line The line.
 */
void expect_fixture_line(Fixture_check& check, const std::string& line)
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
        expect_encoding(check, fields, line);
    }
    else
    {
        ADD_FAILURE() << "unknown fixture line: " << line;
    }
}

} // namespace

TEST(Definition_test, The_special_tokens_of_the_cl100k_rank_file_are_the_vendor_s)
{
    const auto specials{matisse::cl100k::specials()};

    ASSERT_EQ(specials.size(), 5U);
    EXPECT_EQ(specials[0].spelling, "<|endoftext|>");
    EXPECT_EQ(specials[0].id, 100257U);
    EXPECT_EQ(specials[4].spelling, "<|endofprompt|>");
    EXPECT_EQ(specials[4].id, 100276U);
}

TEST(Definition_test, The_pinned_cl100k_file_reads_as_its_vocabulary_and_spells_the_published_known_answer)
{
    const auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    EXPECT_EQ(definition->spellings.size(), 100256U);
    EXPECT_EQ(definition->specials.size(), 5U);
    EXPECT_EQ(spelled(*definition, {83, 1609, 5963, 374, 2294, 0}), "tiktoken is great!");
}

TEST(Definition_test, The_ids_of_every_known_answer_of_the_fixture_spell_its_text_under_the_pinned_vocabulary)
{
    auto cl100k{pinned_cl100k()};

    if (!cl100k)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    std::ifstream file{std::string{SOURCE_DIR} + "/libs/load/tests/data/known_answers_reference.txt"};

    ASSERT_TRUE(file.is_open());

    Fixture_check check{.cl100k = std::move(*cl100k)};

    for (std::string line{}; std::getline(file, line);)
    {
        expect_fixture_line(check, line);
    }

    EXPECT_EQ(check.counts.universes, 1U);
    EXPECT_EQ(check.counts.encodings, 15U);
    EXPECT_EQ(check.counts.corpora, 3U);
}
