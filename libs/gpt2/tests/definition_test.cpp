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

#include "matisse/common/sha256.hpp"
#include "matisse/gpt2/tokenizer.hpp"

using namespace matisse::load;

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
 * @brief The known-answer fixture under check: GPT-2's pinned vocabulary, whether the current universe is GPT-2's, and
 *        the counts of its lines so far.
 */
struct Fixture_check
{
    /**
     * @brief GPT-2's pinned vocabulary.
     */
    Definition gpt2{};

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

    return read_gpt2(*encoder, *vocab);
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
 * @brief Checks an encode line: its ids spell its text, written in hex, under GPT-2's vocabulary.
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

    EXPECT_EQ(spelled(check.gpt2, ids), bytes_of(text)) << context;

    ++check.counts.encodings;
}

/**
 * @brief Checks one fixture line against GPT-2's pinned vocabulary, a comment or blank line checking nothing, a line of
 *        another universe skipped and a corpus line only counted.
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

TEST(Definition_test, The_special_tokens_of_a_rank_file_under_gpt2_are_the_vendor_s)
{
    const auto specials{matisse::gpt2::specials()};

    ASSERT_EQ(specials.size(), 1U);
    EXPECT_EQ(specials[0].spelling, "<|endoftext|>");
    EXPECT_EQ(specials[0].id, 50256U);
}

TEST(Definition_test, The_pinned_gpt2_files_read_as_its_vocabulary_and_spell_the_published_known_answer)
{
    const auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    EXPECT_EQ(definition->spellings.size(), 50257U);
    EXPECT_EQ(definition->merges.size(), 50000U);
    EXPECT_EQ(definition->spellings[50256], "<|endoftext|>");
    EXPECT_EQ(definition->spellings[220], " ");
    EXPECT_EQ(spelled(*definition, {15496, 995}), "Hello world");
}

TEST(Definition_test, The_ids_of_every_known_answer_of_the_fixture_spell_its_text_under_the_pinned_vocabulary)
{
    auto gpt2{pinned_gpt2()};

    if (!gpt2)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    std::ifstream file{std::string{SOURCE_DIR} + "/libs/load/tests/data/known_answers_reference.txt"};

    ASSERT_TRUE(file.is_open());

    Fixture_check check{.gpt2 = std::move(*gpt2)};

    for (std::string line{}; std::getline(file, line);)
    {
        expect_fixture_line(check, line);
    }

    EXPECT_EQ(check.counts.universes, 1U);
    EXPECT_EQ(check.counts.encodings, 15U);
    EXPECT_EQ(check.counts.corpora, 3U);
}
