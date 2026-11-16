#include "matisse/load/definition.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "matisse/common/sha256.hpp"
#include "matisse/llama3/tokenizer.hpp"

using namespace matisse::load;

namespace
{
/**
 * @brief The SHA-256 of llama3.tiktoken as pinned.
 */
constexpr std::string_view llama3_digest{"82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"};

/**
 * @brief The SHA-256 of Llama 3's tokenizer.json as pinned, llama3-tokenizer.json in the vocabulary directory.
 */
constexpr std::string_view llama3_tokenizer_json_digest{
        "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"};

/**
 * @brief The SHA-256 of cl100k_base.tiktoken as pinned.
 */
constexpr std::string_view cl100k_digest{"223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"};

/**
 * @brief The SHA-256 of cl100k's tokenizer.json as pinned, cl100k-tokenizer.json in the vocabulary directory, taken
 *        from huggingface.co/Xenova/gpt-4 at commit 1d9f1f1b1fae88c0e4df1dab0a397f8de6229075.
 */
constexpr std::string_view cl100k_tokenizer_json_digest{
        "239eb2359f79c38497476671aaa835e01fb43d42743c612a8514a0dfa2ac93a2"};

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
 * @brief The known-answer fixture under check: Llama 3's pinned vocabulary, whether the current universe is Llama 3's,
 *        and the counts of its lines so far.
 */
struct Fixture_check
{
    /**
     * @brief Llama 3's pinned vocabulary.
     */
    Definition llama3{};

    /**
     * @brief Whether the current universe is Llama 3's, false before the first universe line.
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
 * @brief Returns Llama 3's pinned rank file read, after checking it against its digest.
 * @return The definition, std::nullopt when the vocabulary directory does not hold the file.
 */
std::optional<Definition> pinned_llama3()
{
    const auto path{pinned("llama3.tiktoken")};

    if (!path)
    {
        return std::nullopt;
    }

    EXPECT_EQ(matisse::common::sha256(contents(*path)), llama3_digest);

    const auto specials{matisse::llama3::specials()};

    return read_tiktoken(*path, specials);
}

/**
 * @brief Starts a universe line's universe, Llama 3's when it names llama3.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.inside = name == "llama3";

    if (check.inside)
    {
        ++check.counts.universes;
    }
}

/**
 * @brief Checks an encode line: its ids spell its text, written in hex, under Llama 3's vocabulary.
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

    EXPECT_EQ(spelled(check.llama3, ids), bytes_of(text)) << context;

    ++check.counts.encodings;
}

/**
 * @brief Checks one fixture line against Llama 3's pinned vocabulary, a comment or blank line checking nothing, a line
 *        of another universe skipped and a corpus line only counted.
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

TEST(Definition_test, The_special_tokens_of_the_llama3_rank_file_are_the_vendor_s)
{
    const auto specials{matisse::llama3::specials()};

    ASSERT_EQ(specials.size(), 256U);
    EXPECT_EQ(specials[0].spelling, "<|begin_of_text|>");
    EXPECT_EQ(specials[0].id, 128000U);
    EXPECT_EQ(specials[9].spelling, "<|eot_id|>");
    EXPECT_EQ(specials[9].id, 128009U);
    EXPECT_EQ(specials[10].spelling, "<|reserved_special_token_5|>");
    EXPECT_EQ(specials[10].id, 128010U);
    EXPECT_EQ(specials[255].spelling, "<|reserved_special_token_250|>");
    EXPECT_EQ(specials[255].id, 128255U);
}

TEST(Definition_test, The_pinned_llama3_file_reads_as_its_vocabulary_and_spells_the_published_known_answer)
{
    const auto definition{pinned_llama3()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding llama3.tiktoken";
    }

    EXPECT_EQ(definition->spellings.size(), 128000U);
    EXPECT_EQ(definition->specials.size(), 256U);
    EXPECT_TRUE(definition->whole_piece_lookup);
    EXPECT_EQ(spelled(*definition, {2028, 374, 264, 1296, 11914, 13}), "This is a test sentence.");
}

TEST(Definition_test, The_ids_of_every_known_answer_of_the_fixture_spell_its_text_under_the_pinned_vocabulary)
{
    auto llama3{pinned_llama3()};

    if (!llama3)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding llama3.tiktoken";
    }

    std::ifstream file{std::string{SOURCE_DIR} + "/libs/load/tests/data/known_answers_reference.txt"};

    ASSERT_TRUE(file.is_open());

    Fixture_check check{.llama3 = std::move(*llama3)};

    for (std::string line{}; std::getline(file, line);)
    {
        expect_fixture_line(check, line);
    }

    EXPECT_EQ(check.counts.universes, 1U);
    EXPECT_EQ(check.counts.encodings, 16U);
    EXPECT_EQ(check.counts.corpora, 3U);
}

TEST(Definition_test, The_pinned_llama3_tokenizer_json_reads_as_the_rank_file_with_hugging_face_s_merges)
{
    const auto llama3{pinned_llama3()};
    const auto path{pinned("llama3-tokenizer.json")};

    if (!llama3 || !path)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding llama3.tiktoken and llama3-tokenizer.json";
    }

    EXPECT_EQ(matisse::common::sha256(contents(*path)), llama3_tokenizer_json_digest);
    EXPECT_EQ(tokenizer_json_spelling(*path), matisse::llama3::spelling());

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{
            read_tokenizer_json(*path, matisse::llama3::spelling())};

    // Llama 3's model sets ignore_merges, so a piece that is a token is that token, as tiktoken looks it up.
    EXPECT_EQ(merge_rule, Merge_rule::pairs);
    EXPECT_TRUE(whole_piece_lookup);
    EXPECT_EQ(spellings, llama3->spellings);
    EXPECT_EQ(merges.size(), 280147U);
    ASSERT_EQ(specials.size(), llama3->specials.size());

    for (const auto& [special, vendor] : std::views::zip(specials, llama3->specials))
    {
        EXPECT_EQ(special.spelling, vendor.spelling);
        EXPECT_EQ(special.id, vendor.id);
    }
}

TEST(Definition_test, The_pinned_cl100k_tokenizer_json_reads_under_llama3_s_spelling_as_cl100k_s_vocabulary)
{
    const auto ranks{pinned("cl100k_base.tiktoken")};
    const auto path{pinned("cl100k-tokenizer.json")};

    if (!ranks || !path)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken and cl100k-tokenizer.json";
    }

    EXPECT_EQ(matisse::common::sha256(contents(*ranks)), cl100k_digest);
    EXPECT_EQ(matisse::common::sha256(contents(*path)), cl100k_tokenizer_json_digest);

    // Xenova's conversion carries cl100k's pattern as tiktoken shipped it before its possessive rewrite, Llama 3's.
    EXPECT_EQ(tokenizer_json_spelling(*path), matisse::llama3::spelling());

    const auto cl100k{read_tiktoken(*ranks, {})};

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{
            read_tokenizer_json(*path, matisse::llama3::spelling())};

    EXPECT_EQ(merge_rule, Merge_rule::pairs);
    EXPECT_FALSE(whole_piece_lookup);
    EXPECT_EQ(spellings, cl100k.spellings);
    EXPECT_EQ(merges.size(), 100000U);

    std::vector<std::pair<std::string, Token_id_t>> named{};

    for (const auto& [spelling, id, normalized] : specials)
    {
        named.emplace_back(spelling, id);
    }

    EXPECT_EQ(
            named, (std::vector<std::pair<std::string, Token_id_t>>{
                           {"<|endoftext|>", 100257},
                           {"<|fim_prefix|>", 100258},
                           {"<|fim_middle|>", 100259},
                           {"<|fim_suffix|>", 100260},
                           {"<|im_start|>", 100264},
                           {"<|im_end|>", 100265},
                           {"<|endofprompt|>", 100276}}));
}
