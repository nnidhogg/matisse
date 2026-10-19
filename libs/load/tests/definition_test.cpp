#include "matisse/load/definition.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace matisse::load;

namespace
{
/**
 * @brief Returns the directory a test writes its files into, one per test and process.
 * @return The directory, created.
 */
std::filesystem::path scratch()
{
    const std::string_view test{testing::UnitTest::GetInstance()->current_test_info()->name()};

    const auto name{std::format("matisse-load-{}-{}", test, ::getpid())};

    const auto dir{std::filesystem::temp_directory_path() / name};

    std::filesystem::create_directories(dir);

    return dir;
}

/**
 * @brief Writes a file into the test's directory.
 * @param name The file's name.
 * @param text The file's bytes.
 * @return The file's path.
 */
std::filesystem::path written(const std::string_view name, const std::string_view text)
{
    const auto path{scratch() / name};

    std::ofstream{path, std::ios::binary} << text;

    return path;
}

/**
 * @brief Returns the refusal read_gpt2() makes of an encoder.json and a vocab.bpe, empty when it makes none.
 * @param encoder The bytes of encoder.json.
 * @param vocab The bytes of vocab.bpe.
 * @return The refusal's message.
 */
std::string gpt2_refusal(const std::string_view encoder, const std::string_view vocab)
{
    try
    {
        std::ignore = read_gpt2(written("encoder.json", encoder), written("vocab.bpe", vocab));
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

/**
 * @brief Returns the refusal read_tiktoken() makes of a rank file, empty when it makes none.
 * @param text The bytes of the rank file.
 * @return The refusal's message.
 */
std::string tiktoken_refusal(const std::string_view text)
{
    try
    {
        std::ignore = read_tiktoken(written("ranks.tiktoken", text), {});
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

} // namespace

TEST(Definition_test, Read_gpt2_undoes_the_byte_map_in_the_spellings_and_the_merges)
{
    const auto encoder{written("encoder.json", R"({"a": 0, "Ġ": 1, "Ġa": 2, "\"": 3, "Ã©": 4})")};
    const auto vocab{written("vocab.bpe", "#version: 0.2\nĠ a\nÃ ©\n")};

    const auto [merge_rule, spellings, merges, specials]{read_gpt2(encoder, vocab)};

    EXPECT_EQ(merge_rule, Merge_rule::pairs);
    EXPECT_EQ(spellings, (std::vector<std::string>{"a", " ", " a", "\"", "\xC3\xA9"}));
    ASSERT_EQ(merges.size(), 2U);
    EXPECT_EQ(merges[0].left, " ");
    EXPECT_EQ(merges[0].right, "a");
    EXPECT_EQ(merges[1].left, "\xC3");
    EXPECT_EQ(merges[1].right, "\xA9");
    EXPECT_TRUE(specials.empty());
}

TEST(Definition_test, Read_gpt2_drops_the_text_after_the_last_newline_of_the_merges_as_encoder_py_does)
{
    const auto encoder{written("encoder.json", R"({"a": 0, "b": 1})")};
    const auto vocab{written("vocab.bpe", "#version: 0.2\na b\nb a")};

    const auto definition{read_gpt2(encoder, vocab)};

    ASSERT_EQ(definition.merges.size(), 1U);
    EXPECT_EQ(definition.merges[0].left, "a");
}

TEST(Definition_test, Read_gpt2_refuses_ids_that_are_not_dense_from_zero)
{
    const auto beyond{gpt2_refusal(R"({"a": 0, "b": 2})", "#version: 0.2\n")};
    const auto repeated{gpt2_refusal(R"({"a": 0, "b": 0})", "#version: 0.2\n")};

    EXPECT_TRUE(beyond.contains("are not dense from zero: 2 tokens, and the id 2")) << beyond;
    EXPECT_TRUE(repeated.contains("are not dense from zero: 2 tokens, and the id 0")) << repeated;
}

TEST(Definition_test, Read_gpt2_refuses_an_empty_token)
{
    const auto refusal{gpt2_refusal(R"({"a": 0, "": 1})", "#version: 0.2\n")};

    EXPECT_TRUE(refusal.contains("holds an empty token at byte 9")) << refusal;
}

TEST(Definition_test, Read_gpt2_refuses_a_character_the_byte_map_does_not_carry)
{
    const auto escaped{gpt2_refusal(R"({"ń": 0})", "#version: 0.2\n")};
    const auto raw{gpt2_refusal(R"({"a b": 0})", "#version: 0.2\n")};
    const auto merged{gpt2_refusal(R"({"a": 0})", "#version: 0.2\na \xC5\x84\n")};

    EXPECT_TRUE(escaped.contains("at byte 2 holds U+0144, which GPT-2's byte map does not carry")) << escaped;
    EXPECT_TRUE(raw.contains("at byte 3 holds U+0020")) << raw;
    EXPECT_TRUE(merged.contains("line 2 of")) << merged;
    EXPECT_TRUE(merged.contains("holds U+0144")) << merged;
}

TEST(Definition_test, Read_gpt2_refuses_a_merge_line_that_is_not_two_parts)
{
    const auto refusal{gpt2_refusal(R"({"a": 0})", "#version: 0.2\na a\na\n")};

    EXPECT_TRUE(refusal.contains("line 3 of")) << refusal;
    EXPECT_TRUE(refusal.contains("is not a merge, two parts separated by white space")) << refusal;
}

TEST(Definition_test, Read_gpt2_refuses_an_encoder_that_is_not_one_flat_object_of_spellings_and_ids)
{
    const auto colon{gpt2_refusal(R"({"a" 0})", "#version: 0.2\n")};
    const auto id{gpt2_refusal(R"({"a": "0"})", "#version: 0.2\n")};
    const auto escape{gpt2_refusal(R"({"\q": 0})", "#version: 0.2\n")};
    const auto trailing{gpt2_refusal(R"({"a": 0} {})", "#version: 0.2\n")};

    EXPECT_TRUE(colon.contains("has no colon at byte 5")) << colon;
    EXPECT_TRUE(id.contains("has no id at byte 6")) << id;
    EXPECT_TRUE(escape.contains("has no JSON escape at byte 3")) << escape;
    EXPECT_TRUE(trailing.contains("has no end of the object at byte 9")) << trailing;
}

TEST(Definition_test, Read_gpt2_reads_an_empty_object_and_a_version_line_alone_as_an_empty_definition)
{
    const auto definition{read_gpt2(written("encoder.json", " { } "), written("vocab.bpe", "#version: 0.2\n"))};

    EXPECT_TRUE(definition.spellings.empty());
    EXPECT_TRUE(definition.merges.empty());
}

TEST(Definition_test, The_readers_refuse_a_file_they_cannot_read)
{
    const auto missing{scratch() / "missing"};

    EXPECT_THROW(std::ignore = read_gpt2(missing, missing), std::runtime_error);
    EXPECT_THROW(std::ignore = read_tiktoken(missing, {}), std::runtime_error);

    try
    {
        std::ignore = read_tiktoken(missing, {});
    }
    catch (const std::runtime_error& refusal)
    {
        EXPECT_EQ(std::string_view{refusal.what()}, std::format("cannot read {}", missing.string()));
    }
}

TEST(Definition_test, Read_tiktoken_reads_the_ranks_as_ids_and_attaches_the_special_tokens_it_is_given)
{
    const auto path{written("ranks.tiktoken", "Ig== 1\nIQ== 0\n\nIGE= 2\r\n")};

    const std::vector<Special_token> given{{.spelling = "<|endoftext|>", .id = 3}, {.spelling = "<|end|>", .id = 4}};

    const auto [merge_rule, spellings, merges, specials]{read_tiktoken(path, given)};

    EXPECT_EQ(merge_rule, Merge_rule::ranks);
    EXPECT_EQ(spellings, (std::vector<std::string>{"!", "\"", " a"}));
    EXPECT_TRUE(merges.empty());
    ASSERT_EQ(specials.size(), 2U);
    EXPECT_EQ(specials[0].spelling, "<|endoftext|>");
    EXPECT_EQ(specials[0].id, 3U);
    EXPECT_EQ(specials[1].spelling, "<|end|>");
    EXPECT_EQ(specials[1].id, 4U);
}

TEST(Definition_test, Read_tiktoken_refuses_a_line_that_is_not_a_base64_token_and_a_decimal_rank)
{
    for (const std::string_view line : {"IQ=", "I*== 1", "IQ==", "IQ== -1", "IQ== 1x", "IQ== 1 2", "==== 1"})
    {
        const auto refusal{tiktoken_refusal(std::format("Ig== 0\n{}\n", line))};

        EXPECT_TRUE(refusal.contains("line 2 of")) << line << ": " << refusal;
        EXPECT_TRUE(refusal.contains("is not a base64 token and a decimal rank")) << line << ": " << refusal;
    }
}

TEST(Definition_test, Read_tiktoken_refuses_ranks_that_are_not_dense_from_zero)
{
    const auto beyond{tiktoken_refusal("IQ== 0\nIg== 2\n")};
    const auto repeated{tiktoken_refusal("IQ== 0\nIg== 0\n")};

    EXPECT_TRUE(beyond.contains("are not dense from zero: 2 tokens, and the rank 2")) << beyond;
    EXPECT_TRUE(repeated.contains("are not dense from zero: 2 tokens, and the rank 0")) << repeated;
}
