#include "matisse/load/definition.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <utility>
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
 * @brief Returns a text with every line feed replaced, for writing a file's line ends another way.
 * @param text The text, its lines ended by `\n`.
 * @param line_end What ends each line instead.
 * @return The text with its line ends replaced.
 */
std::string with_line_ends(const std::string_view text, const std::string_view line_end)
{
    std::string out{};

    for (const auto byte : text)
    {
        out += byte == '\n' ? std::string{line_end} : std::string{byte};
    }

    return out;
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
 * @brief Returns the merges of a definition as pairs of their parts, in rank order.
 * @param definition The definition.
 * @return Every merge's left and right part.
 */
std::vector<std::pair<std::string, std::string>> parts_of(const Definition& definition)
{
    std::vector<std::pair<std::string, std::string>> parts{};

    for (const auto& [left, right] : definition.merges)
    {
        parts.emplace_back(left, right);
    }

    return parts;
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

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{read_gpt2(encoder, vocab)};

    EXPECT_EQ(merge_rule, Merge_rule::pairs);
    EXPECT_FALSE(whole_piece_lookup);
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

TEST(Definition_test, Read_gpt2_reads_cr_and_crlf_line_ends_as_line_feeds_as_python_s_text_mode_does)
{
    // The text after the last line end is dropped in every form, as encoder.py drops it after the last newline.
    constexpr std::string_view encoder{"{\n \"a\": 0,\n \"b\": 1\n}\n"};
    constexpr std::string_view vocab{"#version: 0.2\nĠ a\nÃ ©\na b\nb a"};

    const auto lf{read_gpt2(written("encoder.json", encoder), written("vocab.bpe", vocab))};

    ASSERT_EQ(lf.merges.size(), 3U);

    for (const std::string_view line_end : {"\r", "\r\n"})
    {
        const auto encoder_path{written("encoder.json", with_line_ends(encoder, line_end))};
        const auto vocab_path{written("vocab.bpe", with_line_ends(vocab, line_end))};

        const auto read{read_gpt2(encoder_path, vocab_path)};

        EXPECT_EQ(read.spellings, lf.spellings) << line_end.size();
        EXPECT_EQ(parts_of(read), parts_of(lf)) << line_end.size();
    }
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

TEST(Definition_test, Read_gpt2_refuses_bytes_that_are_not_utf_8_before_undoing_the_byte_map)
{
    const auto overlong{gpt2_refusal("{\"\xC1\xA1\":0}", "#version: 0.2\n")};
    const auto merged{gpt2_refusal(R"({"a": 0})", "#version: 0.2\na \xC5\n")};

    EXPECT_TRUE(overlong.contains("at byte 2 holds bytes that are not UTF-8")) << overlong;
    EXPECT_TRUE(merged.contains("at byte 16 holds bytes that are not UTF-8")) << merged;
}

TEST(Definition_test, Read_gpt2_refuses_a_merge_file_not_utf_8_in_its_version_line_or_tail_as_python_s_text_mode_does)
{
    const auto header{gpt2_refusal(R"({"a": 0, "b": 1})", "#version: 0.2 \xFF\na b\n")};
    const auto tail{gpt2_refusal(R"({"a": 0, "b": 1})", "#version: 0.2\na b\ntail \xC5")};

    EXPECT_TRUE(header.contains("at byte 14 holds bytes that are not UTF-8")) << header;
    EXPECT_TRUE(tail.contains("at byte 23 holds bytes that are not UTF-8")) << tail;
    EXPECT_TRUE(gpt2_refusal(R"({"a": 0, "b": 1})", "#version: 0.2\na b\ntail").empty());
}

TEST(Definition_test, Read_gpt2_refuses_a_merge_line_that_is_not_two_parts)
{
    const auto refusal{gpt2_refusal(R"({"a": 0})", "#version: 0.2\na a\na\n")};

    EXPECT_TRUE(refusal.contains("line 3 of")) << refusal;
    EXPECT_TRUE(refusal.contains("is not a merge, two parts separated by white space")) << refusal;
}

TEST(Definition_test, Read_gpt2_splits_a_merge_line_at_the_white_space_python_s_str_split_splits_at)
{
    // The code points c from 0 to 0x10FFFF for which ("a" + chr(c) + "b").split() is ["a", "b"] under Python 3.12.3,
    // which are those for which chr(c).isspace() holds and those CPython 3.12's _PyUnicode_IsWhitespace() lists.
    constexpr std::array<char32_t, 29> white_space{0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x001C, 0x001D, 0x001E,
                                                   0x001F, 0x0020, 0x0085, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002,
                                                   0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
                                                   0x2028, 0x2029, 0x202F, 0x205F, 0x3000};

    const auto encoder{written("encoder.json", R"({"a": 0, "b": 1})")};

    for (const auto point : white_space)
    {
        // A line feed or a carriage return ends the line first, as Python's text mode and encoder.py read it.
        if (point == U'\n' || point == U'\r')
        {
            continue;
        }

        const auto vocab{written("vocab.bpe", std::format("#version: 0.2\na{}b\n", utf8_of(point)))};

        const auto definition{read_gpt2(encoder, vocab)};

        EXPECT_EQ(parts_of(definition), (std::vector<std::pair<std::string, std::string>>{{"a", "b"}}))
                << static_cast<std::uint32_t>(point);
    }

    // A run of separators is one, and those before the first part and after the last are dropped.
    const auto runs{read_gpt2(encoder, written("vocab.bpe", "#version: 0.2\n\u3000a\u2028\x1C\tb\u0085\n"))};

    EXPECT_EQ(parts_of(runs), (std::vector<std::pair<std::string, std::string>>{{"a", "b"}}));
}

TEST(Definition_test, Read_gpt2_reads_a_character_python_s_str_split_does_not_split_at_into_the_part)
{
    // Code points beside the white space that str.split() keeps in the part: controls, U+180E, a space before Unicode
    // 6.3, the zero-width space and joiner, the bidirectional embeddings, the word joiner, the ideographic comma, the
    // byte order mark and the last code point. The byte map carries none of them, so the part holding one is refused.
    constexpr std::array<char32_t, 23> kept{0x0000, 0x0008, 0x000E, 0x001B, 0x0084, 0x0086, 0x009F,  0x167F,
                                            0x1681, 0x180E, 0x1FFF, 0x200B, 0x200D, 0x2027, 0x202A,  0x202E,
                                            0x2030, 0x205E, 0x2060, 0x2FFF, 0x3001, 0xFEFF, 0x10FFFF};

    for (const auto point : kept)
    {
        const auto refusal{gpt2_refusal(R"({"a": 0, "b": 1})", std::format("#version: 0.2\na{}b\n", utf8_of(point)))};

        const auto expected{std::format(
                "line 2 of {} holds U+{:04X}, which GPT-2's byte map does not carry",
                (scratch() / "vocab.bpe").string(), static_cast<std::uint32_t>(point))};

        EXPECT_TRUE(refusal.contains(expected)) << static_cast<std::uint32_t>(point) << ": " << refusal;
    }

    // A character the byte map carries is read into the part, so the line is one part.
    for (const std::string_view line : {"a!b", "a\u00A1b", "a\u0100b", "a\u0120b"})
    {
        const auto refusal{gpt2_refusal(R"({"a": 0, "b": 1})", std::format("#version: 0.2\n{}\n", line))};

        EXPECT_TRUE(refusal.contains("is not a merge, two parts separated by white space")) << line << ": " << refusal;
    }
}

TEST(Definition_test, Read_gpt2_refuses_a_merge_line_of_white_space_alone_or_of_three_parts_split_outside_ascii)
{
    for (const std::string_view line : {"", "\u3000", "\u2028 \u0085", "a\u3000b\u2029a", "x\x1Cy\x1Fz"})
    {
        const auto refusal{gpt2_refusal(R"({"a": 0, "b": 1})", std::format("#version: 0.2\n{}\n", line))};

        EXPECT_TRUE(refusal.contains("line 2 of")) << line << ": " << refusal;
        EXPECT_TRUE(refusal.contains("is not a merge, two parts separated by white space")) << line << ": " << refusal;
    }
}

TEST(Definition_test, Read_gpt2_refuses_a_merge_whose_part_a_later_merge_forms_since_the_two_pair_rules_differ_there)
{
    // Under (ab, a) then (a, b), encoder.py's pass gives ab ab on abab and one join at a time aba b.
    const auto later{gpt2_refusal(R"({"a": 0, "b": 1})", "#version: 0.2\nab a\na b\n")};
    const auto earlier{gpt2_refusal(R"({"a": 0, "b": 1})", "#version: 0.2\na b\nab a\n")};

    EXPECT_TRUE(later.contains(R"(the merge ("ab", "a") at rank 0, line 2 of)")) << later;
    EXPECT_TRUE(later.contains(R"(has the part "ab", which the merge at rank 1 forms)")) << later;
    EXPECT_TRUE(earlier.empty()) << earlier;
}

TEST(Definition_test, Read_gpt2_refuses_a_part_whose_latest_forming_merge_comes_after_its_use)
{
    // abc is formed twice, by (a, bc) at rank 2 and (ab, c) at rank 4, and (abc, d) at rank 3 joins it in between.
    const auto refusal{gpt2_refusal(R"({"a": 0, "b": 1})", "#version: 0.2\na b\nb c\na bc\nabc d\nab c\n")};

    EXPECT_TRUE(refusal.contains(R"(the merge ("abc", "d") at rank 3, line 5 of)")) << refusal;
    EXPECT_TRUE(refusal.contains(R"(has the part "abc", which the merge at rank 4 forms)")) << refusal;
}

TEST(Definition_test, Read_gpt2_refuses_a_pair_listed_twice)
{
    const auto refusal{gpt2_refusal(R"({"a": 0, "b": 1})", "#version: 0.2\na b\nb a\na b\n")};

    EXPECT_TRUE(refusal.contains(R"(the merge ("a", "b") at rank 2, line 4 of)")) << refusal;
    EXPECT_TRUE(refusal.contains("repeats the merge at rank 0")) << refusal;
}

TEST(Definition_test, Read_gpt2_refuses_a_spelling_listed_twice)
{
    const auto refusal{gpt2_refusal(R"({"a": 0, "b": 1, "a": 2})", "#version: 0.2\n")};

    EXPECT_TRUE(refusal.contains("the token at id 2 of")) << refusal;
    EXPECT_TRUE(refusal.contains("repeats the spelling of the token at id 0")) << refusal;
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

TEST(Definition_test, Read_gpt2_refuses_a_leading_zero_and_white_space_outside_json_s_four_as_python_s_json_does)
{
    const auto zero{gpt2_refusal(R"({"a":00})", "#version: 0.2\n")};
    const auto vertical_tab{gpt2_refusal("{\v\"a\":0}", "#version: 0.2\n")};

    EXPECT_TRUE(zero.contains("has no closing brace at byte 6")) << zero;
    EXPECT_TRUE(vertical_tab.contains("has no quoted spelling at byte 1")) << vertical_tab;
}

TEST(Definition_test, Read_gpt2_reads_an_id_spelled_minus_zero_as_zero_as_python_s_json_does)
{
    const auto vocab{written("vocab.bpe", "#version: 0.2\n")};

    const auto first{read_gpt2(written("encoder.json", R"({"a": -0, "b": 1})"), vocab)};
    const auto last{read_gpt2(written("encoder.json", R"({"a": 1, "b":-0})"), vocab)};

    EXPECT_EQ(first.spellings, (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(last.spellings, (std::vector<std::string>{"b", "a"}));

    // -0 is the id 0, so beside a 0 it repeats it.
    const auto repeated{gpt2_refusal(R"({"a": 0, "b": -0})", "#version: 0.2\n")};

    EXPECT_TRUE(repeated.contains("are not dense from zero: 2 tokens, and the id 0")) << repeated;
}

TEST(Definition_test, Read_gpt2_refuses_an_id_python_s_json_reads_as_negative_or_as_no_integer)
{
    // Python's json reads -1 as a negative integer, 1.0, 1e0 and -0.0 as floats, and refuses +0, - 0, 01 and -01; the
    // number's syntax stops before a fraction, an exponent or a second digit after a zero.
    const std::vector<std::pair<std::string_view, std::string_view>> refusals{
            {"-1", "has no id at byte 6"},
            {"+0", "has no id at byte 6"},
            {"- 0", "has no id at byte 6"},
            {"-", "has no id at byte 6"},
            {"01", "has no closing brace at byte 7"},
            {"-01", "has no closing brace at byte 8"},
            {"1.0", "has no closing brace at byte 7"},
            {"1e0", "has no closing brace at byte 7"},
            {"-0.0", "has no closing brace at byte 8"},
            {"-0e0", "has no closing brace at byte 8"}};

    for (const auto& [number, expected] : refusals)
    {
        const auto refusal{gpt2_refusal(std::format(R"({{"a": {}}})", number), "#version: 0.2\n")};

        EXPECT_TRUE(refusal.contains(expected)) << number << ": " << refusal;
    }
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

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{read_tiktoken(path, given)};

    EXPECT_EQ(merge_rule, Merge_rule::ranks);
    EXPECT_TRUE(whole_piece_lookup);
    EXPECT_EQ(spellings, (std::vector<std::string>{"!", "\"", " a"}));
    EXPECT_TRUE(merges.empty());
    ASSERT_EQ(specials.size(), 2U);
    EXPECT_EQ(specials[0].spelling, "<|endoftext|>");
    EXPECT_EQ(specials[0].id, 3U);
    EXPECT_EQ(specials[1].spelling, "<|end|>");
    EXPECT_EQ(specials[1].id, 4U);
}

TEST(Definition_test, Read_tiktoken_splits_lines_at_cr_and_crlf_as_python_s_bytes_splitlines_does)
{
    constexpr std::string_view ranks{"Ig== 1\nIQ== 0\n\nIGE= 2\n"};

    const auto lf{read_tiktoken(written("ranks.tiktoken", ranks), {})};

    ASSERT_EQ(lf.spellings.size(), 3U);

    for (const std::string_view line_end : {"\r", "\r\n"})
    {
        const auto read{read_tiktoken(written("ranks.tiktoken", with_line_ends(ranks, line_end)), {})};

        EXPECT_EQ(read.spellings, lf.spellings) << line_end.size();

        const auto refusal{tiktoken_refusal(with_line_ends("Ig== 0\nIQ== 1 2\n", line_end))};

        EXPECT_TRUE(refusal.contains("line 2 of")) << line_end.size() << ": " << refusal;
    }
}

TEST(Definition_test, Read_tiktoken_refuses_a_line_that_is_not_a_base64_token_and_a_decimal_rank)
{
    for (const std::string_view line :
         {"IQ=", "I*== 1", "IQ==", "IQ== -1", "IQ== 1x", "IQ== 1 2", "==== 1", " ", "\t\v\f"})
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

TEST(Definition_test, Read_tiktoken_refuses_a_spelling_listed_twice)
{
    const auto refusal{tiktoken_refusal("IQ== 0\nIg== 1\nIQ== 2\n")};

    EXPECT_TRUE(refusal.contains("the token at rank 2 of")) << refusal;
    EXPECT_TRUE(refusal.contains("repeats the spelling of the token at rank 0")) << refusal;
}
