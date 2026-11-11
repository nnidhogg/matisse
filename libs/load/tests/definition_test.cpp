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
 * @brief GPT-2's pattern as encoder.py spells it, which a ByteLevel pre-tokenizer applies with its regex.
 */
constexpr std::string_view gpt2_spelling{
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"};

/**
 * @brief A pattern's spelling for the Split pre-tokenizers of the tests.
 */
constexpr std::string_view split_spelling{R"(\p{L}+|\p{N}{1,3}|\s+(?!\S)|\s+|[^\s\p{L}\p{N}]+)"};

/**
 * @brief The members of a small tokenizer.json, each as JSON text, which a test replaces one at a time: a GPT-2 file in
 *        miniature, its vocabulary spelled under the byte map and its special token in the vocabulary as GPT-2's is.
 */
struct Tokenizer_json_parts
{
    /**
     * @brief The normalizer.
     */
    std::string normalizer{"null"};

    /**
     * @brief The pre-tokenizer.
     */
    std::string pre_tokenizer{R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true})"};

    /**
     * @brief The post-processor.
     */
    std::string post_processor{R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false})"};

    /**
     * @brief The added tokens.
     */
    std::string added_tokens{R"([{"id": 6, "content": "<|end|>", "special": true, "single_word": false,
                                  "lstrip": false, "rstrip": false, "normalized": true}])"};

    /**
     * @brief The model.
     */
    std::string model{R"({"dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                          "end_of_word_suffix": "", "fuse_unk": false,
                          "vocab": {"a": 0, "b": 1, "Ġ": 2, "ab": 3, "Ġab": 4, "Ã©": 5, "<|end|>": 6},
                          "merges": ["a b", "Ġ ab"]})"};

    /**
     * @brief The truncation.
     */
    std::string truncation{"null"};
};

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

/**
 * @brief Returns the text of a tokenizer.json of its members.
 * @param parts The members.
 * @return The text.
 */
std::string text_of(const Tokenizer_json_parts& parts)
{
    return std::format(
            R"({{"version": "1.0", "truncation": {}, "padding": null, "added_tokens": {}, "normalizer": {},
                 "pre_tokenizer": {}, "post_processor": {}, "decoder": {{"type": "ByteLevel"}}, "model": {}}})",
            parts.truncation, parts.added_tokens, parts.normalizer, parts.pre_tokenizer, parts.post_processor,
            parts.model);
}

/**
 * @brief Returns the pre-tokenizer of a Split on a spelling followed by a ByteLevel without its regex, as JSON text.
 * @param spelling The Split's pattern, as JSON spells it.
 * @param behavior The Split's behavior.
 * @param invert The Split's invert.
 * @return The pre-tokenizer.
 */
std::string split_of(const std::string_view spelling, const std::string_view behavior, const std::string_view invert)
{
    return std::format(
            R"({{"type": "Sequence", "pretokenizers": [
                   {{"type": "Split", "pattern": {{"Regex": "{}"}}, "behavior": "{}", "invert": {}}},
                   {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}}]}})",
            spelling, behavior, invert);
}

/**
 * @brief Returns a spelling as a JSON string's characters spell it, each backslash doubled.
 * @param spelling The spelling.
 * @return The JSON characters.
 */
std::string json_escaped(const std::string_view spelling)
{
    std::string out{};

    for (const auto character : spelling)
    {
        out += character == '\\' ? std::string{R"(\\)"} : std::string{character};
    }

    return out;
}

/**
 * @brief Returns the refusal read_tokenizer_json() makes of a file, empty when it makes none.
 * @param text The file's bytes.
 * @param published The published spelling the file is read under.
 * @return The refusal's message.
 */
std::string tokenizer_json_refusal(const std::string_view text, const std::string_view published = gpt2_spelling)
{
    try
    {
        std::ignore = read_tokenizer_json(written("tokenizer.json", text), published);
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
        const auto vocab{std::format("#version: 0.2\na{}b\n", utf8_of(point))};

        const auto refusal{gpt2_refusal(R"({"a": 0, "b": 1})", vocab)};

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
        const auto ended{with_line_ends(ranks, line_end)};

        const auto read{read_tiktoken(written("ranks.tiktoken", ended), {})};

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

TEST(Definition_test, Read_tokenizer_json_reads_gpt2_s_byte_level_pre_tokenizer_under_gpt2_s_spelling)
{
    const auto path{written("tokenizer.json", text_of({}))};

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{read_tokenizer_json(path, gpt2_spelling)};

    EXPECT_EQ(merge_rule, Merge_rule::pairs);
    EXPECT_FALSE(whole_piece_lookup);
    EXPECT_EQ(spellings, (std::vector<std::string>{"a", "b", " ", "ab", " ab", "\xC3\xA9"}));
    EXPECT_EQ(
            parts_of({.merges = merges}), (std::vector<std::pair<std::string, std::string>>{{"a", "b"}, {" ", "ab"}}));
    ASSERT_EQ(specials.size(), 1U);
    EXPECT_EQ(specials[0].spelling, "<|end|>");
    EXPECT_EQ(specials[0].id, 6U);
    EXPECT_EQ(tokenizer_json_spelling(path), gpt2_spelling);
}

TEST(Definition_test, Read_tokenizer_json_reads_a_split_on_the_published_spelling_isolated_or_removed_and_inverted)
{
    for (const auto& [behavior, invert] : {std::pair{"Isolated", "false"}, std::pair{"Removed", "true"}})
    {
        Tokenizer_json_parts parts{};

        parts.pre_tokenizer = split_of(json_escaped(split_spelling), behavior, invert);
        parts.model = R"({"type": "BPE", "ignore_merges": true, "byte_fallback": false,
                          "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]})";
        parts.added_tokens = R"([{"id": 3, "content": "<|end|>", "special": true}])";

        const auto path{written("tokenizer.json", text_of(parts))};

        const auto definition{read_tokenizer_json(path, split_spelling)};

        EXPECT_TRUE(definition.whole_piece_lookup) << behavior;
        EXPECT_EQ(definition.spellings, (std::vector<std::string>{"a", "b", "ab"})) << behavior;
        EXPECT_EQ(parts_of(definition), (std::vector<std::pair<std::string, std::string>>{{"a", "b"}})) << behavior;
        ASSERT_EQ(definition.specials.size(), 1U) << behavior;
        EXPECT_EQ(definition.specials[0].id, 3U) << behavior;
        EXPECT_EQ(tokenizer_json_spelling(path), split_spelling) << behavior;
    }
}

TEST(Definition_test, Read_tokenizer_json_reads_a_byte_level_without_its_regex_alone_under_the_empty_spelling)
{
    Tokenizer_json_parts parts{};

    parts.pre_tokenizer = R"({"type": "ByteLevel", "add_prefix_space": false, "use_regex": false})";

    const auto path{written("tokenizer.json", text_of(parts))};

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{read_tokenizer_json(path, "")};

    EXPECT_EQ(spellings.size(), 6U);
    EXPECT_EQ(tokenizer_json_spelling(path), "");
}

TEST(Definition_test, Read_tokenizer_json_orders_the_special_tokens_by_id_and_takes_them_out_of_the_ordinary_ids)
{
    Tokenizer_json_parts parts{};

    // Xenova's conversion lists a special token of the vocabulary after one of a greater id, and Llama 3's special
    // tokens stand outside its vocabulary, at the ids Hugging Face gives them past it.
    parts.model = R"({"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "<|e|>": 3, "<|end|>": 4}, "merges": ["a b"]})";
    parts.added_tokens = R"([{"id": 4, "content": "<|end|>", "special": true},
                            {"id": 3, "content": "<|e|>", "special": true},
                            {"id": 5, "content": "<|a|>", "special": true},
                            {"id": 6, "content": "<|b|>", "special": true}])";

    const auto path{written("tokenizer.json", text_of(parts))};

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{read_tokenizer_json(path, gpt2_spelling)};

    EXPECT_EQ(spellings, (std::vector<std::string>{"a", "b", "ab"}));
    ASSERT_EQ(specials.size(), 4U);
    EXPECT_EQ(specials[0].spelling, "<|e|>");
    EXPECT_EQ(specials[1].spelling, "<|end|>");
    EXPECT_EQ(specials[2].spelling, "<|a|>");
    EXPECT_EQ(specials[3].spelling, "<|b|>");
    EXPECT_EQ(specials[3].id, 6U);
}

TEST(Definition_test, Read_tokenizer_json_refuses_a_special_token_whose_id_is_not_the_one_hugging_face_gives_it)
{
    // Hugging Face reads no added token's own id: one in the vocabulary gets its id there and any other the next id
    // past the vocabulary and every token given one before it.
    const std::vector<std::pair<std::string_view, std::string_view>> refusals{
            {R"([{"id": 5, "content": "<|end|>", "special": true}])",
             "added_tokens[0].id of {} is 5, where the reader takes 6, the id Hugging Face gives the token"},
            {R"([{"id": 6, "content": "<|end|>", "special": true}, {"id": 8, "content": "<|a|>", "special": true}])",
             "added_tokens[1].id of {} is 8, where the reader takes 7, the id Hugging Face gives the token"},
            {R"([{"id": 7, "content": "<|a|>", "special": true}, {"id": 6, "content": "<|end|>", "special": true},
                 {"id": 7, "content": "<|b|>", "special": true}])",
             "added_tokens[2].id of {} is 7, where the reader takes 8, the id Hugging Face gives the token"}};

    for (const auto& [added_tokens, expected] : refusals)
    {
        Tokenizer_json_parts parts{};

        parts.added_tokens = added_tokens;

        const auto refusal{tokenizer_json_refusal(text_of(parts))};

        const auto path{(scratch() / "tokenizer.json").string()};

        const auto message{std::vformat(expected, std::make_format_args(path))};

        EXPECT_TRUE(refusal.contains(message)) << added_tokens << ": " << refusal;
    }
}

TEST(Definition_test, Read_tokenizer_json_refuses_a_pre_tokenizer_the_published_spelling_does_not_name)
{
    Tokenizer_json_parts split{};

    split.pre_tokenizer = split_of(json_escaped(split_spelling), "Isolated", "false");

    const auto byte_level{tokenizer_json_refusal(text_of({}), split_spelling)};
    const auto other_split{tokenizer_json_refusal(text_of(split), R"(\p{L}+|\s+)")};
    const auto empty{tokenizer_json_refusal(text_of({}), "")};

    EXPECT_TRUE(byte_level.contains("pre_tokenizer of")) << byte_level;
    EXPECT_TRUE(
            byte_level.contains("is a ByteLevel with its regex, GPT-2's pattern, where the reader takes a Split on"))
            << byte_level;
    EXPECT_TRUE(other_split.contains("pre_tokenizer.pretokenizers[0].pattern.Regex of")) << other_split;
    EXPECT_TRUE(other_split.contains(R"(is "\\p{L}+|\\p{N}{1,3}|)")) << other_split;
    EXPECT_TRUE(other_split.contains(R"(where the reader takes a Split on "\\p{L}+|\\s+" followed by a ByteLevel)"))
            << other_split;
    EXPECT_TRUE(empty.contains("where the reader takes a ByteLevel without its regex, for the empty spelling"))
            << empty;
}

TEST(Definition_test, Read_tokenizer_json_refuses_every_other_pre_tokenizer_naming_the_member_and_what_it_holds)
{
    const std::vector<std::pair<std::string, std::string_view>> refusals{
            {"null", "pre_tokenizer of"},
            {R"({"type": "Whitespace"})", R"(is a "Whitespace" object)"},
            {R"({"type": "ByteLevel", "add_prefix_space": true})", "pre_tokenizer.add_prefix_space of"},
            {R"({"type": "ByteLevel"})", "add_prefix_space of"},
            {split_of(json_escaped(split_spelling), "MergedWithPrevious", "false"),
             R"(a Split with the behavior "MergedWithPrevious" and the invert false)"},
            {split_of(json_escaped(split_spelling), "Isolated", "true"), "the invert true, where the reader takes"},
            {split_of(json_escaped(split_spelling), "Removed", "false"), R"(the behavior "Removed" and the invert)"},
            {R"({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"String": " "},
                 "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false,
                 "use_regex": false}]})",
             "pre_tokenizer.pretokenizers[0].pattern.Regex of {} is absent"},
            {R"({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "a"},
                 "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false}]})",
             "pre_tokenizer.pretokenizers[1].use_regex of"},
            {R"({"type": "Sequence", "pretokenizers": [{"type": "ByteLevel", "add_prefix_space": false}]})",
             "pre_tokenizer.pretokenizers of"},
            {split_of("", "Isolated", "false"), R"(pattern.Regex of {} is "", where the reader takes a nonempty)"},
            {R"({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": 5},
                 "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false,
                 "use_regex": false}]})",
             "pre_tokenizer.pretokenizers[0].pattern.Regex of {} is 5, where the reader takes a string"}};

    for (const auto& [pre_tokenizer, expected] : refusals)
    {
        Tokenizer_json_parts parts{};

        parts.pre_tokenizer = pre_tokenizer;

        const auto refusal{tokenizer_json_refusal(text_of(parts), split_spelling)};

        const auto path{(scratch() / "tokenizer.json").string()};

        const auto message{std::vformat(expected, std::make_format_args(path))};

        EXPECT_TRUE(refusal.contains(message)) << pre_tokenizer << ": " << refusal;
        EXPECT_TRUE(refusal.starts_with("read_tokenizer_json: ")) << refusal;
    }
}

TEST(Definition_test, Read_tokenizer_json_refuses_an_interval_followed_by_a_plus_which_oniguruma_reads_as_repeated)
{
    constexpr std::string_view repeated{R"(\p{L}++|\p{N}{1,3}+|\s+)"};

    Tokenizer_json_parts parts{};

    parts.pre_tokenizer = split_of(json_escaped(repeated), "Isolated", "false");

    const auto refusal{tokenizer_json_refusal(text_of(parts), repeated)};

    EXPECT_TRUE(refusal.contains(R"(whose "{1,3}+" Oniguruma, Hugging Face's engine, reads as the interval repeated)"))
            << refusal;

    const auto path{written("tokenizer.json", text_of(parts))};

    EXPECT_THROW(std::ignore = tokenizer_json_spelling(path), std::invalid_argument);

    // A property's braces, an interval followed by no plus and an escaped brace are no such interval.
    for (const std::string_view spelling : {R"(\p{N}+|\s{2,}|a{3}?)", R"(\{1}+|x{,2})"})
    {
        parts.pre_tokenizer = split_of(json_escaped(spelling), "Isolated", "false");

        EXPECT_TRUE(tokenizer_json_refusal(text_of(parts), spelling).empty()) << spelling;
    }

    for (const std::string_view spelling : {"a{2}+", "a{2,}+", "a{,2}+"})
    {
        parts.pre_tokenizer = split_of(json_escaped(spelling), "Isolated", "false");

        EXPECT_TRUE(tokenizer_json_refusal(text_of(parts), spelling).contains("Oniguruma")) << spelling;
    }
}

TEST(Definition_test, Read_tokenizer_json_refuses_a_normalizer_truncation_or_post_processor_it_does_not_take)
{
    Tokenizer_json_parts normalized{};
    normalized.normalizer = R"({"type": "NFC"})";

    Tokenizer_json_parts truncated{};
    truncated.truncation = R"({"max_length": 512})";

    Tokenizer_json_parts roberta{};
    roberta.post_processor = R"({"type": "RobertaProcessing"})";

    Tokenizer_json_parts sequence{};
    sequence.post_processor =
            R"({"type": "Sequence", "processors": [{"type": "ByteLevel"}, {"type": "BertProcessing"}]})";

    EXPECT_TRUE(tokenizer_json_refusal(text_of(normalized)).contains(R"(normalizer of)")) << text_of(normalized);
    EXPECT_TRUE(
            tokenizer_json_refusal(text_of(normalized)).contains(R"(is a "NFC" object, where the reader takes null)"));
    EXPECT_TRUE(tokenizer_json_refusal(text_of(truncated)).contains("truncation of"));
    EXPECT_TRUE(tokenizer_json_refusal(text_of(roberta)).contains(R"(post_processor of)"));
    EXPECT_TRUE(tokenizer_json_refusal(text_of(sequence)).contains(R"(post_processor.processors[1] of)"));
    EXPECT_TRUE(tokenizer_json_refusal(text_of(sequence)).contains("which add tokens only under add_special_tokens"));

    // Llama 3's post-processor adds its beginning of text only under add_special_tokens.
    Tokenizer_json_parts template_processing{};
    template_processing.post_processor = R"({"type": "Sequence", "processors": [{"type": "ByteLevel"},
        {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<|end|>", "type_id": 0}}]}]})";

    EXPECT_TRUE(tokenizer_json_refusal(text_of(template_processing)).empty());
}

TEST(Definition_test, Read_tokenizer_json_refuses_a_model_that_is_not_bpe_as_the_reader_takes_it)
{
    const std::vector<std::pair<std::string_view, std::string_view>> refusals{
            {R"({"type": "WordPiece", "vocab": {}})", R"(model.type of)"},
            {R"({"type": "BPE", "dropout": 0.1, "vocab": {}, "merges": []})", "model.dropout of"},
            {R"({"type": "BPE", "continuing_subword_prefix": "##", "vocab": {}, "merges": []})",
             R"(model.continuing_subword_prefix of)"},
            {R"({"type": "BPE", "end_of_word_suffix": "</w>", "vocab": {}, "merges": []})",
             "model.end_of_word_suffix of"},
            {R"({"type": "BPE", "byte_fallback": true, "vocab": {}, "merges": []})", "model.byte_fallback of"},
            {R"({"type": "BPE", "ignore_merges": 1, "vocab": {}, "merges": []})", "model.ignore_merges of"},
            {R"({"type": "BPE", "vocab": [], "merges": []})", "model.vocab of"},
            {R"({"type": "BPE", "vocab": {}})", "model.merges of"},
            {R"([])", "model of"}};

    for (const auto& [model, expected] : refusals)
    {
        Tokenizer_json_parts parts{};

        parts.model = model;
        parts.added_tokens = "[]";

        const auto refusal{tokenizer_json_refusal(text_of(parts))};

        EXPECT_TRUE(refusal.contains(expected)) << model << ": " << refusal;
    }

    Tokenizer_json_parts word_piece{};

    word_piece.model = R"({"type": "WordPiece", "vocab": {}})";

    EXPECT_TRUE(
            tokenizer_json_refusal(text_of(word_piece)).contains(R"(is "WordPiece", where the reader takes "BPE")"));
}

TEST(Definition_test, Read_tokenizer_json_refuses_an_added_token_that_is_not_special_or_strips_or_matches_words)
{
    const std::vector<std::pair<std::string_view, std::string_view>> refusals{
            {R"([{"id": 5, "content": "<|end|>", "special": false}])", R"(added_tokens[0] of)"},
            {R"([{"id": 5, "content": "<|end|>"}])", R"(is "<|end|>", which is not special)"},
            {R"([{"id": 5, "content": "<|end|>", "special": true, "lstrip": true}])", "added_tokens[0].lstrip of"},
            {R"([{"id": 5, "content": "<|end|>", "special": true, "rstrip": true}])", "added_tokens[0].rstrip of"},
            {R"([{"id": 5, "content": "<|end|>", "special": true, "single_word": true}])",
             "added_tokens[0].single_word of"},
            {R"([{"id": 5, "content": "", "special": true}])", "added_tokens[0].content of"},
            {R"([{"id": -5, "content": "<|end|>", "special": true}])", "added_tokens[0].id of"},
            {R"([{"content": "<|end|>", "special": true}])", "added_tokens[0].id of"},
            {R"([{"id": 6, "special": true}])", "added_tokens[0].content of"},
            {R"({})", "added_tokens of"}};

    for (const auto& [added_tokens, expected] : refusals)
    {
        Tokenizer_json_parts parts{};

        parts.added_tokens = added_tokens;

        const auto refusal{tokenizer_json_refusal(text_of(parts))};

        EXPECT_TRUE(refusal.contains(expected)) << added_tokens << ": " << refusal;
    }
}

TEST(Definition_test, Read_tokenizer_json_refuses_a_vocabulary_or_merges_it_cannot_read_exactly)
{
    const auto model_of{[](const std::string_view vocab, const std::string_view merges) {
        Tokenizer_json_parts parts{};

        parts.added_tokens = "[]";
        parts.model = std::format(R"({{"type": "BPE", "vocab": {}, "merges": {}}})", vocab, merges);

        return text_of(parts);
    }};

    const std::vector<std::pair<std::string, std::string_view>> refusals{
            {model_of(R"({"a": 0, "b": 2})", "[]"), "are not dense from zero: 2 tokens, and the id 2"},
            {model_of(R"({"a": 0, "b": 0})", "[]"), "are not dense from zero: 2 tokens, and the id 0"},
            {model_of(R"({"a": 0, "": 1})", "[]"), "model.vocab of"},
            {model_of(R"({"a": 0, "a": 1})", "[]"), "the token at id 1 of"},
            {model_of(R"({"a b": 0})", "[]"), R"(model.vocab["a b"] of)"},
            {model_of(R"({"a": 0, "ń": 1})", "[]"), "holds U+0144, which GPT-2's byte map does not carry"},
            {model_of(R"({"a": 1.0})", "[]"), R"(model.vocab["a"] of)"},
            {model_of(R"({"a": 4294967296})", "[]"), "where the reader takes an id from 0 to 4294967295"},
            {model_of(R"({"a": 0, "b": 1})", R"(["ab"])"), "model.merges[0] of"},
            {model_of(R"({"a": 0, "b": 1})", R"(["a "])"), "two nonempty parts separated by a space"},
            {model_of(R"({"a": 0, "b": 1})", R"([["a", "b", "c"]])"), "model.merges[0] of"},
            {model_of(R"({"a": 0, "b": 1})", R"(["a b", ["a", "b"]])"), R"(model.merges[1] of )"},
            {model_of(R"({"a": 0, "b": 1})", R"(["a b c"])"), "holds U+0020, which GPT-2's byte map does not carry"}};

    for (const auto& [text, expected] : refusals)
    {
        const auto refusal{tokenizer_json_refusal(text)};

        EXPECT_TRUE(refusal.contains(expected)) << text << ": " << refusal;
    }

    const auto repeated{tokenizer_json_refusal(model_of(R"({"a": 0, "b": 1})", R"(["a b", ["a", "b"]])"))};

    EXPECT_TRUE(repeated.contains(R"(the merge ("a", "b"), repeats the merge at rank 0)")) << repeated;
}

TEST(Definition_test, Read_tokenizer_json_reads_a_merge_whose_part_a_later_merge_forms_as_hugging_face_does)
{
    Tokenizer_json_parts parts{};

    parts.added_tokens = "[]";
    parts.model = R"({"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "aba": 3}, "merges": ["ab a", "a b"]})";

    const auto path{written("tokenizer.json", text_of(parts))};

    const auto [merge_rule, whole_piece_lookup, spellings, merges, specials]{read_tokenizer_json(path, gpt2_spelling)};

    EXPECT_EQ(merges.size(), 2U);
}

TEST(Definition_test, Read_tokenizer_json_refuses_a_file_that_is_not_one_json_object)
{
    const auto not_json{tokenizer_json_refusal(R"({"model": })")};
    const auto array{tokenizer_json_refusal("[]")};

    EXPECT_TRUE(not_json.contains("is not JSON: 1:")) << not_json;
    EXPECT_TRUE(array.contains("is an array of 0, where the reader takes an object")) << array;

    const auto missing{scratch() / "missing"};

    EXPECT_THROW(std::ignore = read_tokenizer_json(missing, gpt2_spelling), std::runtime_error);
    EXPECT_THROW(std::ignore = tokenizer_json_spelling(missing), std::runtime_error);
}
