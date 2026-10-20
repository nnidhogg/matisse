#include "matisse/encode/encoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <munch/regex/utf8.hpp>

#include "matisse/encode/merger.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::encode;
using matisse::load::Definition;
using matisse::load::Merge_rule;
using matisse::load::Special_token;
using matisse::load::Token_id_t;
using matisse::pretok::Alphabet;
using matisse::pretok::Alternative;
using matisse::pretok::Code_point_set;
using matisse::pretok::Pattern;
using munch::regex::utf8::Code_point_range;

namespace
{
/**
 * @brief The code point of the space.
 */
constexpr std::array<Code_point_range, 1> space_points{{{.first = U' ', .last = U' '}}};

/**
 * @brief The letters of the test pattern: Latin, Latin-1 and Latin Extended, Greek, Cyrillic, kana and CJK ideographs.
 */
constexpr std::array<Code_point_range, 7> letter_points{{
        {.first = U'A', .last = U'Z'},
        {.first = U'a', .last = U'z'},
        {.first = U'À', .last = U'ɏ'},
        {.first = U'Ͱ', .last = U'Ͽ'},
        {.first = U'Ѐ', .last = U'ӿ'},
        {.first = U'぀', .last = U'ヿ'},
        {.first = U'一', .last = U'鿿'},
}};

/**
 * @brief The sets the test alphabet reads, the space and the letters.
 */
constexpr std::array<Code_point_set, 2> sets{{
        {.symbol = ' ', .ranges = space_points},
        {.symbol = 'a', .ranges = letter_points},
}};

/**
 * @brief An alphabet of three classes: the space, the letters, and everything else.
 */
constexpr Alphabet alphabet{.symbols = " !a", .sets = sets, .other = '!'};

/**
 * @brief A pattern in GPT-2's shape over the three classes: a letter run or a run of everything else, each after an
 *        optional space, and a run of spaces.
 */
constexpr std::array<Alternative, 3> alternatives{{
        {.language = R"(\x20?a+)", .gate = " !a"},
        {.language = R"(\x20?[!]+)", .gate = " !a"},
        {.language = R"(\x20+)", .gate = " !a"},
}};

/**
 * @brief The test pattern.
 */
constexpr Pattern pattern{.alphabet = alphabet, .alternatives = alternatives};

/**
 * @brief Returns a definition under the rank rule: the single bytes, then every string of two and three letters over a,
 *        b and c and every space followed by one of them, ranked in a seeded random order, then the given tokens, with
 *        two special tokens after them.
 * @param tokens Further multi-byte tokens, ranked last.
 * @return The definition.
 */
Definition test_definition(const std::vector<std::string>& tokens = {})
{
    constexpr std::string_view letters{"abc"};

    std::vector<std::string> merged{};

    for (const auto first : letters)
    {
        merged.push_back(std::string{' '} + first);

        for (const auto second : letters)
        {
            merged.push_back(std::string{first} + second);

            for (const auto third : letters)
            {
                merged.push_back(std::string{first} + second + third);
            }
        }
    }

    std::mt19937 random{11};

    std::ranges::shuffle(merged, random);

    std::vector<std::string> spellings{};

    for (unsigned value{0}; value < 256; ++value)
    {
        spellings.emplace_back(1, static_cast<char>(value));
    }

    spellings.insert(spellings.end(), merged.begin(), merged.end());

    spellings.insert(spellings.end(), tokens.begin(), tokens.end());

    const auto first_special{static_cast<Token_id_t>(spellings.size())};

    std::vector<Special_token> specials{
            {.spelling = "<|end|>", .id = first_special},
            {.spelling = "<|end|><|end|>", .id = first_special + 1}};

    return {.merge_rule = Merge_rule::ranks,
            .whole_piece_lookup = true,
            .spellings = std::move(spellings),
            .merges = {},
            .specials = std::move(specials)};
}

/**
 * @brief Returns the encoder of a definition under the test pattern.
 * @param definition The definition.
 * @param allowed The special tokens allowed.
 * @return The encoder.
 */
Encoder encoder_of(Definition definition, const std::vector<std::string>& allowed = {})
{
    const auto report{matisse::validate::validate(pattern)};

    return {std::move(definition), report, matisse::validate::Inventory::factorization, pattern, allowed};
}

/**
 * @brief Returns the ids of a text with no special token in it, each piece the pattern segments looked up whole or
 *        merged, with no cache.
 * @param definition The definition.
 * @param text The text.
 * @return The ids.
 */
std::vector<Token_id_t> uncached_ids(const Definition& definition, const std::string_view text)
{
    const Merger merger{definition};

    const matisse::pretok::Segmenter segmenter{pattern};

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
 * @brief Returns the refusal an Encoder makes of an allowed spelling, empty when it makes none.
 * @param allowed The spellings allowed.
 * @return The refusal's message.
 */
std::string encoder_refusal(const std::vector<std::string>& allowed)
{
    try
    {
        std::ignore = encoder_of(test_definition(), allowed);
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

/**
 * @brief Returns a seeded random text of the shapes the pattern and the special tokens meet: letter runs over a, b and
 *        c, runs of spaces, tabs and newlines, punctuation, multilingual letters, emoji, digits, and the spellings of
 *        the special tokens, some of them broken.
 * @param random The random sequence.
 * @param fragments The number of fragments joined.
 * @return The text, well-formed UTF-8.
 */
std::string random_text(std::mt19937& random, const std::size_t fragments)
{
    constexpr std::array<std::string_view, 24> shapes{
            "a",      "b",         "c",     "abc",        " ab",     " cab",           "  ",    "   ",
            "\t\t",   "\n\n",      " \n\t", "!",          "?!",      "12345",          "héllo", " wörld",
            "日本語", " Ελληνικά", "рус",   "\U0001F642", "<|end|>", "<|end|><|end|>", "<|en",  "|>"};

    std::uniform_int_distribution<std::size_t> shape{0, shapes.size() - 1};

    std::string text{};

    for (std::size_t count{0}; count < fragments; ++count)
    {
        text += shapes[shape(random)];
    }

    return text;
}

} // namespace

TEST(Encoder_test, Encode_looks_up_or_merges_every_piece_the_pattern_segments_and_decode_spells_the_text_back)
{
    const auto definition{test_definition()};

    const auto encoder{encoder_of(definition)};

    for (const std::string_view text : {"abcab cab", "  ab!!  c", "a\tb\n\nc", "héllo wörld"})
    {
        const auto ids{encoder.encode(text)};

        EXPECT_EQ(ids, uncached_ids(definition, text)) << text;
        EXPECT_EQ(encoder.decode(ids), text);
    }
}

TEST(Encoder_test, The_empty_text_has_no_ids_and_no_ids_spell_the_empty_text)
{
    const auto encoder{encoder_of(test_definition())};

    EXPECT_TRUE(encoder.encode("").empty());
    EXPECT_TRUE(encoder.decode({}).empty());
}

TEST(Encoder_test, An_allowed_special_token_is_cut_out_with_its_id_and_one_not_allowed_is_ordinary_text)
{
    const auto definition{test_definition()};

    const auto end{definition.specials.front().id};

    const auto allowing{encoder_of(definition, {"<|end|>"})};

    const auto refusing{encoder_of(definition)};

    const auto ab{allowing.encode("ab")};

    const auto c{allowing.encode(" c")};

    auto expected{ab};

    expected.push_back(end);

    expected.insert(expected.end(), c.begin(), c.end());

    EXPECT_EQ(allowing.encode("ab<|end|> c"), expected);
    EXPECT_EQ(refusing.encode("ab<|end|> c"), uncached_ids(definition, "ab<|end|> c"));
    EXPECT_EQ(allowing.decode(allowing.encode("ab<|end|> c")), "ab<|end|> c");
}

TEST(Encoder_test, Among_allowed_special_tokens_that_begin_at_one_byte_the_longest_is_cut)
{
    const auto definition{test_definition()};

    const auto end{definition.specials[0].id};

    const auto end_end{definition.specials[1].id};

    const auto encoder{encoder_of(definition, {"<|end|>", "<|end|><|end|>"})};

    EXPECT_EQ(encoder.encode("<|end|><|end|><|end|>"), (std::vector<Token_id_t>{end_end, end}));
}

TEST(Encoder_test, An_allowed_spelling_that_is_not_one_of_the_definition_s_special_tokens_is_refused)
{
    const auto refusal{encoder_refusal({"<|end|>", "<|start|>"})};

    EXPECT_TRUE(refusal.contains(R"(the allowed spelling "<|start|>" is not one of the definition's special tokens)"))
            << refusal;
    EXPECT_TRUE(encoder_refusal({"<|end|>"}).empty());
}

TEST(Encoder_test, Decode_refuses_an_id_that_is_neither_an_ordinary_nor_a_special_token_naming_it_and_its_index)
{
    const auto definition{test_definition()};

    const auto beyond{definition.specials.back().id + 1};

    const auto encoder{encoder_of(definition)};

    try
    {
        std::ignore = encoder.decode(std::vector<Token_id_t>{'a', beyond});

        ADD_FAILURE() << "decode() took an id beyond the definition";
    }
    catch (const std::invalid_argument& refusal)
    {
        const std::string_view message{refusal.what()};

        EXPECT_TRUE(message.contains(std::format("the id {} at index 1 is neither", beyond))) << message;
    }
}

TEST(Encoder_test, A_text_that_is_not_utf_8_is_refused_as_the_segmenter_refuses_it)
{
    const auto encoder{encoder_of(test_definition())};

    EXPECT_THROW(std::ignore = encoder.encode("ab\xFF"), std::invalid_argument);
}

TEST(Encoder_test, A_piece_that_is_a_token_the_merge_does_not_form_is_one_id_only_where_the_whole_piece_lookup_says_so)
{
    // xyz is a token, but neither xy nor yz is one, so the merge leaves x y z.
    auto definition{test_definition({"xyz"})};

    const auto xyz{static_cast<Token_id_t>(definition.spellings.size() - 1)};

    EXPECT_EQ(encoder_of(definition).encode("xyz"), std::vector{xyz});

    definition.whole_piece_lookup = false;

    EXPECT_EQ(encoder_of(definition).encode("xyz"), (std::vector<Token_id_t>{'x', 'y', 'z'}));
}

TEST(Encoder_test, The_ids_are_those_of_no_cache_on_a_text_whose_distinct_pieces_fill_the_cache_past_its_bound)
{
    // Twenty thousand words of eight to fourteen letters over a, b and c, almost all distinct and none a token, so
    // every one is merged and the cache stops filling at its bound.
    constexpr std::string_view letters{"abc"};

    std::mt19937 random{3};

    std::uniform_int_distribution<std::size_t> length{8, 14};

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

    const auto definition{test_definition()};

    const auto encoder{encoder_of(definition)};

    EXPECT_EQ(encoder.encode(text), uncached_ids(definition, text));
}

TEST(Encoder_test, The_ids_are_those_of_no_cache_and_decode_spells_the_text_back_on_seeded_random_texts)
{
    const auto definition{test_definition()};

    const auto allowing{encoder_of(definition, {"<|end|>", "<|end|><|end|>"})};

    const auto refusing{encoder_of(definition)};

    std::mt19937 random{5};

    for (std::size_t count{0}; count < 500; ++count)
    {
        const auto text{random_text(random, 40)};

        const auto ids{refusing.encode(text)};

        EXPECT_EQ(ids, uncached_ids(definition, text)) << text;
        EXPECT_EQ(refusing.decode(ids), text);
        EXPECT_EQ(allowing.decode(allowing.encode(text)), text);
    }
}
