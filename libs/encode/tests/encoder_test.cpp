#include "matisse/encode/encoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <munch/regex/regex.hpp>
#include <munch/regex/set.hpp>
#include <munch/regex/utf8.hpp>

#include "matisse/encode/merger.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/encode/stream_encoder.hpp"
#include "matisse/pretok/classes.hpp"
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
using matisse::validate::Inventory;
using matisse::validate::Window_table;
using munch::regex::any_of;
using munch::regex::concat;
using munch::regex::optional;
using munch::regex::plus;
using munch::regex::Set;
using munch::regex::text;
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
const std::vector<Code_point_set> sets{
        {.symbol = ' ', .ranges = {space_points.begin(), space_points.end()}},
        {.symbol = 'a', .ranges = {letter_points.begin(), letter_points.end()}},
};

/**
 * @brief An alphabet of three classes: the space, the letters, and everything else.
 */
const Alphabet alphabet{.symbols = " !a", .sets = sets, .other = '!'};

/**
 * @brief A pattern in GPT-2's shape over the three classes: a letter run or a run of everything else, each after an
 *        optional space, a run of spaces that only a space or the end follows, as GPT-2's `\s+(?!\S)`, and a run of
 *        spaces.
 */
const std::vector<Alternative> alternatives{
        {.language = concat(optional(text(" ")), plus(text("a"))), .gate = " !a"},
        {.language = concat(optional(text(" ")), plus(any_of(Set{'!'}))), .gate = " !a"},
        {.language = plus(text(" ")), .gate = " "},
        {.language = plus(text(" ")), .gate = " !a"},
};

/**
 * @brief The test pattern.
 */
const Pattern pattern{.alphabet = alphabet, .alternatives = alternatives};

/**
 * @brief An alphabet of one class, every code point in it.
 */
const Alphabet one_symbol{.symbols = "a", .sets = {}, .other = 'a'};

/**
 * @brief A pattern over the one class whose every piece is one code point, so that a window of any width is certified
 *        at its last symbol.
 */
const std::vector<Alternative> single_alternatives{
        {.language = text("a"), .gate = "a"},
};

/**
 * @brief The single-symbol pattern over the one class.
 */
const Pattern single{.alphabet = one_symbol, .alternatives = single_alternatives};

/**
 * @brief Returns a definition under the rank rule: the single bytes, then every string of two and three letters over a,
 *        b and c and every space followed by one of them, ranked in a seeded random order, then the given tokens, with
 *        three special tokens after them, the last a single byte.
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
            {.spelling = "<|end|><|end|>", .id = first_special + 1},
            {.spelling = "!", .id = first_special + 2}};

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
 * @param inventory The inventory whose cuts the encoder takes.
 * @return The encoder.
 */
Encoder encoder_of(
        Definition definition, const std::vector<std::string>& allowed = {},
        const Inventory inventory = Inventory::factorization)
{
    const auto report{matisse::validate::validate(pattern)};

    return {std::move(definition), report, inventory, allowed};
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

/**
 * @brief A span of a text's bytes, an occurrence of a special token or a stretch between them: where it begins and the
 *        byte after it.
 */
struct Span
{
    /**
     * @brief The byte at which the span begins.
     */
    std::size_t begin{};

    /**
     * @brief The byte after the span.
     */
    std::size_t end{};
};

/**
 * @brief An anchor of a text, as brute force finds it: where it stands and where its stretch begins.
 */
struct Anchor
{
    /**
     * @brief The anchor's byte.
     */
    std::size_t at{};

    /**
     * @brief The first byte of the stretch between special tokens that holds it.
     */
    std::size_t stretch{};
};

/**
 * @brief Returns whether a byte continues a UTF-8 sequence rather than beginning one.
 * @param byte The byte.
 * @return True for a continuation byte.
 */
bool is_continuation(const char byte)
{
    return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

/**
 * @brief Returns the occurrences of the allowed special tokens in a text by brute force: at every byte from the front,
 *        the longest allowed spelling that begins there, if any, the search going on after it.
 * @param text The text.
 * @param allowed The allowed spellings.
 * @return The occurrences, ascending.
 */
std::vector<Span> special_spans(const std::string_view text, const std::vector<std::string>& allowed)
{
    std::vector<Span> spans{};

    for (std::size_t at{0}; at < text.size();)
    {
        std::size_t longest{0};

        for (const auto& spelling : allowed)
        {
            if (text.substr(at).starts_with(spelling))
            {
                longest = std::max(longest, spelling.size());
            }
        }

        if (longest == 0)
        {
            ++at;

            continue;
        }

        spans.push_back({.begin = at, .end = at + longest});

        at += longest;
    }

    return spans;
}

/**
 * @brief Returns the stretches of a text between its special tokens.
 * @param text The text.
 * @param spans The occurrences of the allowed special tokens.
 * @return The stretches, empty ones included.
 */
std::vector<Span> stretches_of(const std::string_view text, const std::vector<Span>& spans)
{
    std::vector<Span> stretches{};

    std::size_t begin{0};

    for (const auto& span : spans)
    {
        stretches.push_back({.begin = begin, .end = span.begin});

        begin = span.end;
    }

    stretches.push_back({.begin = begin, .end = text.size()});

    return stretches;
}

/**
 * @brief Returns every anchor of a text by brute force: in every stretch between special tokens, the byte of t + o for
 *        every occurrence at t of every window of a table certified at o, the stretch's start left out.
 * @param table The window table.
 * @param text The text, well-formed UTF-8.
 * @param spans The occurrences of the allowed special tokens.
 * @return The anchors, by stretch and within one ascending.
 */
std::vector<Anchor> anchors_of(const Window_table& table, const std::string_view text, const std::vector<Span>& spans)
{
    std::vector<Anchor> anchors{};

    for (const auto [begin, end] : stretches_of(text, spans))
    {
        const auto stretch{text.substr(begin, end - begin)};

        std::string symbols{};

        for (const auto symbol : matisse::pretok::classes_of(alphabet, stretch))
        {
            symbols += alphabet.symbols[symbol];
        }

        std::vector<std::size_t> offsets{};

        for (const auto [at, byte] : std::views::enumerate(stretch))
        {
            if (!is_continuation(byte))
            {
                offsets.push_back(begin + static_cast<std::size_t>(at));
            }
        }

        offsets.push_back(end);

        std::set<std::size_t> found{};

        for (const auto& [window, origin] : table.windows)
        {
            for (auto at{symbols.find(window)}; at != std::string::npos; at = symbols.find(window, at + 1))
            {
                if (at + origin > 0)
                {
                    found.insert(offsets[at + origin]);
                }
            }
        }

        for (const auto at : found)
        {
            anchors.push_back({.at = at, .stretch = begin});
        }
    }

    return anchors;
}

/**
 * @brief Returns the number of code points of a UTF-8 text.
 * @param text The text.
 * @return The count.
 */
std::size_t code_points(const std::string_view text)
{
    return static_cast<std::size_t>(std::ranges::count_if(text, std::not_fn(is_continuation)));
}

/**
 * @brief Returns the boundaries chunk_boundaries() promises, by brute force over every cut: from each equal-division
 *        target, the least edge of a special token or anchor at or after it and after the previous boundary, an anchor
 *        counted only at least the width less one code points after the previous boundary and its stretch's start.
 * @param text The text.
 * @param spans The occurrences of the allowed special tokens.
 * @param anchors The anchors.
 * @param width The window table's width.
 * @param chunks The number of chunks aimed for.
 * @return The boundaries.
 */
std::vector<std::size_t> promised_boundaries(
        const std::string_view text, const std::vector<Span>& spans, const std::vector<Anchor>& anchors,
        const std::size_t width, const std::size_t chunks)
{
    std::vector<std::size_t> boundaries{0};

    const auto usable{std::min(chunks, text.size())};

    for (std::size_t index{1}; index < usable; ++index)
    {
        const auto target{text.size() * index / usable};

        const auto previous{boundaries.back()};

        const auto floor{std::max(target, previous + 1)};

        std::optional<std::size_t> least{};

        const auto offer{[&](const std::size_t cut) {
            if (cut >= floor && cut < text.size() && (!least || cut < *least))
            {
                least = cut;
            }
        }};

        for (const auto& [begin, end] : spans)
        {
            offer(begin);

            offer(end);
        }

        for (const auto& [at, stretch] : anchors)
        {
            const auto from{std::max(previous, stretch)};

            if (at >= from && code_points(text.substr(from, at - from)) >= width - 1)
            {
                offer(at);
            }
        }

        if (!least)
        {
            break;
        }

        boundaries.push_back(*least);
    }

    boundaries.push_back(text.size());

    return boundaries;
}

/**
 * @brief Expects every interior boundary of a text to be a certified cut: an edge of a special token, or an anchor at a
 *        code point outside every special token and a boundary of the pattern's segmentation of its stretch.
 * @param text The text.
 * @param spans The occurrences of the allowed special tokens.
 * @param anchors The anchors.
 * @param boundaries The boundaries.
 */
void expect_certified(
        const std::string_view text, const std::vector<Span>& spans, const std::vector<Anchor>& anchors,
        const std::vector<std::size_t>& boundaries)
{
    const matisse::pretok::Segmenter segmenter{pattern};

    for (const auto cut : boundaries | std::views::drop(1) | std::views::take(boundaries.size() - 2))
    {
        const auto edge{
                std::ranges::any_of(spans, [cut](const Span& span) { return span.begin == cut || span.end == cut; })};

        const auto inside{
                std::ranges::any_of(spans, [cut](const Span& span) { return span.begin < cut && cut < span.end; })};

        const auto anchor{std::ranges::find(anchors, cut, &Anchor::at)};

        EXPECT_FALSE(inside) << text << " cut " << cut;
        EXPECT_FALSE(is_continuation(text[cut])) << text << " cut " << cut;

        if (edge)
        {
            continue;
        }

        ASSERT_NE(anchor, anchors.end()) << text << " cut " << cut;

        const auto stretch_end{std::ranges::find_if(spans, [cut](const Span& span) { return span.begin > cut; })};

        const auto end{stretch_end == spans.end() ? text.size() : stretch_end->begin};

        const auto pieces{segmenter.boundaries(text.substr(anchor->stretch, end - anchor->stretch))};

        EXPECT_TRUE(std::ranges::contains(pieces, cut - anchor->stretch)) << text << " cut " << cut;
    }
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

TEST(Encoder_test, An_allowed_special_token_whose_spelling_is_empty_is_refused_naming_its_id)
{
    auto definition{test_definition()};

    const auto empty{definition.specials.back().id + 1};

    definition.specials.push_back({.spelling = "", .id = empty});

    try
    {
        std::ignore = encoder_of(definition, {"<|end|>", ""});

        ADD_FAILURE() << "Encoder took an empty allowed spelling";
    }
    catch (const std::invalid_argument& refusal)
    {
        const std::string_view message{refusal.what()};

        EXPECT_TRUE(message.contains(std::format("the allowed special token at id {} has an empty spelling", empty)))
                << message;
    }

    EXPECT_NO_THROW(std::ignore = encoder_of(definition, {"<|end|>"}));
}

TEST(Encoder_test, An_allowed_special_token_whose_id_is_an_ordinary_token_s_is_refused_naming_its_spelling_and_id)
{
    auto definition{test_definition()};

    const auto ordinary{static_cast<Token_id_t>('a')};

    definition.specials.push_back({.spelling = "<x>", .id = ordinary});

    try
    {
        std::ignore = encoder_of(definition, {"<|end|>", "<x>"});

        ADD_FAILURE() << "Encoder took an allowed special token with an ordinary token's id";
    }
    catch (const std::invalid_argument& refusal)
    {
        const std::string_view message{refusal.what()};

        EXPECT_TRUE(message.contains(
                std::format(R"(the allowed special token "<x>" has the id {} of an ordinary token)", ordinary)))
                << message;
    }

    EXPECT_NO_THROW(std::ignore = encoder_of(definition, {"<|end|>"}));
}

TEST(Encoder_test, A_special_token_whose_id_is_another_special_token_s_is_refused_allowed_or_not_naming_both)
{
    auto definition{test_definition()};

    const auto end{definition.specials.front().id};

    definition.specials.push_back({.spelling = "<x>", .id = end});

    const std::vector<std::vector<std::string>> allowed_sets{{}, {"<x>"}, {"<|end|>", "<x>"}};

    for (const auto& allowed : allowed_sets)
    {
        try
        {
            std::ignore = encoder_of(definition, allowed);

            ADD_FAILURE() << "Encoder took two special tokens with one id";
        }
        catch (const std::invalid_argument& refusal)
        {
            const std::string_view message{refusal.what()};

            EXPECT_TRUE(message.contains(
                    std::format(R"(the special token "<x>" has the id {} of the special token "<|end|>")", end)))
                    << message;
        }
    }

    EXPECT_NO_THROW(std::ignore = encoder_of(test_definition(), {"<|end|>"}));
}

TEST(Encoder_test, A_special_token_listed_twice_with_two_ids_is_refused_allowed_or_not_naming_both_ids)
{
    auto definition{test_definition()};

    const auto end{definition.specials.front().id};

    const auto next{definition.specials.back().id + 1};

    definition.specials.push_back({.spelling = "<|end|>", .id = next});

    const std::vector<std::vector<std::string>> allowed_sets{{}, {"<|end|>"}};

    for (const auto& allowed : allowed_sets)
    {
        try
        {
            std::ignore = encoder_of(definition, allowed);

            ADD_FAILURE() << "Encoder took one special token listed with two ids";
        }
        catch (const std::invalid_argument& refusal)
        {
            const std::string_view message{refusal.what()};

            EXPECT_TRUE(message.contains(
                    std::format(R"(the special token "<|end|>" is listed twice, with the ids {} and {})", end, next)))
                    << message;
        }
    }

    EXPECT_NO_THROW(std::ignore = encoder_of(test_definition(), {"<|end|>"}));
}

TEST(Encoder_test, A_special_token_whose_spelling_is_not_utf_8_is_refused_allowed_or_not_naming_its_bytes)
{
    auto definition{test_definition()};

    const auto next{definition.specials.back().id + 1};

    // A continuation byte first: allowed, it would begin inside the sequence of a code point such as é, C3 A9.
    definition.specials.push_back({.spelling = "\xA9X", .id = next});

    const std::vector<std::vector<std::string>> allowed_sets{{}, {"\xA9X"}, {"<|end|>", "\xA9X"}};

    for (const auto& allowed : allowed_sets)
    {
        try
        {
            std::ignore = encoder_of(definition, allowed);

            ADD_FAILURE() << "Encoder took a special token whose spelling is not UTF-8";
        }
        catch (const std::invalid_argument& refusal)
        {
            const std::string_view message{refusal.what()};

            EXPECT_TRUE(message.contains(R"(the special token "\xA9X" is not UTF-8)")) << message;
        }
    }

    EXPECT_NO_THROW(std::ignore = encoder_of(test_definition(), {"<|end|>"}));
}

TEST(Encoder_test, A_special_token_spelled_from_a_code_point_s_bytes_is_encoded_alike_serially_in_parallel_and_streamed)
{
    auto definition{test_definition()};

    const auto next{definition.specials.back().id + 1};

    definition.specials.push_back({.spelling = "éx", .id = next});

    const auto encoder{encoder_of(definition, {"<|end|>", "éx"})};

    for (const std::string_view text : {"é", "éé", "éx", "aéxé", "é<|end|>éx éa"})
    {
        const auto expected{encoder.encode(text)};

        for (const std::size_t chunks : {1UZ, 2UZ, 3UZ, 4UZ})
        {
            EXPECT_EQ(encoder.encode_parallel(text, chunks), expected) << text << " in " << chunks;
        }

        Stream_encoder stream{encoder};

        std::vector<Token_id_t> streamed{};

        for (const auto byte : text)
        {
            const auto ids{stream.push(std::string_view{&byte, 1})};

            streamed.insert(streamed.end(), ids.begin(), ids.end());
        }

        const auto rest{stream.finish()};

        streamed.insert(streamed.end(), rest.begin(), rest.end());

        EXPECT_EQ(streamed, expected) << text;
        EXPECT_EQ(encoder.decode(expected), text);
    }
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
    catch (const std::out_of_range& refusal)
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

    const auto expected{uncached_ids(definition, text)};

    EXPECT_EQ(encoder.encode(text), expected);

    // A kept cache, full after the first call, gives the same ids on the second.
    Piece_cache cache{encoder, piece_cache_capacity};

    EXPECT_EQ(encoder.encode(text, cache), expected);
    EXPECT_EQ(encoder.encode(text, cache), expected);
}

TEST(Encoder_test, A_cache_kept_across_calls_gives_encode_s_ids_on_texts_whose_pieces_repeat_from_call_to_call)
{
    const auto encoder{encoder_of(test_definition(), {"<|end|>"})};

    std::mt19937 random{11};

    Piece_cache cache{encoder, piece_cache_capacity};

    for (std::size_t count{0}; count < 500; ++count)
    {
        const auto text{random_text(random, 20)};

        EXPECT_EQ(encoder.encode(text, cache), encoder.encode(text)) << text;
    }
}

TEST(Encoder_test, Threads_that_each_keep_a_cache_of_their_own_give_encode_s_ids_on_every_document)
{
    const auto encoder{encoder_of(test_definition())};

    std::mt19937 random{13};

    std::vector<std::string> documents{};

    for (std::size_t count{0}; count < 400; ++count)
    {
        documents.push_back(random_text(random, 20));
    }

    constexpr std::size_t threads{4};

    std::vector<std::vector<Token_id_t>> encoded(documents.size());

    // Each thread takes every fourth document and writes its own slots, so the threads share only the encoder and the
    // documents, both read.
    const auto work{[&](const std::size_t first) {
        Piece_cache cache{encoder, piece_cache_capacity};

        for (auto index{first}; index < documents.size(); index += threads)
        {
            encoded[index] = encoder.encode(documents[index], cache);
        }
    }};

    {
        std::vector<std::jthread> workers{};

        for (std::size_t worker{0}; worker < threads; ++worker)
        {
            workers.emplace_back(work, worker);
        }
    }

    for (const auto [index, document] : std::views::enumerate(documents))
    {
        EXPECT_EQ(encoded[static_cast<std::size_t>(index)], encoder.encode(document)) << document;
    }
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

TEST(Encoder_test, Chunk_boundaries_cuts_as_brute_force_over_every_certified_cut_says_and_only_at_certified_cuts)
{
    const auto definition{test_definition()};

    const auto report{matisse::validate::validate(pattern)};

    // The one-byte special token can stand at the text's end and next to another, so a cut can meet both.
    const std::vector<std::vector<std::string>> allowed_lists{{}, {"<|end|>", "<|end|><|end|>"}, {"<|end|>", "!"}};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto& table{inventory == Inventory::factorization ? report.factorization : report.policy};

        for (const auto& allowed : allowed_lists)
        {
            const Encoder encoder{definition, report, inventory, allowed};

            std::mt19937 random{17};

            for (std::size_t count{0}; count < 200; ++count)
            {
                const auto text{random_text(random, 20)};

                const auto spans{special_spans(text, allowed)};

                const auto anchors{anchors_of(table, text, spans)};

                for (const auto chunks : chunk_counts(text))
                {
                    const auto boundaries{encoder.chunk_boundaries(text, chunks)};

                    EXPECT_EQ(boundaries, promised_boundaries(text, spans, anchors, table.width, chunks))
                            << text << " in " << chunks;

                    expect_certified(text, spans, anchors, boundaries);
                }
            }
        }
    }
}

TEST(Encoder_test, Encode_parallel_gives_the_ids_of_encode_at_every_chunk_count_and_every_cut_on_seeded_random_texts)
{
    const auto definition{test_definition()};

    const auto report{matisse::validate::validate(pattern)};

    const std::vector<std::vector<std::string>> allowed_lists{{}, {"<|end|>", "<|end|><|end|>"}, {"<|end|>", "!"}};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        for (const auto& allowed : allowed_lists)
        {
            const Encoder encoder{definition, report, inventory, allowed};

            std::mt19937 random{19};

            for (std::size_t count{0}; count < 100; ++count)
            {
                const auto text{random_text(random, 40)};

                const auto ids{encoder.encode(text)};

                for (const auto chunks : chunk_counts(text))
                {
                    EXPECT_EQ(encoder.encode_parallel(text, chunks), ids) << text << " in " << chunks;
                }
            }
        }
    }
}

TEST(Encoder_test, The_empty_text_and_tiny_texts_are_encoded_in_parallel_as_they_are_encoded)
{
    const auto encoder{encoder_of(test_definition(), {"<|end|>"})};

    EXPECT_EQ(encoder.chunk_boundaries("", 4), (std::vector<std::size_t>{0, 0}));

    for (const std::string_view text : {"", "a", "!", " ", "a!", "!a", " !", "<|end|>", "a<|end|>", "<|end|>!", "é!"})
    {
        for (std::size_t chunks{0}; chunks <= 8; ++chunks)
        {
            EXPECT_EQ(encoder.encode_parallel(text, chunks), encoder.encode(text)) << text << " in " << chunks;
        }
    }
}

TEST(Encoder_test, A_text_with_no_certified_cut_is_one_chunk_however_many_are_asked_for)
{
    const std::string letters(5000, 'b');

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto encoder{encoder_of(test_definition(), {}, inventory)};

        EXPECT_EQ(encoder.chunk_boundaries(letters, 8), (std::vector<std::size_t>{0, letters.size()}));
        EXPECT_EQ(encoder.encode_parallel(letters, 8), encoder.encode(letters));
    }
}

TEST(Encoder_test, An_allowed_special_token_is_a_cut_of_its_own_where_no_window_certifies_one)
{
    const auto encoder{encoder_of(test_definition(), {"<|end|>"})};

    const std::string text{"aaaaaaaa<|end|>aaaaaaaa"};

    // The first target falls in the letters before the token, which certify nothing, and the second at its end.
    EXPECT_EQ(encoder.chunk_boundaries(text, 3), (std::vector<std::size_t>{0, 8, 15, 23}));
    EXPECT_EQ(encoder.encode_parallel(text, 3), encoder.encode(text));
}

TEST(Encoder_test, Encode_parallel_rethrows_a_worker_s_refusal_of_a_text_that_is_not_utf_8)
{
    const auto encoder{encoder_of(test_definition())};

    std::string text{"ab\xFF"};

    for (std::size_t count{0}; count < 100; ++count)
    {
        text += "ab!";
    }

    // The walks from the targets read the text near them, so the text is cut, and the first chunk, which holds the
    // ill-formed byte, is encoded on a thread of its own.
    ASSERT_EQ(encoder.chunk_boundaries(text, 4).size(), 5U);
    EXPECT_THROW(std::ignore = encoder.encode_parallel(text, 4), std::invalid_argument);
}

TEST(Encoder_test, A_cut_in_white_space_holds_with_the_right_context_the_policy_inventory_decided)
{
    const auto definition{test_definition()};

    const auto report{matisse::validate::validate(pattern)};

    const Encoder encoder{definition, report, Inventory::policy, {}};

    // The report's counterexample: a worker reading one code point less past the cut between the spaces reads them as
    // one piece.
    const std::string text{"aaaa  !aaaa"};

    EXPECT_EQ(report.policy.right_context, 2U);
    EXPECT_EQ(encoder.chunk_boundaries(text, 2), (std::vector<std::size_t>{0, 5, 11}));
    EXPECT_EQ(encoder.encode_parallel(text, 2), encoder.encode(text));
}

TEST(Encoder_test, A_one_byte_special_token_at_the_text_s_end_leaves_no_empty_chunk)
{
    const auto encoder{encoder_of(test_definition(), {"!"})};

    EXPECT_EQ(encoder.chunk_boundaries("aaaaaaaa!", 3), (std::vector<std::size_t>{0, 8, 9}));
    EXPECT_EQ(encoder.chunk_boundaries("aaaaaaaa!!", 9), (std::vector<std::size_t>{0, 8, 9, 10}));
}

TEST(Encoder_test, Windows_wider_than_the_block_the_walk_classes_cut_at_their_spacing_and_give_the_ids_of_encode)
{
    const auto report{matisse::validate::validate(single, 257)};

    const Encoder encoder{test_definition(), report, Inventory::factorization, {}};

    const std::string text(1000, 'a');

    // Every cut stands at least 256 code points after the previous one, as far as the 256 bytes the walk classes at a
    // time, so the walk reads on past the block it starts in.
    EXPECT_EQ(encoder.chunk_boundaries(text, 4), (std::vector<std::size_t>{0, 256, 512, 768, 1000}));
    EXPECT_EQ(encoder.encode_parallel(text, 4), encoder.encode(text));
}

TEST(Encoder_test, An_encoder_segments_by_the_pattern_its_report_was_decided_of_and_takes_no_other)
{
    using matisse::validate::Report;

    // A report is made by validate() alone, and an encoder takes the pattern from it, so no report reaches an encoder
    // beside a pattern it was not decided of.
    static_assert(!std::is_constructible_v<Report, Window_table, Window_table, munch::verifier::Boundary_gap, Pattern>);
    static_assert(!std::is_constructible_v<
                  Encoder, Definition, const Report&, Inventory, const Pattern&, std::span<const std::string>>);

    const auto words{encoder_of(test_definition())};

    const Encoder points{test_definition(), matisse::validate::validate(single, 2), Inventory::factorization, {}};

    // The test pattern takes the letters as one piece, a token whole, and the single-symbol pattern each letter alone.
    EXPECT_EQ(words.encode("abc").size(), 1U);
    EXPECT_EQ(points.encode("abc"), (std::vector<Token_id_t>{'a', 'b', 'c'}));

    const std::string text{"abc ab!a ba  cab!!a"};

    for (std::size_t chunks{1}; chunks <= 8; ++chunks)
    {
        EXPECT_EQ(words.encode_parallel(text, chunks), words.encode(text)) << chunks;
        EXPECT_EQ(points.encode_parallel(text, chunks), points.encode(text)) << chunks;
    }
}

TEST(Encoder_test, An_encoder_is_copied_and_moved_but_never_assigned_so_a_cache_serves_one_definition_throughout)
{
    // A cache knows its encoder by address, so an assignment would leave it serving another definition's ids; neither
    // assignment compiles.
    static_assert(!std::is_copy_assignable_v<Encoder>);
    static_assert(!std::is_move_assignable_v<Encoder>);
    static_assert(std::is_copy_constructible_v<Encoder>);
    static_assert(std::is_move_constructible_v<Encoder>);

    const auto encoder{encoder_of(test_definition())};

    Piece_cache cache{encoder, piece_cache_capacity};

    const auto ids{encoder.encode("abc ab!a", cache)};

    const auto copied{encoder};

    EXPECT_EQ(copied.encode("abc ab!a"), ids);
    EXPECT_THROW(std::ignore = copied.encode("abc ab!a", cache), std::invalid_argument);
}

TEST(Encoder_test, A_cache_made_for_another_encoder_is_refused)
{
    // A cache borrows the encoder it is made for, so a temporary one is no encoder to make it for.
    static_assert(!std::is_constructible_v<Piece_cache, Encoder, std::size_t>);

    const auto encoder{encoder_of(test_definition())};

    const auto other{encoder_of(test_definition())};

    Piece_cache cache{other, piece_cache_capacity};

    try
    {
        std::ignore = encoder.encode("abc", cache);

        ADD_FAILURE() << "encode() took a cache made for another encoder";
    }
    catch (const std::invalid_argument& refusal)
    {
        const std::string_view message{refusal.what()};

        EXPECT_TRUE(message.contains("the piece cache was made for another encoder")) << message;
    }

    EXPECT_EQ(other.encode("abc", cache), other.encode("abc"));
}
