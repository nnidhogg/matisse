#include "matisse/encode/stream_encoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <munch/regex/regex.hpp>
#include <munch/regex/set.hpp>
#include <munch/regex/utf8.hpp>

#include "matisse/encode/encoder.hpp"
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
 * @brief The allowed spellings of the tests that allow special tokens, the second beginning with the first.
 */
const std::vector<std::string> both_ends{"<|end|>", "<|end|><|end|>"};

/**
 * @brief Returns a definition under the rank rule with the whole-piece lookup: the single bytes, then every string of
 *        two and three letters over a, b and c and every space followed by one of them, ranked in a seeded random
 *        order, then abcab, xyz and cbx, of which the merge forms cbx alone, with three special tokens after them, the
 *        last a single byte.
 * @return The definition.
 */
Definition test_definition()
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

    for (const auto token : {"abcab", "xyz", "cbx"})
    {
        spellings.emplace_back(token);
    }

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
 * @brief Returns the encoder of the test definition under the test pattern.
 * @param allowed The special tokens allowed.
 * @return The encoder.
 */
Encoder encoder_of(const std::vector<std::string>& allowed)
{
    const auto report{matisse::validate::validate(pattern)};

    return {test_definition(), report, matisse::validate::Inventory::factorization, pattern, allowed};
}

/**
 * @brief Returns a seeded random text of the shapes the pattern, the merge and the special tokens meet: letter runs
 *        over a, b and c and the letters of the tokens the merge does not form, runs of spaces, tabs and newlines,
 *        punctuation, multilingual letters, and the spellings of the special tokens whole and broken.
 * @param random The random sequence.
 * @param fragments The number of fragments joined.
 * @return The text, well-formed UTF-8.
 */
std::string random_text(std::mt19937& random, const std::size_t fragments)
{
    constexpr std::array<std::string_view, 24> shapes{
            "a",     "b", "c",  "abc",   "cab",   "abcab",  " ab",    " cab",    "  ",   "   ", "\t\t", "\n\n",
            " \n\t", "!", "?!", "12345", "héllo", " wörld", "日本語", "<|end|>", "<|en", "|>",  "x",    "\U0001F642"};

    std::uniform_int_distribution<std::size_t> shape{0, shapes.size() - 1};

    std::string text{};

    for (std::size_t count{0}; count < fragments; ++count)
    {
        text += shapes[shape(random)];
    }

    return text;
}

/**
 * @brief Returns seeded random ends of the pushes of a text, each push 0 to a bound of bytes long, so that pushes end
 *        inside UTF-8 sequences and inside spellings and some push nothing.
 * @param random The random sequence.
 * @param size The text's size.
 * @param widest The longest push.
 * @return The byte after each push, ascending to the text's size.
 */
std::vector<std::size_t> random_cuts(std::mt19937& random, const std::size_t size, const std::size_t widest)
{
    std::uniform_int_distribution<std::size_t> width{0, widest};

    std::vector<std::size_t> cuts{};

    for (std::size_t at{0}; at < size;)
    {
        at = std::min(size, at + width(random));

        cuts.push_back(at);
    }

    return cuts;
}

/**
 * @brief Returns the ends of the pushes of a text one byte each.
 * @param size The text's size.
 * @return The byte after each push.
 */
std::vector<std::size_t> one_byte_cuts(const std::size_t size)
{
    const auto ends{std::views::iota(1UZ, size + 1)};

    return {ends.begin(), ends.end()};
}

/**
 * @brief Pushes a text to a stream in parts and expects every push to return ids that extend a prefix of encode() of
 *        the whole text, and the ids of all pushes and finish() to be encode() of it.
 * @param encoder The encoder.
 * @param text The text.
 * @param cuts The byte after each push.
 */
void expect_streamed(const Encoder& encoder, const std::string_view text, const std::span<const std::size_t> cuts)
{
    const auto expected{encoder.encode(text)};

    Stream_encoder stream{encoder};

    std::vector<Token_id_t> returned{};

    std::size_t start{0};

    for (const auto cut : cuts)
    {
        const auto ids{stream.push(text.substr(start, cut - start))};

        returned.insert(returned.end(), ids.begin(), ids.end());

        start = cut;

        ASSERT_LE(returned.size(), expected.size()) << text << " after " << cut;
        ASSERT_TRUE(std::ranges::equal(returned, std::span{expected}.first(returned.size())))
                << text << " after " << cut;
    }

    const auto rest{stream.finish()};

    returned.insert(returned.end(), rest.begin(), rest.end());

    EXPECT_EQ(returned, expected) << text;
}

/**
 * @brief Returns every string of up to a number of fragments, the empty one first.
 * @param fragments The fragments.
 * @param depth The most fragments joined.
 * @return The strings.
 */
std::vector<std::string> strings_up_to(const std::span<const std::string_view> fragments, const std::size_t depth)
{
    std::vector<std::string> all{""};

    std::vector<std::string> layer{""};

    for (std::size_t joined{0}; joined < depth; ++joined)
    {
        std::vector<std::string> next{};

        for (const auto& string : layer)
        {
            for (const auto fragment : fragments)
            {
                next.push_back(string + std::string{fragment});
            }
        }

        all.insert(all.end(), next.begin(), next.end());

        layer = std::move(next);
    }

    return all;
}

/**
 * @brief Returns the boundaries of encode()'s segmentation of a text up to a byte, by brute force: every edge of an
 *        allowed special token's occurrence, the earliest first and the longest among those at one byte, and every
 *        piece end of the pattern's segmentation of each stretch between them, ascending.
 * @param segmenter The pattern's segmenter.
 * @param text The text.
 * @param allowed The allowed spellings.
 * @param upto The last byte counted.
 * @return The boundaries at or before it.
 */
std::vector<std::size_t> boundaries_of(
        const matisse::pretok::Segmenter& segmenter, const std::string_view text,
        const std::vector<std::string>& allowed, const std::size_t upto)
{
    std::vector<std::size_t> boundaries{};

    const auto add_stretch{[&](const std::size_t begin, const std::size_t end) {
        for (const auto piece_end : segmenter.boundaries(text.substr(begin, end - begin)))
        {
            boundaries.push_back(begin + piece_end);
        }
    }};

    std::size_t begin{0};

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

        add_stretch(begin, at);

        boundaries.push_back(at);

        boundaries.push_back(at + longest);

        at += longest;

        begin = at;
    }

    add_stretch(begin, text.size());

    std::erase_if(boundaries, [upto](const std::size_t boundary) { return boundary > upto; });

    std::ranges::sort(boundaries);

    const auto [first_repeat, last]{std::ranges::unique(boundaries)};

    boundaries.erase(first_repeat, last);

    return boundaries;
}

/**
 * @brief Returns whether a text ends in a nonempty proper prefix of an allowed spelling.
 * @param text The text.
 * @param allowed The allowed spellings.
 * @return True when some suffix of the text begins an allowed spelling and is not all of it.
 */
bool ends_in_spelling(const std::string_view text, const std::vector<std::string>& allowed)
{
    for (const auto& spelling : allowed)
    {
        for (std::size_t length{1}; length < spelling.size() && length <= text.size(); ++length)
        {
            if (spelling.starts_with(text.substr(text.size() - length)))
            {
                return true;
            }
        }
    }

    return false;
}

/**
 * @brief Returns the length of a text's longest prefix that ends at a code point.
 * @param text The text, well-formed UTF-8 but for its end.
 * @param end The byte the prefix may not pass.
 * @return The prefix's length.
 */
std::size_t point_floor(const std::string_view text, std::size_t end)
{
    while (end > 0 && end < text.size() && (static_cast<unsigned char>(text[end]) & 0xC0U) == 0x80U)
    {
        --end;
    }

    return end;
}

/**
 * @brief Pushes seeded random texts to a stream in random parts and holds every push to brute force over every
 *        continuation of up to four fragments: the ids returned are a prefix of encode() of the text read followed by
 *        each of them, and, unless the text read ends in a proper prefix of an allowed spelling, they spell at least
 *        every piece and special token whose boundaries the encodings of all of them keep.
 * @param allowed The allowed spellings.
 * @param seed The seed of the texts and the parts.
 * @param texts The number of texts.
 */
void expect_brute_force(const std::vector<std::string>& allowed, const unsigned seed, const std::size_t texts)
{
    constexpr std::array<std::string_view, 7> fragments{" ", "a", "b", "c", "!", "<|end|>", "x"};

    const auto continuations{strings_up_to(fragments, 4)};

    const auto encoder{encoder_of(allowed)};

    const matisse::pretok::Segmenter segmenter{pattern};

    Piece_cache cache{piece_cache_capacity};

    std::mt19937 random{seed};

    for (std::size_t count{0}; count < texts; ++count)
    {
        const auto text{random_text(random, 12)};

        Stream_encoder stream{encoder};

        std::vector<Token_id_t> returned{};

        std::size_t start{0};

        for (const auto cut : random_cuts(random, text.size(), 9))
        {
            const auto ids{stream.push(std::string_view{text}.substr(start, cut - start))};

            returned.insert(returned.end(), ids.begin(), ids.end());

            start = cut;

            // A push that ends inside a code point has read the whole code points before it.
            const auto read{text.substr(0, point_floor(text, cut))};

            auto bound{encoder.encode(read, cache)};

            std::vector<std::size_t> kept{boundaries_of(segmenter, read, allowed, read.size())};

            for (const auto& continuation : continuations)
            {
                const auto continued{read + continuation};

                const auto ids_continued{encoder.encode(continued, cache)};

                const auto [bound_end, continued_end]{std::ranges::mismatch(bound, ids_continued)};

                bound.erase(bound_end, bound.end());

                const auto boundaries{boundaries_of(segmenter, continued, allowed, read.size())};

                const auto [kept_end, boundaries_end]{std::ranges::mismatch(kept, boundaries)};

                kept.erase(kept_end, kept.end());
            }

            ASSERT_LE(returned.size(), bound.size()) << text << " after " << cut;
            ASSERT_TRUE(std::ranges::equal(returned, std::span{bound}.first(returned.size())))
                    << text << " after " << cut;

            if (ends_in_spelling(read, allowed) || kept.empty())
            {
                continue;
            }

            const auto spelled{encoder.decode(returned)};

            EXPECT_GE(spelled.size(), kept.back()) << text << " after " << cut;
        }

        const auto rest{stream.finish()};

        returned.insert(returned.end(), rest.begin(), rest.end());

        EXPECT_EQ(returned, encoder.encode(text)) << text;
    }
}

} // namespace

TEST(Stream_encoder_test, Pushes_of_every_size_return_prefixes_of_encode_and_concatenate_to_it_on_seeded_random_texts)
{
    for (const auto& allowed : {std::vector<std::string>{}, both_ends})
    {
        const auto encoder{encoder_of(allowed)};

        std::mt19937 random{3};

        for (std::size_t count{0}; count < 200; ++count)
        {
            const auto text{random_text(random, 30)};

            expect_streamed(encoder, text, random_cuts(random, text.size(), 9));
            expect_streamed(encoder, text, random_cuts(random, text.size(), 64));
            expect_streamed(encoder, text, one_byte_cuts(text.size()));
            expect_streamed(encoder, text, std::array{text.size()});
        }
    }
}

TEST(Stream_encoder_test, Every_push_returns_a_prefix_of_every_continuation_s_encoding_and_every_piece_they_all_keep)
{
    expect_brute_force(both_ends, 7, 40);
    expect_brute_force({}, 8, 30);
}

TEST(Stream_encoder_test, Inside_a_letter_run_the_ids_returned_are_those_every_continuation_of_the_run_keeps)
{
    const auto encoder{encoder_of({})};

    const auto& spellings{encoder.definition().spellings};

    const auto ids_of{[&encoder](const std::string_view piece) {
        if (const auto whole{encoder.merger().whole_piece(piece)})
        {
            return std::vector<Token_id_t>{*whole};
        }

        return encoder.merger().merged(piece);
    }};

    // The finite continuation set of the stable prefix: the empty string and every proper suffix of every token.
    std::set<std::string> continuations{""};

    for (const auto& spelling : spellings)
    {
        for (std::size_t at{1}; at < spelling.size(); ++at)
        {
            continuations.insert(spelling.substr(at));
        }
    }

    constexpr std::string_view letters{"abcaaxyz"};

    std::mt19937 random{5};

    std::uniform_int_distribution<std::size_t> length{1, 12};

    std::uniform_int_distribution<std::size_t> letter{0, letters.size() - 1};

    for (std::size_t count{0}; count < 1000; ++count)
    {
        std::string run{};

        for (auto left{length(random)}; left > 0; --left)
        {
            run += letters[letter(random)];
        }

        auto kept{ids_of(run)};

        for (const auto& continuation : continuations)
        {
            const auto continued{ids_of(run + continuation)};

            const auto [kept_end, continued_end]{std::ranges::mismatch(kept, continued)};

            kept.erase(kept_end, kept.end());
        }

        Stream_encoder stream{encoder};

        std::vector<Token_id_t> returned{};

        std::size_t start{0};

        for (const auto cut : random_cuts(random, run.size(), 4))
        {
            const auto ids{stream.push(std::string_view{run}.substr(start, cut - start))};

            returned.insert(returned.end(), ids.begin(), ids.end());

            start = cut;
        }

        EXPECT_EQ(returned, kept) << run;
    }
}

TEST(Stream_encoder_test, A_code_point_split_across_pushes_is_read_whole_once_its_last_byte_arrives)
{
    const auto encoder{encoder_of({})};

    const std::string text{"日本 語 \U0001F642 é"};

    Stream_encoder stream{encoder};

    std::vector<Token_id_t> returned{};

    for (std::size_t at{0}; at < text.size(); ++at)
    {
        const auto ids{stream.push(std::string_view{text}.substr(at, 1))};

        returned.insert(returned.end(), ids.begin(), ids.end());

        const auto spelled{encoder.decode(returned)};

        EXPECT_TRUE(text.starts_with(spelled)) << at;
        EXPECT_LE(spelled.size(), point_floor(text, at + 1)) << at;
    }

    const auto rest{stream.finish()};

    returned.insert(returned.end(), rest.begin(), rest.end());

    EXPECT_EQ(returned, encoder.encode(text));
}

TEST(Stream_encoder_test, A_spelling_split_across_pushes_is_held_until_it_completes_or_breaks)
{
    const auto encoder{encoder_of(both_ends)};

    const auto end_id{encoder.definition().specials[0].id};

    const auto end_end_id{encoder.definition().specials[1].id};

    Stream_encoder stream{encoder};

    auto returned{stream.push("ab <|en")};

    // The spelling completed is a proper prefix of the longer one, so it is held still.
    EXPECT_TRUE(stream.push("d|>").empty());

    const auto completed{stream.push("<|end|> c")};

    EXPECT_TRUE(std::ranges::contains(completed, end_end_id));

    returned.insert(returned.end(), completed.begin(), completed.end());

    const auto rest{stream.finish()};

    returned.insert(returned.end(), rest.begin(), rest.end());

    EXPECT_EQ(returned, encoder.encode("ab <|end|><|end|> c"));

    EXPECT_TRUE(stream.push("<|end|>").empty());

    const auto broken{stream.push("x")};

    ASSERT_FALSE(broken.empty());
    EXPECT_EQ(broken.front(), end_id);

    auto broken_all{broken};

    const auto broken_rest{stream.finish()};

    broken_all.insert(broken_all.end(), broken_rest.begin(), broken_rest.end());

    EXPECT_EQ(broken_all, encoder.encode("<|end|>x"));
}

TEST(Stream_encoder_test, Finish_ends_the_text_and_the_stream_reads_a_new_one)
{
    const auto encoder{encoder_of(both_ends)};

    Stream_encoder stream{encoder};

    for (const std::string_view text : {"abc <|end|>  ab", "", "<|en", " cab cab!", "\n\n日本"})
    {
        auto returned{stream.push(text)};

        const auto rest{stream.finish()};

        returned.insert(returned.end(), rest.begin(), rest.end());

        EXPECT_EQ(returned, encoder.encode(text)) << text;
    }
}

TEST(Stream_encoder_test, A_text_that_is_not_utf_8_is_refused_where_it_is_read)
{
    const auto encoder{encoder_of({})};

    Stream_encoder ill_formed{encoder};

    EXPECT_THROW(std::ignore = ill_formed.push("ab\xFF"), std::invalid_argument);

    Stream_encoder unfinished{encoder};

    std::ignore = unfinished.push("ab\xE6\x97");
    EXPECT_THROW(std::ignore = unfinished.finish(), std::invalid_argument);
}

TEST(Stream_encoder_test, Streams_on_threads_of_their_own_over_one_encoder_give_encode_s_ids)
{
    const auto encoder{encoder_of(both_ends)};

    std::mt19937 random{13};

    std::vector<std::string> documents{};

    for (std::size_t count{0}; count < 200; ++count)
    {
        documents.push_back(random_text(random, 30));
    }

    constexpr std::size_t threads{4};

    std::vector<std::vector<Token_id_t>> streamed(documents.size());

    // Each thread takes every fourth document with a stream of its own, pushed five bytes at a time.
    const auto work{[&](const std::size_t first) {
        Stream_encoder stream{encoder};

        for (auto index{first}; index < documents.size(); index += threads)
        {
            const std::string_view document{documents[index]};

            for (std::size_t at{0}; at < document.size(); at += 5)
            {
                const auto ids{stream.push(document.substr(at, 5))};

                streamed[index].insert(streamed[index].end(), ids.begin(), ids.end());
            }

            const auto rest{stream.finish()};

            streamed[index].insert(streamed[index].end(), rest.begin(), rest.end());
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
        EXPECT_EQ(streamed[static_cast<std::size_t>(index)], encoder.encode(document)) << document;
    }
}
