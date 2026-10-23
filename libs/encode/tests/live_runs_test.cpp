#include "matisse/encode/live_runs.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <munch/regex/regex.hpp>
#include <munch/regex/set.hpp>
#include <munch/regex/utf8.hpp>

#include "matisse/encode/verifier_table.hpp"
#include "matisse/pretok/pattern.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/pretok/verifier.hpp"

using namespace matisse::encode;
using matisse::pretok::Alphabet;
using matisse::pretok::Alternative;
using matisse::pretok::Code_point_set;
using matisse::pretok::Pattern;
using munch::regex::any_of;
using munch::regex::concat;
using munch::regex::optional;
using munch::regex::plus;
using munch::regex::range;
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
 * @brief The ASCII letters.
 */
constexpr std::array<Code_point_range, 2> letter_points{{
        {.first = U'A', .last = U'Z'},
        {.first = U'a', .last = U'z'},
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
 * @brief A pattern in GPT-2's shape over the three classes, whose white space runs hold their last space for what
 *        follows.
 */
const std::vector<Alternative> gpt2_shaped{
        {.language = concat(optional(text(" ")), plus(text("a"))), .gate = " !a"},
        {.language = concat(optional(text(" ")), plus(any_of(Set{'!'}))), .gate = " !a"},
        {.language = plus(text(" ")), .gate = " "},
        {.language = plus(text(" ")), .gate = " !a"},
};

/**
 * @brief A pattern whose letter runs are cut every three letters, as cl100k cuts digits, with a dead end after three.
 */
const std::vector<Alternative> bounded{
        {.language = range(text("a"), 1, 3), .gate = " !a"},
        {.language = plus(any_of(Set{'!'})), .gate = " !a"},
        {.language = plus(text(" ")), .gate = " !a"},
};

/**
 * @brief Returns the piece ends of a text's segmentation before its last byte, the ones its restriction to a prefix of
 *        that length can say.
 * @param segmenter The pattern's segmenter.
 * @param text The text.
 * @param before The prefix's length.
 * @return The piece ends below it, ascending.
 */
std::vector<std::size_t> ends_before(
        const matisse::pretok::Segmenter& segmenter, const std::string_view text, const std::size_t before)
{
    auto ends{segmenter.boundaries(text)};

    std::erase_if(ends, [before](const std::size_t end) { return end >= before; });

    return ends;
}

/**
 * @brief Expects the live runs over seeded random texts of the three classes to be the segmentations of every
 *        continuation of up to five code points, restricted to the text, and the pieces they close to be the ones all
 *        those segmentations share from the start.
 * @param pattern The pattern.
 * @param seed The seed of the texts.
 */
void expect_futures(const Pattern& pattern, const unsigned seed)
{
    const auto verifier{matisse::pretok::split_verifier(pattern)};

    const Verifier_table table{verifier, pattern.alphabet};

    const matisse::pretok::Segmenter segmenter{pattern};

    constexpr std::string_view symbols{" !a"};

    std::vector<std::string> continuations{""};

    for (std::size_t length{1}; length <= 5; ++length)
    {
        std::vector<std::string> longer{};

        for (const auto& continuation : continuations)
        {
            if (continuation.size() + 1 != length)
            {
                continue;
            }

            for (const auto symbol : symbols)
            {
                longer.push_back(continuation + symbol);
            }
        }

        continuations.insert(continuations.end(), longer.begin(), longer.end());
    }

    std::mt19937 random{seed};

    std::uniform_int_distribution<std::size_t> symbol{0, symbols.size() - 1};

    for (std::size_t count{0}; count < 300; ++count)
    {
        Live_runs runs{table};

        std::string text{};

        std::vector<std::size_t> closed{};

        for (std::size_t length{0}; length < 12; ++length)
        {
            const auto drawn{symbol(random)};

            text += symbols[drawn];

            runs.step(static_cast<matisse::pretok::Class_t>(drawn), text.size());

            while (const auto end{runs.next_closed()})
            {
                closed.push_back(*end);
            }

            std::set<std::vector<std::size_t>> futures{};

            std::optional<std::vector<std::size_t>> shared{};

            for (const auto& continuation : continuations)
            {
                const auto ends{ends_before(segmenter, text + continuation, text.size())};

                futures.insert(ends);

                if (!shared)
                {
                    shared = ends;

                    continue;
                }

                const auto [shared_end, ends_end]{std::ranges::mismatch(*shared, ends)};

                shared->erase(shared_end, shared->end());
            }

            std::set<std::vector<std::size_t>> marked{};

            for (const auto& [ends, dead_end] : runs.markings())
            {
                std::vector<std::size_t> all{closed};

                all.insert(all.end(), ends.begin(), ends.end());

                std::erase_if(all, [&text](const std::size_t end) { return end >= text.size(); });

                std::ranges::sort(all);

                const auto [first_repeat, last]{std::ranges::unique(all)};

                all.erase(first_repeat, last);

                marked.insert(all);
            }

            auto closed_before{closed};

            std::erase_if(closed_before, [&text](const std::size_t end) { return end >= text.size(); });

            EXPECT_EQ(marked, futures) << text;
            EXPECT_EQ(closed_before, *shared) << text;
        }
    }
}

} // namespace

TEST(Live_runs_test, The_runs_are_the_restrictions_of_every_continuation_s_segmentation_and_close_what_they_all_share)
{
    expect_futures({.alphabet = alphabet, .alternatives = gpt2_shaped}, 1);
    expect_futures({.alphabet = alphabet, .alternatives = bounded}, 2);
}

TEST(Live_runs_test, A_run_of_three_letters_closes_its_piece_at_the_dead_end_before_the_next_code_point)
{
    const Pattern pattern{.alphabet = alphabet, .alternatives = bounded};

    const auto verifier{matisse::pretok::split_verifier(pattern)};

    const Verifier_table table{verifier, pattern.alphabet};

    Live_runs runs{table};

    constexpr matisse::pretok::Class_t letter{2};

    for (std::size_t end{1}; end <= 2; ++end)
    {
        runs.step(letter, end);

        EXPECT_FALSE(runs.next_closed());
    }

    runs.step(letter, 3);

    EXPECT_EQ(runs.next_closed(), 3);
    EXPECT_FALSE(runs.next_closed());

    runs.step(letter, 4);

    EXPECT_FALSE(runs.next_closed());
    EXPECT_TRUE(runs.accepted().empty());
}
