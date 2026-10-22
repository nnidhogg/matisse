#include "matisse/encode/verifier_table.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <ranges>

#include <munch/regex/utf8.hpp>
#include <munch/verifier/verifier.hpp>

#include "matisse/pretok/pattern.hpp"
#include "matisse/pretok/verifier.hpp"

using namespace matisse::encode;
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
 * @brief A pattern in GPT-2's shape over the three classes, with no bounded run and so no dead end.
 */
constexpr std::array<Alternative, 4> gpt2_shaped{{
        {.language = R"(\x20?a+)", .gate = " !a"},
        {.language = R"(\x20?[!]+)", .gate = " !a"},
        {.language = R"(\x20+)", .gate = " "},
        {.language = R"(\x20+)", .gate = " !a"},
}};

/**
 * @brief A pattern whose letter runs are cut every three letters, as cl100k cuts digits, so that a run of three letters
 *        unmarked ends in a dead end.
 */
constexpr std::array<Alternative, 3> bounded{{
        {.language = R"(a{1,3})", .gate = " !a"},
        {.language = R"([!]+)", .gate = " !a"},
        {.language = R"(\x20+)", .gate = " !a"},
}};

/**
 * @brief Expects a pattern's verifier table to step, accept and end as munch's verifier does, on every state, class and
 *        mark.
 * @param pattern The pattern.
 * @return The number of dead ends.
 */
std::size_t expect_as_verifier(const Pattern& pattern)
{
    const auto verifier{matisse::pretok::split_verifier(pattern)};

    const Verifier_table table{verifier, pattern.alphabet};

    EXPECT_EQ(table.start(), verifier.start());

    std::size_t dead_ends{0};

    for (munch::verifier::Verifier::State_t state{0}; state < verifier.state_count(); ++state)
    {
        const auto at{static_cast<Verifier_table::State_t>(state)};

        auto moves{false};

        for (const auto [symbol_class, symbol] : std::views::enumerate(pattern.alphabet.symbols))
        {
            for (const auto boundary_after : {false, true})
            {
                const munch::verifier::Marked marked{
                        .byte = static_cast<unsigned char>(symbol),
                        .boundary_after = boundary_after};

                const auto expected{verifier.step(state, marked)};

                const auto stepped{table.step(at, static_cast<matisse::pretok::Class_t>(symbol_class), boundary_after)};

                const auto as_optional{
                        stepped == Verifier_table::no_state ? std::nullopt : std::optional<std::size_t>{stepped}};

                EXPECT_EQ(as_optional, expected) << state << " " << symbol;

                if (expected)
                {
                    moves = true;
                }
            }
        }

        EXPECT_EQ(table.accepts(at), verifier.accepts(state)) << state;
        EXPECT_EQ(table.is_dead_end(at), verifier.accepts(state) && !moves) << state;

        if (table.is_dead_end(at))
        {
            ++dead_ends;
        }
    }

    return dead_ends;
}

} // namespace

TEST(Verifier_table_test, The_table_steps_accepts_and_ends_as_munch_s_verifier_on_every_state_class_and_mark)
{
    EXPECT_EQ(expect_as_verifier({.alphabet = alphabet, .alternatives = gpt2_shaped}), 0);
    EXPECT_GT(expect_as_verifier({.alphabet = alphabet, .alternatives = bounded}), 0);
}
