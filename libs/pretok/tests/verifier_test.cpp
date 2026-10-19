#include "matisse/pretok/verifier.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "matisse/pretok/segmenter.hpp"

using namespace matisse::pretok;

namespace
{
/**
 * @brief A boundary marking, one bit per symbol, set when a piece ends after it.
 */
using Marking_t = std::vector<bool>;

/**
 * @brief The longest class string the tests enumerate.
 */
constexpr std::size_t longest_text{7};

/**
 * @brief An alphabet of two classes, `a` a literal and `b` everything else.
 */
constexpr Alphabet two_symbols{.symbols = "ab", .literals = "a", .caseless = false, .sets = {}, .other = 'b'};

/**
 * @brief A pattern whose first alternative passes only before another `a`, as `\s+(?!\S)` passes only before white
 *        space, so that a run of `a` before a `b` leaves its last `a` to the second alternative.
 */
constexpr std::array<Alternative, 3> gated_alternatives{{
        {.language = "a+", .gate = "a"},
        {.language = "a+", .gate = "ab"},
        {.language = "b", .gate = "ab"},
}};

/**
 * @brief The gated pattern over the two classes.
 */
constexpr Pattern gated{.alphabet = two_symbols, .alternatives = gated_alternatives};

/**
 * @brief A pattern whose first alternative passes only at the end of the input, as `\s++$` does, and whose earlier
 *        alternative takes a word a later one would extend, so that the order of the alternatives decides the pieces.
 */
constexpr std::array<Alternative, 4> anchored_alternatives{{
        {.language = "a+", .gate = ""},
        {.language = "ab", .gate = "ab"},
        {.language = "a", .gate = "ab"},
        {.language = "b+", .gate = "ab"},
}};

/**
 * @brief The anchored pattern over the two classes.
 */
constexpr Pattern anchored{.alphabet = two_symbols, .alternatives = anchored_alternatives};

/**
 * @brief Returns whether a verifier accepts a class string under a marking.
 * @param verifier The verifier.
 * @param text The class string.
 * @param marking The marking, one bit per symbol.
 * @return True when the run from the start ends accepting.
 */
bool accepts(const munch::verifier::Verifier& verifier, const std::string_view text, const Marking_t& marking)
{
    auto state{std::optional{verifier.start()}};

    for (std::size_t index{0}; index < text.size() && state; ++index)
    {
        const munch::verifier::Marked symbol{
                .byte = static_cast<unsigned char>(text[index]),
                .boundary_after = marking[index]};

        state = verifier.step(*state, symbol);
    }

    return state && verifier.accepts(*state);
}

/**
 * @brief Returns every string over a pattern's class symbols of lengths zero to longest_text, shorter first.
 * @param symbols The class symbols.
 * @return The strings.
 */
std::vector<std::string> texts(const std::string_view symbols)
{
    std::vector<std::string> all{{}};

    for (std::size_t from{0}; from < all.size(); ++from)
    {
        if (all[from].size() == longest_text)
        {
            continue;
        }

        for (const auto symbol : symbols)
        {
            all.push_back(all[from] + symbol);
        }
    }

    return all;
}

/**
 * @brief Returns the marking of a segmentation of a class string: a bit after every piece but the last.
 * @param ends The piece ends.
 * @param length The length of the class string.
 * @return The marking.
 */
Marking_t marking_of(const std::vector<std::size_t>& ends, const std::size_t length)
{
    Marking_t marking(length, false);

    for (const auto end : ends)
    {
        if (end < length)
        {
            marking[end - 1] = true;
        }
    }

    return marking;
}

/**
 * @brief Expects a pattern's verifier to accept the segmenter's marking of every class string the tests enumerate and
 *        no other marking of it.
 * @param pattern The pattern.
 */
void expect_segmentation(const Pattern& pattern)
{
    const auto verifier{split_verifier(pattern)};

    const Segmenter segmenter{pattern};

    for (const auto& text : texts(pattern.alphabet.symbols))
    {
        const auto ends{segmenter.boundaries(text)};

        const auto segmented{marking_of(ends, text.size())};

        const auto marking_count{text.empty() ? 1UZ : 1UZ << (text.size() - 1)};

        for (std::size_t bits{0}; bits < marking_count; ++bits)
        {
            Marking_t marking(text.size(), false);

            for (std::size_t index{0}; index + 1 < text.size(); ++index)
            {
                marking[index] = ((bits >> index) & 1U) != 0;
            }

            EXPECT_EQ(accepts(verifier, text, marking), marking == segmented) << text;
        }
    }
}

} // namespace

TEST(Verifier_test, A_gate_passes_an_accept_only_before_one_of_its_symbols)
{
    const auto verifier{split_verifier(gated)};

    EXPECT_TRUE(accepts(verifier, "aab", {true, true, false}));
    EXPECT_FALSE(accepts(verifier, "aab", {false, true, false}));
    EXPECT_TRUE(accepts(verifier, "aa", {false, false}));
}

TEST(Verifier_test, The_verifier_accepts_the_segmenter_s_marking_of_every_string_and_no_other)
{
    expect_segmentation(gated);
    expect_segmentation(anchored);
}

TEST(Verifier_test, The_factorization_lexer_certifies_a_window_whose_covering_word_begins_at_one_origin)
{
    const auto lexer{factorization_lexer(anchored)};

    // No word holds an a after a b, so the word covering an a after a b begins at that a.
    EXPECT_EQ(lexer.is_split_window("ba"), std::optional<std::size_t>{1});
    EXPECT_EQ(lexer.is_split_window("bba"), std::optional<std::size_t>{2});
    EXPECT_FALSE(lexer.is_split_window("a").has_value());
    EXPECT_FALSE(lexer.is_split_window("ab").has_value());
}
