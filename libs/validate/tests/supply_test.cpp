#include "matisse/validate/supply.hpp"

#include <array>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <munch/regex/regex.hpp>
#include <munch/regex/utf8.hpp>

#include "matisse/validate/report.hpp"

using namespace matisse::validate;
using matisse::pretok::Alphabet;
using matisse::pretok::Alternative;
using matisse::pretok::Code_point_set;
using matisse::pretok::Pattern;
using munch::regex::plus;
using munch::regex::text;
using munch::regex::utf8::Code_point_range;

namespace
{
/**
 * @brief The code point of the literal `a`.
 */
constexpr std::array<Code_point_range, 1> a_points{{{.first = U'a', .last = U'a'}}};

/**
 * @brief The set of the literal `a`, the one set the two classes read.
 */
constexpr std::array<Code_point_set, 1> a_set{{{.symbol = 'a', .ranges = a_points}}};

/**
 * @brief An alphabet of two classes, `a` a literal and `b` everything else.
 */
constexpr Alphabet two_symbols{.symbols = "ab", .sets = a_set, .other = 'b'};

/**
 * @brief A pattern whose every piece is one symbol, so that every window is certified at its last symbol and every
 *        interior code point of a text is a boundary and an anchor.
 */
const std::vector<Alternative> single_alternatives{
        {.language = text("a"), .gate = "ab"},
        {.language = text("b"), .gate = "ab"},
};

/**
 * @brief The single-symbol pattern over the two classes.
 */
const Pattern single{.alphabet = two_symbols, .alternatives = single_alternatives};

/**
 * @brief A pattern whose first alternative passes only before another `a`, so that a run of `a` before a `b` leaves its
 *        last `a` to the second alternative: the boundary before the last `a` of the run is certified by no window,
 *        since a run of `a` at the end of the text is one piece, and the boundary before the `b` is.
 */
const std::vector<Alternative> gated_alternatives{
        {.language = plus(text("a")), .gate = "a"},
        {.language = plus(text("a")), .gate = "ab"},
        {.language = text("b"), .gate = "ab"},
};

/**
 * @brief The gated pattern over the two classes.
 */
const Pattern gated{.alphabet = two_symbols, .alternatives = gated_alternatives};

} // namespace

TEST(Supply_test, Every_interior_code_point_of_one_symbol_pieces_is_an_anchor_at_its_byte_offset)
{
    // "a\u00E9 b" is four code points in five bytes, every one its own piece, so the anchors are the three interior
    // code points, bytes 1, 3 and 4, however many windows place each: 1024 * 3 / 5 = 614.4 per KiB, the gaps 2 and 1,
    // whose median by the floor of q n is the larger, every boundary an anchor and the e acute the longest piece.
    const auto report{validate(single)};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto [bytes, anchors, boundary_share, longest_piece]{supply(report, inventory, "a\u00E9 b")};

        EXPECT_EQ(bytes, 5U);
        EXPECT_EQ(anchors.count, 3U);
        EXPECT_DOUBLE_EQ(anchors.per_kibibyte, 1024.0 * 3 / 5);
        ASSERT_TRUE(anchors.gaps.has_value());
        EXPECT_EQ(anchors.gaps->median, 2U);
        EXPECT_EQ(anchors.gaps->ninetieth, 2U);
        EXPECT_EQ(anchors.gaps->ninety_ninth, 2U);
        EXPECT_EQ(anchors.gaps->longest, 2U);
        EXPECT_DOUBLE_EQ(boundary_share, 1.0);
        EXPECT_EQ(longest_piece, 2U);
    }
}

TEST(Supply_test, Percentiles_are_the_order_statistic_at_the_floor_of_q_n)
{
    // Code points of one, two, three and four bytes then one more, "a\u00E9\u65E5\U0001F600a": the anchors stand at
    // bytes 1, 3, 6 and 10, so the three gaps ascending are 2, 3 and 4. floor(0.5 * 3) = 1 gives 3, floor(0.9 * 3) = 2
    // gives 4, floor(0.99 * 3) = 2 gives 4, the maximum 4; the longest piece is the four-byte code point.
    const auto [bytes, anchors, boundary_share, longest_piece]{
            supply(validate(single), Inventory::policy, "a\u00E9\u65E5\U0001F600a")};

    EXPECT_EQ(bytes, 11U);
    EXPECT_EQ(anchors.count, 4U);
    ASSERT_TRUE(anchors.gaps.has_value());
    EXPECT_EQ(anchors.gaps->median, 3U);
    EXPECT_EQ(anchors.gaps->ninetieth, 4U);
    EXPECT_EQ(anchors.gaps->ninety_ninth, 4U);
    EXPECT_EQ(anchors.gaps->longest, 4U);
    EXPECT_EQ(longest_piece, 4U);
}

TEST(Supply_test, A_boundary_no_window_certifies_is_counted_in_the_share_and_not_among_the_anchors)
{
    // "aab" segments as a, a, b: the boundary before the b is the origin of "ab" at 1 and of "aab" at 2, one anchor
    // however many windows place it, and the boundary between the two a is certified by no window, since "aa" at the
    // end of a text is one piece. One anchor of two boundaries, 1024 / 3 = 341.3 per KiB, and no gap.
    const auto report{validate(gated)};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto [bytes, anchors, boundary_share, longest_piece]{supply(report, inventory, "aab")};

        EXPECT_EQ(bytes, 3U);
        EXPECT_EQ(anchors.count, 1U);
        EXPECT_DOUBLE_EQ(anchors.per_kibibyte, 1024.0 / 3);
        EXPECT_FALSE(anchors.gaps.has_value());
        EXPECT_DOUBLE_EQ(boundary_share, 0.5);
        EXPECT_EQ(longest_piece, 1U);
    }

    // A run of a ending the text is one piece, so it has neither a boundary nor an anchor.
    const auto [bytes, anchors, boundary_share, longest_piece]{supply(report, Inventory::policy, "aaaa")};

    EXPECT_EQ(anchors.count, 0U);
    EXPECT_DOUBLE_EQ(boundary_share, 0.0);
    EXPECT_EQ(longest_piece, 4U);
}

TEST(Supply_test, The_ends_of_the_input_are_no_anchors_and_an_input_without_boundaries_shares_none)
{
    const auto report{validate(single)};

    const auto [empty_bytes, empty_anchors, empty_share, empty_piece]{supply(report, Inventory::factorization, "")};

    EXPECT_EQ(empty_bytes, 0U);
    EXPECT_EQ(empty_anchors.count, 0U);
    EXPECT_DOUBLE_EQ(empty_anchors.per_kibibyte, 0.0);
    EXPECT_FALSE(empty_anchors.gaps.has_value());
    EXPECT_DOUBLE_EQ(empty_share, 0.0);
    EXPECT_EQ(empty_piece, 0U);

    // One code point is one piece, with no interior position to cut at.
    const auto [one_bytes, one_anchors, one_share, one_piece]{supply(report, Inventory::factorization, "\u00E9")};

    EXPECT_EQ(one_bytes, 2U);
    EXPECT_EQ(one_anchors.count, 0U);
    EXPECT_DOUBLE_EQ(one_share, 0.0);
    EXPECT_EQ(one_piece, 2U);
}

TEST(Supply_test, The_section_and_the_object_carry_the_same_figures)
{
    // The input's length and longest piece are the factorization supply's, both being measured on one input.
    const Supply factorization{
            .bytes = 3,
            .anchors = {.count = 1, .per_kibibyte = 1024.0 / 3, .gaps = std::nullopt},
            .boundary_share = 0.5,
            .longest_piece = 1};

    const Gaps spread{.median = 1, .ninetieth = 2, .ninety_ninth = 3, .longest = 4};

    const Supply policy{
            .bytes = 3,
            .anchors = {.count = 2, .per_kibibyte = 2048.0 / 3, .gaps = spread},
            .boundary_share = 1.0,
            .longest_piece = 1};

    EXPECT_EQ(
            supply_section(factorization, policy, "in.txt"),
            "\ncertified-anchor supply on in.txt, 3 bytes\n"
            "  longest piece              1 byte, the floor no cut rule goes below\n"
            "  factorization              1 anchor, 341.3 per KiB, 50.00% of the pre-token boundaries, no gaps, fewer "
            "than two anchors\n"
            "  policy                     2 anchors, 682.7 per KiB, 100.00% of the pre-token boundaries, gaps p50 1, "
            "p90 2, p99 3, max 4\n");

    EXPECT_EQ(
            supply_json(factorization, policy, "in.txt"),
            R"({"input": "in.txt", "bytes": 3, "longest_piece": 1, )"
            R"("factorization": {"anchors": 1, "per_kibibyte": 341.3, "boundary_share": 0.5000, "gap_p50": null, )"
            R"("gap_p90": null, "gap_p99": null, "gap_max": null}, )"
            R"("policy": {"anchors": 2, "per_kibibyte": 682.7, "boundary_share": 1.0000, "gap_p50": 1, "gap_p90": 2, )"
            R"("gap_p99": 3, "gap_max": 4}})");
}

TEST(Supply_test, An_input_that_is_not_utf8_is_refused_as_classes_of_refuses_it)
{
    const auto report{validate(single)};

    EXPECT_THROW(std::ignore = supply(report, Inventory::policy, "a\xFF"), std::invalid_argument);
}
