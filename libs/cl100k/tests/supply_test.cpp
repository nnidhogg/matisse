#include "matisse/validate/supply.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <string_view>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::validate;

namespace
{
/**
 * @brief Returns cl100k's report at the default width, decided once for every test of this file.
 * @return The report.
 */
const Report& decided_report()
{
    static const Report report{validate(matisse::cl100k::pattern())};

    return report;
}

} // namespace

TEST(Supply_test, The_published_tiktoken_example_is_cut_at_each_of_its_boundaries)
{
    // "tiktoken is great!" is "tiktoken", " is", " great" and "!": a letter before a space or punctuation certifies the
    // cut under both certificates, so the anchors are the boundaries, bytes 8, 11 and 17, 170.7 per KiB, with the gaps
    // 3 and 6, whose median by the floor of q n is the larger, and "tiktoken" is the longest piece.
    const auto& report{decided_report()};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto [bytes, anchors, boundary_share, longest_piece]{
                supply(report, inventory, matisse::cl100k::pattern(), "tiktoken is great!")};

        EXPECT_EQ(bytes, 18U);
        EXPECT_EQ(anchors.count, 3U);
        EXPECT_DOUBLE_EQ(anchors.per_kibibyte, 1024.0 * 3 / 18);
        ASSERT_TRUE(anchors.gaps.has_value());
        EXPECT_EQ(anchors.gaps->median, 6U);
        EXPECT_EQ(anchors.gaps->longest, 6U);
        EXPECT_DOUBLE_EQ(boundary_share, 1.0);
        EXPECT_EQ(longest_piece, 8U);
    }
}

TEST(Supply_test, A_window_certified_at_its_first_symbol_places_no_anchor_at_the_start_of_the_input)
{
    // The policy certificate certifies a space followed by a letter at origin 0, where the space opens a piece; at the
    // start of " world" that origin is the input's own start, which is no cut, and the one piece has no boundary.
    const auto& report{decided_report()};

    const auto opening{std::ranges::find(report.policy.windows, std::string_view{" A"}, &Certified_window::window)};

    ASSERT_NE(opening, report.policy.windows.end());
    EXPECT_EQ(opening->origin, 0U);

    const auto [bytes, anchors, boundary_share, longest_piece]{
            supply(report, Inventory::policy, matisse::cl100k::pattern(), " world")};

    EXPECT_EQ(anchors.count, 0U);
    EXPECT_DOUBLE_EQ(boundary_share, 0.0);
    EXPECT_EQ(longest_piece, 6U);
}
