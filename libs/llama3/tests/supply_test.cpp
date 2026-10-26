#include "matisse/validate/supply.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <string_view>

#include "matisse/llama3/tokenizer.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::validate;

namespace
{
/**
 * @brief Returns Llama 3's report at the default width, decided once for every test of this file.
 * @return The report.
 */
const Report& decided_report()
{
    static const Report report{validate(matisse::llama3::pattern())};

    return report;
}

} // namespace

TEST(Supply_test, The_published_llama3_example_is_cut_at_each_of_its_boundaries)
{
    // "This is a test sentence." is "This", " is", " a", " test", " sentence" and ".": a letter before a space or
    // punctuation certifies the cut under both certificates, so the anchors are the boundaries, bytes 4, 7, 9, 14 and
    // 23, 213.3 per KiB, with the gaps 3, 2, 5 and 9, whose median by the floor of q n is the third ascending, 5, and
    // the longest piece is the space and the word sentence, nine bytes.
    const auto& report{decided_report()};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto [bytes, anchors, boundary_share, longest_piece]{
                supply(report, inventory, "This is a test sentence.")};

        EXPECT_EQ(bytes, 24U);
        EXPECT_EQ(anchors.count, 5U);
        EXPECT_DOUBLE_EQ(anchors.per_kibibyte, 1024.0 * 5 / 24);
        ASSERT_TRUE(anchors.gaps.has_value());
        EXPECT_EQ(anchors.gaps->median, 5U);
        EXPECT_EQ(anchors.gaps->longest, 9U);
        EXPECT_DOUBLE_EQ(boundary_share, 1.0);
        EXPECT_EQ(longest_piece, 9U);
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

    const auto [bytes, anchors, boundary_share, longest_piece]{supply(report, Inventory::policy, " world")};

    EXPECT_EQ(anchors.count, 0U);
    EXPECT_DOUBLE_EQ(boundary_share, 0.0);
    EXPECT_EQ(longest_piece, 6U);
}
