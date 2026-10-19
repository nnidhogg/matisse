#include "matisse/validate/supply.hpp"

#include <cstddef>
#include <gtest/gtest.h>

#include "matisse/gpt2/tokenizer.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::validate;

namespace
{
/**
 * @brief The widest window the tests decide: the windows of width two already certify every boundary of their texts,
 *        and they decide in a fraction of the time the default width takes.
 */
constexpr std::size_t narrow_width{2};

} // namespace

TEST(Supply_test, The_published_hello_world_is_cut_where_its_second_piece_begins)
{
    // GPT-2's pattern splits "Hello world" into "Hello" and " world", the space opening the second piece, which a
    // letter followed by a space certifies under both certificates: one anchor at byte 5, the one boundary, 1024 / 11 =
    // 93.1 per KiB and no gap. The longest piece is " world".
    const auto report{validate(matisse::gpt2::pattern(), narrow_width)};

    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto [bytes, anchors, boundary_share, longest_piece]{
                supply(report, inventory, matisse::gpt2::pattern(), "Hello world")};

        EXPECT_EQ(bytes, 11U);
        EXPECT_EQ(anchors.count, 1U);
        EXPECT_DOUBLE_EQ(anchors.per_kibibyte, 1024.0 / 11);
        EXPECT_FALSE(anchors.gaps.has_value());
        EXPECT_DOUBLE_EQ(boundary_share, 1.0);
        EXPECT_EQ(longest_piece, 6U);
    }
}

TEST(Supply_test, An_anchor_after_letters_beyond_ascii_stands_at_their_byte_offset)
{
    // "h\u00E9llo w\u00F6rld" is "h\u00E9llo" and " w\u00F6rld", six and seven bytes, eleven code points: the cut
    // certified at the sixth code point stands at byte 6, and the gap and the share are as for its ASCII twin.
    const auto report{validate(matisse::gpt2::pattern(), narrow_width)};

    const auto [bytes, anchors, boundary_share, longest_piece]{
            supply(report, Inventory::policy, matisse::gpt2::pattern(), "h\u00E9llo w\u00F6rld")};

    EXPECT_EQ(bytes, 13U);
    EXPECT_EQ(anchors.count, 1U);
    EXPECT_DOUBLE_EQ(anchors.per_kibibyte, 1024.0 / 13);
    EXPECT_DOUBLE_EQ(boundary_share, 1.0);
    EXPECT_EQ(longest_piece, 7U);
}

TEST(Supply_test, A_cut_only_a_window_of_the_default_width_certifies_is_an_anchor)
{
    // "\t'dog" is "\t", "'d" and "og": the policy certificate certifies the cut before the o by "\t'dA" at 3 and by no
    // shorter window, a space before the apostrophe joining it as ` ?[^\s\p{L}\p{N}]+`; under the factorization
    // certificate the apostrophe may be a piece of its own, so only the cut after the tab is certified.
    const auto report{validate(matisse::gpt2::pattern())};

    const auto [policy_bytes, policy_anchors, policy_share, policy_piece]{
            supply(report, Inventory::policy, matisse::gpt2::pattern(), "\t'dog")};

    EXPECT_EQ(policy_anchors.count, 2U);
    ASSERT_TRUE(policy_anchors.gaps.has_value());
    EXPECT_EQ(policy_anchors.gaps->longest, 2U);
    EXPECT_DOUBLE_EQ(policy_share, 1.0);

    const auto [factorization_bytes, factorization_anchors, factorization_share, factorization_piece]{
            supply(report, Inventory::factorization, matisse::gpt2::pattern(), "\t'dog")};

    EXPECT_EQ(factorization_anchors.count, 1U);
    EXPECT_DOUBLE_EQ(factorization_share, 0.5);
}
