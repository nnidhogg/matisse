#include "matisse/encode/join_table.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <random>

using namespace matisse::encode;
using matisse::load::Token_id_t;

namespace
{
/**
 * @brief Returns the join a map holds for a pair, as the table is to return it.
 * @param joins The map.
 * @param left The left part's id.
 * @param right The right part's id.
 * @return The join, std::nullopt when the map holds none for the pair.
 */
std::optional<Join_table::Join> held(const Join_table::Joins_t& joins, const Token_id_t left, const Token_id_t right)
{
    const auto found{joins.find(Join_table::key_of(left, right))};

    if (found == joins.end())
    {
        return std::nullopt;
    }

    return found->second;
}

} // namespace

TEST(Join_table_test, A_table_finds_the_join_of_every_pair_its_map_holds_and_of_no_pair_it_does_not)
{
    // A third of the pairs of 300 ids, drawn at random, so that homes collide and walks run past each other; every pair
    // of the 90,000 is then looked up.
    std::mt19937 random{1};

    std::bernoulli_distribution drawn{1.0 / 3.0};

    std::uniform_int_distribution<std::uint32_t> rank{0, 1000000};

    Join_table::Joins_t joins{};

    constexpr Token_id_t ids{300};

    for (Token_id_t left{0}; left < ids; ++left)
    {
        for (Token_id_t right{0}; right < ids; ++right)
        {
            if (!drawn(random))
            {
                continue;
            }

            const auto product{static_cast<Token_id_t>(left * ids + right)};

            joins.emplace(Join_table::key_of(left, right), Join_table::Join{.rank = rank(random), .product = product});
        }
    }

    const Join_table table{joins};

    std::size_t found{0};

    for (Token_id_t left{0}; left < ids; ++left)
    {
        for (Token_id_t right{0}; right < ids; ++right)
        {
            const auto answer{table.find(left, right)};

            EXPECT_EQ(answer, held(joins, left, right)) << left << " " << right;

            found += answer ? 1UZ : 0UZ;
        }
    }

    EXPECT_EQ(found, joins.size());
}

TEST(Join_table_test, The_largest_ids_short_of_all_ones_are_pairs_like_any_other)
{
    constexpr Token_id_t largest{0xFFFFFFFEU};

    const Join_table::Joins_t joins{
            {Join_table::key_of(largest, largest), {.rank = 1, .product = 2}},
            {Join_table::key_of(0, largest), {.rank = 3, .product = 4}},
            {Join_table::key_of(largest, 0), {.rank = 5, .product = 6}},
    };

    const Join_table table{joins};

    for (const auto left : {Token_id_t{0}, largest})
    {
        for (const auto right : {Token_id_t{0}, largest})
        {
            EXPECT_EQ(table.find(left, right), held(joins, left, right)) << left << " " << right;
        }
    }
}

TEST(Join_table_test, A_table_of_no_joins_finds_no_pair)
{
    const Join_table table{{}};

    EXPECT_FALSE(table.find(0, 0).has_value());
    EXPECT_FALSE(table.find(97, 98).has_value());
}
