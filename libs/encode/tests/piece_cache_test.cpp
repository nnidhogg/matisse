#include "matisse/encode/piece_cache.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace matisse::encode;
using matisse::load::Token_id_t;

namespace
{
/**
 * @brief Returns the ids a cache holds for a piece, as a vector.
 * @param cache The cache.
 * @param piece The piece.
 * @return The ids, std::nullopt when the piece is not stored.
 */
std::optional<std::vector<Token_id_t>> held(const Piece_cache& cache, const std::string_view piece)
{
    const auto found{cache.find(piece)};

    if (!found)
    {
        return std::nullopt;
    }

    return std::vector<Token_id_t>{found->begin(), found->end()};
}

} // namespace

TEST(Piece_cache_test, A_stored_piece_is_found_with_its_ids_and_no_other_piece_is)
{
    Piece_cache cache{4};

    EXPECT_FALSE(held(cache, "ab").has_value());

    cache.store("ab", {3, 4});

    EXPECT_EQ(held(cache, "ab"), (std::vector<Token_id_t>{3, 4}));
}

TEST(Piece_cache_test, A_piece_is_compared_whole_so_its_prefixes_and_extensions_are_not_found)
{
    Piece_cache cache{4};

    cache.store("abc", {7});

    EXPECT_FALSE(held(cache, "ab").has_value());
    EXPECT_FALSE(held(cache, "abcd").has_value());
    EXPECT_FALSE(held(cache, "abd").has_value());
    EXPECT_EQ(held(cache, "abc"), std::vector<Token_id_t>{7});
}

TEST(Piece_cache_test, A_cache_filled_to_its_bound_stops_filling_and_keeps_what_it_holds)
{
    Piece_cache cache{2};

    cache.store("a", {1});
    cache.store("b", {2});
    cache.store("c", {3});

    EXPECT_EQ(held(cache, "a"), std::vector<Token_id_t>{1});
    EXPECT_EQ(held(cache, "b"), std::vector<Token_id_t>{2});
    EXPECT_FALSE(held(cache, "c").has_value());
}

TEST(Piece_cache_test, A_piece_longer_than_the_longest_cached_is_not_stored)
{
    Piece_cache cache{4};

    const std::string longest(longest_cached_piece, 'a');

    const std::string longer(longest_cached_piece + 1, 'a');

    cache.store(longest, {1});
    cache.store(longer, {2});

    EXPECT_EQ(held(cache, longest), std::vector<Token_id_t>{1});
    EXPECT_FALSE(held(cache, longer).has_value());
}
