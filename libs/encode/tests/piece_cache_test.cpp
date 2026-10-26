#include "matisse/encode/piece_cache.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <munch/regex/regex.hpp>

#include "matisse/encode/encoder.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::encode;
using matisse::load::Definition;
using matisse::load::Merge_rule;
using matisse::load::Token_id_t;
using matisse::pretok::Alphabet;
using matisse::pretok::Alternative;
using matisse::pretok::Pattern;
using matisse::validate::Inventory;
using munch::regex::text;

namespace
{
/**
 * @brief An alphabet of one class, every code point in it.
 */
constexpr Alphabet one_symbol{.symbols = "a", .sets = {}, .other = 'a'};

/**
 * @brief A pattern over the one class whose every piece is one code point.
 */
const std::vector<Alternative> single_alternatives{
        {.language = text("a"), .gate = "a"},
};

/**
 * @brief The single-symbol pattern over the one class.
 */
const Pattern single{.alphabet = one_symbol, .alternatives = single_alternatives};

/**
 * @brief Returns a definition of the single bytes alone.
 * @return The definition.
 */
Definition byte_definition()
{
    Definition definition{.merge_rule = Merge_rule::ranks, .whole_piece_lookup = true};

    for (unsigned value{0}; value < 256; ++value)
    {
        definition.spellings.emplace_back(1, static_cast<char>(value));
    }

    return definition;
}

/**
 * @brief Returns the encoder the tests' caches are made for, the single bytes under the single-symbol pattern.
 * @return The encoder.
 */
const Encoder& test_encoder()
{
    static const auto report{matisse::validate::validate(single, 2)};

    static const Encoder encoder{byte_definition(), report, Inventory::factorization, {}};

    return encoder;
}

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
    Piece_cache cache{test_encoder(), 4};

    EXPECT_FALSE(held(cache, "ab").has_value());

    cache.store("ab", {3, 4});

    EXPECT_EQ(held(cache, "ab"), (std::vector<Token_id_t>{3, 4}));
}

TEST(Piece_cache_test, A_piece_is_compared_whole_so_its_prefixes_and_extensions_are_not_found)
{
    Piece_cache cache{test_encoder(), 4};

    cache.store("abc", {7});

    EXPECT_FALSE(held(cache, "ab").has_value());
    EXPECT_FALSE(held(cache, "abcd").has_value());
    EXPECT_FALSE(held(cache, "abd").has_value());
    EXPECT_EQ(held(cache, "abc"), std::vector<Token_id_t>{7});
}

TEST(Piece_cache_test, A_cache_filled_to_its_bound_stops_filling_and_keeps_what_it_holds)
{
    Piece_cache cache{test_encoder(), 2};

    cache.store("a", {1});
    cache.store("b", {2});
    cache.store("c", {3});

    EXPECT_EQ(held(cache, "a"), std::vector<Token_id_t>{1});
    EXPECT_EQ(held(cache, "b"), std::vector<Token_id_t>{2});
    EXPECT_FALSE(held(cache, "c").has_value());
}

TEST(Piece_cache_test, A_piece_longer_than_the_longest_cached_is_not_stored)
{
    Piece_cache cache{test_encoder(), 4};

    const std::string longest(longest_cached_piece, 'a');

    const std::string longer(longest_cached_piece + 1, 'a');

    cache.store(longest, {1});
    cache.store(longer, {2});

    EXPECT_EQ(held(cache, longest), std::vector<Token_id_t>{1});
    EXPECT_FALSE(held(cache, longer).has_value());
}

TEST(Piece_cache_test, Every_stored_piece_is_found_with_its_ids_as_the_table_grows)
{
    // 1,000 pieces take the slots from 64 to 2,048 through five doublings.
    constexpr std::size_t pieces{1000};

    Piece_cache cache{test_encoder(), pieces};

    for (std::size_t index{0}; index < pieces; ++index)
    {
        const auto piece{std::to_string(index)};

        const auto id{static_cast<Token_id_t>(index)};

        cache.store(piece, {id, id + 1});
    }

    for (std::size_t index{0}; index < pieces; ++index)
    {
        const auto piece{std::to_string(index)};

        const auto id{static_cast<Token_id_t>(index)};

        EXPECT_EQ(held(cache, piece), (std::vector<Token_id_t>{id, id + 1})) << piece;
    }

    EXPECT_FALSE(held(cache, "1000").has_value());
}
