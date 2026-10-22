#include "matisse/encode/token_trie.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace matisse::encode;
using matisse::load::Token_id_t;

namespace
{
/**
 * @brief Returns a random string over an alphabet, of a length drawn between two bounds.
 * @param random The generator.
 * @param alphabet The symbols.
 * @param shortest The shortest length drawn.
 * @param longest The longest length drawn.
 * @return The string.
 */
std::string random_string(
        std::mt19937& random, const std::string_view alphabet, const std::size_t shortest, const std::size_t longest)
{
    std::uniform_int_distribution<std::size_t> length{shortest, longest};

    std::uniform_int_distribution<std::size_t> symbol{0, alphabet.size() - 1};

    std::string out{};

    for (auto left{length(random)}; left > 0; --left)
    {
        out += alphabet[symbol(random)];
    }

    return out;
}

/**
 * @brief Returns the longest of a set of spellings that begins a text, found by trying every prefix.
 * @param held The id of every spelling held.
 * @param text The text.
 * @param excluded_from The least length not tried, the text's length plus one to try every prefix.
 * @return The id and length of the longest held prefix, std::nullopt when none is held.
 */
std::optional<Token_trie::Match> brute_longest(
        const std::unordered_map<std::string, Token_id_t>& held, const std::string_view text,
        const std::size_t excluded_from)
{
    for (auto length{std::min(text.size(), excluded_from - 1)}; length > 0; --length)
    {
        const auto found{held.find(std::string{text.substr(0, length)})};

        if (found != held.end())
        {
            return Token_trie::Match{.id = found->second, .length = static_cast<std::uint32_t>(length)};
        }
    }

    return std::nullopt;
}

} // namespace

TEST(Token_trie_test, A_trie_gives_the_longest_token_it_holds_that_begins_a_text_and_each_token_s_shorter_one)
{
    // 3,000 random spellings of one to eight symbols over three, the high byte among them, of which a random half is
    // held; 20,000 random texts of up to ten symbols are walked, and every held token's shorter token is asked.
    constexpr std::string_view alphabet{"ab\xFF"};

    std::mt19937 random{1};

    std::bernoulli_distribution drawn{0.5};

    std::vector<std::string> spellings{};

    std::set<std::string> seen{};

    std::vector<Token_id_t> ids{};

    std::unordered_map<std::string, Token_id_t> held{};

    for (std::size_t count{0}; count < 3000; ++count)
    {
        auto spelling{random_string(random, alphabet, 1, 8)};

        const auto id{static_cast<Token_id_t>(spellings.size())};

        const auto first{seen.insert(spelling).second};

        if (first && drawn(random))
        {
            ids.push_back(id);

            held.emplace(spelling, id);
        }

        spellings.push_back(std::move(spelling));
    }

    const Token_trie trie{spellings, ids};

    for (std::size_t count{0}; count < 20000; ++count)
    {
        const auto text{random_string(random, alphabet, 0, 10)};

        const auto expected{brute_longest(held, text, text.size() + 1)};

        const auto [id, length]{trie.longest(text)};

        EXPECT_EQ(length, expected ? expected->length : 0U) << text;

        if (expected)
        {
            EXPECT_EQ(id, expected->id) << text;
        }
    }

    for (const auto id : ids)
    {
        const auto& spelling{spellings[id]};

        const auto expected{brute_longest(held, spelling, spelling.size())};

        const auto shorter{trie.shorter(id)};

        EXPECT_EQ(trie.length(id), spelling.size()) << spelling;
        ASSERT_EQ(shorter.has_value(), expected.has_value()) << spelling;

        if (expected)
        {
            EXPECT_EQ(shorter->id, expected->id) << spelling;
            EXPECT_EQ(shorter->length, expected->length) << spelling;
        }
    }
}

TEST(Token_trie_test, The_tokens_beginning_a_text_are_its_longest_and_then_each_one_s_shorter_one_in_turn)
{
    // ab is not held, so abc's shorter token is a; c is no token, so a text beginning with it has none.
    const std::vector<std::string> spellings{"a", "ab", "abc", "b", "abcd"};

    const std::vector<Token_id_t> ids{0, 2, 3, 4};

    const Token_trie trie{spellings, ids};

    const auto [id, length]{trie.longest("abcdx")};

    EXPECT_EQ(id, Token_id_t{4});
    EXPECT_EQ(length, 4U);
    EXPECT_EQ(trie.shorter(4)->id, Token_id_t{2});
    EXPECT_EQ(trie.shorter(2)->id, Token_id_t{0});
    EXPECT_EQ(trie.shorter(2)->length, 1U);
    EXPECT_FALSE(trie.shorter(0).has_value());
    EXPECT_EQ(trie.longest("abx").id, Token_id_t{0});
    EXPECT_EQ(trie.longest("cab").length, 0U);
    EXPECT_EQ(trie.longest("").length, 0U);
}

TEST(Token_trie_test, A_trie_of_every_byte_gives_each_byte_and_nothing_longer)
{
    std::vector<std::string> spellings{};

    std::vector<Token_id_t> ids{};

    for (unsigned value{0}; value < 256; ++value)
    {
        ids.push_back(static_cast<Token_id_t>(spellings.size()));

        spellings.emplace_back(1, static_cast<char>(value));
    }

    const Token_trie trie{spellings, ids};

    for (unsigned value{0}; value < 256; ++value)
    {
        const std::string text{static_cast<char>(value), static_cast<char>(255 - value)};

        const auto [id, length]{trie.longest(text)};

        EXPECT_EQ(id, Token_id_t{value});
        EXPECT_EQ(length, 1U);
        EXPECT_FALSE(trie.shorter(value).has_value());
    }
}
