#include "matisse/encode/spelling_table.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace matisse::encode;
using matisse::load::Token_id_t;

namespace
{
/**
 * @brief Returns a random string over an alphabet, of a length drawn up to a bound.
 * @param random The generator.
 * @param alphabet The symbols.
 * @param longest The longest length drawn.
 * @return The string.
 */
std::string random_string(std::mt19937& random, const std::string_view alphabet, const std::size_t longest)
{
    std::uniform_int_distribution<std::size_t> length{0, longest};

    std::uniform_int_distribution<std::size_t> symbol{0, alphabet.size() - 1};

    std::string out{};

    for (auto left{length(random)}; left > 0; --left)
    {
        out += alphabet[symbol(random)];
    }

    return out;
}

} // namespace

TEST(Spelling_table_test, A_table_finds_the_first_id_of_every_spelling_it_holds_and_no_other_string)
{
    // 20,000 random spellings of up to twelve symbols over three, so that the short ones repeat and many share homes,
    // held to a map that keeps the first id of each; then 20,000 further random strings of up to fourteen are looked
    // up.
    constexpr std::string_view alphabet{"ab\xFF"};

    std::mt19937 random{1};

    std::vector<std::string> spellings{};

    std::unordered_map<std::string, Token_id_t> first_ids{};

    for (std::size_t id{0}; id < 20000; ++id)
    {
        auto spelling{random_string(random, alphabet, 12)};

        first_ids.emplace(spelling, static_cast<Token_id_t>(id));

        spellings.push_back(std::move(spelling));
    }

    const Spelling_table table{spellings};

    for (std::size_t count{0}; count < 20000; ++count)
    {
        const auto text{random_string(random, alphabet, 14)};

        const auto found{first_ids.find(text)};

        const auto expected{found == first_ids.end() ? std::nullopt : std::optional{found->second}};

        EXPECT_EQ(table.find(text), expected) << text;
    }

    for (const auto& [spelling, id] : first_ids)
    {
        EXPECT_EQ(table.find(spelling), id) << spelling;
    }
}

TEST(Spelling_table_test, A_spelling_is_compared_whole_so_its_prefixes_and_extensions_are_not_found)
{
    const std::vector<std::string> spellings{"abc", "x"};

    const Spelling_table table{spellings};

    EXPECT_EQ(table.find("abc"), Token_id_t{0});
    EXPECT_EQ(table.find("x"), Token_id_t{1});
    EXPECT_FALSE(table.find("ab").has_value());
    EXPECT_FALSE(table.find("abcd").has_value());
    EXPECT_FALSE(table.find("").has_value());
}

TEST(Spelling_table_test, A_table_of_no_spellings_finds_nothing)
{
    const Spelling_table table{std::vector<std::string>{}};

    EXPECT_FALSE(table.find("").has_value());
    EXPECT_FALSE(table.find("a").has_value());
}
