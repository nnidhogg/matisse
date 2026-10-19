#include "matisse/lookup/tokenizers.hpp"

#include <gtest/gtest.h>
#include <string_view>
#include <vector>

using namespace matisse::lookup;

TEST(Tokenizers_test, Every_tokenizer_is_listed_by_name_in_order)
{
    std::vector<std::string_view> names{};

    for (const auto& [name, pattern, specials] : tokenizers())
    {
        names.push_back(name);
    }

    EXPECT_EQ(names, (std::vector<std::string_view>{"gpt2", "cl100k"}));
}

TEST(Tokenizers_test, A_name_finds_its_tokenizer_s_pattern_and_special_tokens)
{
    const auto gpt2{tokenizer("gpt2")};
    const auto cl100k{tokenizer("cl100k")};

    ASSERT_TRUE(gpt2);
    ASSERT_TRUE(cl100k);
    EXPECT_EQ(gpt2->pattern.alphabet.symbols, "\t !'0Adelmrstv");
    EXPECT_EQ(gpt2->specials.size(), 1U);
    EXPECT_EQ(cl100k->pattern.alphabet.symbols, "\t\n\r !'0Adelmrstv");
    EXPECT_EQ(cl100k->specials.size(), 5U);
}

TEST(Tokenizers_test, A_name_no_tokenizer_has_finds_none)
{
    EXPECT_FALSE(tokenizer("o200k"));
    EXPECT_FALSE(tokenizer(""));
}
