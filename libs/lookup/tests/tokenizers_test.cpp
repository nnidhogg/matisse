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

    EXPECT_EQ(names, (std::vector<std::string_view>{"gpt2", "cl100k", "o200k", "llama3"}));
}

TEST(Tokenizers_test, A_name_finds_its_tokenizer_s_pattern_and_special_tokens)
{
    const auto gpt2{tokenizer("gpt2")};
    const auto cl100k{tokenizer("cl100k")};
    const auto o200k{tokenizer("o200k")};
    const auto llama3{tokenizer("llama3")};

    ASSERT_TRUE(gpt2);
    ASSERT_TRUE(cl100k);
    ASSERT_TRUE(o200k);
    ASSERT_TRUE(llama3);
    EXPECT_EQ(gpt2->pattern.alphabet.symbols, "\t !'0Adelmrstv");
    EXPECT_EQ(gpt2->specials.size(), 1U);
    EXPECT_EQ(cl100k->pattern.alphabet.symbols, "\t\n\r !'0Adelmrstv");
    EXPECT_EQ(cl100k->specials.size(), 5U);
    EXPECT_EQ(o200k->pattern.alphabet.symbols, "\t\n\r !#$%&'/0ADELMRSTVadelmrstv");
    EXPECT_EQ(o200k->specials.size(), 2U);
    EXPECT_EQ(llama3->pattern.alphabet.symbols, "\t\n\r !'0Adelmrstv");
    EXPECT_EQ(llama3->specials.size(), 256U);
}

TEST(Tokenizers_test, A_name_no_tokenizer_has_finds_none)
{
    EXPECT_FALSE(tokenizer("o200k_base"));
    EXPECT_FALSE(tokenizer(""));
}
