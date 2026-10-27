#include "matisse/pretok/segmenter.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <munch/regex/regex.hpp>

using namespace matisse::pretok;
using munch::regex::text;

namespace
{
/**
 * @brief The one alternative of a pattern over one class, every code point a piece, ungated.
 */
const std::vector<Alternative> one_piece_each{{.language = text("!"), .gate = "!"}};

/**
 * @brief An alphabet of one class, every code point's.
 */
const Alphabet one_symbol{.symbols = "!", .sets = {}, .other = '!'};

/**
 * @brief A pattern over one class, every code point's, which is all the decoding of a text needs.
 */
const Pattern one_class{.alphabet = one_symbol, .alternatives = one_piece_each};

/**
 * @brief Returns the refusal a Segmenter makes of a text, empty when it makes none.
 * @param text The text.
 * @return The refusal's message.
 */
std::string refusal_of(const std::string_view text)
{
    try
    {
        std::ignore = Segmenter{one_class}.boundaries(text);
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

} // namespace

TEST(Segmenter_test, Refuses_text_that_is_not_utf8)
{
    EXPECT_TRUE(refusal_of("ab\xFF").contains("the text is not UTF-8 at byte 2"));
    EXPECT_TRUE(refusal_of("ab").empty());
}
