#include "matisse/pretok/classes.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

using namespace matisse::pretok;

namespace
{
/**
 * @brief An alphabet of one class, every code point's, which is all the decoding of a text needs.
 */
constexpr Alphabet one_class{.symbols = "!", .sets = {}, .other = '!'};

/**
 * @brief Returns the refusal classes_of() makes of a text, empty when it makes none.
 * @param text The text.
 * @return The refusal's message.
 */
std::string refusal_of(const std::string_view text)
{
    try
    {
        std::ignore = classes_of(one_class, text);
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

} // namespace

TEST(Classes_test, Refuses_text_that_is_not_utf8_at_the_offset_of_its_first_ill_formed_sequence)
{
    EXPECT_TRUE(refusal_of("ab\xC0\x80").contains("classes_of: the text is not UTF-8 at byte 2"));
    EXPECT_TRUE(refusal_of("a\xE2\x82").contains("not UTF-8 at byte 1"));
    EXPECT_TRUE(refusal_of("\xED\xA0\x80").contains("not UTF-8 at byte 0"));
    EXPECT_TRUE(refusal_of("\xF4\x90\x80\x80").contains("not UTF-8 at byte 0"));
    EXPECT_TRUE(refusal_of("").empty());
}
