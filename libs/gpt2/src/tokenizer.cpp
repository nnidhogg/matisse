#include "matisse/gpt2/tokenizer.hpp"

#include <array>
#include <string_view>
#include <vector>

#include <munch/regex/utf8.hpp>

namespace matisse::gpt2
{
namespace
{
using munch::regex::utf8::Code_point_range;

#include "class_ranges.inc"

/**
 * @brief The symbol of every class, ascending: the white space that is no literal, the space, everything else, the
 *        apostrophe, the numbers, the letters that are no literal and the contraction letters.
 */
constexpr std::string_view symbols{"\t !'0Adelmrstv"};

/**
 * @brief The class symbols of the white space, `\s`, the gate of `(?!\S)`.
 */
constexpr std::string_view white_space{"\t "};

/**
 * @brief The sets the classes read, in the order a code point is classed by them: the white space, the letters and
 *        the numbers.
 */
constexpr std::array<pretok::Code_point_set, 3> sets{{
        {.symbol = '\t', .ranges = white_space_ranges},
        {.symbol = 'A', .ranges = letter_ranges},
        {.symbol = '0', .ranges = number_ranges},
}};

/**
 * @brief GPT-2's alphabet: the space, the apostrophe and the contraction letters of `'s|'t|'re|'ve|'m|'ll|'d` are
 *        literals, case sensitive, and the rest of the code points are classed by the sets or are `!`.
 */
constexpr pretok::Alphabet alphabet{
        .symbols = symbols,
        .literals = " 'delmrstv",
        .caseless = false,
        .sets = sets,
        .other = '!',
        .unicode_version = class_unicode_version};

/**
 * @brief GPT-2's pattern alternative by alternative over its classes: the letters are `A` and the contraction letters,
 *        the numbers `0`, the white space `\t` and the space, and everything else `!` and the apostrophe. The run
 *        `\s+(?!\S)` passes only before white space, and every other alternative is ungated, its gate every symbol.
 */
constexpr std::array<pretok::Alternative, 12> alternatives{{
        {.language = "'s", .gate = symbols},
        {.language = "'t", .gate = symbols},
        {.language = "'re", .gate = symbols},
        {.language = "'ve", .gate = symbols},
        {.language = "'m", .gate = symbols},
        {.language = "'ll", .gate = symbols},
        {.language = "'d", .gate = symbols},
        {.language = R"(\x20?[Adelmrstv]+)", .gate = symbols},
        {.language = R"(\x20?0+)", .gate = symbols},
        {.language = R"(\x20?[!']+)", .gate = symbols},
        {.language = R"([\t\x20]+)", .gate = white_space},
        {.language = R"([\t\x20]+)", .gate = symbols},
}};

} // namespace

pretok::Pattern pattern() noexcept
{
    return {.alphabet = alphabet, .alternatives = alternatives};
}

std::vector<load::Special_token> specials()
{
    return {{.spelling = "<|endoftext|>", .id = 50256}};
}

} // namespace matisse::gpt2
