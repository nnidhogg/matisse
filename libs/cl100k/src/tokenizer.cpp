#include "matisse/cl100k/tokenizer.hpp"

#include <array>
#include <string_view>
#include <vector>

#include <munch/regex/utf8.hpp>

namespace matisse::cl100k
{
namespace
{
using munch::regex::utf8::Code_point_range;

#include "class_ranges.inc"

/**
 * @brief The symbol of every class, ascending: the white space that is no literal, the line feed, the carriage return,
 *        the space, everything else, the apostrophe, the numbers, the letters that are no literal and the contraction
 *        letters.
 */
constexpr std::string_view symbols{"\t\n\r !'0Adelmrstv"};

/**
 * @brief The class symbols of the white space, `\s`, the gate of `(?!\S)`.
 */
constexpr std::string_view white_space{"\t\n\r "};

/**
 * @brief The gate of `\s++$`, which no symbol passes, only the end of the input.
 */
constexpr std::string_view end_of_input{};

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
 * @brief cl100k's alphabet: the carriage return, the line feed, the space, the apostrophe and the contraction letters
 *        of `'(?i:[sdmt]|ll|ve|re)` are literals, caseless, and the rest of the code points are classed by the sets or
 *        are `!`.
 */
constexpr pretok::Alphabet alphabet{
        .symbols = symbols,
        .literals = "\n\r 'delmrstv",
        .caseless = true,
        .sets = sets,
        .other = '!',
        .unicode_version = class_unicode_version};

/**
 * @brief cl100k's pattern alternative by alternative over its classes: the letters are `A` and the contraction
 *        letters, each holding both cases, the numbers `0`, the white space `\t`, `\n`, `\r` and the space, and
 *        everything else `!` and the apostrophe. The run `\s+(?!\S)` passes only before white space, `\s++$` only at
 *        the end of the input, and every other alternative is ungated, its gate every symbol.
 */
constexpr std::array<pretok::Alternative, 8> alternatives{{
        {.language = "'([sdmt]|ll|ve|re)", .gate = symbols},
        {.language = R"([\t\x20!']?[Adelmrstv]+)", .gate = symbols},
        {.language = "0{1,3}", .gate = symbols},
        {.language = R"(\x20?[!']+[\r\n]*)", .gate = symbols},
        {.language = R"([\t\n\r\x20]+)", .gate = end_of_input},
        {.language = R"([\t\n\r\x20]*[\r\n])", .gate = symbols},
        {.language = R"([\t\n\r\x20]+)", .gate = white_space},
        {.language = R"([\t\n\r\x20])", .gate = symbols},
}};

} // namespace

pretok::Pattern pattern() noexcept
{
    return {.alphabet = alphabet, .alternatives = alternatives};
}

std::vector<load::Special_token> specials()
{
    return {{.spelling = "<|endoftext|>", .id = 100257},
            {.spelling = "<|fim_prefix|>", .id = 100258},
            {.spelling = "<|fim_middle|>", .id = 100259},
            {.spelling = "<|fim_suffix|>", .id = 100260},
            {.spelling = "<|endofprompt|>", .id = 100276}};
}

} // namespace matisse::cl100k
