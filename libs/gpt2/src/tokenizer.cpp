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
 * @brief The space, a literal.
 */
constexpr std::array<Code_point_range, 1> space_points{{{.first = U' ', .last = U' '}}};

/**
 * @brief The apostrophe, a literal.
 */
constexpr std::array<Code_point_range, 1> apostrophe_points{{{.first = U'\'', .last = U'\''}}};

/**
 * @brief `d`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> d_points{{{.first = U'd', .last = U'd'}}};

/**
 * @brief `e`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> e_points{{{.first = U'e', .last = U'e'}}};

/**
 * @brief `l`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> l_points{{{.first = U'l', .last = U'l'}}};

/**
 * @brief `m`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> m_points{{{.first = U'm', .last = U'm'}}};

/**
 * @brief `r`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> r_points{{{.first = U'r', .last = U'r'}}};

/**
 * @brief `s`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> s_points{{{.first = U's', .last = U's'}}};

/**
 * @brief `t`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> t_points{{{.first = U't', .last = U't'}}};

/**
 * @brief `v`, a contraction letter, case sensitive.
 */
constexpr std::array<Code_point_range, 1> v_points{{{.first = U'v', .last = U'v'}}};

/**
 * @brief The sets the classes read, in the order a code point is classed by them: the literals, the white space, the
 *        letters and the numbers.
 */
constexpr std::array<pretok::Code_point_set, 13> sets{{
        {.symbol = ' ', .ranges = space_points},
        {.symbol = '\'', .ranges = apostrophe_points},
        {.symbol = 'd', .ranges = d_points},
        {.symbol = 'e', .ranges = e_points},
        {.symbol = 'l', .ranges = l_points},
        {.symbol = 'm', .ranges = m_points},
        {.symbol = 'r', .ranges = r_points},
        {.symbol = 's', .ranges = s_points},
        {.symbol = 't', .ranges = t_points},
        {.symbol = 'v', .ranges = v_points},
        {.symbol = '\t', .ranges = white_space_ranges},
        {.symbol = 'A', .ranges = letter_ranges},
        {.symbol = '0', .ranges = number_ranges},
}};

/**
 * @brief GPT-2's alphabet: the space, the apostrophe and the contraction letters of `'s|'t|'re|'ve|'m|'ll|'d` are
 *        literals, case sensitive, each a class of its one code point, and the rest of the code points are classed by
 *        the Unicode sets or are `!`.
 */
constexpr pretok::Alphabet alphabet{
        .symbols = symbols,
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
