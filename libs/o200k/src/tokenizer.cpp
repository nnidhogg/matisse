#include "matisse/o200k/tokenizer.hpp"

#include <array>
#include <string_view>
#include <vector>

#include <munch/regex/utf8.hpp>

namespace matisse::o200k
{
namespace
{
using munch::regex::utf8::Code_point_range;

#include "class_ranges.inc"

/**
 * @brief The symbol of every class, ascending: the white space that is no literal, the line feed, the carriage return,
 *        the space, everything else, then the four classes with no ASCII member, each written by a byte of everything
 *        else, the other letters, the title case letters, the modifier letters and the marks, then the apostrophe, the
 *        slash, the numbers, the upper case letters that are no literal and the upper case contraction letters, and the
 *        lower case letters that are no literal and the lower case contraction letters.
 */
constexpr std::string_view symbols{"\t\n\r !#$%&'/0ADELMRSTVadelmrstv"};

/**
 * @brief The class symbols of the white space, `\s`, the gate of `(?!\S)`.
 */
constexpr std::string_view white_space{"\t\n\r "};

/**
 * @brief The line feed, a literal.
 */
constexpr std::array<Code_point_range, 1> line_feed_points{{{.first = U'\n', .last = U'\n'}}};

/**
 * @brief The carriage return, a literal.
 */
constexpr std::array<Code_point_range, 1> carriage_return_points{{{.first = U'\r', .last = U'\r'}}};

/**
 * @brief The space, a literal.
 */
constexpr std::array<Code_point_range, 1> space_points{{{.first = U' ', .last = U' '}}};

/**
 * @brief The apostrophe, a literal.
 */
constexpr std::array<Code_point_range, 1> apostrophe_points{{{.first = U'\'', .last = U'\''}}};

/**
 * @brief The slash, a literal.
 */
constexpr std::array<Code_point_range, 1> slash_points{{{.first = U'/', .last = U'/'}}};

/**
 * @brief `D`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_d_points{{{.first = U'D', .last = U'D'}}};

/**
 * @brief `E`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_e_points{{{.first = U'E', .last = U'E'}}};

/**
 * @brief `L`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_l_points{{{.first = U'L', .last = U'L'}}};

/**
 * @brief `M`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_m_points{{{.first = U'M', .last = U'M'}}};

/**
 * @brief `R`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_r_points{{{.first = U'R', .last = U'R'}}};

/**
 * @brief `S`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_s_points{{{.first = U'S', .last = U'S'}}};

/**
 * @brief `T`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_t_points{{{.first = U'T', .last = U'T'}}};

/**
 * @brief `V`, a contraction letter matched caselessly and an upper case letter.
 */
constexpr std::array<Code_point_range, 1> upper_v_points{{{.first = U'V', .last = U'V'}}};

/**
 * @brief `d`, a contraction letter matched caselessly and a lower case letter.
 */
constexpr std::array<Code_point_range, 1> d_points{{{.first = U'd', .last = U'd'}}};

/**
 * @brief `e`, a contraction letter matched caselessly and a lower case letter.
 */
constexpr std::array<Code_point_range, 1> e_points{{{.first = U'e', .last = U'e'}}};

/**
 * @brief `l`, a contraction letter matched caselessly and a lower case letter.
 */
constexpr std::array<Code_point_range, 1> l_points{{{.first = U'l', .last = U'l'}}};

/**
 * @brief `m`, a contraction letter matched caselessly and a lower case letter.
 */
constexpr std::array<Code_point_range, 1> m_points{{{.first = U'm', .last = U'm'}}};

/**
 * @brief `r`, a contraction letter matched caselessly and a lower case letter.
 */
constexpr std::array<Code_point_range, 1> r_points{{{.first = U'r', .last = U'r'}}};

/**
 * @brief `s`, a contraction letter matched caselessly and a lower case letter: `s` and U+017F, the long s, which `s`
 *        matches under the simple case folding both the regex crate and Python's re apply and which is a lower case
 *        letter too.
 */
constexpr std::array<Code_point_range, 2> s_points{{{.first = U's', .last = U's'}, {.first = 0x17F, .last = 0x17F}}};

/**
 * @brief `t`, a contraction letter matched caselessly and a lower case letter.
 */
constexpr std::array<Code_point_range, 1> t_points{{{.first = U't', .last = U't'}}};

/**
 * @brief `v`, a contraction letter matched caselessly and a lower case letter.
 */
constexpr std::array<Code_point_range, 1> v_points{{{.first = U'v', .last = U'v'}}};

/**
 * @brief The sets the classes read, in the order a code point is classed by them: the literals, the white space, the
 *        letters by their general category, the marks and the numbers.
 */
constexpr std::array<pretok::Code_point_set, 29> sets{{
        {.symbol = '\n', .ranges = line_feed_points},
        {.symbol = '\r', .ranges = carriage_return_points},
        {.symbol = ' ', .ranges = space_points},
        {.symbol = '\'', .ranges = apostrophe_points},
        {.symbol = '/', .ranges = slash_points},
        {.symbol = 'D', .ranges = upper_d_points},
        {.symbol = 'E', .ranges = upper_e_points},
        {.symbol = 'L', .ranges = upper_l_points},
        {.symbol = 'M', .ranges = upper_m_points},
        {.symbol = 'R', .ranges = upper_r_points},
        {.symbol = 'S', .ranges = upper_s_points},
        {.symbol = 'T', .ranges = upper_t_points},
        {.symbol = 'V', .ranges = upper_v_points},
        {.symbol = 'd', .ranges = d_points},
        {.symbol = 'e', .ranges = e_points},
        {.symbol = 'l', .ranges = l_points},
        {.symbol = 'm', .ranges = m_points},
        {.symbol = 'r', .ranges = r_points},
        {.symbol = 's', .ranges = s_points},
        {.symbol = 't', .ranges = t_points},
        {.symbol = 'v', .ranges = v_points},
        {.symbol = '\t', .ranges = white_space_ranges},
        {.symbol = 'A', .ranges = upper_ranges},
        {.symbol = 'a', .ranges = lower_ranges},
        {.symbol = '#', .ranges = other_letter_ranges},
        {.symbol = '$', .ranges = title_ranges},
        {.symbol = '%', .ranges = modifier_ranges},
        {.symbol = '&', .ranges = mark_ranges},
        {.symbol = '0', .ranges = number_ranges},
}};

/**
 * @brief o200k's alphabet: the carriage return, the line feed, the space, the apostrophe, the slash and the contraction
 *        letters of `(?i:'s|'t|'re|'ve|'m|'ll|'d)` are literals, each letter's case partners split by the general
 *        category the pattern reads them by, and the rest of the code points are classed by the Unicode sets or are
 *        `!`.
 */
constexpr pretok::Alphabet alphabet{
        .symbols = symbols,
        .sets = sets,
        .other = '!',
        .unicode_version = class_unicode_version};

/**
 * @brief o200k's pattern alternative by alternative over its classes: `[^\r\n\p{L}\p{N}]` is `\t`, the space, `!`, the
 *        marks `&`, the apostrophe and the slash, `[^\s\p{L}\p{N}]` is `!`, `&`, the apostrophe and the slash, the
 *        upper case side `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` is `A`, the upper case contraction letters and `#`, `$`, `%`
 *        and `&`, the lower case side `[\p{Ll}\p{Lm}\p{Lo}\p{M}]` is `a`, the lower case contraction letters and `#`,
 *        `%` and `&`, the numbers are `0` and the white space `\t`, `\n`, `\r` and the space. The run `\s+(?!\S)`
 *        passes only before white space, and every other alternative is ungated, its gate every symbol.
 */
constexpr std::array<pretok::Alternative, 7> alternatives{{
        {.language = R"([\t\x20!&'/]?[ADELMRSTV#$%&]*[adelmrstv#%&]+('([DMSTdmst]|[Rr][Ee]|[Vv][Ee]|[Ll][Ll]))?)",
         .gate = symbols},
        {.language = R"([\t\x20!&'/]?[ADELMRSTV#$%&]+[adelmrstv#%&]*('([DMSTdmst]|[Rr][Ee]|[Vv][Ee]|[Ll][Ll]))?)",
         .gate = symbols},
        {.language = "0{1,3}", .gate = symbols},
        {.language = R"(\x20?[!&'/]+[\n\r/]*)", .gate = symbols},
        {.language = R"([\t\n\r\x20]*[\n\r]+)", .gate = symbols},
        {.language = R"([\t\n\r\x20]+)", .gate = white_space},
        {.language = R"([\t\n\r\x20]+)", .gate = symbols},
}};

} // namespace

pretok::Pattern pattern() noexcept
{
    return {.alphabet = alphabet, .alternatives = alternatives};
}

std::vector<load::Special_token> specials()
{
    return {{.spelling = "<|endoftext|>", .id = 199999}, {.spelling = "<|endofprompt|>", .id = 200018}};
}

} // namespace matisse::o200k
