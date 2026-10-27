#include "matisse/gpt2/tokenizer.hpp"

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <munch/regex/regex.hpp>
#include <munch/regex/set.hpp>
#include <munch/regex/utf8.hpp>

namespace matisse::gpt2
{
namespace
{
using munch::regex::any_of;
using munch::regex::concat;
using munch::regex::optional;
using munch::regex::plus;
using munch::regex::Regex;
using munch::regex::Set;
using munch::regex::text;
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
constexpr std::string_view white_space_symbols{"\t "};

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
 * @brief Returns the sets the classes read, in the order a code point is classed by them: the literals, the white
 *        space, the letters and the numbers.
 * @return The sets.
 */
[[nodiscard]] std::vector<pretok::Code_point_set> sets()
{
    return {
            {.symbol = ' ', .ranges = {space_points.begin(), space_points.end()}},
            {.symbol = '\'', .ranges = {apostrophe_points.begin(), apostrophe_points.end()}},
            {.symbol = 'd', .ranges = {d_points.begin(), d_points.end()}},
            {.symbol = 'e', .ranges = {e_points.begin(), e_points.end()}},
            {.symbol = 'l', .ranges = {l_points.begin(), l_points.end()}},
            {.symbol = 'm', .ranges = {m_points.begin(), m_points.end()}},
            {.symbol = 'r', .ranges = {r_points.begin(), r_points.end()}},
            {.symbol = 's', .ranges = {s_points.begin(), s_points.end()}},
            {.symbol = 't', .ranges = {t_points.begin(), t_points.end()}},
            {.symbol = 'v', .ranges = {v_points.begin(), v_points.end()}},
            {.symbol = '\t', .ranges = {white_space_ranges.begin(), white_space_ranges.end()}},
            {.symbol = 'A', .ranges = {letter_ranges.begin(), letter_ranges.end()}},
            {.symbol = '0', .ranges = {number_ranges.begin(), number_ranges.end()}},
    };
}

/**
 * @brief Returns GPT-2's alphabet: the space, the apostrophe and the contraction letters of `'s|'t|'re|'ve|'m|'ll|'d`
 *        are literals, case sensitive, each a class of its one code point, and the rest of the code points are classed
 *        by the Unicode sets or are `!`.
 * @return The alphabet.
 */
[[nodiscard]] pretok::Alphabet alphabet()
{
    return {.symbols = std::string{symbols},
            .sets = sets(),
            .other = '!',
            .unicode_version = std::string{class_unicode_version}};
}

/**
 * @brief Returns the space, a literal, as the one class symbol it is.
 * @return The space.
 */
[[nodiscard]] Regex space()
{
    return text(" ");
}

/**
 * @brief Returns one symbol of the letters, `\p{L}`: `A` or a contraction letter.
 * @return The letter.
 */
[[nodiscard]] Regex letter()
{
    return any_of(Set{'A', 'd', 'e', 'l', 'm', 'r', 's', 't', 'v'});
}

/**
 * @brief Returns one symbol of the numbers, `\p{N}`: `0`.
 * @return The number.
 */
[[nodiscard]] Regex number()
{
    return any_of(Set{'0'});
}

/**
 * @brief Returns one symbol of `[^\s\p{L}\p{N}]`: `!` or the apostrophe.
 * @return The symbol.
 */
[[nodiscard]] Regex other()
{
    return any_of(Set{'!', '\''});
}

/**
 * @brief Returns one symbol of the white space, `\s`: `\t` or the space.
 * @return The white space.
 */
[[nodiscard]] Regex white_space()
{
    return any_of(Set{'\t', ' '});
}

/**
 * @brief Returns GPT-2's pattern alternative by alternative over its classes, each below the vendor's alternative it
 *        describes: the letters are `A` and the contraction letters, the numbers `0`, the white space `\t` and the
 *        space, and everything else `!` and the apostrophe. The run `\s+(?!\S)` passes only before white space, and
 *        every other alternative is ungated, its gate every symbol.
 * @return The alternatives, in the order of the published pattern.
 */
[[nodiscard]] std::vector<pretok::Alternative> alternatives()
{
    return {
            // `'s`
            {.language = text("'s"), .gate = std::string{symbols}},
            // `'t`
            {.language = text("'t"), .gate = std::string{symbols}},
            // `'re`
            {.language = text("'re"), .gate = std::string{symbols}},
            // `'ve`
            {.language = text("'ve"), .gate = std::string{symbols}},
            // `'m`
            {.language = text("'m"), .gate = std::string{symbols}},
            // `'ll`
            {.language = text("'ll"), .gate = std::string{symbols}},
            // `'d`
            {.language = text("'d"), .gate = std::string{symbols}},
            // ` ?\p{L}+`
            {.language = concat(optional(space()), plus(letter())), .gate = std::string{symbols}},
            // ` ?\p{N}+`
            {.language = concat(optional(space()), plus(number())), .gate = std::string{symbols}},
            // ` ?[^\s\p{L}\p{N}]+`
            {.language = concat(optional(space()), plus(other())), .gate = std::string{symbols}},
            // `\s+(?!\S)`
            {.language = plus(white_space()), .gate = std::string{white_space_symbols}},
            // `\s+`
            {.language = plus(white_space()), .gate = std::string{symbols}},
    };
}

} // namespace

pretok::Pattern pattern()
{
    return {.alphabet = alphabet(), .alternatives = alternatives()};
}

std::vector<load::Special_token> specials()
{
    return {{.spelling = "<|endoftext|>", .id = 50256}};
}

} // namespace matisse::gpt2
