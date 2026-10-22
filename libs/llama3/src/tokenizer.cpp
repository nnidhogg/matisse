#include "matisse/llama3/tokenizer.hpp"

#include <array>
#include <cstddef>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <munch/regex/utf8.hpp>

namespace matisse::llama3
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
 * @brief `d`, a contraction letter matched caselessly: `D` and `d`.
 */
constexpr std::array<Code_point_range, 2> d_points{{{.first = U'D', .last = U'D'}, {.first = U'd', .last = U'd'}}};

/**
 * @brief `e`, a contraction letter matched caselessly: `E` and `e`.
 */
constexpr std::array<Code_point_range, 2> e_points{{{.first = U'E', .last = U'E'}, {.first = U'e', .last = U'e'}}};

/**
 * @brief `l`, a contraction letter matched caselessly: `L` and `l`.
 */
constexpr std::array<Code_point_range, 2> l_points{{{.first = U'L', .last = U'L'}, {.first = U'l', .last = U'l'}}};

/**
 * @brief `m`, a contraction letter matched caselessly: `M` and `m`.
 */
constexpr std::array<Code_point_range, 2> m_points{{{.first = U'M', .last = U'M'}, {.first = U'm', .last = U'm'}}};

/**
 * @brief `r`, a contraction letter matched caselessly: `R` and `r`.
 */
constexpr std::array<Code_point_range, 2> r_points{{{.first = U'R', .last = U'R'}, {.first = U'r', .last = U'r'}}};

/**
 * @brief `s`, a contraction letter matched caselessly: `S`, `s` and U+017F, the long s, which `s` matches under the
 *        simple case folding both the regex crate and Python's re apply.
 */
constexpr std::array<Code_point_range, 3> s_points{{
        {.first = U'S', .last = U'S'},
        {.first = U's', .last = U's'},
        {.first = 0x17F, .last = 0x17F},
}};

/**
 * @brief `t`, a contraction letter matched caselessly: `T` and `t`.
 */
constexpr std::array<Code_point_range, 2> t_points{{{.first = U'T', .last = U'T'}, {.first = U't', .last = U't'}}};

/**
 * @brief `v`, a contraction letter matched caselessly: `V` and `v`.
 */
constexpr std::array<Code_point_range, 2> v_points{{{.first = U'V', .last = U'V'}, {.first = U'v', .last = U'v'}}};

/**
 * @brief The sets the classes read, in the order a code point is classed by them: the literals, the white space, the
 *        letters and the numbers.
 */
constexpr std::array<pretok::Code_point_set, 15> sets{{
        {.symbol = '\n', .ranges = line_feed_points},
        {.symbol = '\r', .ranges = carriage_return_points},
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
 * @brief Llama 3's alphabet, cl100k's: the carriage return, the line feed, the space, the apostrophe and the
 *        contraction letters of `(?i:'s|'t|'re|'ve|'m|'ll|'d)` are literals, the letters caseless, each a class of the
 *        code points it matches, and the rest of the code points are classed by the Unicode sets or are `!`.
 */
constexpr pretok::Alphabet alphabet{
        .symbols = symbols,
        .sets = sets,
        .other = '!',
        .unicode_version = class_unicode_version};

/**
 * @brief Llama 3's pattern alternative by alternative over its classes: the letters are `A` and the contraction
 *        letters, each holding both cases, the numbers `0`, the white space `\t`, `\n`, `\r` and the space, and
 *        everything else `!` and the apostrophe. The run `\s+(?!\S)` passes only before white space, and every other
 *        alternative is ungated, its gate every symbol.
 */
constexpr std::array<pretok::Alternative, 7> alternatives{{
        {.language = "'([sdmt]|ll|ve|re)", .gate = symbols},
        {.language = R"([\t\x20!']?[Adelmrstv]+)", .gate = symbols},
        {.language = "0{1,3}", .gate = symbols},
        {.language = R"(\x20?[!']+[\r\n]*)", .gate = symbols},
        {.language = R"([\t\n\r\x20]*[\r\n]+)", .gate = symbols},
        {.language = R"([\t\n\r\x20]+)", .gate = white_space},
        {.language = R"([\t\n\r\x20]+)", .gate = symbols},
}};

/**
 * @brief The id of the first special token, the count of the rank file's ordinary tokens.
 */
constexpr load::Token_id_t first_special{128000};

/**
 * @brief The special tokens Meta's tokenizer.py names first, in id order from first_special.
 */
constexpr std::array<std::string_view, 10> named_specials{
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>"};

/**
 * @brief The first of the reserved special tokens Meta's tokenizer.py numbers after the named ones.
 */
constexpr std::size_t first_unnamed_reserved{5};

/**
 * @brief One past the last of them, the 256 special tokens less the five named reserved ones.
 */
constexpr std::size_t last_unnamed_reserved{251};

} // namespace

pretok::Pattern pattern() noexcept
{
    return {.alphabet = alphabet, .alternatives = alternatives};
}

std::vector<load::Special_token> specials()
{
    std::vector<load::Special_token> tokens{};

    for (const auto spelling : named_specials)
    {
        const auto id{static_cast<load::Token_id_t>(first_special + tokens.size())};

        tokens.push_back({.spelling = std::string{spelling}, .id = id});
    }

    for (auto reserved{first_unnamed_reserved}; reserved < last_unnamed_reserved; ++reserved)
    {
        const auto id{static_cast<load::Token_id_t>(first_special + tokens.size())};

        auto spelling{std::format("<|reserved_special_token_{}|>", reserved)};

        tokens.push_back({.spelling = std::move(spelling), .id = id});
    }

    return tokens;
}

} // namespace matisse::llama3
