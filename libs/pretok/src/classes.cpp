#include "matisse/pretok/classes.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <munch/regex/utf8.hpp>

namespace matisse::pretok
{
namespace
{
using munch::regex::utf8::Code_point_range;

#include "class_ranges.inc"

/**
 * @brief The class alphabet of a pattern: its class symbols and which of them are literals of the pattern.
 */
struct Alphabet
{
    /**
     * @brief The symbol of every class, ascending, so a class is its symbol's place among them.
     */
    std::string_view symbols{};

    /**
     * @brief The symbols that are literals of the pattern, each a class of that code point alone, or of its two cases
     *        when the alphabet is caseless.
     */
    std::string_view literals{};

    /**
     * @brief Whether the pattern's contraction letters match either case, as cl100k's `(?i:...)` matches them.
     */
    bool caseless{};
};

/**
 * @brief GPT-2's alphabet: the space, the apostrophe and the contraction letters of `'s|'t|'re|'ve|'m|'ll|'d`, case
 *        sensitive.
 */
constexpr Alphabet gpt2_alphabet{.symbols = "\t !'0Adelmrstv", .literals = " 'delmrstv", .caseless = false};

/**
 * @brief cl100k's alphabet: the carriage return, the line feed, the space, the apostrophe and the contraction
 *        letters of `'(?i:[sdmt]|ll|ve|re)`, caseless.
 */
constexpr Alphabet cl100k_alphabet{.symbols = "\t\n\r !'0Adelmrstv", .literals = "\n\r 'delmrstv", .caseless = true};

/**
 * @brief The symbol of the white space that is no literal.
 */
constexpr char white_space_symbol{'\t'};

/**
 * @brief The symbol of the letters that are no literal.
 */
constexpr char letter_symbol{'A'};

/**
 * @brief The symbol of the numbers.
 */
constexpr char number_symbol{'0'};

/**
 * @brief The symbol of every code point in no set and no literal.
 */
constexpr char other_symbol{'!'};

/**
 * @brief U+017F, the long s, which `s` matches caselessly under the simple case folding both the regex crate and
 *        Python's re apply.
 */
constexpr char32_t long_s{0x17F};

/**
 * @brief One past the last ASCII code point.
 */
constexpr char32_t ascii_end{0x80};

/**
 * @brief Returns the alphabet of a pattern.
 * @param pattern The pattern.
 * @param caller The function asking, for the refusal.
 * @return The alphabet.
 * @throws std::invalid_argument If the pattern is Pattern::none.
 */
[[nodiscard]] const Alphabet& alphabet_of(const load::Pattern pattern, const std::string_view caller)
{
    if (pattern == load::Pattern::none)
    {
        throw std::invalid_argument{std::format("{}: Pattern::none has no class alphabet", caller)};
    }

    return pattern == load::Pattern::gpt2 ? gpt2_alphabet : cl100k_alphabet;
}

/**
 * @brief Returns whether a byte continues a UTF-8 sequence rather than beginning one.
 * @param byte The byte.
 * @return True for a continuation byte, `10xxxxxx`.
 */
[[nodiscard]] constexpr bool is_continuation(const unsigned char byte) noexcept
{
    return (byte & 0xC0U) == 0x80U;
}

/**
 * @brief Returns the length of the well-formed UTF-8 sequence at an index, zero when none stands there: an ASCII byte,
 *        or a lead byte of the shape 110xxxxx, 1110xxxx or 11110xxx followed by as many continuation bytes as its shape
 *        says, the overlong, surrogate and beyond-U+10FFFF encodings excluded as the standard excludes them, which the
 *        second byte's narrower range after a lead of E0, ED, F0 or F4 rules out.
 * @param text The text.
 * @param at The index, within the text.
 * @return One to four, or zero.
 */
[[nodiscard]] constexpr std::size_t well_formed_length(const std::string_view text, const std::size_t at) noexcept
{
    const auto lead{static_cast<unsigned char>(text[at])};

    const auto length{
            lead < 0x80U                  ? 1UZ :
            lead >= 0xC2U && lead < 0xE0U ? 2UZ :
            lead >= 0xE0U && lead < 0xF0U ? 3UZ :
            lead >= 0xF0U && lead < 0xF5U ? 4UZ :
                                            0UZ};

    if (length == 0 || at + length > text.size())
    {
        return 0;
    }

    for (std::size_t index{1}; index < length; ++index)
    {
        if (!is_continuation(static_cast<unsigned char>(text[at + index])))
        {
            return 0;
        }
    }

    const auto second{static_cast<unsigned char>(text[at + (length > 1 ? 1 : 0)])};

    const auto malformed{
            (lead == 0xE0U && second < 0xA0U) || (lead == 0xEDU && second >= 0xA0U) ||
            (lead == 0xF0U && second < 0x90U) || (lead == 0xF4U && second >= 0x90U)};

    return malformed ? 0 : length;
}

/**
 * @brief Returns the code point a well-formed UTF-8 sequence encodes.
 * @param sequence The sequence, as well_formed_length() validates it.
 * @return The code point.
 */
[[nodiscard]] constexpr char32_t decoded(const std::string_view sequence) noexcept
{
    const auto lead{static_cast<unsigned char>(sequence.front())};

    if (sequence.size() == 1)
    {
        return lead;
    }

    // The length marker is the sequence's length in high one bits and a zero below them.
    const auto marker_bits{sequence.size() + 1};

    char32_t point{lead & (0xFFU >> marker_bits)};

    for (const char byte : sequence.substr(1))
    {
        point = (point << 6U) | (static_cast<unsigned char>(byte) & 0x3FU);
    }

    return point;
}

/**
 * @brief Returns whether a table of ranges, ascending and apart, holds a code point.
 * @param ranges The table.
 * @param point The code point.
 * @return True when a range holds it.
 */
[[nodiscard]] bool holds(const std::span<const Code_point_range> ranges, const char32_t point) noexcept
{
    const auto above{std::ranges::upper_bound(ranges, point, {}, &Code_point_range::first)};

    if (above == ranges.begin())
    {
        return false;
    }

    return point <= std::prev(above)->last;
}

/**
 * @brief Returns the symbol of a code point's class: its own as a literal of the alphabet, the lowercase letter's for
 *        either case of a contraction letter of a caseless alphabet and the `s`'s for U+017F, and otherwise the symbol
 *        of the first set holding it, white space, letters or numbers, or the symbol of everything else.
 * @param alphabet The alphabet.
 * @param point The code point.
 * @return The symbol.
 */
[[nodiscard]] char symbol_of(const Alphabet& alphabet, const char32_t point) noexcept
{
    constexpr char32_t case_bit{0x20};

    const auto& [symbols, literals, caseless]{alphabet};

    const auto lowered{point >= U'A' && point <= U'Z' && caseless ? point | case_bit : point};

    if (lowered < ascii_end && literals.contains(static_cast<char>(lowered)))
    {
        return static_cast<char>(lowered);
    }

    if (caseless && point == long_s)
    {
        return 's';
    }

    if (holds(white_space_ranges, point))
    {
        return white_space_symbol;
    }

    if (holds(letter_ranges, point))
    {
        return letter_symbol;
    }

    if (holds(number_ranges, point))
    {
        return number_symbol;
    }

    return other_symbol;
}

} // namespace

std::vector<Class_t> classes_of(const load::Pattern pattern, const std::string_view text)
{
    const auto& alphabet{alphabet_of(pattern, "classes_of")};

    std::vector<Class_t> classes{};

    for (std::size_t at{0}; at < text.size();)
    {
        const auto length{well_formed_length(text, at)};

        if (length == 0)
        {
            throw std::invalid_argument{std::format("classes_of: the text is not UTF-8 at byte {}", at)};
        }

        const auto point{decoded(text.substr(at, length))};

        const auto symbol{symbol_of(alphabet, point)};

        classes.push_back(static_cast<Class_t>(alphabet.symbols.find(symbol)));

        at += length;
    }

    return classes;
}

std::size_t class_count(const load::Pattern pattern)
{
    return alphabet_of(pattern, "class_count").symbols.size();
}

std::string_view class_symbols(const load::Pattern pattern)
{
    return alphabet_of(pattern, "class_symbols").symbols;
}

std::string_view unicode_version() noexcept
{
    return class_unicode_version;
}

} // namespace matisse::pretok
