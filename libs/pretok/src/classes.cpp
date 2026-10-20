#include "matisse/pretok/classes.hpp"

#include <algorithm>
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
 * @brief Returns the symbol of a code point's class: the symbol of the first set holding it, the literals' sets coming
 *        first, or the symbol of everything else.
 * @param alphabet The alphabet.
 * @param point The code point.
 * @return The symbol.
 */
[[nodiscard]] char symbol_of(const Alphabet& alphabet, const char32_t point) noexcept
{
    const auto& [symbols, sets, other, unicode_version]{alphabet};

    for (const auto& [symbol, ranges] : sets)
    {
        if (holds(ranges, point))
        {
            return symbol;
        }
    }

    return other;
}

} // namespace

std::vector<Class_t> classes_of(const Alphabet& alphabet, const std::string_view text)
{
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

} // namespace matisse::pretok
