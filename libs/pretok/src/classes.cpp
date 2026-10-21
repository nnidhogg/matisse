#include "matisse/pretok/classes.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <munch/regex/utf8.hpp>

namespace matisse::pretok
{
namespace
{
/**
 * @brief The code points U+0000 to U+10FFFF, the surrogates among them, so that a code point indexes the table.
 */
constexpr std::size_t code_point_count{0x110000};

/**
 * @brief The distinct blocks a byte indexes.
 */
constexpr std::size_t most_blocks{256};

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

} // namespace

Class_table::Class_table(const Alphabet& alphabet)
{
    const auto& [symbols, sets, other, unicode_version]{alphabet};

    // The sets are painted last to first, so that the last paint of a code point is by the first set holding it.
    std::string painted(code_point_count, other);

    for (const auto& [symbol, ranges] : sets | std::views::reverse)
    {
        for (const auto& [first, last] : ranges)
        {
            const auto count{last + 1 - first};

            const auto range_cells{std::span{painted}.subspan(first, count)};

            std::ranges::fill(range_cells, symbol);
        }
    }

    const std::string_view cells{painted};

    // The index of every distinct block, the count of the blocks before it.
    std::map<std::string_view, std::size_t> seen{};

    for (std::size_t start{0}; start < code_point_count; start += block_points)
    {
        const auto block{cells.substr(start, block_points)};

        const auto [found, inserted]{seen.try_emplace(block, seen.size())};

        if (inserted)
        {
            blocks_.insert(blocks_.end(), block.begin(), block.end());
        }

        const auto& [kept, index]{*found};

        block_of_.push_back(static_cast<std::uint8_t>(index));
    }

    if (seen.size() > most_blocks)
    {
        throw std::invalid_argument{std::format(
                "Class_table: the alphabet has {} distinct blocks of code points, more than the {} a byte indexes",
                seen.size(), most_blocks)};
    }

    const auto ascii_cells{cells.substr(0, ascii_.size())};

    std::ranges::copy(ascii_cells, ascii_.begin());

    for (const auto [index, symbol] : std::views::enumerate(symbols))
    {
        class_of_symbol_[static_cast<unsigned char>(symbol)] = static_cast<Class_t>(index);
    }
}

std::string Class_table::symbols(const std::string_view text) const
{
    std::string class_string{};

    // A code point takes at least one byte, so the class string is no longer than the text.
    class_string.reserve(text.size());

    for (std::size_t at{0}; at < text.size();)
    {
        const auto lead{static_cast<unsigned char>(text[at])};

        if (lead < ascii_.size())
        {
            class_string += ascii_[lead];

            ++at;

            continue;
        }

        const auto length{well_formed_length(text, at)};

        if (length == 0)
        {
            throw std::invalid_argument{std::format("classes_of: the text is not UTF-8 at byte {}", at)};
        }

        const auto point{decoded(text.substr(at, length))};

        class_string += symbol(point);

        at += length;
    }

    return class_string;
}

char Class_table::symbol(const char32_t point) const noexcept
{
    const auto block{block_of_[point / block_points]};

    const auto cell{point % block_points};

    const auto at{block * block_points + cell};

    return blocks_[at];
}

std::vector<Class_t> Class_table::classes(const std::string_view text) const
{
    const auto class_string{symbols(text)};

    std::vector<Class_t> classes{};

    classes.reserve(class_string.size());

    for (const auto class_symbol : class_string)
    {
        classes.push_back(class_of_symbol_[static_cast<unsigned char>(class_symbol)]);
    }

    return classes;
}

std::vector<Class_t> classes_of(const Alphabet& alphabet, const std::string_view text)
{
    const Class_table table{alphabet};

    return table.classes(text);
}

} // namespace matisse::pretok
