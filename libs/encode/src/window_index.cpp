#include "matisse/encode/window_index.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace matisse::encode
{
namespace
{
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
 * @brief Returns the byte offset a number of code points before an offset of a UTF-8 text, or the text's start when
 *        fewer stand before it.
 * @param text The text.
 * @param from The offset, at the start of a code point or at the text's end.
 * @param points The number of code points.
 * @return The offset, at the start of a code point.
 */
[[nodiscard]] std::size_t byte_before(
        const std::string_view text, const std::size_t from, const std::size_t points) noexcept
{
    auto byte{from};

    for (std::size_t point{0}; point < points && byte > 0; ++point)
    {
        --byte;

        while (byte > 0 && is_continuation(static_cast<unsigned char>(text[byte])))
        {
            --byte;
        }
    }

    return byte;
}

/**
 * @brief Returns the first offset of a UTF-8 text at or after an offset at which a code point begins, or the text's
 *        end.
 * @param text The text.
 * @param at The offset, at most the text's size.
 * @return The offset.
 */
[[nodiscard]] std::size_t point_start(const std::string_view text, const std::size_t at) noexcept
{
    auto byte{at};

    while (byte < text.size() && is_continuation(static_cast<unsigned char>(text[byte])))
    {
        ++byte;
    }

    return byte;
}

/**
 * @brief Returns the byte offset of every code point of a well-formed UTF-8 text, and the text's size after the last.
 * @param text The text, well-formed UTF-8.
 * @return The offsets, one more than the code points.
 */
[[nodiscard]] std::vector<std::size_t> code_point_offsets(const std::string_view text)
{
    std::vector<std::size_t> offsets{};

    for (const auto [at, byte] : std::views::enumerate(text))
    {
        // A code point begins at every byte that does not continue one.
        if (!is_continuation(static_cast<unsigned char>(byte)))
        {
            offsets.push_back(static_cast<std::size_t>(at));
        }
    }

    offsets.push_back(text.size());

    return offsets;
}

} // namespace

Window_index::Window_index(const validate::Window_table& table, const pretok::Alphabet& alphabet)
    : width_{table.width}, classes_{alphabet}
{
    std::set<std::size_t> widths{};

    for (const auto& [window, origin] : table.windows)
    {
        origins_.emplace(window, origin);

        widths.insert(window.size());
    }

    widths_.assign(widths.begin(), widths.end());
}

std::size_t Window_index::width() const noexcept
{
    return width_;
}

std::optional<std::size_t> Window_index::first_anchor(const std::string_view stretch, const std::size_t from) const
{
    // The stretch's start is no anchor; an anchor stands at a code point, so one at or after the byte after it is one
    // after it.
    const auto floor{std::max(from, 1UZ)};

    // An occurrence up to the width less one code points before the byte can place its origin at or after it.
    auto begin{byte_before(stretch, from, width_ - 1)};

    auto end{point_start(stretch, begin + std::min(stretch.size() - begin, block_bytes))};

    while (true)
    {
        if (const auto anchor{block_anchor(stretch, begin, end, floor)})
        {
            return anchor;
        }

        if (end == stretch.size())
        {
            return std::nullopt;
        }

        // The next block classes the bytes after this one and repeats its last width less one code points, so that a
        // window across the edge is read whole and the walk moves on however wide the windows are.
        begin = byte_before(stretch, end, width_ - 1);

        end = point_start(stretch, end + std::min(stretch.size() - end, block_bytes));
    }
}

std::optional<std::size_t> Window_index::block_anchor(
        const std::string_view stretch, const std::size_t begin, const std::size_t end, const std::size_t floor) const
{
    const auto block{stretch.substr(begin, end - begin)};

    const auto symbols{classes_.symbols(block)};

    const std::string_view classes{symbols};

    const auto offsets{code_point_offsets(block)};

    std::optional<std::size_t> least{};

    for (std::size_t at{0}; at < classes.size(); ++at)
    {
        // An anchor stands at or after its occurrence, so no later occurrence places one before the least found.
        if (least && begin + offsets[at] >= *least)
        {
            break;
        }

        for (const auto width : widths_)
        {
            if (at + width > classes.size())
            {
                break;
            }

            const auto found{origins_.find(classes.substr(at, width))};

            if (found == origins_.end())
            {
                continue;
            }

            const auto& [window, origin]{*found};

            const auto anchor{begin + offsets[at + origin]};

            if (anchor >= floor && (!least || anchor < *least))
            {
                least = anchor;
            }
        }
    }

    return least;
}

} // namespace matisse::encode
