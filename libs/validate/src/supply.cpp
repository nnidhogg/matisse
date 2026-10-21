#include "matisse/validate/supply.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "matisse/pretok/classes.hpp"
#include "matisse/pretok/segmenter.hpp"

namespace matisse::validate
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

/**
 * @brief Returns the interior code point positions of a class string at which an occurrence of a certified window of a
 *        table places its origin.
 * @param table The window table.
 * @param symbols The class string.
 * @return The positions, ascending, each once.
 */
[[nodiscard]] std::vector<std::size_t> anchored_points(const Window_table& table, const std::string_view symbols)
{
    std::map<std::string, std::size_t, std::less<>> origin_of{};

    std::set<std::size_t> widths{};

    for (const auto& [window, origin] : table.windows)
    {
        origin_of.emplace(window, origin);

        widths.insert(window.size());
    }

    std::vector<bool> anchored(symbols.size(), false);

    for (std::size_t at{0}; at < symbols.size(); ++at)
    {
        for (const auto width : widths)
        {
            if (at + width > symbols.size())
            {
                break;
            }

            const auto found{origin_of.find(symbols.substr(at, width))};

            if (found == origin_of.end())
            {
                continue;
            }

            const auto& [window, origin]{*found};

            // The start of the input is no cut, and an origin lies within its window, so never at the input's end.
            if (at + origin > 0)
            {
                anchored[at + origin] = true;
            }
        }
    }

    std::vector<std::size_t> points{};

    for (const auto [at, is_anchor] : std::views::enumerate(anchored))
    {
        if (is_anchor)
        {
            points.push_back(static_cast<std::size_t>(at));
        }
    }

    return points;
}

/**
 * @brief Returns the gaps between consecutive anchors, each percentile the order statistic at floor(q n) capped at the
 *        last.
 * @param anchors The anchor positions, ascending.
 * @return The gaps, std::nullopt when fewer than two anchors leave none.
 */
[[nodiscard]] std::optional<Gaps> gaps_of(const std::vector<std::size_t>& anchors)
{
    if (anchors.size() < 2)
    {
        return std::nullopt;
    }

    const auto gap{[](const std::size_t before, const std::size_t after) { return after - before; }};

    std::vector<std::size_t> gaps{};

    gaps.reserve(anchors.size() - 1);

    std::ranges::copy(anchors | std::views::pairwise_transform(gap), std::back_inserter(gaps));

    std::ranges::sort(gaps);

    const auto at_percentile{[&gaps](const std::size_t percent) {
        const auto rank{std::min(gaps.size() - 1, gaps.size() * percent / 100)};

        return gaps[rank];
    }};

    return Gaps{
            .median = at_percentile(50),
            .ninetieth = at_percentile(90),
            .ninety_ninth = at_percentile(99),
            .longest = gaps.back()};
}

/**
 * @brief Returns what a set of anchors supplies on an input.
 * @param anchors The anchors' byte offsets, ascending, the ends never among them.
 * @param bytes The input's length in bytes.
 * @return The count, the density and the gaps.
 */
[[nodiscard]] Anchors anchors_of(const std::vector<std::size_t>& anchors, const std::size_t bytes)
{
    if (bytes == 0)
    {
        return Anchors{.count = 0, .per_kibibyte = 0.0, .gaps = std::nullopt};
    }

    const auto density{1024.0 * static_cast<double>(anchors.size()) / static_cast<double>(bytes)};

    auto gaps{gaps_of(anchors)};

    return Anchors{.count = anchors.size(), .per_kibibyte = density, .gaps = std::move(gaps)};
}

/**
 * @brief Returns the figures of one inventory as its JSON object.
 * @param supply The inventory's supply.
 * @return The JSON text.
 */
[[nodiscard]] std::string json_inventory(const Supply& supply)
{
    const auto& [bytes, anchors, boundary_share, longest_piece]{supply};

    const auto& [count, per_kibibyte, gaps]{anchors};

    const auto [p50, p90, p99, longest]{gaps.value_or(Gaps{})};

    const auto figure{[&gaps](const std::size_t value) { return gaps ? std::to_string(value) : "null"; }};

    return std::format(
            R"({{"anchors": {}, "per_kibibyte": {:.1f}, "boundary_share": {:.4f}, "gap_p50": {}, "gap_p90": {}, )"
            R"("gap_p99": {}, "gap_max": {}}})",
            count, per_kibibyte, boundary_share, figure(p50), figure(p90), figure(p99), figure(longest));
}

/**
 * @brief Returns the figures of one inventory as the text row prints them.
 * @param supply The inventory's supply.
 * @return The row's value.
 */
[[nodiscard]] std::string shown(const Supply& supply)
{
    const auto& [bytes, anchors, boundary_share, longest_piece]{supply};

    const auto& [count, per_kibibyte, gaps]{anchors};

    const auto [median, ninetieth, ninety_ninth, longest]{gaps.value_or(Gaps{})};

    const auto spread{
            gaps ? std::format("gaps p50 {}, p90 {}, p99 {}, max {}", median, ninetieth, ninety_ninth, longest) :
                   std::string{"no gaps, fewer than two anchors"}};

    const auto percent{100.0 * boundary_share};

    return std::format(
            "{} anchor{}, {:.1f} per KiB, {:.2f}% of the pre-token boundaries, {}", count, plural(count), per_kibibyte,
            percent, spread);
}

} // namespace

Supply supply(
        const Report& report, const Inventory inventory, const pretok::Pattern& pattern, const std::string_view input)
{
    const pretok::Class_table classes{pattern.alphabet};

    const auto symbols{classes.symbols(input)};

    const auto& table{inventory == Inventory::factorization ? report.factorization : report.policy};

    const auto offsets{code_point_offsets(input)};

    const auto points{anchored_points(table, symbols)};

    const auto at_offset{[&offsets](const std::size_t point) { return offsets[point]; }};

    std::vector<std::size_t> anchor_offsets{};

    std::ranges::transform(points, std::back_inserter(anchor_offsets), at_offset);

    const pretok::Segmenter segmenter{pattern};

    const auto ends{segmenter.boundaries(input)};

    // The last end is the input's own, no boundary between two pieces.
    const auto boundaries{ends.empty() ? 0 : ends.size() - 1};

    const auto share{boundaries == 0 ? 0.0 : static_cast<double>(points.size()) / static_cast<double>(boundaries)};

    std::size_t longest_piece{0};

    std::size_t start{0};

    for (const auto end : ends)
    {
        longest_piece = std::max(longest_piece, end - start);

        start = end;
    }

    auto anchors{anchors_of(anchor_offsets, input.size())};

    return Supply{
            .bytes = input.size(),
            .anchors = std::move(anchors),
            .boundary_share = share,
            .longest_piece = longest_piece};
}

std::string supply_json(const Supply& factorization, const Supply& policy, const std::string_view path)
{
    const auto factorization_json{json_inventory(factorization)};

    const auto policy_json{json_inventory(policy)};

    return std::format(
            R"({{"input": {}, "bytes": {}, "longest_piece": {}, "factorization": {}, "policy": {}}})",
            json_string(path), factorization.bytes, factorization.longest_piece, factorization_json, policy_json);
}

std::string supply_section(const Supply& factorization, const Supply& policy, const std::string_view path)
{
    const auto& [bytes, anchors, boundary_share, longest_piece]{factorization};

    auto out{std::format("\ncertified-anchor supply on {}, {} byte{}\n", path, bytes, plural(bytes))};

    const auto floor{std::format("{} byte{}, the floor no cut rule goes below", longest_piece, plural(longest_piece))};

    out += std::format("  {:<26} {}\n", "longest piece", floor);

    out += std::format("  {:<26} {}\n", "factorization", shown(factorization));

    out += std::format("  {:<26} {}\n", "policy", shown(policy));

    return out;
}

} // namespace matisse::validate
