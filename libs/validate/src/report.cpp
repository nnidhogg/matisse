#include "matisse/validate/report.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "matisse/pretok/verifier.hpp"

namespace matisse::validate
{
namespace
{
/**
 * @brief The certified windows decided so far, each with its origin, found by its class symbols.
 */
using Origins_t = std::map<std::string, std::size_t, std::less<>>;

/**
 * @brief Returns every string of a width over the class symbols, in the order the symbols give.
 * @param symbols The class symbols, ascending.
 * @param width The width.
 * @return The strings.
 */
[[nodiscard]] std::vector<std::string> windows_of(const std::string_view symbols, const std::size_t width)
{
    std::vector<std::string> windows{{}};

    for (std::size_t length{0}; length < width; ++length)
    {
        std::vector<std::string> longer{};

        for (const auto& window : windows)
        {
            for (const auto symbol : symbols)
            {
                longer.push_back(window + symbol);
            }
        }

        windows = std::move(longer);
    }

    return windows;
}

/**
 * @brief Returns the windows the factorization certificate certifies, by is_split_window() on the factorization lexer.
 * @param pattern The pattern.
 * @param width The widest window decided.
 * @return The certified windows, widths ascending, each width's in the order the class symbols give.
 */
[[nodiscard]] std::vector<Certified_window> factorization_windows(
        const pretok::Pattern& pattern, const std::size_t width)
{
    const auto lexer{pretok::factorization_lexer(pattern)};

    std::vector<Certified_window> certified{};

    for (std::size_t length{2}; length <= width; ++length)
    {
        for (auto& window : windows_of(pattern.alphabet.symbols, length))
        {
            const auto origin{lexer.is_split_window(window)};

            if (origin)
            {
                certified.push_back({.window = std::move(window), .origin = *origin});
            }
        }
    }

    return certified;
}

/**
 * @brief Returns the origin a refutation leaves a window: where the piece holding the refuted occurrence's last symbol
 *        begins, counted from the occurrence, or none when it begins before the occurrence.
 * @param refutation The refutation.
 * @param width The window's width.
 * @return The origin, or std::nullopt.
 */
[[nodiscard]] std::optional<std::size_t> origin_left(
        const munch::verifier::Miscovering& refutation, const std::size_t width)
{
    const auto& [segmentation, occurrence]{refutation};

    const auto last{occurrence + width - 1};

    // The piece holding the last symbol begins after the last boundary before it, or at the start.
    auto begins{last};

    while (begins > 0 && !segmentation.boundaries[begins - 1])
    {
        --begins;
    }

    if (begins < occurrence)
    {
        return std::nullopt;
    }

    return begins - occurrence;
}

/**
 * @brief Returns the origin the policy certificate certifies a window at, by munch::verifier::miscovering() on the
 *        split verifier: its last symbol, else the one origin a refutation there leaves, else none.
 * @param verifier The split verifier.
 * @param window The window, two symbols at least.
 * @return The certified origin, or std::nullopt when the window is refused.
 */
[[nodiscard]] std::optional<std::size_t> policy_origin(
        const munch::verifier::Verifier& verifier, const std::string_view window)
{
    const auto last{window.size() - 1};

    const auto refutation{munch::verifier::miscovering(verifier, window, last)};

    if (!refutation)
    {
        return last;
    }

    const auto left{origin_left(*refutation, window.size())};

    if (!left)
    {
        return std::nullopt;
    }

    const auto refuted_again{munch::verifier::miscovering(verifier, window, *left)};

    if (refuted_again)
    {
        return std::nullopt;
    }

    return left;
}

/**
 * @brief Returns the windows the policy certificate certifies, a window whose last symbols are certified taking their
 *        origin plus one and every other one decided by policy_origin().
 * @param pattern The pattern.
 * @param verifier The pattern's split verifier.
 * @param width The widest window decided.
 * @return The certified windows, widths ascending, each width's in the order the class symbols give.
 */
[[nodiscard]] std::vector<Certified_window> policy_windows(
        const pretok::Pattern& pattern, const munch::verifier::Verifier& verifier, const std::size_t width)
{
    Origins_t origins{};

    std::vector<Certified_window> certified{};

    for (std::size_t length{2}; length <= width; ++length)
    {
        for (auto& window : windows_of(pattern.alphabet.symbols, length))
        {
            const auto last_symbols{std::string_view{window}.substr(1)};

            const auto shorter{origins.find(last_symbols)};

            const auto origin{[&] {
                if (shorter != origins.end())
                {
                    return std::optional{shorter->second + 1};
                }

                return policy_origin(verifier, window);
            }()};

            if (!origin)
            {
                continue;
            }

            origins.emplace(window, *origin);

            certified.push_back({.window = std::move(window), .origin = *origin});
        }
    }

    return certified;
}

/**
 * @brief Returns whether a certified window holds a shorter certified window whose origin lands on the same symbol, so
 *        that every cut of the window is a cut of the shorter one at the same place.
 * @param origins The certified windows with their origins.
 * @param window The window.
 * @param origin The window's origin.
 * @return True when it holds such a window.
 */
[[nodiscard]] bool holds_shorter(const Origins_t& origins, const std::string_view window, const std::size_t origin)
{
    for (std::size_t length{2}; length < window.size(); ++length)
    {
        for (std::size_t from{0}; from <= origin && from + length <= window.size(); ++from)
        {
            const auto inner{window.substr(from, length)};

            const auto shorter{origins.find(inner)};

            if (shorter != origins.end() && from + shorter->second == origin)
            {
                return true;
            }
        }
    }

    return false;
}

/**
 * @brief Returns the window table of an inventory: its windows with the right context their cuts need and the shortest
 *        truncation at every context that fails, the first in the table's order among the shortest, decided by
 *        munch::verifier::right_context() on the split verifier.
 * @param inventory The inventory.
 * @param width The widest window decided.
 * @param windows The certified windows.
 * @param verifier The split verifier.
 * @return The table.
 * @throws std::invalid_argument If the right context of a window is unbounded, or the split verifier does not certify
 *         a window.
 */
[[nodiscard]] Window_table table_of(
        const Inventory inventory, const std::size_t width, std::vector<Certified_window> windows,
        const munch::verifier::Verifier& verifier)
{
    Origins_t origins{};

    for (const auto& [window, origin] : windows)
    {
        origins.emplace(window, origin);
    }

    std::size_t right_context{0};

    std::map<std::size_t, munch::verifier::Truncation> shortest{};

    for (const auto& [window, origin] : windows)
    {
        if (holds_shorter(origins, window, origin))
        {
            continue;
        }

        const auto decided{munch::verifier::right_context(verifier, window, origin)};

        if (!std::holds_alternative<munch::verifier::Finite_context>(decided))
        {
            throw std::invalid_argument{
                    std::format("validate: the right context of the window ({}, {}) is unbounded", window, origin)};
        }

        const auto& [context, least, truncations]{std::get<munch::verifier::Finite_context>(decided)};

        right_context = std::max(right_context, context);

        for (const auto& truncation : truncations)
        {
            const auto known{shortest.find(truncation.context)};

            const auto shortest_so_far{
                    known == shortest.end() ||
                    truncation.segmentation.bytes.size() < known->second.segmentation.bytes.size()};

            if (shortest_so_far)
            {
                shortest.insert_or_assign(truncation.context, truncation);
            }
        }
    }

    std::vector<munch::verifier::Truncation> failing{};

    for (auto& truncation : shortest | std::views::values)
    {
        failing.push_back(std::move(truncation));
    }

    return {.inventory = inventory,
            .width = width,
            .windows = std::move(windows),
            .right_context = right_context,
            .failing = std::move(failing)};
}

} // namespace

Report validate(const pretok::Pattern& pattern, const std::size_t width)
{
    const auto verifier{pretok::split_verifier(pattern)};

    auto factorization{factorization_windows(pattern, width)};

    auto policy{policy_windows(pattern, verifier, width)};

    auto factorization_table{table_of(Inventory::factorization, width, std::move(factorization), verifier)};

    auto policy_table{table_of(Inventory::policy, width, std::move(policy), verifier)};

    auto gap{munch::verifier::boundary_gap(verifier)};

    return {.factorization = std::move(factorization_table), .policy = std::move(policy_table), .gap = std::move(gap)};
}

} // namespace matisse::validate
