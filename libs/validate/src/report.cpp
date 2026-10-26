#include "matisse/validate/report.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <functional>
#include <iterator>
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
 * @brief One entry of a window's policy cloud: a state the split verifier can be in after an occurrence of the window
 *        read so far, where the piece holding the occurrence's last symbol begins, and whether a piece ends after it.
 */
struct Cloud_entry
{
    /**
     * @brief Ordered by the origin, then by the state and the bit, so that a sorted cloud's first and last entries hold
     *        its least and greatest origin.
     */
    auto operator<=>(const Cloud_entry&) const = default;

    /**
     * @brief Where the piece holding the last symbol read begins, a symbol index of the window, or std::nullopt when it
     *        begins before the window.
     */
    std::optional<std::size_t> origin{};

    /**
     * @brief The split verifier's state after the last symbol read.
     */
    std::size_t state{};

    /**
     * @brief Whether a piece ends after the last symbol read.
     */
    bool boundary{};
};

/**
 * @brief A window's policy cloud, its entries sorted and distinct.
 */
using Cloud_t = std::vector<Cloud_entry>;

/**
 * @brief The search that decides the policy certificate of every window to a width at once, walking the trie of
 *        windows, each window's cloud the one-symbol extension of its prefix's.
 *
 * A window's cloud holds an entry for every way an accepted marked string can read an occurrence of it, the split
 * verifier being trimmed, so every state it reaches accepts some extension. The window is certified at an origin
 * exactly when every entry places the piece holding the occurrence's last symbol there, which is
 * munch::verifier::miscovering()'s certificate at that origin, decided as the Python reference decides it and as
 * munch::verifier::modulo_cuts() decides its cuts, once per verifier rather than once per window. A window with an
 * empty cloud occurs in no accepted marked string, nor does any window extending it, so the search does not go past it;
 * a total pattern has none. validate() to width five takes about one second for GPT-2 and two for cl100k this way,
 * where a miscovering() search per window would take 92 and 213 seconds.
 */
class Policy_search
{
public:
    /**
     * @brief Starts the search, nothing decided yet, the split verifier's moves read into a table.
     * @param verifier The split verifier.
     * @param symbols The class symbols, ascending, the bytes the verifier reads, which outlive the search.
     * @param width The widest window decided.
     */
    Policy_search(const munch::verifier::Verifier& verifier, std::string_view symbols, std::size_t width);

    /**
     * @brief Decides every window of width two to the widest.
     * @return The certified windows, widths ascending, each width's in the order the class symbols give.
     */
    [[nodiscard]] std::vector<Certified_window> run() &&;

private:
    /**
     * @brief Returns the cloud of the empty window: every state of the split verifier reachable from the start, with
     *        whether a piece ends at the symbol read last, the start of the input counting as a piece end, and no
     *        origin.
     * @return The cloud.
     */
    [[nodiscard]] Cloud_t initial() const;

    /**
     * @brief Returns the state the split verifier moves to from a state on a class symbol and a bit.
     * @param state The state.
     * @param symbol The symbol's index among the class symbols.
     * @param boundary Whether a piece ends after the symbol.
     * @return The state moved to, or std::nullopt where the verifier has no move.
     */
    [[nodiscard]] std::optional<std::size_t> move(std::size_t state, std::size_t symbol, bool boundary) const noexcept;

    /**
     * @brief Extends a window by every symbol in turn, records each extension of width two or more that the policy
     *        certificate certifies, and walks past each extension that occurs.
     * @param window The window extended, restored before returning.
     * @param cloud The window's cloud.
     */
    void extend(std::string& window, const Cloud_t& cloud);

    /**
     * @brief Returns the cloud of a window extended by one symbol.
     * @param cloud The window's cloud.
     * @param at The symbol's index in the extended window.
     * @param symbol The symbol's index among the class symbols.
     * @return The extended window's cloud.
     */
    [[nodiscard]] Cloud_t extended(const Cloud_t& cloud, std::size_t at, std::size_t symbol) const;

    /**
     * @brief The widest window decided.
     */
    std::size_t width_{};

    /**
     * @brief How many states the split verifier has.
     */
    std::size_t state_count_{};

    /**
     * @brief The class symbols, ascending.
     */
    std::string_view symbols_;

    /**
     * @brief The split verifier's moves, by state, then by symbol index, then by bit.
     */
    std::vector<std::optional<std::size_t>> moves_{};

    /**
     * @brief The certified windows found, by width.
     */
    std::vector<std::vector<Certified_window>> found_{};
};

Policy_search::Policy_search(
        const munch::verifier::Verifier& verifier, const std::string_view symbols, const std::size_t width)
    : width_{width}, state_count_{verifier.state_count()}, symbols_{symbols}, found_(width + 1)
{
    for (std::size_t state{0}; state < verifier.state_count(); ++state)
    {
        for (const auto symbol : symbols)
        {
            for (const auto boundary : {false, true})
            {
                const munch::verifier::Marked read{
                        .byte = static_cast<unsigned char>(symbol),
                        .boundary_after = boundary};

                moves_.push_back(verifier.step(state, read));
            }
        }
    }
}

std::vector<Certified_window> Policy_search::run() &&
{
    std::string window{};

    extend(window, initial());

    std::vector<Certified_window> certified{};

    for (auto& width : found_)
    {
        std::ranges::move(width, std::back_inserter(certified));
    }

    return certified;
}

Cloud_t Policy_search::initial() const
{
    // The start is reached at the start of the input, which a piece end precedes.
    std::vector seen(state_count_, std::array{false, false});

    seen[0][1] = true;

    Cloud_t cloud{{.origin = std::nullopt, .state = 0, .boundary = true}};

    for (std::size_t index{0}; index < cloud.size(); ++index)
    {
        const auto [origin, state, boundary]{cloud[index]};

        for (std::size_t symbol{0}; symbol < symbols_.size(); ++symbol)
        {
            for (const auto bit : {false, true})
            {
                const auto next{move(state, symbol, bit)};

                if (!next || seen[*next][bit ? 1 : 0])
                {
                    continue;
                }

                seen[*next][bit ? 1 : 0] = true;
                cloud.push_back({.origin = std::nullopt, .state = *next, .boundary = bit});
            }
        }
    }

    return cloud;
}

std::optional<std::size_t> Policy_search::move(
        const std::size_t state, const std::size_t symbol, const bool boundary) const noexcept
{
    const auto index{(((state * symbols_.size()) + symbol) * 2) + (boundary ? 1 : 0)};

    return moves_[index];
}

void Policy_search::extend(std::string& window, const Cloud_t& cloud)
{
    for (std::size_t symbol{0}; symbol < symbols_.size(); ++symbol)
    {
        const auto next{extended(cloud, window.size(), symbol)};

        if (next.empty())
        {
            continue;
        }

        window.push_back(symbols_[symbol]);

        // Sorted by origin, the cloud agrees on one when its first and last entries do.
        const auto origin{next.front().origin};

        if (window.size() >= 2 && origin && origin == next.back().origin)
        {
            found_[window.size()].push_back({.window = window, .origin = *origin});
        }

        if (window.size() < width_)
        {
            extend(window, next);
        }

        window.pop_back();
    }
}

Cloud_t Policy_search::extended(const Cloud_t& cloud, const std::size_t at, const std::size_t symbol) const
{
    Cloud_t next{};

    for (const auto& [origin, state, boundary] : cloud)
    {
        // A piece that ended at the symbol before begins at this one.
        const auto begins{boundary ? std::optional{at} : origin};

        for (const auto bit : {false, true})
        {
            const auto moved{move(state, symbol, bit)};

            if (moved)
            {
                next.push_back({.origin = begins, .state = *moved, .boundary = bit});
            }
        }
    }

    std::ranges::sort(next);

    const auto [first, last]{std::ranges::unique(next)};

    next.erase(first, last);

    return next;
}

/**
 * @brief Returns the windows the policy certificate certifies, decided by one walk over the windows with the policy
 *        cloud.
 * @param pattern The pattern.
 * @param verifier The pattern's split verifier.
 * @param width The widest window decided.
 * @return The certified windows, widths ascending, each width's in the order the class symbols give.
 */
[[nodiscard]] std::vector<Certified_window> policy_windows(
        const pretok::Pattern& pattern, const munch::verifier::Verifier& verifier, const std::size_t width)
{
    return Policy_search{verifier, pattern.alphabet.symbols, width}.run();
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
    if (width == 0)
    {
        throw std::invalid_argument{"validate: the widest window decided is zero symbols wide"};
    }

    const auto verifier{pretok::split_verifier(pattern)};

    auto factorization{factorization_windows(pattern, width)};

    auto policy{policy_windows(pattern, verifier, width)};

    auto factorization_table{table_of(Inventory::factorization, width, std::move(factorization), verifier)};

    auto policy_table{table_of(Inventory::policy, width, std::move(policy), verifier)};

    auto gap{munch::verifier::boundary_gap(verifier)};

    return {.factorization = std::move(factorization_table), .policy = std::move(policy_table), .gap = std::move(gap)};
}

} // namespace matisse::validate
