#include "matisse/encode/live_runs.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ranges>
#include <vector>

namespace matisse::encode
{
namespace
{
/**
 * @brief The fewest dropped layers or nodes erased at once, so that a short stretch erases nothing.
 */
constexpr std::size_t least_erased{64};

} // namespace

Live_runs::Live_runs(const Verifier_table& table) : table_{table}, front_{0}, erased_{0}, stable_{table.start()}
{}

void Live_runs::step(const pretok::Class_t symbol_class, const std::size_t end)
{
    const auto first{erased_ + nodes_.size()};

    const auto extend{[&](const Verifier_table::State_t state, const std::size_t parent) {
        for (const auto boundary_after : {false, true})
        {
            const auto next{table_.step(state, symbol_class, boundary_after)};

            if (next == Verifier_table::no_state)
            {
                continue;
            }

            nodes_.push_back(
                    {.state = next, .parent = parent, .boundary_after = boundary_after, .children = 0, .live = true});

            if (parent != root)
            {
                ++node(parent).children;
            }
        }
    }};

    if (front_ == layers_.size())
    {
        extend(stable_, root);

        const auto count{erased_ + nodes_.size() - first};

        layers_.push_back({.first = first, .count = count, .live = count, .end = end, .closed = false});

        return;
    }

    const auto [back_first, back_count, back_live, back_end, back_closed]{layers_.back()};

    for (auto number{back_first}; number < back_first + back_count; ++number)
    {
        const auto [state, parent, boundary_after, children, live]{node(number)};

        if (live)
        {
            extend(state, number);
        }
    }

    const auto count{erased_ + nodes_.size() - first};

    layers_.push_back({.first = first, .count = count, .live = count, .end = end, .closed = false});

    const auto previous{layers_.size() - 2};

    for (auto number{back_first}; number < back_first + back_count; ++number)
    {
        const auto& stepped{node(number)};

        if (stepped.live && stepped.children == 0)
        {
            kill(number, previous);
        }
    }
}

Live_runs::Node& Live_runs::node(const std::size_t number)
{
    return nodes_[number - erased_];
}

const Live_runs::Node& Live_runs::node(const std::size_t number) const
{
    return nodes_[number - erased_];
}

void Live_runs::kill(std::size_t number, std::size_t layer)
{
    while (true)
    {
        auto& dying{node(number)};

        dying.live = false;

        --layers_[layer].live;

        if (dying.parent == root)
        {
            return;
        }

        auto& parent{node(dying.parent)};

        --parent.children;

        if (parent.children > 0)
        {
            return;
        }

        number = dying.parent;

        --layer;
    }
}

std::optional<std::size_t> Live_runs::next_closed()
{
    const auto live_in{[this](const Layer& layer) {
        const auto numbers{std::views::iota(layer.first, layer.first + layer.count)};

        return numbers | std::views::filter([this](const std::size_t number) { return node(number).live; });
    }};

    while (front_ < layers_.size())
    {
        auto& front{layers_[front_]};

        auto live{live_in(front)};

        // A front layer before the last is stable when every run passes through one of its nodes.
        if (front_ + 1 < layers_.size())
        {
            if (front.live != 1)
            {
                return std::nullopt;
            }

            const auto& stable_node{node(*live.begin())};

            const auto ends{stable_node.boundary_after && !front.closed};

            const auto end{front.end};

            drop_front();

            if (ends)
            {
                return end;
            }

            continue;
        }

        if (front.closed)
        {
            return std::nullopt;
        }

        // The last code point read: every run must say the same of a piece ending after it.
        const auto unmarked{
                std::ranges::find_if(live, [this](const std::size_t number) { return !node(number).boundary_after; })};

        if (unmarked == live.end())
        {
            const auto end{front.end};

            drop_front();

            return end;
        }

        const auto state{node(*unmarked).state};

        if (table_.is_dead_end(state))
        {
            front.closed = true;

            return front.end;
        }

        if (front.live == 1 && !table_.accepts(state))
        {
            drop_front();
        }

        return std::nullopt;
    }

    return std::nullopt;
}

void Live_runs::drop_front()
{
    const auto [first, count, live, end, closed]{layers_[front_]};

    for (auto number{first}; number < first + count; ++number)
    {
        const auto& dropped{node(number)};

        if (dropped.live)
        {
            stable_ = dropped.state;
        }
    }

    ++front_;

    if (front_ < layers_.size())
    {
        const auto& next{layers_[front_]};

        for (auto number{next.first}; number < next.first + next.count; ++number)
        {
            node(number).parent = root;
        }
    }

    const auto dropped_nodes{first + count - erased_};

    if (front_ >= least_erased && 2 * front_ >= layers_.size())
    {
        layers_.erase(layers_.begin(), layers_.begin() + static_cast<std::ptrdiff_t>(front_));

        front_ = 0;
    }

    if (dropped_nodes >= least_erased && 2 * dropped_nodes >= nodes_.size())
    {
        nodes_.erase(nodes_.begin(), nodes_.begin() + static_cast<std::ptrdiff_t>(dropped_nodes));

        erased_ += dropped_nodes;
    }
}

std::vector<Live_runs::Marking> Live_runs::markings() const
{
    if (front_ == layers_.size())
    {
        return {Marking{.ends = {}, .dead_end = table_.is_dead_end(stable_)}};
    }

    const auto& back{layers_.back()};

    std::vector<Marking> found{};

    for (auto number{back.first}; number < back.first + back.count; ++number)
    {
        if (node(number).live)
        {
            found.push_back(marking_of(number));
        }
    }

    return found;
}

Live_runs::Marking Live_runs::marking_of(const std::size_t number) const
{
    std::vector<std::size_t> ends{};

    auto layer{layers_.size()};

    for (auto at{number}; at != root; at = node(at).parent)
    {
        --layer;

        if (node(at).boundary_after)
        {
            ends.push_back(layers_[layer].end);
        }
    }

    std::ranges::reverse(ends);

    return {.ends = std::move(ends), .dead_end = table_.is_dead_end(node(number).state)};
}

std::vector<std::size_t> Live_runs::accepted() const
{
    if (front_ == layers_.size())
    {
        return {};
    }

    const auto& back{layers_.back()};

    for (auto number{back.first}; number < back.first + back.count; ++number)
    {
        const auto& [state, parent, boundary_after, children, live]{node(number)};

        if (live && table_.accepts(state))
        {
            return marking_of(number).ends;
        }
    }

    return {};
}

void Live_runs::reset()
{
    nodes_.clear();

    layers_.clear();

    front_ = 0;

    erased_ = 0;

    stable_ = table_.start();
}

} // namespace matisse::encode
