#include "matisse/pretok/automata.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace matisse::pretok::automata
{
namespace
{
/**
 * @brief A set of NFA states, ascending, as a state of the subset construction.
 */
using Subset_t = std::vector<State_t>;

/**
 * @brief Returns the empty closure of a set of NFA states: the states it reaches by empty moves alone, itself
 *        included.
 * @param nfa The NFA.
 * @param states The states.
 * @return The closure, ascending.
 */
[[nodiscard]] Subset_t closure(const Nfa& nfa, const Subset_t& states)
{
    std::vector<bool> seen(nfa.edges.size(), false);

    for (const auto state : states)
    {
        seen[state] = true;
    }

    auto pending{states};

    while (!pending.empty())
    {
        const auto state{pending.back()};

        pending.pop_back();

        for (const auto next : nfa.epsilons[state])
        {
            if (seen[next])
            {
                continue;
            }

            seen[next] = true;
            pending.push_back(next);
        }
    }

    Subset_t reached{};

    for (State_t state{0}; state < seen.size(); ++state)
    {
        if (seen[state])
        {
            reached.push_back(state);
        }
    }

    return reached;
}

/**
 * @brief Returns the NFA states a set of them moves to on each symbol, before closure.
 * @param nfa The NFA.
 * @param states The states.
 * @return The states moved to, by symbol, ascending and without repeats.
 */
[[nodiscard]] std::vector<Subset_t> moves_of(const Nfa& nfa, const Subset_t& states)
{
    std::vector<Subset_t> moves(nfa.symbol_count);

    for (const auto state : states)
    {
        for (const auto& [symbol, to] : nfa.edges[state])
        {
            moves[symbol].push_back(to);
        }
    }

    for (auto& targets : moves)
    {
        std::ranges::sort(targets);

        const auto repeated{std::ranges::unique(targets)};

        targets.erase(repeated.begin(), repeated.end());
    }

    return moves;
}

/**
 * @brief Returns the states of a DFA reachable from its start, ascending.
 * @param dfa The DFA.
 * @return The reachable states.
 */
[[nodiscard]] std::vector<State_t> reachable(const Dfa& dfa)
{
    std::vector<bool> seen(dfa.transitions.size(), false);

    seen[0] = true;

    std::deque<State_t> frontier{0};

    while (!frontier.empty())
    {
        const auto state{frontier.front()};

        frontier.pop_front();

        for (const auto next : dfa.transitions[state])
        {
            if (seen[next])
            {
                continue;
            }

            seen[next] = true;
            frontier.push_back(next);
        }
    }

    std::vector<State_t> states{};

    for (State_t state{0}; state < seen.size(); ++state)
    {
        if (seen[state])
        {
            states.push_back(state);
        }
    }

    return states;
}

/**
 * @brief Returns a DFA restricted to its reachable states, renumbered densely in ascending order, the start staying
 *        zero.
 * @param dfa The DFA.
 * @return The restricted DFA.
 */
[[nodiscard]] Dfa reachable_part(const Dfa& dfa)
{
    const auto states{reachable(dfa)};

    std::vector<State_t> renumbered(dfa.transitions.size(), 0);

    for (std::size_t index{0}; index < states.size(); ++index)
    {
        renumbered[states[index]] = index;
    }

    Dfa part{.symbol_count = dfa.symbol_count, .transitions = {}, .accepting = {}};

    for (const auto state : states)
    {
        std::vector<State_t> row{};

        for (const auto next : dfa.transitions[state])
        {
            row.push_back(renumbered[next]);
        }

        part.transitions.push_back(std::move(row));
        part.accepting.push_back(dfa.accepting[state]);
    }

    return part;
}

/**
 * @brief Returns how many blocks a partition holds.
 * @param blocks The block of each state.
 * @return The number of distinct blocks.
 */
[[nodiscard]] std::size_t block_count(std::vector<std::size_t> blocks)
{
    std::ranges::sort(blocks);

    const auto repeated{std::ranges::unique(blocks)};

    return static_cast<std::size_t>(repeated.begin() - blocks.begin());
}

/**
 * @brief Splits the blocks of a partition of a DFA's states once: two states stay together when they share a block and
 *        move to the same blocks on every symbol.
 * @param dfa The DFA.
 * @param blocks The block of each state.
 * @return The refined block of each state, numbered in the order the blocks' first states appear.
 */
[[nodiscard]] std::vector<std::size_t> refine(const Dfa& dfa, const std::vector<std::size_t>& blocks)
{
    std::map<std::vector<std::size_t>, std::size_t> numbered{};

    std::vector<std::size_t> refined{};

    for (State_t state{0}; state < dfa.transitions.size(); ++state)
    {
        std::vector<std::size_t> signature{blocks[state]};

        for (const auto next : dfa.transitions[state])
        {
            signature.push_back(blocks[next]);
        }

        const auto [entry, inserted]{numbered.try_emplace(std::move(signature), numbered.size())};

        refined.push_back(entry->second);
    }

    return refined;
}

/**
 * @brief Returns whether a product's pair accepts, by the operation and by whether each operand accepts.
 * @param accepts The operation.
 * @param first Whether the first operand accepts.
 * @param second Whether the second operand accepts.
 * @return True when the pair accepts.
 */
[[nodiscard]] constexpr bool combined(const Accepts accepts, const bool first, const bool second) noexcept
{
    switch (accepts)
    {
    case Accepts::both:
        return first && second;
    case Accepts::either:
        return first || second;
    case Accepts::first_only:
        return first && !second;
    }

    return false;
}

} // namespace

State_t add_state(Nfa& nfa)
{
    nfa.edges.emplace_back();
    nfa.epsilons.emplace_back();

    return nfa.edges.size() - 1;
}

State_t embed(Nfa& nfa, const Dfa& dfa)
{
    const auto base{nfa.edges.size()};

    for (std::size_t index{0}; index < dfa.transitions.size(); ++index)
    {
        std::ignore = add_state(nfa);
    }

    for (State_t state{0}; state < dfa.transitions.size(); ++state)
    {
        for (Symbol_t symbol{0}; symbol < dfa.symbol_count; ++symbol)
        {
            const auto next{dfa.transitions[state][symbol]};

            nfa.edges[base + state].push_back({.symbol = symbol, .to = base + next});
        }
    }

    return base;
}

Dfa determinize(const Nfa& nfa)
{
    std::vector<bool> accepting(nfa.edges.size(), false);

    for (const auto state : nfa.accepting)
    {
        accepting[state] = true;
    }

    const auto start{closure(nfa, {nfa.start})};

    std::map<Subset_t, State_t> numbered{{start, 0}};

    std::vector<Subset_t> subsets{start};

    Dfa dfa{.symbol_count = nfa.symbol_count, .transitions = {}, .accepting = {}};

    for (State_t state{0}; state < subsets.size(); ++state)
    {
        const auto moves{moves_of(nfa, subsets[state])};

        std::vector<State_t> row{};

        for (const auto& targets : moves)
        {
            auto next{closure(nfa, targets)};

            const auto [entry, inserted]{numbered.try_emplace(next, subsets.size())};

            if (inserted)
            {
                subsets.push_back(std::move(next));
            }

            row.push_back(entry->second);
        }

        const auto accepts{
                std::ranges::any_of(subsets[state], [&](const State_t member) { return accepting[member]; })};

        dfa.transitions.push_back(std::move(row));
        dfa.accepting.push_back(accepts);
    }

    return minimize(dfa);
}

Dfa minimize(const Dfa& dfa)
{
    const auto part{reachable_part(dfa)};

    std::vector<std::size_t> blocks{};

    for (const auto accepts : part.accepting)
    {
        blocks.push_back(accepts ? 1 : 0);
    }

    for (auto count{block_count(blocks)};;)
    {
        blocks = refine(part, blocks);

        const auto refined_count{block_count(blocks)};

        if (refined_count == count)
        {
            break;
        }

        count = refined_count;
    }

    // The start's block is state zero, and the rest follow in the order their first states appear.
    std::map<std::size_t, State_t> renumbered{{blocks[0], 0}};

    for (const auto block : blocks)
    {
        renumbered.try_emplace(block, renumbered.size());
    }

    Dfa minimal{
            .symbol_count = part.symbol_count,
            .transitions = std::vector<std::vector<State_t>>(renumbered.size()),
            .accepting = std::vector<bool>(renumbered.size(), false)};

    std::vector<bool> filled(renumbered.size(), false);

    for (State_t state{0}; state < part.transitions.size(); ++state)
    {
        const auto merged{renumbered[blocks[state]]};

        if (filled[merged])
        {
            continue;
        }

        filled[merged] = true;

        for (const auto next : part.transitions[state])
        {
            minimal.transitions[merged].push_back(renumbered[blocks[next]]);
        }

        minimal.accepting[merged] = part.accepting[state];
    }

    return minimal;
}

Dfa complement(const Dfa& dfa)
{
    auto flipped{dfa};

    flipped.accepting.flip();

    return flipped;
}

Dfa product(const Dfa& first, const Dfa& second, const Accepts accepts)
{
    using Pair_t = std::pair<State_t, State_t>;

    std::map<Pair_t, State_t> numbered{{Pair_t{0, 0}, 0}};

    std::vector<Pair_t> pairs{{0, 0}};

    Dfa paired{.symbol_count = first.symbol_count, .transitions = {}, .accepting = {}};

    for (State_t state{0}; state < pairs.size(); ++state)
    {
        const auto [left, right]{pairs[state]};

        std::vector<State_t> row{};

        for (Symbol_t symbol{0}; symbol < first.symbol_count; ++symbol)
        {
            const Pair_t next{first.transitions[left][symbol], second.transitions[right][symbol]};

            const auto [entry, inserted]{numbered.try_emplace(next, pairs.size())};

            if (inserted)
            {
                pairs.push_back(next);
            }

            row.push_back(entry->second);
        }

        const auto pair_accepts{combined(accepts, first.accepting[left], second.accepting[right])};

        paired.transitions.push_back(std::move(row));
        paired.accepting.push_back(pair_accepts);
    }

    return minimize(paired);
}

Dfa concatenation(const Dfa& first, const Dfa& second)
{
    Nfa nfa{.symbol_count = first.symbol_count, .edges = {}, .epsilons = {}, .start = 0, .accepting = {}};

    const auto first_base{embed(nfa, first)};

    const auto second_base{embed(nfa, second)};

    nfa.start = first_base;

    for (State_t state{0}; state < first.transitions.size(); ++state)
    {
        if (first.accepting[state])
        {
            nfa.epsilons[first_base + state].push_back(second_base);
        }
    }

    for (State_t state{0}; state < second.transitions.size(); ++state)
    {
        if (second.accepting[state])
        {
            nfa.accepting.push_back(second_base + state);
        }
    }

    return determinize(nfa);
}

} // namespace matisse::pretok::automata
