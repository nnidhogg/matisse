#include "matisse/encode/token_trie.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace matisse::encode
{
namespace
{
/**
 * @brief The number of byte values, the most children a node has.
 */
constexpr std::size_t byte_count{256};

/**
 * @brief A node of the trie waiting to be placed: its slot, the tokens whose spellings begin with its spelling, a run
 *        of the tokens sorted by spelling, and the length of its spelling.
 */
struct Pending_node
{
    /**
     * @brief The node's slot.
     */
    std::uint32_t slot{};

    /**
     * @brief The index of the first of its tokens among the sorted ones.
     */
    std::size_t first{};

    /**
     * @brief The index after the last of its tokens among the sorted ones.
     */
    std::size_t last{};

    /**
     * @brief The length of the node's spelling, its depth in the trie.
     */
    std::size_t depth{};
};

/**
 * @brief One child of a node waiting to be placed: the byte that leads to it and the run of tokens below it.
 */
struct Pending_child
{
    /**
     * @brief The byte from the node to the child.
     */
    unsigned char byte{};

    /**
     * @brief The index of the first of the child's tokens among the sorted ones.
     */
    std::size_t first{};

    /**
     * @brief The index after the last of the child's tokens among the sorted ones.
     */
    std::size_t last{};
};

/**
 * @brief Returns the least base at which every child of a node lands on a free slot, searched from the first free slot
 *        on.
 * @param children The node's children, by byte ascending, at least one.
 * @param used Whether each slot holds a node; a slot past its end is free.
 * @param first_free The first slot that holds no node.
 * @return The base.
 */
[[nodiscard]] std::size_t least_base(
        const std::span<const Pending_child> children, const std::vector<bool>& used, const std::size_t first_free)
{
    const auto is_free{[&](const std::size_t slot) { return slot >= used.size() || !used[slot]; }};

    const auto least_byte{children.front().byte};

    for (auto slot{std::max<std::size_t>(first_free, least_byte)};; ++slot)
    {
        if (!is_free(slot))
        {
            continue;
        }

        const auto base{slot - least_byte};

        const auto lands_free{
                std::ranges::all_of(children, [&](const Pending_child& child) { return is_free(base + child.byte); })};

        if (lands_free)
        {
            return base;
        }
    }
}

} // namespace

Token_trie::Token_trie(const std::span<const std::string> spellings, const std::span<const load::Token_id_t> ids)
    : slots_{slots_of(spellings, ids)}, entries_{entries_of(spellings, ids)}
{}

std::vector<Token_trie::Slot> Token_trie::slots_of(
        const std::span<const std::string> spellings, const std::span<const load::Token_id_t> ids)
{
    std::vector sorted(ids.begin(), ids.end());

    const auto spelling_of{[&](const load::Token_id_t id) { return std::string_view{spellings[id]}; }};

    std::ranges::sort(sorted, {}, spelling_of);

    std::vector slots(1, Slot{.base = 0, .parent = no_slot, .id = no_token});

    std::vector<bool> used(1, true);

    std::size_t first_free{1};

    std::size_t greatest_base{0};

    std::deque pending{Pending_node{.slot = 0, .first = 0, .last = sorted.size(), .depth = 0}};

    std::vector<Pending_child> children{};

    while (!pending.empty())
    {
        const auto [slot, first, last, depth]{pending.front()};

        pending.pop_front();

        // The token the node spells sorts before every longer spelling that begins with it.
        auto below{first};

        if (below < last && spelling_of(sorted[below]).size() == depth)
        {
            slots[slot].id = sorted[below];

            ++below;
        }

        children.clear();

        for (auto at{below}; at < last; ++at)
        {
            const auto byte{static_cast<unsigned char>(spelling_of(sorted[at])[depth])};

            if (children.empty() || children.back().byte != byte)
            {
                children.push_back({.byte = byte, .first = at, .last = at});
            }

            children.back().last = at + 1;
        }

        if (children.empty())
        {
            continue;
        }

        const auto base{least_base(children, used, first_free)};

        const auto reach{base + children.back().byte + 1};

        if (reach > slots.size())
        {
            slots.resize(reach, Slot{.base = 0, .parent = no_slot, .id = no_token});

            used.resize(reach, false);
        }

        slots[slot].base = static_cast<std::uint32_t>(base);

        greatest_base = std::max(greatest_base, base);

        for (const auto& [byte, child_first, child_last] : children)
        {
            const auto child{static_cast<std::uint32_t>(base + byte)};

            slots[child].parent = slot;

            used[child] = true;

            pending.push_back({.slot = child, .first = child_first, .last = child_last, .depth = depth + 1});
        }

        while (first_free < used.size() && used[first_free])
        {
            ++first_free;
        }
    }

    // Every base plus every byte names a slot, so a walk reads no slot outside them.
    const auto least_size{std::max(slots.size(), greatest_base + byte_count)};

    slots.resize(least_size, Slot{.base = 0, .parent = no_slot, .id = no_token});

    return slots;
}

std::vector<Token_trie::Entry> Token_trie::entries_of(
        const std::span<const std::string> spellings, const std::span<const load::Token_id_t> ids) const
{
    std::vector entries(spellings.size(), Entry{.length = 0, .shorter = no_token});

    for (const auto id : ids)
    {
        const std::string_view spelling{spellings[id]};

        const auto shorter_match{longest(spelling.substr(0, spelling.size() - 1))};

        const auto shorter{shorter_match.length == 0 ? no_token : shorter_match.id};

        entries[id] = {.length = static_cast<std::uint32_t>(spelling.size()), .shorter = shorter};
    }

    return entries;
}

Token_trie::Match Token_trie::longest(const std::string_view text) const noexcept
{
    Match longest_match{.id = no_token, .length = 0};

    std::uint32_t node{0};

    for (std::size_t at{0}; at < text.size(); ++at)
    {
        const auto byte{static_cast<unsigned char>(text[at])};

        const auto child{slots_[node].base + byte};

        const auto& [base, parent, id]{slots_[child]};

        if (parent != node)
        {
            break;
        }

        node = child;

        if (id != no_token)
        {
            longest_match = {.id = id, .length = static_cast<std::uint32_t>(at + 1)};
        }
    }

    return longest_match;
}

std::optional<Token_trie::Match> Token_trie::shorter(const load::Token_id_t id) const noexcept
{
    const auto shorter_id{entries_[id].shorter};

    if (shorter_id == no_token)
    {
        return std::nullopt;
    }

    return Match{.id = shorter_id, .length = entries_[shorter_id].length};
}

std::uint32_t Token_trie::length(const load::Token_id_t id) const noexcept
{
    return entries_[id].length;
}

} // namespace matisse::encode
