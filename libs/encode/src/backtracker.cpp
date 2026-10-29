#include "matisse/encode/backtracker.hpp"

#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace matisse::encode
{
namespace
{
/**
 * @brief The rank of no join, above every rank a join table holds, a definition holding fewer than 2^32 tokens and
 *        merges.
 */
constexpr std::uint32_t no_rank{std::numeric_limits<std::uint32_t>::max()};

/**
 * @brief One part of a spelling under the merge that reads its split: its id and the join it makes with the part after
 *        it.
 */
struct Split_part
{
    /**
     * @brief The id of the token the part spells.
     */
    load::Token_id_t id{};

    /**
     * @brief The rank of the join with the part after it, no_rank when there is none.
     */
    std::uint32_t rank{};

    /**
     * @brief The id of the token that join makes, read only when the rank is not no_rank.
     */
    load::Token_id_t product{};
};

} // namespace

Backtracker::Backtracker(Token_trie tokens, std::vector<Split> splits, Join_table pairs)
    : tokens_{std::move(tokens)}, splits_{std::move(splits)}, pairs_{std::move(pairs)}
{}

Backtracker Backtracker::of(
        const std::span<const std::string> spellings, const std::span<const load::Token_id_t, 256> byte_ids,
        const Join_table& joins)
{
    std::vector<std::optional<Split>> found(spellings.size());

    for (const auto id : byte_ids)
    {
        found[id] =
                Split{.order = 0, .left = id, .right = id, .in_order = true, .bit = 0, .followers = 0, .leaders = 0};
    }

    for (std::size_t id{0}; id < spellings.size(); ++id)
    {
        const std::string_view spelling{spellings[id]};

        if (spelling.size() < 2)
        {
            continue;
        }

        const auto parts{last_parts_of(spelling, byte_ids, joins)};

        if (!parts)
        {
            continue;
        }

        const auto [left, right]{*parts};

        const auto join{joins.find(left, right)};

        if (!join || join->product != id)
        {
            continue;
        }

        found[id] =
                Split{.order = join->rank + 1,
                      .left = left,
                      .right = right,
                      .in_order = false,
                      .bit = 0,
                      .followers = 0,
                      .leaders = 0};
    }

    const auto lower{[&](const load::Token_id_t part, const std::uint32_t order) {
        return found[part] && found[part]->order < order;
    }};

    std::vector<load::Token_id_t> ids{};

    std::vector<Split> splits(spellings.size());

    Join_table::Joins_t pairs{};

    for (std::size_t id{0}; id < spellings.size(); ++id)
    {
        if (!found[id])
        {
            continue;
        }

        const auto [order, left, right, in_order, bit, followers, leaders]{*found[id]};

        const auto token{static_cast<load::Token_id_t>(id)};

        ids.push_back(token);

        splits[id].order = order;

        splits[id].left = left;

        splits[id].right = right;

        splits[id].in_order = in_order;

        splits[id].bit = partner_bit(token);

        if (order == 0)
        {
            continue;
        }

        const auto key{Join_table::key_of(left, right)};

        pairs.emplace(key, Join_table::Join{.rank = order - 1, .product = token});

        splits[left].followers |= partner_bit(right);

        splits[right].leaders |= partner_bit(left);
    }

    // A part is shorter than its token, so with the tokens taken shortest first both parts of a split are decided
    // before the token they make: its merge joins in order when theirs do and their orders are below its own.
    auto shortest_first{ids};

    std::ranges::stable_sort(shortest_first, {}, [&](const load::Token_id_t id) { return spellings[id].size(); });

    for (const auto id : shortest_first)
    {
        auto& [order, left, right, in_order, bit, followers, leaders]{splits[id]};

        if (order == 0)
        {
            continue;
        }

        in_order = lower(left, order) && lower(right, order) && splits[left].in_order && splits[right].in_order;
    }

    Token_trie tokens{spellings, ids};

    Join_table split_pairs{pairs};

    return Backtracker{std::move(tokens), std::move(splits), std::move(split_pairs)};
}

std::optional<std::pair<load::Token_id_t, load::Token_id_t>> Backtracker::last_parts_of(
        const std::string_view spelling, const std::span<const load::Token_id_t, 256> byte_ids, const Join_table& joins)
{
    std::vector<Split_part> parts(spelling.size());

    // Sets the join of the part at an index with the part after it, none after the last part.
    const auto look_up{[&](const std::size_t at) {
        auto& [id, rank, product]{parts[at]};

        rank = no_rank;

        if (at + 1 == parts.size())
        {
            return;
        }

        const auto next_id{parts[at + 1].id};

        const auto join{joins.find(id, next_id)};

        if (!join)
        {
            return;
        }

        rank = join->rank;

        product = join->product;
    }};

    for (std::size_t at{0}; at < spelling.size(); ++at)
    {
        const auto byte{static_cast<unsigned char>(spelling[at])};

        parts[at].id = byte_ids[byte];
    }

    for (std::size_t at{0}; at < parts.size(); ++at)
    {
        look_up(at);
    }

    while (parts.size() > 2)
    {
        auto least{parts.size()};

        auto least_rank{no_rank};

        for (std::size_t at{0}; at < parts.size(); ++at)
        {
            if (parts[at].rank < least_rank)
            {
                least = at;

                least_rank = parts[at].rank;
            }
        }

        if (least == parts.size())
        {
            return std::nullopt;
        }

        parts[least].id = parts[least].product;

        const auto right{parts.begin() + static_cast<std::ptrdiff_t>(least) + 1};

        parts.erase(right);

        look_up(least);

        if (least > 0)
        {
            look_up(least - 1);
        }
    }

    return std::pair{parts[0].id, parts[1].id};
}

std::uint32_t Backtracker::partner_bit(const load::Token_id_t id) noexcept
{
    constexpr std::size_t partner_bits{32};

    std::size_t seed{};

    boost::hash_combine(seed, id);

    const auto bit{seed % partner_bits};

    return 1U << bit;
}

std::optional<std::vector<load::Token_id_t>> Backtracker::merged(const std::string_view piece) const
{
    std::vector<load::Token_id_t> ids{};

    if (piece.empty())
    {
        return ids;
    }

    ids.reserve(piece.size());

    // The positions from which no token fits, whatever tokens were laid before them, sized at the first.
    std::vector<bool> dead{};

    std::size_t at{0};

    auto token{tokens_.longest(piece)};

    while (true)
    {
        const auto end{at + token.length};

        const auto open{dead.empty() || !dead[end]};

        const auto asked{open && !ids.empty()};

        // The pair test is exact on two tokens whose own merges join in order of rank, so the search gives way before
        // it would act on the test's answer about any other.
        if (asked && (!splits_[ids.back()].in_order || !splits_[token.id].in_order))
        {
            return std::nullopt;
        }

        const auto fits{open && (ids.empty() || compatible(ids.back(), token.id))};

        if (fits)
        {
            ids.push_back(token.id);

            at = end;

            if (at == piece.size())
            {
                return ids;
            }

            token = tokens_.longest(piece.substr(at));

            continue;
        }

        auto shorter{tokens_.shorter(token.id)};

        // No shorter token is left to try here, so the last token laid is taken back and its shorter ones tried.
        while (!shorter)
        {
            dead.resize(piece.size() + 1);

            dead[at] = true;

            const auto last{ids.back()};

            ids.pop_back();

            at -= tokens_.length(last);

            shorter = tokens_.shorter(last);
        }

        token = *shorter;
    }
}

bool Backtracker::compatible(load::Token_id_t left, load::Token_id_t right) const noexcept
{
    auto limit{no_rank};

    while (true)
    {
        const auto& [left_order, left_left, left_right, left_in_order, left_bit, left_followers, left_leaders]{
                splits_[left]};

        const auto& [right_order, right_left, right_right, right_in_order, right_bit, right_followers, right_leaders]{
                splits_[right]};

        const auto right_follows{(left_followers & right_bit) != 0};

        const auto left_leads{(right_leaders & left_bit) != 0};

        const auto partners{right_follows && left_leads};

        const auto join{partners ? pairs_.find(left, right) : std::nullopt};

        if (join && join->rank < limit)
        {
            return false;
        }

        if (left_order == 0 && right_order == 0)
        {
            return true;
        }

        if (left_order > right_order)
        {
            limit = left_order - 1;

            left = left_right;

            continue;
        }

        limit = right_order;

        right = right_left;
    }
}

} // namespace matisse::encode
