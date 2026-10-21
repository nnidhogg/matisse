#include "matisse/encode/merger.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace matisse::encode
{
namespace
{
/**
 * @brief The number of byte values, each a token of one byte.
 */
constexpr std::size_t byte_count{256};

/**
 * @brief Returns a spelling as a refusal quotes it: a printable ASCII byte as itself, the quote and the backslash
 *        escaped, and every other byte as `\xHH`, so that a spelling outside ASCII reads the same in every terminal.
 * @param spelling The spelling's bytes.
 * @return The spelling, quoted.
 */
[[nodiscard]] std::string quoted_spelling(const std::string_view spelling)
{
    constexpr unsigned char first_printable{0x20};
    constexpr unsigned char last_printable{0x7E};

    std::string out{'"'};

    for (const auto byte : spelling)
    {
        const auto value{static_cast<unsigned char>(byte)};

        const auto printable{value >= first_printable && value <= last_printable};

        if (byte == '"' || byte == '\\')
        {
            out += std::format(R"(\{})", byte);
        }
        else if (printable)
        {
            out += byte;
        }
        else
        {
            out += std::format(R"(\x{:02X})", value);
        }
    }

    out += '"';

    return out;
}

/**
 * @brief The longest piece merged by a scan of its parts, a longer piece by the heap.
 *
 * Measured with no cache, the encoding of Chinese gains as the threshold rises to 128 bytes, 6 to 9 per cent on wiki-zh
 * under both tokenizers over a threshold of 64, and no further, 256, 512 and 1,024 measuring level with 128; every
 * other corpus measures level from 16 bytes on.
 */
constexpr std::size_t longest_scanned_piece{128};

/**
 * @brief The rank of no join, above every rank a join table holds, a definition holding fewer than 2^32 tokens and
 *        merges.
 */
constexpr std::uint32_t no_rank{std::numeric_limits<std::uint32_t>::max()};

/**
 * @brief One part of a piece under a scan: its id and the join it makes with the part after it.
 */
struct Scanned_part
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

/**
 * @brief Returns the index of the pair of a piece's bytes at an offset in the table of byte pairs.
 * @param piece The piece's bytes.
 * @param at The offset of the left byte, the right byte after it.
 * @return The left byte's value times 256 plus the right byte's.
 */
[[nodiscard]] std::size_t byte_pair_of(const std::string_view piece, const std::size_t at) noexcept
{
    const auto left{static_cast<unsigned char>(piece[at])};

    const auto right{static_cast<unsigned char>(piece[at + 1])};

    return left * byte_count + right;
}

/**
 * @brief Returns the ids of a piece merged by a scan of its parts: each part holding the join it makes with the part
 *        after it, the least rank found by a scan from the left, that join made, and the joins of the joined part with
 *        its neighbours looked up again.
 *
 * The scan makes the join the heap makes: the heap's entries that stand are one for every adjacent pair with a join,
 * and it joins the least of them by rank and then by start, which is the least rank the scan finds, the leftmost on a
 * tie since a later part replaces the least only at a rank below it; and both look up the same two pairs after a join,
 * the joined part's with the part before it and with the part after it, every other pair standing as it stood.
 * @param piece The piece's bytes, at most longest_scanned_piece.
 * @param byte_ids The id of every byte as a token of one byte.
 * @param byte_pairs The join of every pair of bytes as tokens of one byte, by the left byte's value times 256 plus the
 *        right byte's.
 * @param joins The join of every pair of parts the merge may join.
 * @return The ids of the parts the merge ends with, in order.
 */
[[nodiscard]] std::vector<load::Token_id_t> merged_by_scan(
        const std::string_view piece, const std::span<const load::Token_id_t, byte_count> byte_ids,
        const std::span<const Join_table::Join> byte_pairs, const Join_table& joins)
{
    std::array<Scanned_part, longest_scanned_piece> parts{};

    auto count{piece.size()};

    // Sets the join of the part at an index with the part after it, none after the last part.
    const auto look_up{[&](const std::size_t at) {
        auto& [id, rank, product]{parts[at]};

        rank = no_rank;

        if (at + 1 == count)
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

    for (std::size_t at{0}; at < count; ++at)
    {
        const auto byte{static_cast<unsigned char>(piece[at])};

        parts[at].id = byte_ids[byte];

        parts[at].rank = no_rank;
    }

    // Every part is a byte to begin with, so the first pairs are read from the table of byte pairs.
    for (std::size_t at{0}; at + 1 < count; ++at)
    {
        const auto pair{byte_pair_of(piece, at)};

        const auto [rank, product]{byte_pairs[pair]};

        parts[at].rank = rank;

        parts[at].product = product;
    }

    while (true)
    {
        auto least{count};

        auto least_rank{no_rank};

        for (std::size_t at{0}; at < count; ++at)
        {
            if (parts[at].rank < least_rank)
            {
                least = at;

                least_rank = parts[at].rank;
            }
        }

        if (least == count)
        {
            break;
        }

        parts[least].id = parts[least].product;

        const auto rest{std::ranges::subrange(parts.begin() + least + 2, parts.begin() + count)};

        std::ranges::copy(rest, parts.begin() + least + 1);

        --count;

        look_up(least);

        if (least > 0)
        {
            look_up(least - 1);
        }
    }

    const auto merged{std::span{parts}.first(count) | std::views::transform(&Scanned_part::id)};

    return {merged.begin(), merged.end()};
}

/**
 * @brief A byte offset into a piece the heap merges, which is shorter than 2^32 bytes.
 */
using Offset_t = std::uint32_t;

/**
 * @brief A candidate join of the merge: the claim that the part starting at `start` and ending at `end_left` and the
 *        part after it, ending at `end_right`, are adjacent parts of the piece, with the join's rank and product.
 *
 * Entries are ordered by rank and then by start, the order the heap pops them in. At most one entry of a start names
 * the parts that stand there, so the fields after the start never decide which join is made.
 */
struct Entry
{
    /**
     * @brief The join's rank.
     */
    std::uint32_t rank{};

    /**
     * @brief The byte at which the left part starts.
     */
    Offset_t start{};

    /**
     * @brief The byte after the left part.
     */
    Offset_t end_left{};

    /**
     * @brief The byte after the right part.
     */
    Offset_t end_right{};

    /**
     * @brief The id of the token the join makes.
     */
    load::Token_id_t product{};

    /** @brief Ordered by rank, then by start, then by the fields after them. */
    [[nodiscard]] auto operator<=>(const Entry& other) const = default;
};

/**
 * @brief The parts of a piece under merging: a doubly linked list over its bytes, node i the part that starts at byte
 *        i, live while it has not been joined into the part before it.
 *
 * A node's end only grows and a node once joined into its predecessor never lives again, so parts that are no longer
 * adjacent never stand together again; that is what lets the merge discard an entry when it is popped rather than when
 * its parts change.
 */
class Parts
{
public:
    /**
     * @brief The index that names no node, before the first and after the last.
     */
    static constexpr Offset_t none{std::numeric_limits<Offset_t>::max()};

    /**
     * @brief Makes every byte of a piece a part of its own, with the id of its byte.
     * @param piece The piece's bytes.
     * @param byte_ids The id of every byte as a token of one byte.
     */
    Parts(std::string_view piece, std::span<const load::Token_id_t, byte_count> byte_ids);

    /**
     * @brief Returns the part after a live part.
     * @param start The byte at which the part starts.
     * @return The byte at which the next part starts, none after the last.
     */
    [[nodiscard]] Offset_t next(Offset_t start) const noexcept;

    /**
     * @brief Returns the byte after a live part.
     * @param start The byte at which the part starts.
     * @return The byte after its last.
     */
    [[nodiscard]] Offset_t end(Offset_t start) const noexcept;

    /**
     * @brief Returns the id of a live part.
     * @param start The byte at which the part starts.
     * @return The id of the token the part spells.
     */
    [[nodiscard]] load::Token_id_t id(Offset_t start) const noexcept;

    /**
     * @brief Returns whether an entry's parts are still adjacent parts of the piece: its left part live and ending
     *        where the entry recorded, followed by a part ending where it recorded.
     * @param start The byte at which the entry's left part starts.
     * @param end_left The byte after the left part, as recorded.
     * @param end_right The byte after the right part, as recorded.
     * @return True when both parts stand as recorded.
     */
    [[nodiscard]] bool holds(Offset_t start, Offset_t end_left, Offset_t end_right) const noexcept;

    /**
     * @brief Joins a live part with the part after it.
     * @param start The byte at which the left part starts, a part with one after it.
     * @param product The id of the token the join makes.
     */
    void join(Offset_t start, load::Token_id_t product) noexcept;

    /**
     * @brief Returns the part before a live part.
     * @param start The byte at which the part starts.
     * @return The byte at which the part before it starts, none before the first.
     */
    [[nodiscard]] Offset_t previous(Offset_t start) const noexcept;

    /**
     * @brief Returns the ids of the live parts, in order.
     * @return The ids.
     */
    [[nodiscard]] std::vector<load::Token_id_t> ids() const;

private:
    /**
     * @brief One node of the list, the part that starts at its index.
     */
    struct Node
    {
        /**
         * @brief The byte after the part's last.
         */
        Offset_t end{};

        /**
         * @brief The start of the part before it, none before the first.
         */
        Offset_t previous{};

        /**
         * @brief The start of the part after it, none after the last.
         */
        Offset_t next{};

        /**
         * @brief The id of the token the part spells.
         */
        load::Token_id_t id{};

        /**
         * @brief Whether the part stands, not joined into the part before it.
         */
        bool live{};
    };

    /**
     * @brief The nodes, one per byte of the piece.
     */
    std::vector<Node> nodes_;
};

Parts::Parts(const std::string_view piece, const std::span<const load::Token_id_t, byte_count> byte_ids)
    : nodes_(piece.size())
{
    for (Offset_t start{0}; start < piece.size(); ++start)
    {
        const auto byte{static_cast<unsigned char>(piece[start])};

        const auto previous{start == 0 ? none : start - 1};

        const auto next{start + 1 < piece.size() ? start + 1 : none};

        nodes_[start] = {.end = start + 1, .previous = previous, .next = next, .id = byte_ids[byte], .live = true};
    }
}

Offset_t Parts::next(const Offset_t start) const noexcept
{
    return nodes_[start].next;
}

Offset_t Parts::end(const Offset_t start) const noexcept
{
    return nodes_[start].end;
}

load::Token_id_t Parts::id(const Offset_t start) const noexcept
{
    return nodes_[start].id;
}

bool Parts::holds(const Offset_t start, const Offset_t end_left, const Offset_t end_right) const noexcept
{
    const auto& [end, previous, next, id, live]{nodes_[start]};

    return live && end == end_left && next != none && nodes_[next].end == end_right;
}

void Parts::join(const Offset_t start, const load::Token_id_t product) noexcept
{
    auto& left{nodes_[start]};

    auto& right{nodes_[left.next]};

    left.end = right.end;

    left.next = right.next;

    left.id = product;

    right.live = false;

    if (left.next != none)
    {
        nodes_[left.next].previous = start;
    }
}

Offset_t Parts::previous(const Offset_t start) const noexcept
{
    return nodes_[start].previous;
}

std::vector<load::Token_id_t> Parts::ids() const
{
    std::vector<load::Token_id_t> ids{};

    for (const auto& [end, previous, next, id, live] : nodes_)
    {
        if (live)
        {
            ids.push_back(id);
        }
    }

    return ids;
}

/**
 * @brief Returns the ids of a piece merged by the heap: every adjacent pair with a join pushed as an entry, the least
 *        entry popped, discarded when its parts no longer stand as it recorded and otherwise joined, and the two pairs
 *        the join makes pushed.
 * @param piece The piece's bytes, shorter than 2^32 bytes.
 * @param byte_ids The id of every byte as a token of one byte.
 * @param byte_pairs The join of every pair of bytes as tokens of one byte, by the left byte's value times 256 plus the
 *        right byte's.
 * @param joins The join of every pair of parts the merge may join.
 * @return The ids of the parts the merge ends with, in order.
 * @throws std::length_error If the piece is 2^32 bytes or longer; the message names its length.
 */
[[nodiscard]] std::vector<load::Token_id_t> merged_by_heap(
        const std::string_view piece, const std::span<const load::Token_id_t, byte_count> byte_ids,
        const std::span<const Join_table::Join> byte_pairs, const Join_table& joins)
{
    if (piece.size() > std::numeric_limits<Offset_t>::max())
    {
        throw std::length_error{
                std::format("Merger: a piece of {} bytes is longer than the merge's 32-bit offsets", piece.size())};
    }

    Parts parts{piece, byte_ids};

    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> heap{};

    const auto push{[&](const Offset_t start) {
        const auto next{parts.next(start)};

        const auto left{parts.id(start)};

        const auto right{parts.id(next)};

        const auto join{joins.find(left, right)};

        if (!join)
        {
            return;
        }

        const auto [rank, product]{*join};

        const auto end_left{parts.end(start)};

        const auto end_right{parts.end(next)};

        heap.push({.rank = rank, .start = start, .end_left = end_left, .end_right = end_right, .product = product});
    }};

    // Every part is a byte to begin with, so the first pairs are read from the table of byte pairs.
    for (Offset_t start{0}; start + 1 < piece.size(); ++start)
    {
        const auto pair{byte_pair_of(piece, start)};

        const auto [rank, product]{byte_pairs[pair]};

        if (rank == no_rank)
        {
            continue;
        }

        const auto end_left{start + 1};

        const auto end_right{start + 2};

        heap.push({.rank = rank, .start = start, .end_left = end_left, .end_right = end_right, .product = product});
    }

    while (!heap.empty())
    {
        const auto [rank, start, end_left, end_right, product]{heap.top()};

        heap.pop();

        if (!parts.holds(start, end_left, end_right))
        {
            continue;
        }

        parts.join(start, product);

        const auto previous{parts.previous(start)};

        if (previous != Parts::none)
        {
            push(previous);
        }

        if (parts.next(start) != Parts::none)
        {
            push(start);
        }
    }

    return parts.ids();
}

} // namespace

Merger::Merger(const load::Definition& definition)
    : whole_piece_lookup_{definition.whole_piece_lookup}
    , ids_{definition.spellings}
    , byte_ids_{byte_ids_of(ids_)}
    , joins_{joins_of(definition, ids_)}
    , byte_pairs_{byte_pairs_of(byte_ids_, joins_)}
{}

Merger::Byte_ids_t Merger::byte_ids_of(const Spelling_table& ids)
{
    Byte_ids_t byte_ids{};

    for (std::size_t value{0}; value < byte_count; ++value)
    {
        const std::string byte{static_cast<char>(value)};

        const auto found{ids.find(byte)};

        if (!found)
        {
            throw std::invalid_argument{
                    std::format("Merger: the byte {} is not a token of the definition", quoted_spelling(byte))};
        }

        byte_ids[value] = *found;
    }

    return byte_ids;
}

Join_table::Joins_t Merger::joins_of(const load::Definition& definition, const Spelling_table& ids)
{
    const auto& [merge_rule, whole_piece_lookup, spellings, merges, specials]{definition};

    Join_table::Joins_t joins{};

    if (merge_rule == load::Merge_rule::ranks)
    {
        // Every split of a spelling into two spellings joins them into it, at its rank, its id; a pair of parts is met
        // only at the spellings equal to its concatenation, and emplace keeps the first, the id ids holds.
        for (const auto [id, spelling] : std::views::enumerate(spellings))
        {
            const std::string_view whole{spelling};

            const auto product_id{static_cast<load::Token_id_t>(id)};

            for (std::size_t split{1}; split < whole.size(); ++split)
            {
                const auto left{whole.substr(0, split)};

                const auto right{whole.substr(split)};

                const auto left_found{ids.find(left)};

                const auto right_found{ids.find(right)};

                if (!left_found || !right_found)
                {
                    continue;
                }

                const auto key{Join_table::key_of(*left_found, *right_found)};

                joins.emplace(key, Join_table::Join{.rank = product_id, .product = product_id});
            }
        }

        return joins;
    }

    for (std::size_t rank{0}; rank < merges.size(); ++rank)
    {
        const auto& [left, right]{merges[rank]};

        const auto product{left + right};

        const auto id_of{[&](const std::string_view spelling) {
            const auto found{ids.find(spelling)};

            if (!found)
            {
                throw std::invalid_argument{std::format(
                        "Merger: the merge ({}, {}) at rank {} names {}, which is not a token of the definition",
                        quoted_spelling(left), quoted_spelling(right), rank, quoted_spelling(spelling))};
            }

            return *found;
        }};

        const auto left_id{id_of(left)};

        const auto right_id{id_of(right)};

        const auto product_id{id_of(product)};

        const auto key{Join_table::key_of(left_id, right_id)};

        const auto merge_rank{static_cast<std::uint32_t>(rank)};

        joins.insert_or_assign(key, Join_table::Join{.rank = merge_rank, .product = product_id});
    }

    return joins;
}

Merger::Byte_pairs_t Merger::byte_pairs_of(const Byte_ids_t& byte_ids, const Join_table& joins)
{
    Byte_pairs_t byte_pairs(byte_count * byte_count, Join_table::Join{.rank = no_rank, .product = {}});

    for (std::size_t left{0}; left < byte_count; ++left)
    {
        for (std::size_t right{0}; right < byte_count; ++right)
        {
            const auto left_id{byte_ids[left]};

            const auto right_id{byte_ids[right]};

            const auto join{joins.find(left_id, right_id)};

            if (!join)
            {
                continue;
            }

            const auto pair{left * byte_count + right};

            byte_pairs[pair] = *join;
        }
    }

    return byte_pairs;
}

std::optional<load::Token_id_t> Merger::whole_piece(const std::string_view piece) const
{
    if (!whole_piece_lookup_)
    {
        return std::nullopt;
    }

    return ids_.find(piece);
}

std::vector<load::Token_id_t> Merger::merged(const std::string_view piece) const
{
    if (piece.size() <= longest_scanned_piece)
    {
        return merged_by_scan(piece, byte_ids_, byte_pairs_, joins_);
    }

    return merged_by_heap(piece, byte_ids_, byte_pairs_, joins_);
}

} // namespace matisse::encode
