#include "matisse/encode/merger.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
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
 * @brief Returns the key of a pair of parts in the join table.
 * @param left The left part's id.
 * @param right The right part's id.
 * @return The left id in the upper 32 bits and the right id in the lower.
 */
[[nodiscard]] constexpr std::uint64_t key_of(const load::Token_id_t left, const load::Token_id_t right) noexcept
{
    return (std::uint64_t{left} << 32U) | right;
}

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
    std::size_t rank{};

    /**
     * @brief The byte at which the left part starts.
     */
    std::size_t start{};

    /**
     * @brief The byte after the left part.
     */
    std::size_t end_left{};

    /**
     * @brief The byte after the right part.
     */
    std::size_t end_right{};

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
    static constexpr std::size_t none{std::numeric_limits<std::size_t>::max()};

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
    [[nodiscard]] std::size_t next(std::size_t start) const noexcept;

    /**
     * @brief Returns the byte after a live part.
     * @param start The byte at which the part starts.
     * @return The byte after its last.
     */
    [[nodiscard]] std::size_t end(std::size_t start) const noexcept;

    /**
     * @brief Returns the id of a live part.
     * @param start The byte at which the part starts.
     * @return The id of the token the part spells.
     */
    [[nodiscard]] load::Token_id_t id(std::size_t start) const noexcept;

    /**
     * @brief Returns whether an entry's parts are still adjacent parts of the piece: its left part live and ending
     *        where the entry recorded, followed by a part ending where it recorded.
     * @param start The byte at which the entry's left part starts.
     * @param end_left The byte after the left part, as recorded.
     * @param end_right The byte after the right part, as recorded.
     * @return True when both parts stand as recorded.
     */
    [[nodiscard]] bool holds(std::size_t start, std::size_t end_left, std::size_t end_right) const noexcept;

    /**
     * @brief Joins a live part with the part after it.
     * @param start The byte at which the left part starts, a part with one after it.
     * @param product The id of the token the join makes.
     */
    void join(std::size_t start, load::Token_id_t product) noexcept;

    /**
     * @brief Returns the part before a live part.
     * @param start The byte at which the part starts.
     * @return The byte at which the part before it starts, none before the first.
     */
    [[nodiscard]] std::size_t previous(std::size_t start) const noexcept;

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
        std::size_t end{};

        /**
         * @brief The start of the part before it, none before the first.
         */
        std::size_t previous{};

        /**
         * @brief The start of the part after it, none after the last.
         */
        std::size_t next{};

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
    for (std::size_t start{0}; start < piece.size(); ++start)
    {
        const auto byte{static_cast<unsigned char>(piece[start])};

        const auto previous{start == 0 ? none : start - 1};

        const auto next{start + 1 < piece.size() ? start + 1 : none};

        nodes_[start] = {.end = start + 1, .previous = previous, .next = next, .id = byte_ids[byte], .live = true};
    }
}

std::size_t Parts::next(const std::size_t start) const noexcept
{
    return nodes_[start].next;
}

std::size_t Parts::end(const std::size_t start) const noexcept
{
    return nodes_[start].end;
}

load::Token_id_t Parts::id(const std::size_t start) const noexcept
{
    return nodes_[start].id;
}

bool Parts::holds(const std::size_t start, const std::size_t end_left, const std::size_t end_right) const noexcept
{
    const auto& [end, previous, next, id, live]{nodes_[start]};

    return live && end == end_left && next != none && nodes_[next].end == end_right;
}

void Parts::join(const std::size_t start, const load::Token_id_t product) noexcept
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

std::size_t Parts::previous(const std::size_t start) const noexcept
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

} // namespace

Merger::Merger(const load::Definition& definition)
    : merge_rule_{definition.merge_rule}
    , whole_piece_lookup_{definition.whole_piece_lookup}
    , ids_{ids_of(definition.spellings)}
    , byte_ids_{byte_ids_of(ids_)}
    , joins_{joins_of(definition.merges, ids_)}
{}

Merger::Ids_t Merger::ids_of(const std::span<const std::string> spellings)
{
    Ids_t ids{};

    for (std::size_t id{0}; id < spellings.size(); ++id)
    {
        ids.emplace(spellings[id], static_cast<load::Token_id_t>(id));
    }

    return ids;
}

std::size_t Merger::Spelling_hash::operator()(const std::string_view spelling) const noexcept
{
    return std::hash<std::string_view>{}(spelling);
}

Merger::Byte_ids_t Merger::byte_ids_of(const Ids_t& ids)
{
    Byte_ids_t byte_ids{};

    for (std::size_t value{0}; value < byte_count; ++value)
    {
        const std::string byte{static_cast<char>(value)};

        const auto found{ids.find(byte)};

        if (found == ids.end())
        {
            throw std::invalid_argument{
                    std::format("Merger: the byte {} is not a token of the definition", quoted_spelling(byte))};
        }

        byte_ids[value] = found->second;
    }

    return byte_ids;
}

Merger::Joins_t Merger::joins_of(const std::span<const munch::verifier::Merge> merges, const Ids_t& ids)
{
    Joins_t joins{};

    for (std::size_t rank{0}; rank < merges.size(); ++rank)
    {
        const auto& [left, right]{merges[rank]};

        const auto product{left + right};

        const auto id_of{[&](const std::string_view spelling) {
            const auto found{ids.find(spelling)};

            if (found == ids.end())
            {
                throw std::invalid_argument{std::format(
                        "Merger: the merge ({}, {}) at rank {} names {}, which is not a token of the definition",
                        quoted_spelling(left), quoted_spelling(right), rank, quoted_spelling(spelling))};
            }

            return found->second;
        }};

        const auto left_id{id_of(left)};

        const auto right_id{id_of(right)};

        const auto product_id{id_of(product)};

        const auto key{key_of(left_id, right_id)};

        joins.insert_or_assign(key, Join{.rank = rank, .product = product_id});
    }

    return joins;
}

std::optional<load::Token_id_t> Merger::whole_piece(const std::string_view piece) const
{
    if (!whole_piece_lookup_)
    {
        return std::nullopt;
    }

    const auto found{ids_.find(piece)};

    if (found == ids_.end())
    {
        return std::nullopt;
    }

    return found->second;
}

std::vector<load::Token_id_t> Merger::merged(const std::string_view piece) const
{
    Parts parts{piece, byte_ids_};

    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> heap{};

    // The rule enters here alone: the rank of the parts' concatenation, or the rank of the pair of parts.
    const auto join_at{[&](const std::size_t start) -> std::optional<Join> {
        const auto next{parts.next(start)};

        if (merge_rule_ == load::Merge_rule::ranks)
        {
            const auto end{parts.end(next)};

            const auto concatenation{piece.substr(start, end - start)};

            const auto found{ids_.find(concatenation)};

            if (found == ids_.end())
            {
                return std::nullopt;
            }

            return Join{.rank = found->second, .product = found->second};
        }

        const auto left{parts.id(start)};

        const auto right{parts.id(next)};

        const auto key{key_of(left, right)};

        const auto found{joins_.find(key)};

        if (found == joins_.end())
        {
            return std::nullopt;
        }

        return found->second;
    }};

    const auto push{[&](const std::size_t start) {
        const auto join{join_at(start)};

        if (!join)
        {
            return;
        }

        const auto [rank, product]{*join};

        const auto end_left{parts.end(start)};

        const auto next{parts.next(start)};

        const auto end_right{parts.end(next)};

        heap.push({.rank = rank, .start = start, .end_left = end_left, .end_right = end_right, .product = product});
    }};

    for (std::size_t start{0}; start + 1 < piece.size(); ++start)
    {
        push(start);
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

} // namespace matisse::encode
