#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_JOIN_TABLE_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_JOIN_TABLE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The joins of a merge by the ids of their parts, Join_table: a flat table with open addressing that the merge
 *        looks every pair up in, the pairs of two bytes once, at construction.
 *
 * The table is built from a map of the joins and holds exactly its entries. Every key is placed once, at the first free
 * slot of its walk, the slots from its home on; the slots that placement passed over were full then and no slot is ever
 * emptied, so a lookup, which walks from the key's home and stops at the key or at a free slot, meets every key the map
 * holds before a free slot, and stops at a free slot for any other. The slots are a power of two in number, at least
 * twice the joins, the home the key's hash by Boost's hash_combine taken modulo their number, and the walk the next
 * slot after each, linear probing; a free slot is marked by the key of two ids of all ones, which no pair of a
 * definition's ids is, a definition holding fewer than 2^32 tokens. On cl100k's file the table holds 233,378 pairs in
 * 524,288 slots.
 */
namespace matisse::encode
{
/**
 * @brief The joins of a merge by the ids of their parts, in a flat table with open addressing.
 */
class Join_table
{
public:
    /**
     * @brief A join the merge may make: its rank and the id of the token it makes.
     */
    struct Join
    {
        /** @brief Equal when the rank and the product are. */
        bool operator==(const Join&) const = default;

        /**
         * @brief The rank, the id of the concatenation under the rank rule and the merge's index under the pair rule.
         */
        std::uint32_t rank{};

        /**
         * @brief The id of the token the join makes.
         */
        load::Token_id_t product{};
    };

    /**
     * @brief The joins a table is built from, by key_of() of their parts.
     */
    using Joins_t = std::unordered_map<std::uint64_t, Join>;

    /**
     * @brief Places every join of a map in the table.
     * @param joins The joins, by key_of() of their parts, none of the key of two ids of all ones.
     */
    explicit Join_table(const Joins_t& joins);

    /**
     * @brief Returns the key of a pair of parts.
     * @param left The left part's id.
     * @param right The right part's id.
     * @return The left id in the upper 32 bits and the right id in the lower.
     */
    [[nodiscard]] static constexpr std::uint64_t key_of(
            const load::Token_id_t left, const load::Token_id_t right) noexcept
    {
        return (std::uint64_t{left} << 32U) | right;
    }

    /**
     * @brief Returns the join of a pair of parts.
     * @param left The left part's id.
     * @param right The right part's id.
     * @return The join the map held for the pair, std::nullopt when it held none.
     */
    [[nodiscard]] std::optional<Join> find(load::Token_id_t left, load::Token_id_t right) const noexcept;

private:
    /**
     * @brief One slot of the table: a key and its join, or the free key.
     */
    struct Slot
    {
        /**
         * @brief The key of the pair the slot holds, free_key when it holds none.
         */
        std::uint64_t key{};

        /**
         * @brief The pair's join.
         */
        Join join{};
    };

    /**
     * @brief The key of a free slot, the pair of two ids of all ones.
     */
    static constexpr std::uint64_t free_key{std::numeric_limits<std::uint64_t>::max()};

    /**
     * @brief Returns the number of slots a table of joins takes: the least power of two at least twice their number.
     * @param joins The number of joins.
     * @return The number of slots, at least one, and more than the joins when there are any.
     */
    [[nodiscard]] static std::size_t slot_count_of(std::size_t joins) noexcept;

    /**
     * @brief Returns the slot a key's walk starts at.
     * @param key The key.
     * @return The key's hash by Boost's hash_combine, modulo the number of slots.
     */
    [[nodiscard]] std::size_t home(std::uint64_t key) const noexcept;

    /**
     * @brief The number of slots less one, which a hash or a slot is masked by to wrap.
     */
    std::size_t mask_;

    /**
     * @brief The slots.
     */
    std::vector<Slot> slots_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_JOIN_TABLE_HPP
