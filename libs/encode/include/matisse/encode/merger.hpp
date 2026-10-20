#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_MERGER_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_MERGER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The ids of one piece under a definition, Merger: the whole-piece lookup, whole_piece(), and the merge of the
 *        definition's merge rule, merged().
 *
 * The merge keeps the parts of the piece as a doubly linked list over its bytes, as Hugging Face's Word keeps its
 * symbols, and the candidate joins in a heap ordered by rank and then by where the left part starts, each entry naming
 * the start of its left part and the ends of both parts. An entry popped whose parts are no longer adjacent parts of
 * the piece is discarded, and a join pushes the two pairs it makes. The rule enters only in the lookup of a pair: under
 * Merge_rule::ranks the rank of the two parts' concatenation, under Merge_rule::pairs the rank of the pair of parts in
 * the merge table. So under the rank rule the merge is tiktoken's _byte_pair_merge, which joins the pair of least rank,
 * leftmost first, a join that makes a pair of lower rank than its own included, and under the pair rule it is Hugging
 * Face's merge_all, one join at a time, the pair of least rank at its leftmost occurrence; each on every table and
 * every piece, since a discarded entry names parts that never stand together again and every adjacent pair the rule can
 * join has an entry. A piece of n bytes takes at most n - 1 joins and fewer than 3n entries.
 */
namespace matisse::encode
{
/**
 * @brief The ids of one piece under a definition: the whole-piece lookup and the merge of its merge rule.
 */
class Merger
{
public:
    /**
     * @brief Builds the tables of a definition's merge: the id of every spelling and of every byte, and under the pair
     *        rule the rank and product of every pair of the merge table, keyed by the ids of its parts.
     *
     * A pair listed twice takes its last rank, as encoder.py and Hugging Face's BPE model read it.
     * @param definition The definition.
     * @throws std::invalid_argument If a byte is not a token of the definition, or under the pair rule a merge names a
     *         part or a product that is not; the message names the byte, or the merge, its rank and the spelling.
     */
    explicit Merger(const load::Definition& definition);

    /**
     * @brief Returns the id of a piece looked up whole, before any merge, where the definition's whole-piece lookup
     *        says so.
     * @param piece The piece's bytes.
     * @return The id of the token the piece spells, std::nullopt when the definition does not look pieces up whole or
     *         the piece is no token.
     */
    [[nodiscard]] std::optional<load::Token_id_t> whole_piece(std::string_view piece) const;

    /**
     * @brief Returns the ids of a piece merged by the definition's merge rule from its bytes, without the whole-piece
     *        lookup.
     * @param piece The piece's bytes.
     * @return The ids of the parts the merge ends with, in order; none for the empty piece.
     */
    [[nodiscard]] std::vector<load::Token_id_t> merged(std::string_view piece) const;

private:
    /**
     * @brief A join the merge may make: its rank and the id of the token it makes.
     */
    struct Join
    {
        /**
         * @brief The rank, the id of the concatenation under the rank rule and the merge's index under the pair rule.
         */
        std::size_t rank{};

        /**
         * @brief The id of the token the join makes.
         */
        load::Token_id_t product{};
    };

    /**
     * @brief A hash of spellings that a string_view looks up without a copy.
     */
    struct Spelling_hash
    {
        /**
         * @brief Marks the hash as taking every type that converts to a string_view.
         */
        using is_transparent = void;

        /**
         * @brief Returns the hash of a spelling.
         * @param spelling The spelling's bytes.
         * @return The standard hash of its bytes.
         */
        [[nodiscard]] std::size_t operator()(std::string_view spelling) const noexcept;
    };

    /**
     * @brief The id of every spelling.
     */
    using Ids_t = std::unordered_map<std::string, load::Token_id_t, Spelling_hash, std::equal_to<>>;

    /**
     * @brief The join of every pair of the merge table, keyed by its left part's id above its right part's.
     */
    using Joins_t = std::unordered_map<std::uint64_t, Join>;

    /**
     * @brief The id of every byte as a token of one byte, by its value.
     */
    using Byte_ids_t = std::array<load::Token_id_t, 256>;

    /**
     * @brief Returns the id of every spelling.
     * @param spellings The bytes of every ordinary token, by id.
     * @return The ids, by spelling.
     */
    [[nodiscard]] static Ids_t ids_of(std::span<const std::string> spellings);

    /**
     * @brief Returns the id of every byte as a token of one byte.
     * @param ids The id of every spelling.
     * @return The ids, by byte value.
     * @throws std::invalid_argument If a byte is not a token; the message names the byte.
     */
    [[nodiscard]] static Byte_ids_t byte_ids_of(const Ids_t& ids);

    /**
     * @brief Returns the join of every pair of a merge table, a pair listed twice taking its last rank.
     * @param merges The merges in rank order.
     * @param ids The id of every spelling.
     * @return The joins, by the ids of the parts.
     * @throws std::invalid_argument If a merge names a part or a product that is not a token; the message names the
     *         merge, its rank and the spelling.
     */
    [[nodiscard]] static Joins_t joins_of(std::span<const munch::verifier::Merge> merges, const Ids_t& ids);

    /**
     * @brief How the bytes of a piece are merged.
     */
    load::Merge_rule merge_rule_;

    /**
     * @brief Whether a piece that is a token is looked up whole before any merge.
     */
    bool whole_piece_lookup_;

    /**
     * @brief The id of every spelling.
     */
    Ids_t ids_;

    /**
     * @brief The id of every byte as a token of one byte.
     */
    Byte_ids_t byte_ids_;

    /**
     * @brief The join of every pair of the merge table under the pair rule, empty under the rank rule.
     */
    Joins_t joins_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_MERGER_HPP
