#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_MERGER_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_MERGER_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "matisse/encode/join_table.hpp"
#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The ids of one piece under a definition, Merger: the whole-piece lookup, whole_piece(), and the merge of the
 *        definition's merge rule, merged().
 *
 * The merge keeps the parts of the piece as a doubly linked list over its bytes, as Hugging Face's Word keeps its
 * symbols, and the candidate joins in a heap ordered by rank and then by where the left part starts, each entry naming
 * the start of its left part and the ends of both parts. An entry popped whose parts are no longer adjacent parts of
 * the piece is discarded, and a join pushes the two pairs it makes. A pair is looked up by the ids of its parts in a
 * Join_table, in join_table.hpp, and the rule enters only in the joins that table holds: under Merge_rule::ranks every
 * split of a token into two tokens, the pair joining at the rank of the token they spell, which is the rank of the two
 * parts' concatenation, and under Merge_rule::pairs every pair of the merge table at its rank. So under the rank rule
 * the merge is tiktoken's _byte_pair_merge, which joins the pair of least rank, leftmost first, a join that makes a
 * pair of lower rank than its own included, and under the pair rule it is Hugging Face's merge_all, one join at a time,
 * the pair of least rank at its leftmost occurrence; each on every table and every piece, since a discarded entry names
 * parts that never stand together again and every adjacent pair the rule can join has an entry. A piece of n bytes
 * takes at most n - 1 joins and fewer than 3n entries. A piece of at most 128 bytes is merged instead by a scan of its
 * parts held in an array on the stack, each with the join it makes with the next, which makes the same joins in the
 * same order: the least rank present, the leftmost on a tie, and after each join the two pairs it makes looked up.
 * Every part is a byte to begin with, so both read the first pairs from a table of the joins of all 65,536 pairs of
 * bytes, filled from the join table at construction, and look later pairs up in the join table.
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
     * @brief Builds the tables of a definition's merge: the id of every spelling and of every byte, and the rank and
     *        product of every pair of parts the merge may join, keyed by the ids of the parts.
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
     * @throws std::length_error If the piece is 2^32 bytes or longer, past the merge's 32-bit offsets; the message
     *         names its length.
     */
    [[nodiscard]] std::vector<load::Token_id_t> merged(std::string_view piece) const;

private:
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
     * @brief The id of every byte as a token of one byte, by its value.
     */
    using Byte_ids_t = std::array<load::Token_id_t, 256>;

    /**
     * @brief The join of every pair of bytes as tokens of one byte, by the left byte's value times 256 plus the right
     *        byte's, a rank of all ones where the pair has none.
     */
    using Byte_pairs_t = std::vector<Join_table::Join>;

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
     * @brief Returns the join of every pair of parts a definition's merge may join: under the rank rule every split of
     *        a spelling into two spellings, joining at the rank of the spelling, its id; under the pair rule every pair
     *        of the merge table, a pair listed twice taking its last rank.
     * @param definition The definition.
     * @param ids The id of every spelling.
     * @return The joins, by the key of the parts' ids, which the join table is built from.
     * @throws std::invalid_argument If under the pair rule a merge names a part or a product that is not a token; the
     *         message names the merge, its rank and the spelling.
     */
    [[nodiscard]] static Join_table::Joins_t joins_of(const load::Definition& definition, const Ids_t& ids);

    /**
     * @brief Returns the join of every pair of bytes as tokens of one byte, which every piece's merge looks up first,
     *        read from the join table once so that those lookups index a table of 65,536 joins rather than hash.
     * @param byte_ids The id of every byte as a token of one byte.
     * @param joins The join of every pair of parts the merge may join.
     * @return The joins, by the left byte's value times 256 plus the right byte's.
     */
    [[nodiscard]] static Byte_pairs_t byte_pairs_of(const Byte_ids_t& byte_ids, const Join_table& joins);

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
     * @brief The join of every pair of parts the merge may join, in the flat table the merge looks pairs up in.
     */
    Join_table joins_;

    /**
     * @brief The join of every pair of bytes as tokens of one byte.
     */
    Byte_pairs_t byte_pairs_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_MERGER_HPP
