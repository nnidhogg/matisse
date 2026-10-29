#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_BACKTRACKER_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_BACKTRACKER_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "matisse/encode/join_table.hpp"
#include "matisse/encode/token_trie.hpp"
#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The merge of a piece by backtracking, Backtracker: the tokens that encode to themselves laid along the piece
 *        from the left, the longest that begins the rest first, each held to the token before it by a test of their
 *        pair, the search giving way where the test would be asked about a token whose own merge does not join in
 *        order of rank.
 *
 * A token encodes to itself when the merge of its spelling ends with it alone; its split is the pair of parts that
 * merge joins last, and its order is the rank of that join plus one, zero for a token of one byte. By interval
 * projection every run of the merge that forms a token forms it by its split, and no run forms a token that does not
 * encode to itself. A token's own merge joins in order of rank, the leftmost first among equal ranks, exactly when the
 * parts of every split it makes, its own among them, have lower orders than the token they make, which of() decides
 * once for every token; the run on the spellings of two such tokens is the joins of their own merges in that order
 * until a join crosses between them, and the pair test reads that run back from the two tokens through their splits.
 * Two tokens are compatible when the merge of their spellings ends with them, and a sequence of tokens that encode to
 * themselves, every adjacent pair compatible, is the merge of its spellings, for the rank rule and for the pair rule,
 * with no condition on the table. The search lays the longest token that begins the rest of the piece and tries
 * shorter ones in turn, and takes the last token back when none fits, marking the position it went back from: the
 * tokens laid before a position are the merge of the bytes before it, one sequence however the search reaches it, so a
 * position from which no token fits once fits never. Where the pair test would be asked about a token whose own merge
 * does not join in order the search gives way, and until then every answer it acted on is the two tokens'
 * compatibility, so a search that does not give way ends where a search with an exact test ends, with the merge. The
 * method is GitHub's bpe crate's, the Corollaries III and IIIa of its README and its backtracking encoder; deciding for
 * each token whether its merge joins in order, which the crate's README assumes of every token of every table and its
 * constructor does not check, giving way where it does not, and reading every split from the token's own merge are
 * matisse's. Backtracker is the merge's own and is not part of matisse's stable surface. The header is installed only
 * because every header of the library lives under include, as munch keeps none under src, and it may change in minor
 * releases.
 */
namespace matisse::encode
{
/**
 * @brief The merge of a piece by backtracking over the tokens that encode to themselves, which gives way where its pair
 *        test would be asked about a token whose own merge does not join in order of rank.
 */
class Backtracker
{
public:
    /**
     * @brief Returns the backtracking merge of a definition, with whether the merge of each token that encodes to
     *        itself joins in order of rank: whether every split of two or more bytes it makes, its own among them, has
     *        parts of lower orders than the token they make.
     *
     * Each token's split is read from the merge of its spelling, run with the definition's joins until two parts are
     * left, which the definition's rule then joins into the token or does not.
     * @param spellings The bytes of every ordinary token, by id.
     * @param byte_ids The id of every byte as a token of one byte.
     * @param joins The join of every pair of parts the merge may join.
     * @return The backtracking merge.
     */
    [[nodiscard]] static Backtracker of(
            std::span<const std::string> spellings, std::span<const load::Token_id_t, 256> byte_ids,
            const Join_table& joins);

    /**
     * @brief Returns the ids of a piece merged by backtracking, which are the ids the definition's merge rule gives,
     *        unless the search gives way.
     *
     * The search gives way where its pair test would be asked about a token whose own merge does not join in order of
     * rank, on which the test is not exact.
     * @param piece The piece's bytes.
     * @return The ids of the tokens of the piece's merge, in order, none for the empty piece; std::nullopt when the
     *         search gives way.
     */
    [[nodiscard]] std::optional<std::vector<load::Token_id_t>> merged(std::string_view piece) const;

private:
    /**
     * @brief The split of a token that encodes to itself: its order, the two parts its own merge joins last, whether
     *        that merge joins in order of rank, and which tokens it is a part of a split beside.
     */
    struct Split
    {
        /**
         * @brief The rank of the join that makes the token, plus one; zero for a token of one byte.
         */
        std::uint32_t order{};

        /**
         * @brief The id of the left part, the token itself for a token of one byte.
         */
        load::Token_id_t left{};

        /**
         * @brief The id of the right part, the token itself for a token of one byte.
         */
        load::Token_id_t right{};

        /**
         * @brief Whether the token's own merge joins in order of rank, both parts' orders below the token's and both
         *        parts' own merges joining in order; true for a token of one byte.
         */
        bool in_order{};

        /**
         * @brief The token's own partner_bit(), which the pair test reads beside the token's split.
         */
        std::uint32_t bit{};

        /**
         * @brief The partner_bit() of every token that is the right part of a split whose left part is this token.
         */
        std::uint32_t followers{};

        /**
         * @brief The partner_bit() of every token that is the left part of a split whose right part is this token.
         */
        std::uint32_t leaders{};
    };

    /**
     * @brief Keeps the tokens that encode to themselves, their splits and the splits by their parts.
     * @param tokens The trie of the tokens that encode to themselves.
     * @param splits The split of every token that encodes to itself, by id.
     * @param pairs The split of every token of two or more bytes that encodes to itself, by its parts' ids, the rank of
     *        its join and the token.
     */
    Backtracker(Token_trie tokens, std::vector<Split> splits, Join_table pairs);

    /**
     * @brief Returns the parts the merge of a spelling leaves when two are left.
     * @param spelling The spelling, at least two bytes.
     * @param byte_ids The id of every byte as a token of one byte.
     * @param joins The join of every pair of parts the merge may join.
     * @return The ids of the two parts, std::nullopt when the merge stops at more than two.
     */
    [[nodiscard]] static std::optional<std::pair<load::Token_id_t, load::Token_id_t>> last_parts_of(
            std::string_view spelling, std::span<const load::Token_id_t, 256> byte_ids, const Join_table& joins);

    /**
     * @brief Returns the one bit of 32 a token stands for among the partners a split records of another.
     * @param id The token's id.
     * @return The bit, the id's hash by Boost's hash_combine modulo 32.
     */
    [[nodiscard]] static std::uint32_t partner_bit(load::Token_id_t id) noexcept;

    /**
     * @brief Returns whether two tokens that encode to themselves, each by a merge that joins in order of rank, are
     *        compatible: the merge of their spellings ends with them, no join crossing between them.
     *
     * From the pair of the two tokens back to the pair of their bytes beside the boundary, each step undoes the later
     * of the joins that made the two parts beside it, the right part's on a tie of orders, since a run joins the
     * leftmost first among equal ranks; the pair of parts it reaches stood beside the boundary until that join, and a
     * join of that pair crosses the boundary when its rank is below the undone join's on the left, which precedes it on
     * a tie, or at most the undone join's on the right, which it precedes. The pair of the two tokens themselves
     * crosses when it has any join. Only a split's pair can cross, since a join a run makes is the split of the token
     * it makes, and a pair is looked up only when each part records the other as a partner. On a token whose merge
     * does not join in order the later join is not always the one of higher order, and the answer can be wrong.
     * @param left The left token's id.
     * @param right The right token's id.
     * @return True when no join crosses between them.
     */
    [[nodiscard]] bool compatible(load::Token_id_t left, load::Token_id_t right) const noexcept;

    /**
     * @brief The tokens that encode to themselves, longest first at a position.
     */
    Token_trie tokens_;

    /**
     * @brief The split of every token that encodes to itself, by id.
     */
    std::vector<Split> splits_;

    /**
     * @brief The split of every token of two or more bytes that encodes to itself, by its parts' ids.
     */
    Join_table pairs_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_BACKTRACKER_HPP
