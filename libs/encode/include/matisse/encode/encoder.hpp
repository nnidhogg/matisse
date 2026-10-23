#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "matisse/encode/merger.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/encode/verifier_table.hpp"
#include "matisse/encode/window_index.hpp"
#include "matisse/load/definition.hpp"
#include "matisse/pretok/classes.hpp"
#include "matisse/pretok/pattern.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/validate/report.hpp"

/**
 * @file
 * @brief The encoder, Encoder with encode() and decode(): the token ids of a text under a definition, and the text of a
 *        run of ids; and the parallel encoding, chunk_boundaries() and encode_parallel(), which cuts the text only
 *        where the inventory's window table certifies a cut.
 *
 * encode() splits the text at every allowed special token, segments each stretch between them as a complete input, and
 * merges every piece alone by the definition's merge rule, the pair rule or tiktoken's rank merge, a piece that is a
 * token taking its id whole where the definition's whole-piece lookup says so; Merger, in merger.hpp, holds the lookup
 * and the merge, and a Piece_cache, in piece_cache.hpp, keeps the merged pieces of one call or of one caller's calls.
 * chunk_boundaries() walks forward from each equal byte target to the first cut, an anchor of the window table, which
 * Window_index, in window_index.hpp, finds, or an edge of an allowed special token, and encode_parallel() gives each
 * chunk to a worker that segments it with the table's right context and keeps the pieces ending at its cut. The encoder
 * claims one thing: for every text the definition accepts, encode_parallel() returns the ids encode() returns, at every
 * chunk count. That is the splice theorem of the report on pre-tokenizer certification with the proposition that a
 * certified cut is a pre-token boundary and the decided right context. The encoder also builds what the streaming
 * encoder of stream_encoder.hpp reads, once for every stream over it: the class table of the pattern's alphabet, the
 * pattern's split verifier as a Verifier_table, in verifier_table.hpp, the ordinary tokens in the order of their
 * spellings and the tokens the merge does not form, which its accessors give.
 */
namespace matisse::encode
{
/**
 * @brief The encoder of a definition under its pattern: the ids of a text, with the special tokens it allows cut out.
 */
class Encoder
{
public:
    /**
     * @brief Builds the encoder of a definition under the pattern a report was decided of, keeping the inventory's
     *        window table and the special tokens allowed.
     * @param definition The definition.
     * @param report The report validate() decided of the pattern.
     * @param inventory The inventory whose window table and right context the encoder keeps.
     * @param pattern The pattern.
     * @param allowed The spellings of the special tokens cut out of a text; every other special token is ordinary text.
     * @throws std::invalid_argument If an allowed spelling is not one of the definition's special tokens, or the
     *         definition cannot be merged, as Merger refuses it; the message names the spelling, the byte or the merge.
     */
    Encoder(load::Definition definition, const validate::Report& report, validate::Inventory inventory,
            const pretok::Pattern& pattern, std::span<const std::string> allowed);

    /**
     * @brief Returns the ids of a text, as the definition's tokenizer encodes it with the allowed special tokens.
     *
     * The text is cut at every occurrence of an allowed special token, the earliest first and the longest among those
     * that begin at one byte, each emitting its id; each stretch between them is segmented by the pattern as a complete
     * input. A piece that is a token is that one token where the definition's whole-piece lookup says so, and every
     * other piece is merged alone by the definition's merge rule. That is tiktoken's encode() with these special tokens
     * allowed and none disallowed, where no allowed spelling begins another, as on every definition in scope, and
     * encoder.py's encode() when none is allowed.
     * @param text The text, well-formed UTF-8.
     * @return The ids.
     * @throws std::invalid_argument If the text is not well-formed UTF-8, as the Segmenter refuses it, the byte offset
     *         counted from the start of the stretch that holds the ill-formed sequence.
     * @throws std::length_error If a piece is 2^32 bytes or longer, as Merger::merged() refuses it.
     */
    [[nodiscard]] std::vector<load::Token_id_t> encode(std::string_view text) const;

    /**
     * @brief Returns the ids of a text as encode() does, the merged pieces kept in a cache the caller owns across its
     *        calls.
     *
     * The cache holds the ids of a piece as this encoder's definition merges them, so a cache filled by one encoder is
     * not passed to another; the encoder is not changed by a call and the cache is the only state a call shares with
     * its caller, so it serves one thread at a time with no lock, each thread of a pool holding its own.
     * @param text The text, well-formed UTF-8.
     * @param cache The cache, filled by this encoder alone.
     * @return The ids.
     * @throws std::invalid_argument If the text is not well-formed UTF-8, as encode() refuses it.
     * @throws std::length_error If a piece is 2^32 bytes or longer, as encode() refuses it.
     */
    [[nodiscard]] std::vector<load::Token_id_t> encode(std::string_view text, Piece_cache& cache) const;

    /**
     * @brief Returns where encode_parallel() cuts a text into chunks: each interior boundary the first cut at or after
     *        the later of its equal-division target and one past the previous boundary.
     *
     * A cut is an anchor of the inventory's window table, the origin of an occurrence of a certified window inside one
     * stretch between allowed special tokens, at least the width less one code points after the previous boundary, so
     * that the occurrence begins in the chunk the cut ends, and as far after the stretch's start; or it is where an
     * allowed special token begins or ends, a cut of its own. Every cut is a pre-token boundary of the text, at a code
     * point and outside every special token. The spacing leaves out an anchor that stands closer to the previous
     * boundary, and a target with no cut after it ends the walk, the last chunk then running to the text's end.
     * @param text The text, well-formed UTF-8.
     * @param chunks The number of chunks aimed for; fewer result when cuts are scarce, and zero behaves as one, the
     *        whole text as a single chunk.
     * @return Offsets from 0 to the text's size inclusive; adjacent pairs delimit the chunks.
     * @throws std::invalid_argument If the text the walk classes is not well-formed UTF-8, as classes_of() refuses it.
     */
    [[nodiscard]] std::vector<std::size_t> chunk_boundaries(std::string_view text, std::size_t chunks) const;

    /**
     * @brief Returns the ids of a text encoded as concurrent chunks cut by chunk_boundaries(), which are the ids
     *        encode() returns.
     *
     * Each chunk is encoded on its own thread, the last on the calling thread, with a piece cache of its own: its
     * stretches are segmented as encode() segments them, but the last one with the right context of the inventory's
     * window table, that many further code points of the text, read as a complete input there unless an allowed special
     * token or the text's end comes first, and only the pieces ending at or before the chunk's end are kept. The ids of
     * the chunks, concatenated, are the result. An exception a worker throws is kept and rethrown on the calling thread
     * once every worker has joined, the first one when several throw.
     * @param text The text, well-formed UTF-8.
     * @param chunks As for chunk_boundaries().
     * @return The ids.
     * @throws std::invalid_argument If the text is not well-formed UTF-8, as encode() refuses it, the byte offset
     *         counted from the start of what the walk or the worker classes.
     * @throws std::length_error If a piece is 2^32 bytes or longer, as encode() refuses it.
     */
    [[nodiscard]] std::vector<load::Token_id_t> encode_parallel(std::string_view text, std::size_t chunks) const;

    /**
     * @brief Returns the text a run of ids spells: each ordinary token's bytes and each special token's spelling.
     * @param ids The ids.
     * @return The text.
     * @throws std::invalid_argument If an id is neither an ordinary nor a special token of the definition; the message
     *         names the id and its index.
     */
    [[nodiscard]] std::string decode(std::span<const load::Token_id_t> ids) const;

    /**
     * @brief Returns the definition, whose spellings, whole-piece lookup and special tokens a stream reads.
     * @return The definition.
     */
    [[nodiscard]] const load::Definition& definition() const noexcept;

    /**
     * @brief Returns the definition's whole-piece lookup and merge, which a stream merges its pieces by.
     * @return The merger.
     */
    [[nodiscard]] const Merger& merger() const noexcept;

    /**
     * @brief Returns the special tokens cut out of a text, in the order their spellings were allowed.
     * @return The allowed special tokens.
     */
    [[nodiscard]] std::span<const load::Special_token> allowed() const noexcept;

    /**
     * @brief Returns the class table of the pattern's alphabet, which a stream classes its code points by.
     * @return The class table.
     */
    [[nodiscard]] const pretok::Class_table& class_table() const noexcept;

    /**
     * @brief Returns the pattern's split verifier M in a flat table, which a stream steps its live runs by.
     * @return The verifier's table.
     */
    [[nodiscard]] const Verifier_table& verifier() const noexcept;

    /**
     * @brief Returns the ids of the ordinary tokens in the order of their spellings, bytewise ascending, which a stream
     *        finds the tokens that begin with a string in.
     * @return The ids.
     */
    [[nodiscard]] std::span<const load::Token_id_t> spellings_sorted() const noexcept;

    /**
     * @brief Returns the ids of the ordinary tokens of two or more bytes that the merge does not form from their own
     *        bytes, which only the whole-piece lookup gives, in the order of their spellings; none when the definition
     *        does not look pieces up whole.
     * @return The ids.
     */
    [[nodiscard]] std::span<const load::Token_id_t> unformed_sorted() const noexcept;

private:
    /**
     * @brief An occurrence of an allowed special token in a text: where it begins and which token it is.
     */
    struct Occurrence
    {
        /**
         * @brief The byte at which the occurrence begins.
         */
        std::size_t at{};

        /**
         * @brief The token's index among the allowed special tokens.
         */
        std::size_t special{};
    };

    /**
     * @brief Returns the special tokens of a definition that are allowed, in the order the spellings are given.
     * @param specials The definition's special tokens.
     * @param allowed The spellings allowed.
     * @return The allowed special tokens.
     * @throws std::invalid_argument If an allowed spelling is not one of the special tokens; the message names it.
     */
    [[nodiscard]] static std::vector<load::Special_token> allowed_of(
            std::span<const load::Special_token> specials, std::span<const std::string> allowed);

    /**
     * @brief Returns the ids of a definition's ordinary tokens in the order of their spellings, bytewise ascending.
     * @param spellings The bytes of every ordinary token, by id.
     * @return The ids.
     */
    [[nodiscard]] static std::vector<load::Token_id_t> sorted_of(std::span<const std::string> spellings);

    /**
     * @brief Returns the ids of the ordinary tokens of two or more bytes that a merger does not form from their own
     *        bytes, in the order of their spellings, when the definition looks pieces up whole.
     * @param definition The definition.
     * @param merger The definition's merger.
     * @param sorted The ids of the ordinary tokens in the order of their spellings.
     * @return The ids, none when the definition does not look pieces up whole.
     */
    [[nodiscard]] static std::vector<load::Token_id_t> unformed_of(
            const load::Definition& definition, const Merger& merger, std::span<const load::Token_id_t> sorted);

    /**
     * @brief Returns the occurrences of the allowed special tokens in a text that encode() cuts out, front to back:
     *        from the end of each, the earliest next one, the longest among those that begin at one byte.
     *
     * Each token's next occurrence is kept until the search passes it, so that the text is read once for every token
     * however many of its occurrences the search finds.
     * @param text The text.
     * @return The occurrences, ascending, none overlapping another.
     */
    [[nodiscard]] std::vector<Occurrence> occurrences(std::string_view text) const;

    /**
     * @brief Returns the ids of a chunk of a text, its stretches segmented as complete inputs but the last one with the
     *        right context, only its pieces ending at or before the chunk's end kept.
     * @param text The text.
     * @param found The occurrences of the allowed special tokens in the text, as occurrences() finds them.
     * @param begin The chunk's first byte, the text's start or a cut.
     * @param end The byte after the chunk, a cut or the text's end.
     * @param cache The cache of merged pieces the chunk's pieces are looked up in and stored to.
     * @return The ids.
     * @throws std::invalid_argument If what the chunk segments is not well-formed UTF-8, as the Segmenter refuses it.
     * @throws std::length_error If a piece is 2^32 bytes or longer, as Merger::merged() refuses it.
     */
    [[nodiscard]] std::vector<load::Token_id_t> encode_chunk(
            std::string_view text, std::span<const Occurrence> found, std::size_t begin, std::size_t end,
            Piece_cache& cache) const;

    /**
     * @brief Returns the boundaries chunk_boundaries() returns, the occurrences of the allowed special tokens found.
     * @param text The text.
     * @param found The occurrences of the allowed special tokens in the text, as occurrences() finds them.
     * @param chunks The number of chunks aimed for.
     * @return The boundaries.
     * @throws std::invalid_argument If the text the walk classes is not well-formed UTF-8, as classes_of() refuses it.
     */
    [[nodiscard]] std::vector<std::size_t> boundaries_of(
            std::string_view text, std::span<const Occurrence> found, std::size_t chunks) const;

    /**
     * @brief Returns the first cut of a text at or after a byte: an anchor spaced from the previous boundary and from
     *        its stretch's start, or an edge of an allowed special token.
     * @param text The text.
     * @param found The occurrences of the allowed special tokens in the text, as occurrences() finds them.
     * @param previous The previous boundary.
     * @param floor The byte, after the previous boundary.
     * @return The cut, std::nullopt when none stands at or after the byte before the text's end.
     * @throws std::invalid_argument If the text the walk classes is not well-formed UTF-8, as classes_of() refuses it.
     */
    [[nodiscard]] std::optional<std::size_t> next_cut(
            std::string_view text, std::span<const Occurrence> found, std::size_t previous, std::size_t floor) const;

    /**
     * @brief The definition, whose spellings and special tokens decode() reads.
     */
    load::Definition definition_;

    /**
     * @brief The definition's whole-piece lookup and merge.
     */
    Merger merger_;

    /**
     * @brief The pattern's segmenter.
     */
    pretok::Segmenter segmenter_;

    /**
     * @brief The special tokens cut out of a text.
     */
    std::vector<load::Special_token> allowed_;

    /**
     * @brief The inventory's certified windows by their class strings, which the parallel encoding cuts by.
     */
    Window_index windows_;

    /**
     * @brief The right context of the inventory, the code points a worker reads past its chunk's end.
     */
    std::size_t right_context_;

    /**
     * @brief The class table of the pattern's alphabet, which a stream classes its code points by.
     */
    pretok::Class_table class_table_;

    /**
     * @brief The pattern's split verifier in a flat table.
     */
    Verifier_table verifier_;

    /**
     * @brief The ids of the ordinary tokens in the order of their spellings.
     */
    std::vector<load::Token_id_t> spellings_sorted_;

    /**
     * @brief The ids of the ordinary tokens the merge does not form, in the order of their spellings.
     */
    std::vector<load::Token_id_t> unformed_sorted_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP
