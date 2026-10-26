#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_STREAM_ENCODER_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_STREAM_ENCODER_HPP

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "matisse/encode/encoder.hpp"
#include "matisse/encode/live_runs.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The streaming encoder, Stream_encoder: the ids of a text read in pushes of bytes, each id returned as soon as
 *        no continuation of the text can change it.
 *
 * Every id a stream has returned after a push is a prefix of encode() of the whole text, whatever the rest of the text
 * turns out to be, and finish() returns the rest, so that the ids returned between two finish() calls, concatenated,
 * are encode() of the bytes pushed between them. What a push returns is the longest such prefix, the greatest stable
 * prefix of Dalirrooyfard and Ye's online tokenization, with two exceptions: while the bytes read end in a proper
 * prefix of an allowed special token's spelling those bytes are held, fewer than the longest allowed spelling, and not
 * classed, so that a push returns the longest prefix final for the bytes before them, and the ids can lag by more than
 * the held bytes: with `<|endoftext|>` allowed, a word followed by `<` stays an open piece whose ids are held until the
 * bytes after it show whether the `<` begins the special token; and inside a piece the continuations a stable id is
 * held against are all byte strings, where the pattern allows fewer.
 *
 * The encoding has three layers, each exact at its level. A special token's occurrence is committed once no proper
 * prefix of an allowed spelling can still begin before it, so the stretches between special tokens are encode()'s. In a
 * stretch the live runs of the pattern's split verifier M, Live_runs in live_runs.hpp, are the restrictions of every
 * continuation's segmentation, and a piece whose end every run keeps is closed and its ids returned. Inside each run's
 * open piece the ids returned are the ones every byte continuation of the piece keeps, which by projection need only
 * the continuations that a token can straddle the piece's end with, the rest of every token that begins with a suffix
 * of the piece: from the right, the first boundary of the open piece's merge that the merge of the piece followed by
 * each of them keeps is stable, and the ids before it are returned. A boundary across which no token can begin inside
 * the token before it is stable with no merge at all, which is tested first. The ids returned are the common prefix
 * over the runs of each run's closed pieces followed by its open piece's stable ids. The construction is the zero-delay
 * streamer of the report on the left ripple of byte-pair encoding, run on M for the pre-tokenizer and on the merge
 * inside each run's open piece, as docs/design.md describes it.
 */
namespace matisse::encode
{
/**
 * @brief The streaming encoder of an encoder's definition: the ids of a text pushed in parts, each returned as soon as
 *        no continuation of the text can change it.
 *
 * A stream reads one text at a time and is used by one thread at a time; any number of streams may read over one
 * encoder at once, each on its own thread, since the encoder is not changed by them and every stream keeps its runs,
 * its held bytes and its piece cache to itself, with no lock.
 */
class Stream_encoder
{
public:
    /**
     * @brief Starts a stream over an encoder, which the stream borrows: the encoder must outlive it and must not be
     *        moved while the stream reads it; the special tokens allowed are the encoder's.
     * @param encoder The encoder.
     */
    explicit Stream_encoder(const Encoder& encoder);

    /**
     * @brief A stream is not started over a temporary encoder, which would not outlive it.
     */
    explicit Stream_encoder(const Encoder&& encoder) = delete;

    /**
     * @brief Reads the next bytes of the text and returns the ids that have become final: every id returned since the
     *        last finish() is a prefix of encode() of the whole text, whatever the rest of it, and what is returned is
     *        the longest such prefix but for two exceptions: while the bytes read end in a proper prefix of an allowed
     *        special token's spelling, it is the longest prefix final for the bytes before that prefix, and inside a
     *        piece an id is held against every byte continuation of the piece, where the pattern allows fewer.
     *
     * A push may end inside a UTF-8 sequence or inside a special token's spelling; those bytes are held until a later
     * push or finish() completes them.
     * @param bytes The next bytes of the text, well-formed UTF-8 once the sequence a push ends inside is completed.
     * @return The ids newly final, in order.
     * @throws std::invalid_argument If the bytes read are not well-formed UTF-8, as classes_of() refuses them, the byte
     *         offset counted from the first byte the stream had not classed; the stream is then not read again.
     * @throws std::length_error If a piece is 2^32 bytes or longer, as encode() refuses it.
     */
    [[nodiscard]] std::vector<load::Token_id_t> push(std::string_view bytes);

    /**
     * @brief Ends the text and returns the rest of its ids, so that everything returned since the last finish() is
     *        encode() of the bytes pushed since then; the stream then reads a new text.
     * @return The remaining ids.
     * @throws std::invalid_argument If the text ends inside a UTF-8 sequence or what the stream held is not well-formed
     *         UTF-8.
     * @throws std::length_error If a piece is 2^32 bytes or longer, as encode() refuses it.
     */
    [[nodiscard]] std::vector<load::Token_id_t> finish();

private:
    /**
     * @brief Reads bytes as far as it can: every allowed special token's occurrence that no held spelling can still
     *        begin before is cut out with its id, and the code points before it, or before the bytes held, are fed to
     *        the live runs.
     * @param text The bytes, the held ones first.
     * @param ended Whether the text ends after them, so that no spelling is held and no sequence can be completed.
     * @param ids The ids closed so far, appended to.
     * @return The number of bytes read, the rest held.
     */
    [[nodiscard]] std::size_t read(std::string_view text, bool ended, std::vector<load::Token_id_t>& ids);

    /**
     * @brief Feeds the code points of bytes to the live runs and closes every piece that becomes stable.
     * @param bytes The bytes, between special tokens.
     * @param ended Whether no later byte can complete a sequence they end inside.
     * @param ids The ids closed so far, appended to.
     * @return The number of bytes fed, all but an incomplete sequence at their end when one can still be completed.
     * @throws std::invalid_argument If the bytes fed are not well-formed UTF-8.
     */
    [[nodiscard]] std::size_t feed(std::string_view bytes, bool ended, std::vector<load::Token_id_t>& ids);

    /**
     * @brief Closes the piece that ends at a byte of the stretch: appends its ids less those already returned of it.
     * @param end The byte after the piece, counted from the stretch's start.
     * @param ids The ids closed so far, appended to.
     */
    void close_piece(std::size_t end, std::vector<load::Token_id_t>& ids);

    /**
     * @brief Appends the ids of a piece as encode() gives them: looked up whole where the definition says so, else
     *        found in the cache or merged and stored.
     * @param piece The piece's bytes.
     * @param ids The ids, appended to.
     */
    void append_ids(std::string_view piece, std::vector<load::Token_id_t>& ids);

    /**
     * @brief Ends the stretch: closes the pieces of the run that accepts and starts the empty stretch.
     * @param ids The ids closed so far, appended to.
     */
    void flush(std::vector<load::Token_id_t>& ids);

    /**
     * @brief Appends the ids every live run's futures begin with past those already returned: the common prefix of each
     *        run's closed pieces followed by its open piece's stable ids.
     * @param ids The ids returned by the push, appended to.
     */
    void emit(std::vector<load::Token_id_t>& ids);

    /**
     * @brief Returns the ids a run's futures all begin with past those already returned: from the stretch's last
     *        closed piece, the ids of the pieces it closes, then the stable ids of its open piece, or all of them when
     *        the run stands in a dead end, a piece the ids returned reach into merged from where they end.
     * @param marking The run's marking.
     * @return The ids.
     */
    [[nodiscard]] std::vector<load::Token_id_t> run_ids(const Live_runs::Marking& marking);

    /**
     * @brief The encoder, whose definition, merger and tables the stream reads.
     */
    const Encoder& encoder_;

    /**
     * @brief The live runs of the pattern's split verifier over the stretch read.
     */
    Live_runs runs_;

    /**
     * @brief The merged pieces, kept across the stream's texts.
     */
    Piece_cache cache_;

    /**
     * @brief The bytes read and not yet classed: a proper prefix of an allowed spelling, or a UTF-8 sequence not yet
     *        complete.
     */
    std::string held_;

    /**
     * @brief The stretch's bytes from the end of its last closed piece.
     */
    std::string stretch_;

    /**
     * @brief Where the stretch's last closed piece ends, counted from the stretch's start.
     */
    std::size_t closed_end_;

    /**
     * @brief The number of ids returned past the stretch's last closed piece.
     */
    std::size_t returned_ids_;

    /**
     * @brief The bytes the ids returned past the stretch's last closed piece spell.
     */
    std::size_t returned_bytes_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_STREAM_ENCODER_HPP
