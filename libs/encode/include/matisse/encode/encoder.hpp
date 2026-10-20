#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "matisse/encode/merger.hpp"
#include "matisse/load/definition.hpp"
#include "matisse/pretok/pattern.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/validate/report.hpp"

/**
 * @file
 * @brief The encoder, Encoder with encode() and decode(): the token ids of a text under a definition, and the text of a
 *        run of ids.
 *
 * encode() splits the text at every allowed special token, segments each stretch between them as a complete input, and
 * merges every piece alone by the definition's merge rule, the pair rule or tiktoken's rank merge, a piece that is a
 * token taking its id whole where the definition's whole-piece lookup says so; Merger, in merger.hpp, holds the lookup
 * and the merge, and a Piece_cache, in piece_cache.hpp, keeps the merged pieces of one call. chunk_boundaries() will
 * walk forward from each equal byte target to the first anchor of the inventory's window table, and encode_parallel()
 * will give each chunk to a worker that segments it with the decided right context and keeps the pieces ending at its
 * cut. The encoder claims one thing: for every text the definition accepts, encode_parallel() returns the ids encode()
 * returns, at every chunk count. chunk_boundaries() and encode_parallel() are not declared yet; docs/design.md sketches
 * them.
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
     * @param inventory The inventory whose window table the encoder keeps.
     * @param pattern The pattern, whose views and spans outlive the encoder, as a tokenizer library's static
     *        description does.
     * @param allowed The spellings of the special tokens cut out of a text; every other special token is ordinary text.
     * @throws std::invalid_argument If an allowed spelling is not one of the definition's special tokens, or the
     *         definition cannot be merged, as Merger refuses it; the message names the spelling, the byte or the merge.
     * @throws munch::pattern::Syntax_error If an alternative's language is not in munch's pattern syntax.
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
     */
    [[nodiscard]] std::vector<load::Token_id_t> encode(std::string_view text) const;

    /**
     * @brief Returns the text a run of ids spells: each ordinary token's bytes and each special token's spelling.
     * @param ids The ids.
     * @return The text.
     * @throws std::invalid_argument If an id is neither an ordinary nor a special token of the definition; the message
     *         names the id and its index.
     */
    [[nodiscard]] std::string decode(std::span<const load::Token_id_t> ids) const;

private:
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
     * @brief The inventory's window table with its right context, which the parallel encoding cuts by.
     */
    validate::Window_table table_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP
