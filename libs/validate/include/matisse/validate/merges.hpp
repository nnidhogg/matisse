#ifndef MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_MERGES_HPP
#define MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_MERGES_HPP

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <munch/verifier/verifier_decisions.hpp>

#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The lazy certification of a byte-level merge table without a pre-tokenizer, validate_merges(), which decides
 *        each candidate window the caller gives, Merge_window, and the windows decided rendered as text,
 *        render_merges(), and as JSON, merges_json().
 *
 * A merge table read with no pre-tokenizer, as read_tokenizer_json() reads one under the empty spelling, merges a text
 * as one piece, so it has no pattern, no report and no Encoder, and its cuts are the windows certified of the table
 * itself. validate_merges() builds munch::verifier::token_automaton() of the table once and decides each candidate by
 * munch::verifier::miscovering() on it, under canonical byte-pair encoding, the merge of least rank applied at its
 * leftmost occurrence until none applies, which is the pair rule. Every window of width three is too many to decide on
 * a table of GPT-2's size, so the candidates are the caller's, and matisse-validate draws them from an input.
 */
namespace matisse::validate
{
/**
 * @brief A candidate window of a merge table decided: its bytes, the origin it is decided at, and its refutation or
 *        none.
 *
 * Every byte string is encoded, and the window is one of them, so at its occurrence in itself the token covering its
 * last byte begins where the window's own encoding begins it, and no other origin can be certified. A window refuted at
 * that origin therefore certifies no cut at any origin.
 */
struct Merge_window
{
    /**
     * @brief The window's bytes.
     */
    std::string window{};

    /**
     * @brief The origin decided, a byte index of the window: where the token covering its last byte begins in its own
     *        encoding.
     */
    std::size_t origin{};

    /**
     * @brief std::nullopt when the window is certified at the origin, the token covering its last byte beginning there
     *        at every occurrence in every byte string; otherwise the encoder's segmentation of a string with an
     *        occurrence whose covering token begins elsewhere.
     */
    std::optional<munch::verifier::Miscovering> refutation{};
};

/**
 * @brief Decides each candidate window of a definition's merge table, a table read with no pre-tokenizer, at the one
 *        origin it can be certified at, by munch::verifier::miscovering() on the table's token automaton.
 *
 * The token automaton is built once for all the candidates, and each decision reads every token against the window: on
 * GPT-2's table of 50,000 merges the automaton has 50,001 states, is built in 0.03 s and holds 8 MB, and a decision
 * takes on average about 0.1 ms on one byte, 0.2 ms on a pair and 0.5 ms on a triple. The verdicts are of the merge
 * alone, on a text with no special token cut out, and a refutation is the encoder's segmentation of a string of at most
 * 2L + |window| - 2 bytes, L the longest token, the first the search finds and not necessarily the shortest.
 * @param definition The definition, merged by the pair rule without the whole-piece lookup.
 * @param candidates The candidate windows, each nonempty, in the order they are decided.
 * @return Each candidate decided, in the order given.
 * @throws std::invalid_argument If the definition merges by the rank rule, which carries no merge table, or looks a
 *         whole piece up, which with no pre-tokenizer takes a whole text that is a token as that token; if a candidate
 *         is empty, the message naming its index; or as munch::verifier::token_automaton() refuses the table, a part
 *         that neither a byte nor an earlier merge carries or a product that a byte or an earlier merge already
 *         carries, the message munch's, naming the merge and its rank.
 */
[[nodiscard]] std::vector<Merge_window> validate_merges(
        const load::Definition& definition, std::span<const std::string> candidates);

/**
 * @brief Renders the windows decided as text, the verdict first, then the certified windows and then the refuted ones,
 *        each counted by width with the first six shown as examples, printable ones first.
 *
 * A window and a refutation's text are shown quoted with the common controls escaped, as render() shows a class string,
 * a certified window with its origin and a refuted one with its origin, the occurrence the refutation places it at and
 * the boundaries of the encoder's segmentation, or one token where it has none.
 * @param windows The windows decided.
 * @return The text, a newline after every row.
 */
[[nodiscard]] std::string render_merges(std::span<const Merge_window> windows);

/**
 * @brief Renders the windows decided as JSON, one object of the verdict and the windows in the order decided, each its
 *        bytes, its origin and its refutation or null.
 *
 * A window is a JSON string holding each byte as the code point of its value, as json() holds a class string, and a
 * refutation an object of its segmentation, the text and the offsets its boundaries stand at, and its occurrence.
 * @param windows The windows decided.
 * @return The JSON text, no trailing newline.
 */
[[nodiscard]] std::string merges_json(std::span<const Merge_window> windows);

} // namespace matisse::validate

#endif // MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_MERGES_HPP
