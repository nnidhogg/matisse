#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_VERIFIER_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_VERIFIER_HPP

#include <munch/core/lexer.hpp>
#include <munch/verifier/verifier.hpp>

#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The automata a pattern's two certificates are decided on: split_verifier(), the verifier of the pattern's
 *        split marking, on which the policy certificate is decided, and factorization_lexer(), the gate-erased union of
 *        its alternatives as a munch token set, on which the factorization certificate is decided.
 *
 * Both read the pattern alone, its alphabet, languages and gates, as the class theorem of the report on pre-tokenizer
 * certification constructs them, and both run over the class symbols, so that a position or an origin counts code
 * points of the text.
 */
namespace matisse::pretok
{
/**
 * @brief Builds the verifier M of a pattern: the automaton over class symbols, each marked with whether a piece ends
 *        after it, that accepts a marked string exactly when its marking is the pattern's segmentation of its symbols.
 *
 * For each alternative k with language L and gate g, its endpoints B_k are the strings u|z over the class symbols and a
 * marker with u in L and z empty or beginning with a symbol of g; A_k is B_k less those with a strictly longer endpoint
 * on the same residual, the longest gate-passing accept, and Match_k, the residuals on which k has an accept, is B_k
 * with its marker erased. The one-step relation Q, holding u|z exactly when the pattern commits u on uz, is the union
 * over k of A_k restricted to the residuals no earlier alternative matches. The pattern is total when every nonempty
 * residual is matched by some alternative, which the construction decides on the residuals left unmatched and refuses
 * otherwise. M accepts the marked strings whose last symbol is unmarked and in which every position a piece begins at,
 * the start and every position after a marked symbol, is followed by the piece Q assigns to the residual there; it is
 * built as those strings less every string with a piece start followed by a residual whose first marked piece disagrees
 * with Q, and minimized. For a total pattern M is functional and total, so it is the graph of the segmentation. munch's
 * constructor trims the table, dropping the dead state, and numbers it breadth-first from the start over the marked
 * symbols ascending.
 * @param pattern The pattern.
 * @return The verifier, over the class symbols marked.
 * @throws std::invalid_argument If the pattern is not total, the message naming a shortest nonempty residual no
 *         alternative matches.
 */
[[nodiscard]] munch::verifier::Verifier split_verifier(const Pattern& pattern);

/**
 * @brief Compiles the gate-erased union U of a pattern's alternatives over its class symbols, each alternative's
 *        language one token, ranked in the order of the pattern.
 *
 * Every piece the pattern commits is a word of U, a gate only withholding accepts, so is_split_window() on this lexer
 * decides the factorization certificate, which holds for every complete factorization into words of U and so binds the
 * pattern's segmentation whatever its discipline.
 * @param pattern The pattern.
 * @return The lexer.
 */
[[nodiscard]] munch::core::Lexer factorization_lexer(const Pattern& pattern);

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_VERIFIER_HPP
