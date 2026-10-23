#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP

#include <cstddef>
#include <string_view>
#include <vector>

#include <munch/dfa/simulator.hpp>

#include "matisse/pretok/classes.hpp"
#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The runtime pre-tokenizer of a pattern, the Segmenter.
 *
 * The Segmenter is the gated commitment order of the gate-sufficiency proposition read over code points, which
 * reproduces the engine's segmentation. Its marking of a text's class string is the one marking of it that the
 * pattern's split_verifier(), declared in verifier.hpp, accepts.
 */
namespace matisse::pretok
{
/**
 * @brief The runtime pre-tokenizer of a pattern: where the engine running the pattern's regex ends each piece of a
 *        text.
 *
 * At each boundary it commits the longest gate-passing accept of the first alternative that has one, an accept passing
 * its alternative's gate when the class symbol after it is one the gate holds or the input ends there. A gate reading
 * one code point suffices for every pattern described, so each alternative is compiled by munch as the DFA of its
 * language followed by one symbol of its gate or a sentinel byte, which the segmenter appends to the text's class
 * string, and munch's longest-match scan of that DFA, less the one symbol, is the alternative's longest gate-passing
 * accept. The alternatives' DFAs are walked at once as their product, one munch DFA, so that one scan at each boundary
 * commits the longest accept of the first alternative that has one, as automaton_of() argues. The pattern must be
 * total, some alternative accepting on every nonempty text, so that the commit always advances; split_verifier()
 * decides totality and refuses a pattern that is not, and the segmenter does not check it.
 */
class Segmenter
{
public:
    /**
     * @brief Compiles the alternatives of a pattern into the automaton of its segmentation.
     * @param pattern The pattern, total.
     */
    explicit Segmenter(const Pattern& pattern);

    /**
     * @brief Returns where the pattern ends each piece of a text, the text read as a complete input.
     * @param text The text, well-formed UTF-8.
     * @return The byte offset after each piece, ascending, the last the text's size; none for the empty text.
     * @throws std::invalid_argument If the text is not well-formed UTF-8, as classes_of() refuses it.
     */
    [[nodiscard]] std::vector<std::size_t> boundaries(std::string_view text) const;

private:
    /**
     * @brief Returns the automaton of a pattern's whole segmentation: the product of its alternatives' automata, each
     *        the DFA of the alternative's language followed by one symbol of its gate or the sentinel, walked at once
     *        and accepting where the first alternative that has accepted accepts.
     *
     * Its longest accept on a class string is the longest gate-passing accept, plus the one symbol, of the first
     * alternative that has one. An alternative stops where it cannot accept again, and one after the first that has
     * accepted stops too, since it can no longer decide the piece; the walk stops when every alternative up to the
     * first that has accepted has stopped. The first that has accepted only ever moves to an earlier alternative, and
     * none stops while an accept lies ahead of it, so when the walk stops it is the first alternative with an accept
     * anywhere and the walk has passed its last accept. From that alternative's first accept on the product accepts
     * exactly where it does, and every accept of the product before that is shorter, so the product's longest accept is
     * that alternative's longest. The product's states are numbered in the order they are found, the start zero.
     * @param pattern The pattern.
     * @return The automaton, a munch DFA over the class symbols and the sentinel.
     */
    [[nodiscard]] static munch::dfa::Simulator automaton_of(const Pattern& pattern);

    /**
     * @brief Returns the length of the piece the pattern commits at a boundary of a class string.
     * @param symbols The class string, in class symbols, followed by the sentinel.
     * @param at The boundary, before the sentinel.
     * @return The length, in code points, at least one.
     */
    [[nodiscard]] std::size_t piece_length(std::string_view symbols, std::size_t at) const;

    /**
     * @brief The class table of the pattern's alphabet.
     */
    Class_table classes_;

    /**
     * @brief The automaton of the whole segmentation.
     */
    munch::dfa::Simulator automaton_;
};

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
