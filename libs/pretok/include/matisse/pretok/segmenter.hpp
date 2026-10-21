#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP

#include <cstddef>
#include <span>
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
 * accept. The pattern must be total, some alternative accepting on every nonempty text, so that the commit always
 * advances; split_verifier() decides totality and refuses a pattern that is not, and the segmenter does not check it.
 */
class Segmenter
{
public:
    /**
     * @brief Compiles the alternatives of a pattern.
     * @param pattern The pattern, total.
     * @throws munch::pattern::Syntax_error If an alternative's language is not in munch's pattern syntax.
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
     * @brief One alternative of the pattern as the segmenter runs it: the DFA of its language followed by one symbol
     *        of its gate or the sentinel.
     */
    struct Compiled
    {
        /**
         * @brief The DFA of the alternative's language and its lookahead.
         */
        munch::dfa::Simulator simulator;
    };

    /**
     * @brief Returns the alternatives of a pattern compiled, in the order of the published pattern.
     * @param alternatives The alternatives.
     * @return The compiled alternatives.
     */
    [[nodiscard]] static std::vector<Compiled> compiled_of(std::span<const Alternative> alternatives);

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
     * @brief The pattern's alternatives compiled, in order.
     */
    std::vector<Compiled> alternatives_;
};

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
