#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include <munch/dfa/simulator.hpp>

#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The vendor pre-tokenizers of GPT-2 and cl100k_base: the runtime Segmenter; split_verifier(), building the
 *        pattern's verifier, and factorization_lexer(), compiling the gate-erased union of its alternatives, are not
 *        declared yet.
 *
 * The Segmenter is the gated commitment order of the gate-sufficiency proposition read over code points, which
 * reproduces the engine's segmentation. split_verifier() is to be the minimized verifier of the pattern's one-step
 * relation, handed to munch as a Verifier table over the class symbols, which munch's constructor makes 21 states for
 * GPT-2 and 32 for cl100k, 22 and 33 counting the dead state; factorization_lexer() is to be the munch token set on
 * which is_split_window() decides the factorization certificate. docs/design.md sketches both.
 */
namespace matisse::pretok
{
/**
 * @brief The runtime pre-tokenizer of a pattern: where the engine running the pattern's regex ends each piece of a
 *        text.
 *
 * At each boundary it commits the longest gate-passing accept of the first alternative that has one, each alternative
 * the gate-erased language of one alternative of the published pattern, compiled by munch as a DFA over the class
 * symbols of classes.hpp and run on the text's class string. The run alternative `\s+(?!\S)` is gated by "the next
 * code point is white space", passing at the end of the input too, and cl100k's end-anchored `\s++$` passes only at
 * the end of the input; every other alternative passes everywhere. A gate reading one code point suffices for both
 * patterns, and every nonempty text has an accept, so the commit always advances.
 */
class Segmenter
{
public:
    /**
     * @brief Compiles the alternatives of a pattern.
     * @param pattern The pattern, GPT-2's or cl100k's.
     * @throws std::invalid_argument If the pattern is Pattern::none.
     */
    explicit Segmenter(load::Pattern pattern);

    /**
     * @brief Returns where the pattern ends each piece of a text, the text read as a complete input.
     * @param text The text, well-formed UTF-8.
     * @return The byte offset after each piece, ascending, the last the text's size; none for the empty text.
     * @throws std::invalid_argument If the text is not well-formed UTF-8, as classes_of() refuses it.
     */
    [[nodiscard]] std::vector<std::size_t> boundaries(std::string_view text) const;

private:
    /**
     * @brief The condition an accept of an alternative must meet on what follows it.
     */
    enum class Gate : std::uint8_t
    {
        /**
         * @brief Every accept passes.
         */
        none,

        /**
         * @brief An accept passes when white space or the end of the input follows it, `(?!\S)`.
         */
        next_white_space,

        /**
         * @brief An accept passes only at the end of the input, `$`.
         */
        end_of_input
    };

    /**
     * @brief One alternative of the pattern: the DFA of its gate-erased language over the class symbols, and its gate.
     */
    struct Alternative
    {
        /**
         * @brief The DFA of the alternative's gate-erased language.
         */
        munch::dfa::Simulator simulator;

        /**
         * @brief The alternative's gate.
         */
        Gate gate{};
    };

    /**
     * @brief Returns the alternatives of a pattern, in the order of the published pattern.
     * @param pattern The pattern.
     * @return The alternatives.
     * @throws std::invalid_argument If the pattern is Pattern::none.
     */
    [[nodiscard]] static std::vector<Alternative> alternatives_of(load::Pattern pattern);

    /**
     * @brief Returns the length of the piece the pattern commits at a boundary of a class string.
     * @param symbols The class string, in class symbols.
     * @param at The boundary, before the end of the string.
     * @return The length, in code points, at least one.
     */
    [[nodiscard]] std::size_t piece_length(std::string_view symbols, std::size_t at) const;

    /**
     * @brief Returns whether an accept ending at an offset of a class string passes a gate.
     * @param gate The gate.
     * @param symbols The class string, in class symbols.
     * @param end The offset after the accept.
     * @return True when the accept passes.
     */
    [[nodiscard]] bool passes(Gate gate, std::string_view symbols, std::size_t end) const noexcept;

    /**
     * @brief The pattern.
     */
    load::Pattern pattern_;

    /**
     * @brief The pattern's alternatives, in order.
     */
    std::vector<Alternative> alternatives_;

    /**
     * @brief The class symbols of the pattern's white space, `\s`.
     */
    std::string_view white_space_;
};

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
