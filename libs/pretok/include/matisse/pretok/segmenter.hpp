#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include <munch/dfa/simulator.hpp>

#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The runtime pre-tokenizer of a pattern, the Segmenter; split_verifier(), building the pattern's verifier, and
 *        factorization_lexer(), compiling the gate-erased union of its alternatives, are not declared yet.
 *
 * The Segmenter is the gated commitment order of the gate-sufficiency proposition read over code points, which
 * reproduces the engine's segmentation. split_verifier() is to be the minimized verifier of the pattern's one-step
 * relation, handed to munch as a Verifier table over the class symbols; factorization_lexer() is to be the munch token
 * set on which is_split_window() decides the factorization certificate. docs/design.md sketches both.
 */
namespace matisse::pretok
{
/**
 * @brief The runtime pre-tokenizer of a pattern: where the engine running the pattern's regex ends each piece of a
 *        text.
 *
 * At each boundary it commits the longest gate-passing accept of the first alternative that has one, each alternative's
 * language compiled by munch as a DFA over the class symbols and run on the text's class string, and an accept passing
 * its alternative's gate when the class symbol after it is one the gate holds or the input ends there. A gate reading
 * one code point suffices for every pattern described, and a total pattern has an accept on every nonempty text, so the
 * commit always advances.
 */
class Segmenter
{
public:
    /**
     * @brief Compiles the alternatives of a pattern.
     * @param pattern The pattern, whose views and spans outlive the segmenter, as a tokenizer library's static
     *        description does.
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
     * @brief One alternative of the pattern as the segmenter runs it: the DFA of its language over the class symbols,
     *        and its gate.
     */
    struct Compiled
    {
        /**
         * @brief The DFA of the alternative's language.
         */
        munch::dfa::Simulator simulator;

        /**
         * @brief The class symbols that may follow an accept.
         */
        std::string_view gate{};
    };

    /**
     * @brief Returns the alternatives of a pattern compiled, in the order of the published pattern.
     * @param alternatives The alternatives.
     * @return The compiled alternatives.
     */
    [[nodiscard]] static std::vector<Compiled> compiled_of(std::span<const Alternative> alternatives);

    /**
     * @brief Returns the length of the piece the pattern commits at a boundary of a class string.
     * @param symbols The class string, in class symbols.
     * @param at The boundary, before the end of the string.
     * @return The length, in code points, at least one.
     */
    [[nodiscard]] std::size_t piece_length(std::string_view symbols, std::size_t at) const;

    /**
     * @brief Returns the length of an alternative's longest gate-passing accept at a boundary of a class string.
     * @param alternative The alternative.
     * @param symbols The class string, in class symbols.
     * @param at The boundary, before the end of the string.
     * @return The length, in code points, zero when the alternative has no such accept there.
     */
    [[nodiscard]] static std::size_t longest_accept(
            const Compiled& alternative, std::string_view symbols, std::size_t at) noexcept;

    /**
     * @brief Returns whether an accept ending at an offset of a class string passes a gate.
     * @param gate The class symbols that may follow the accept.
     * @param symbols The class string, in class symbols.
     * @param end The offset after the accept.
     * @return True when the accept passes.
     */
    [[nodiscard]] static bool passes(std::string_view gate, std::string_view symbols, std::size_t end) noexcept;

    /**
     * @brief The pattern's class alphabet.
     */
    Alphabet alphabet_;

    /**
     * @brief The pattern's alternatives compiled, in order.
     */
    std::vector<Compiled> alternatives_;
};

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
