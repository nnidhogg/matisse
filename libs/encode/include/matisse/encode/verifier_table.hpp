#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_VERIFIER_TABLE_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_VERIFIER_TABLE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <munch/verifier/verifier.hpp>

#include "matisse/pretok/classes.hpp"
#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The split verifier M of a pattern in a flat table, Verifier_table: the state M moves to from every state on
 *        every class marked with whether a piece ends after it, and which states accept and which have no move.
 *
 * munch's Verifier keeps M's transitions in a hash map keyed by a state and a marked symbol. The table holds the same
 * transitions in one array indexed by the state, the class and the mark, so that a step is one load, and the streaming
 * encoder steps every live run of M twice on every code point it reads. M has 21 states for GPT-2, 32 for cl100k, 46
 * for o200k and 33 for Llama 3, so the table is at most 46 states by 30 classes by two marks.
 */
namespace matisse::encode
{
/**
 * @brief The split verifier of a pattern in a flat table over the classes of its alphabet.
 */
class Verifier_table
{
public:
    /**
     * @brief A state of M, numbered as munch's Verifier numbers it, the start zero.
     */
    using State_t = std::uint32_t;

    /**
     * @brief The step M has no move for, all ones, which no state is.
     */
    static constexpr State_t no_state{std::numeric_limits<State_t>::max()};

    /**
     * @brief Reads every transition of a verifier over the class symbols of an alphabet, each with both marks.
     * @param verifier The verifier, split_verifier() of the pattern.
     * @param alphabet The pattern's class alphabet.
     */
    Verifier_table(const munch::verifier::Verifier& verifier, const pretok::Alphabet& alphabet);

    /**
     * @brief Returns the start state, from which a stretch is read.
     * @return The start state, zero.
     */
    [[nodiscard]] State_t start() const noexcept;

    /**
     * @brief Returns the state M moves to from a state on a class, marked with whether a piece ends after it.
     *
     * A missing move is a state of its own rather than an empty std::optional, which GCC 13 returns through a byte and
     * a word stored apart and read back as one, which stalls every step; the streaming encoder steps every live run
     * twice a code point.
     * @param state The state.
     * @param symbol_class The class of the code point read.
     * @param boundary_after Whether a piece ends after the code point.
     * @return The next state, no_state when M has no move.
     */
    [[nodiscard]] State_t step(State_t state, pretok::Class_t symbol_class, bool boundary_after) const noexcept;

    /**
     * @brief Returns whether a state accepts, the input ending there with its last piece.
     * @param state The state.
     * @return True for an accepting state.
     */
    [[nodiscard]] bool accepts(State_t state) const noexcept;

    /**
     * @brief Returns whether a state is a dead end: it accepts and has no move, so that only the end of the input
     *        follows it.
     * @param state The state.
     * @return True for an accepting state with no move.
     */
    [[nodiscard]] bool is_dead_end(State_t state) const noexcept;

private:
    /**
     * @brief The number of classes of the alphabet, the stride of a state's steps.
     */
    std::size_t class_count_;

    /**
     * @brief The state every step moves to, by the state times the classes plus the class, times two, plus the mark.
     */
    std::vector<State_t> next_;

    /**
     * @brief Whether each state accepts.
     */
    std::vector<bool> accepting_;

    /**
     * @brief Whether each state is a dead end.
     */
    std::vector<bool> dead_ends_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_VERIFIER_TABLE_HPP
