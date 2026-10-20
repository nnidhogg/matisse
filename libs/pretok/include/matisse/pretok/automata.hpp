#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_AUTOMATA_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_AUTOMATA_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @file
 * @brief The small automaton kit the construction of a pattern's verifier is built with: complete DFAs, Dfa, and NFAs,
 *        Nfa, over an alphabet of a few symbols, with subset construction, minimization, complement, products and
 *        concatenation.
 *
 * A symbol is an index into the alphabet, from zero below its symbol count, so that the class symbols of a pattern, a
 * marker beside them, or the class symbols with a boundary bit each are all alphabets of the same kind. Every DFA is
 * complete, a move from every state on every symbol, its start state zero, and every operation that builds one but the
 * complement returns it minimized, so that two DFAs of one language are equal state for state up to the order of their
 * states. The kit is the verifier construction's own and is not part of matisse's stable surface. The header is
 * installed only because every header of the library lives under include, as munch keeps none under src, and it may
 * change, or move to munch, in minor releases.
 */
namespace matisse::pretok::automata
{
/**
 * @brief A state of an automaton, by its index.
 */
using State_t = std::size_t;

/**
 * @brief A symbol of an alphabet, by its index.
 */
using Symbol_t = std::size_t;

/**
 * @brief A complete deterministic automaton, its start state zero.
 */
struct Dfa
{
    /**
     * @brief How many symbols the alphabet holds.
     */
    std::size_t symbol_count{};

    /**
     * @brief The state each state moves to, by state and then by symbol.
     */
    std::vector<std::vector<State_t>> transitions{};

    /**
     * @brief Whether each state accepts, by state.
     */
    std::vector<bool> accepting{};
};

/**
 * @brief A nondeterministic automaton with empty moves.
 */
struct Nfa
{
    /**
     * @brief A move on a symbol.
     */
    struct Edge
    {
        /**
         * @brief The symbol read.
         */
        Symbol_t symbol{};

        /**
         * @brief The state moved to.
         */
        State_t to{};
    };

    /**
     * @brief How many symbols the alphabet holds.
     */
    std::size_t symbol_count{};

    /**
     * @brief The moves on a symbol out of each state, by state.
     */
    std::vector<std::vector<Edge>> edges{};

    /**
     * @brief The states each state moves to on no symbol, by state.
     */
    std::vector<std::vector<State_t>> epsilons{};

    /**
     * @brief The start state.
     */
    State_t start{};

    /**
     * @brief The accepting states.
     */
    std::vector<State_t> accepting{};
};

/**
 * @brief How a product accepts, by whether each of its two operands accepts.
 */
enum class Accepts : std::uint8_t
{
    /**
     * @brief Both accept: the intersection.
     */
    both,

    /**
     * @brief Either accepts: the union.
     */
    either,

    /**
     * @brief The first accepts and the second does not: the difference.
     */
    first_only
};

/**
 * @brief Adds a state to an NFA, with no moves out of it.
 * @param nfa The NFA, extended in place.
 * @return The state added.
 */
[[nodiscard]] State_t add_state(Nfa& nfa);

/**
 * @brief Adds a copy of a DFA to an NFA: a state for each of its states and a move for each of its transitions, its
 *        acceptance left to the caller.
 * @param nfa The NFA, extended in place, over the DFA's alphabet.
 * @param dfa The DFA.
 * @return The NFA state of the DFA's state zero, the others following it in order.
 */
[[nodiscard]] State_t embed(Nfa& nfa, const Dfa& dfa);

/**
 * @brief Returns the DFA of an NFA's language, by subset construction from the empty closure of its start state.
 * @param nfa The NFA.
 * @return The DFA, minimized.
 */
[[nodiscard]] Dfa determinize(const Nfa& nfa);

/**
 * @brief Returns the minimal DFA of a DFA's language: its reachable states merged by Moore partition refinement, the
 *        states first split by acceptance and each block then by the blocks its states move to, until no block splits,
 *        and renumbered with the start's block zero and the rest in the order their first states appear.
 * @param dfa The DFA.
 * @return The minimal DFA, which keeps one dead state when some string has no extension in the language.
 */
[[nodiscard]] Dfa minimize(const Dfa& dfa);

/**
 * @brief Returns the DFA of a DFA's complement, every state's acceptance flipped, which needs the DFA complete.
 * @param dfa The DFA.
 * @return The complement.
 */
[[nodiscard]] Dfa complement(const Dfa& dfa);

/**
 * @brief Returns the product of two DFAs over one alphabet, its states the pairs reachable from the pair of starts and
 *        each pair accepting as the operation says.
 * @param first The first DFA.
 * @param second The second DFA, over the first's alphabet.
 * @param accepts How a pair accepts.
 * @return The product, minimized.
 */
[[nodiscard]] Dfa product(const Dfa& first, const Dfa& second, Accepts accepts);

/**
 * @brief Returns the DFA of the concatenation of two DFAs' languages: both embedded in one NFA, an empty move from
 *        every accepting state of the first to the start of the second, determinized.
 * @param first The DFA of the first part.
 * @param second The DFA of the second part, over the first's alphabet.
 * @return The concatenation, minimized.
 */
[[nodiscard]] Dfa concatenation(const Dfa& first, const Dfa& second);

} // namespace matisse::pretok::automata

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_AUTOMATA_HPP
