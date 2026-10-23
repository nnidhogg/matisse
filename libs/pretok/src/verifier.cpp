#include "matisse/pretok/verifier.hpp"

#include <cstddef>
#include <deque>
#include <format>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <munch/core/builder.hpp>
#include <munch/dfa/simulator.hpp>
#include <munch/regex/regex.hpp>

#include "matisse/pretok/automata.hpp"

namespace matisse::pretok
{
namespace
{
using automata::Accepts;
using automata::Dfa;
using automata::Nfa;
using automata::State_t;
using automata::Symbol_t;

/**
 * @brief Returns the DFA of one state, over an alphabet, that accepts every string or none.
 * @param symbol_count How many symbols the alphabet holds.
 * @param accepts Whether the state accepts.
 * @return The DFA.
 */
[[nodiscard]] Dfa one_state(const std::size_t symbol_count, const bool accepts)
{
    return {.symbol_count = symbol_count,
            .transitions = {std::vector<State_t>(symbol_count, 0)},
            .accepting = {accepts}};
}

/**
 * @brief Returns the DFA of one alternative's language, compiled by munch as a single token.
 * @param language The language, over the class symbols.
 * @return The DFA.
 */
[[nodiscard]] munch::dfa::Simulator simulator_of(const munch::regex::Regex& language)
{
    munch::core::Builder builder{};

    builder.add_token(language, std::size_t{0}, 0);

    const auto lexer{builder.build()};

    return lexer.simulator();
}

/**
 * @brief Returns the escape a byte of a class string is written as inside the quotes a refusal shows it in, when it has
 *        one: a common control, the backslash, and the quote.
 * @param byte The byte.
 * @return The escape, or std::nullopt for a byte written as itself or in hex.
 */
[[nodiscard]] std::optional<std::string_view> quoted_escape(const unsigned char byte)
{
    switch (byte)
    {
    case '\n':
        return R"(\n)";
    case '\t':
        return R"(\t)";
    case '\r':
        return R"(\r)";
    case '\\':
        return R"(\\)";
    case '"':
        return R"(\")";
    default:
        break;
    }

    return std::nullopt;
}

/**
 * @brief Returns a class string as a refusal shows it, as the report prints one: each byte the character when it is
 *        printable, an escape when it is a common control, and the escape `\xHH` otherwise.
 * @param bytes The class string.
 * @return The rendering, quoted.
 */
[[nodiscard]] std::string shown(const std::string_view bytes)
{
    std::string out{'"'};

    for (const auto byte : bytes)
    {
        const auto value{static_cast<unsigned char>(byte)};

        const auto printable{value >= ' ' && value <= '~'};

        if (const auto escape{quoted_escape(value)})
        {
            out += *escape;
        }
        else if (printable)
        {
            out += byte;
        }
        else
        {
            out += std::format(R"(\x{:02X})", value);
        }
    }

    return out + '"';
}

/**
 * @brief The construction of a pattern's verifier over its class symbols, after the class theorem: each alternative's
 *        endpoints and longest accepts, the one-step relation Q, and the verifier M.
 *
 * The alphabets are indices into the class symbols: the symbols themselves, then the marker after them, which splits a
 * residual where a piece ends, and the marked symbols, class symbol i being 2i unmarked and 2i + 1 marked.
 */
class Construction
{
public:
    /**
     * @brief Sets up the construction over an alphabet's class symbols.
     * @param symbols The class symbols, ascending, which outlive the construction.
     */
    explicit Construction(std::string_view symbols) noexcept;

    /**
     * @brief Builds the verifier M of a pattern's alternatives: the marked strings ending unmarked less those with a
     *        piece start followed by a residual whose first piece disagrees with Q.
     * @param alternatives The alternatives, in the order of the pattern.
     * @return The verifier, its table trimmed and numbered by munch.
     */
    [[nodiscard]] munch::verifier::Verifier verifier(std::span<const Alternative> alternatives) const;

private:
    /**
     * @brief Returns the one-step relation Q: the strings u|z for which u is the longest gate-passing accept of the
     *        first alternative that has one on the residual uz.
     *
     * The residuals no alternative matches are what is left once every alternative's are taken away, so the pattern is
     * total exactly when the empty residual is the only one left.
     * @param alternatives The alternatives, in order.
     * @return Q, over the class symbols and the marker.
     * @throws std::invalid_argument If the pattern is not total, the message naming a shortest nonempty residual no
     *         alternative matches.
     */
    [[nodiscard]] Dfa one_step(std::span<const Alternative> alternatives) const;

    /**
     * @brief Returns the marker, the symbol after the class symbols.
     * @return The marker.
     */
    [[nodiscard]] Symbol_t marker() const noexcept;

    /**
     * @brief Returns an alternative's endpoints B: the strings u|z with u in its language and z empty or beginning with
     *        a symbol of its gate.
     * @param alternative The alternative.
     * @return B, over the class symbols and the marker.
     */
    [[nodiscard]] Dfa endpoints_of(const Alternative& alternative) const;

    /**
     * @brief Returns the DFA of an alternative's language over the class symbols and the marker, which no word holds,
     *        read from munch's compilation of it.
     * @param language The language.
     * @return The DFA.
     */
    [[nodiscard]] Dfa language_of(const munch::regex::Regex& language) const;

    /**
     * @brief Returns the state munch's DFA of a language moves to from a state on a symbol.
     * @param simulator The DFA.
     * @param state The state, std::nullopt for the dead state.
     * @param symbol The symbol, a class symbol or the marker.
     * @return The state moved to, std::nullopt for the dead state, where the marker always leads.
     */
    [[nodiscard]] std::optional<std::size_t> step(
            const munch::dfa::Simulator& simulator, std::optional<std::size_t> state, Symbol_t symbol) const;

    /**
     * @brief Returns the strings a marker begins followed by a tail that passes a gate: empty, or beginning with a
     *        symbol of the gate.
     * @param gate The gate's class symbols.
     * @return The DFA, over the class symbols and the marker.
     */
    [[nodiscard]] Dfa marker_and_tail(std::string_view gate) const;

    /**
     * @brief Returns the longest endpoints A: the endpoints less those with a strictly longer endpoint on the same
     *        residual.
     * @param endpoints The endpoints B.
     * @return A, over the class symbols and the marker.
     */
    [[nodiscard]] Dfa longest(const Dfa& endpoints) const;

    /**
     * @brief Returns Later(B), the strings u|rz with r nonempty and ur|z in B: read u, the proposed marker, then at
     *        least one symbol, and pass B's own marker on at any point after it.
     * @param endpoints The endpoints B.
     * @return Later(B), over the class symbols and the marker.
     */
    [[nodiscard]] Dfa later(const Dfa& endpoints) const;

    /**
     * @brief Returns ins(L), the strings u|z with uz in a language L over the class symbols, exactly one marker each.
     * @param residuals The DFA of L, over the class symbols.
     * @return ins(L), over the class symbols and the marker.
     */
    [[nodiscard]] Dfa with_marker(const Dfa& residuals) const;

    /**
     * @brief Returns an alternative's residuals Match: its endpoints with the marker erased.
     * @param endpoints The endpoints B.
     * @return Match, over the class symbols.
     */
    [[nodiscard]] Dfa erased(const Dfa& endpoints) const;

    /**
     * @brief Returns a shortest nonempty string a DFA over the class symbols accepts, by breadth-first search from the
     *        states one symbol reaches, the symbols tried in ascending order.
     * @param residuals The DFA.
     * @return The string, in class symbols, or std::nullopt when the DFA accepts no nonempty string.
     */
    [[nodiscard]] std::optional<std::string> shortest_nonempty(const Dfa& residuals) const;

    /**
     * @brief Returns K, the nonempty marked strings whose first boundary is where Q ends the first piece of their
     *        symbols, the later bits free, or with no boundary the whole of them Q's first piece.
     * @param relation Q.
     * @return K, over the marked symbols.
     */
    [[nodiscard]] Dfa first_pieces(const Dfa& relation) const;

    /**
     * @brief Returns the nonempty marked strings.
     * @return The DFA, over the marked symbols.
     */
    [[nodiscard]] Dfa nonempty() const;

    /**
     * @brief Returns the marked strings that are empty or whose last symbol carries a given boundary bit.
     * @param boundary The bit.
     * @return The DFA, over the marked symbols.
     */
    [[nodiscard]] Dfa ending_in(bool boundary) const;

    /**
     * @brief Returns the munch verifier of a DFA over the marked symbols, each marked symbol read as its class symbol's
     *        byte and its bit.
     * @param marked The DFA.
     * @return The verifier, its table trimmed and numbered by munch's constructor.
     */
    [[nodiscard]] munch::verifier::Verifier table(const Dfa& marked) const;

    /**
     * @brief Returns the marked symbol of a class symbol and a boundary bit.
     * @param symbol The class symbol.
     * @param boundary Whether a piece ends after it.
     * @return The marked symbol.
     */
    [[nodiscard]] static Symbol_t marked_symbol(Symbol_t symbol, bool boundary) noexcept;

    /**
     * @brief The class symbols, ascending.
     */
    std::string_view symbols_;
};

Construction::Construction(const std::string_view symbols) noexcept : symbols_{symbols}
{}

munch::verifier::Verifier Construction::verifier(const std::span<const Alternative> alternatives) const
{
    const auto relation{one_step(alternatives)};

    const auto agreeing{first_pieces(relation)};

    const auto pieces{nonempty()};

    const auto disagreeing{automata::product(pieces, agreeing, Accepts::first_only)};

    const auto starts{ending_in(true)};

    const auto bad{automata::concatenation(starts, disagreeing)};

    const auto well_formed{ending_in(false)};

    const auto marking{automata::product(well_formed, bad, Accepts::first_only)};

    return table(marking);
}

Dfa Construction::one_step(const std::span<const Alternative> alternatives) const
{
    // Before the first alternative Q holds nothing, and every residual is one no earlier alternative matches.
    auto relation{one_state(marker() + 1, false)};

    auto unmatched{one_state(marker(), true)};

    for (const auto& alternative : alternatives)
    {
        const auto endpoints{endpoints_of(alternative)};

        const auto accepts{longest(endpoints)};

        const auto unmatched_before{with_marker(unmatched)};

        const auto commits{automata::product(accepts, unmatched_before, Accepts::both)};

        relation = automata::product(relation, commits, Accepts::either);

        const auto matched_here{erased(endpoints)};

        const auto unmatched_here{automata::complement(matched_here)};

        unmatched = automata::product(unmatched, unmatched_here, Accepts::both);
    }

    const auto residual{shortest_nonempty(unmatched)};

    if (residual)
    {
        throw std::invalid_argument{std::format(
                "split_verifier: no alternative matches the residual {}, so the pattern is not total",
                shown(*residual))};
    }

    return relation;
}

Symbol_t Construction::marker() const noexcept
{
    return symbols_.size();
}

Dfa Construction::endpoints_of(const Alternative& alternative) const
{
    const auto& [language, gate]{alternative};

    const auto words{language_of(language)};

    const auto tails{marker_and_tail(gate)};

    return automata::concatenation(words, tails);
}

Dfa Construction::language_of(const munch::regex::Regex& language) const
{
    const auto simulator{simulator_of(language)};

    const std::optional<std::size_t> start{simulator.init_state()};

    std::map<std::optional<std::size_t>, State_t> numbered{{start, 0}};

    std::vector<std::optional<std::size_t>> sources{start};

    Dfa words{.symbol_count = marker() + 1, .transitions = {}, .accepting = {}};

    for (State_t state{0}; state < sources.size(); ++state)
    {
        const auto source{sources[state]};

        std::vector<State_t> row{};

        for (Symbol_t symbol{0}; symbol <= marker(); ++symbol)
        {
            const auto next{step(simulator, source, symbol)};

            const auto [entry, inserted]{numbered.try_emplace(next, sources.size())};

            if (inserted)
            {
                sources.push_back(next);
            }

            row.push_back(entry->second);
        }

        const auto accepts{source && simulator.is_accepting(*source)};

        words.transitions.push_back(std::move(row));
        words.accepting.push_back(accepts);
    }

    return automata::minimize(words);
}

std::optional<std::size_t> Construction::step(
        const munch::dfa::Simulator& simulator, const std::optional<std::size_t> state, const Symbol_t symbol) const
{
    if (!state || symbol == marker())
    {
        return std::nullopt;
    }

    const auto byte{static_cast<unsigned char>(symbols_[symbol])};

    return simulator.step(*state, byte);
}

Dfa Construction::marker_and_tail(const std::string_view gate) const
{
    // The states before the marker, just after it, in a tail that passed the gate, and dead.
    constexpr State_t dead{3};

    Dfa tails{
            .symbol_count = marker() + 1,
            .transitions = std::vector(4, std::vector<State_t>(marker() + 1, dead)),
            .accepting = {false, true, true, false}};

    constexpr State_t before_marker{0};

    constexpr State_t after_marker{1};

    constexpr State_t in_tail{2};

    tails.transitions[before_marker][marker()] = after_marker;

    for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
    {
        tails.transitions[in_tail][symbol] = in_tail;

        if (gate.contains(symbols_[symbol]))
        {
            tails.transitions[after_marker][symbol] = in_tail;
        }
    }

    return tails;
}

Dfa Construction::longest(const Dfa& endpoints) const
{
    const auto longer{later(endpoints)};

    return automata::product(endpoints, longer, Accepts::first_only);
}

Dfa Construction::later(const Dfa& endpoints) const
{
    Nfa nfa{.symbol_count = endpoints.symbol_count, .edges = {}, .epsilons = {}, .start = 0, .accepting = {}};

    const auto copy{[&] {
        std::vector<State_t> states{};

        for (std::size_t index{0}; index < endpoints.transitions.size(); ++index)
        {
            states.push_back(automata::add_state(nfa));
        }

        return states;
    }};

    // Reading u; the proposed marker read; past it by at least one symbol; past B's own marker.
    const auto before{copy()};

    const auto proposed{copy()};

    const auto extended{copy()};

    const auto after{copy()};

    for (State_t state{0}; state < endpoints.transitions.size(); ++state)
    {
        for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
        {
            const auto next{endpoints.transitions[state][symbol]};

            nfa.edges[before[state]].push_back({.symbol = symbol, .to = before[next]});
            nfa.edges[proposed[state]].push_back({.symbol = symbol, .to = extended[next]});
            nfa.edges[extended[state]].push_back({.symbol = symbol, .to = extended[next]});
            nfa.edges[after[state]].push_back({.symbol = symbol, .to = after[next]});
        }

        const auto marked{endpoints.transitions[state][marker()]};

        nfa.edges[before[state]].push_back({.symbol = marker(), .to = proposed[state]});
        nfa.epsilons[extended[state]].push_back(after[marked]);

        if (endpoints.accepting[state])
        {
            nfa.accepting.push_back(after[state]);
        }
    }

    nfa.start = before[0];

    return automata::determinize(nfa);
}

Dfa Construction::with_marker(const Dfa& residuals) const
{
    // A state before the marker is the residuals' state, one after it that plus their count, and the last is dead.
    const auto count{residuals.transitions.size()};

    const auto dead{2 * count};

    Dfa marked{.symbol_count = marker() + 1, .transitions = {}, .accepting = {}};

    for (const auto after : {false, true})
    {
        const auto base{after ? count : 0};

        for (State_t state{0}; state < count; ++state)
        {
            std::vector<State_t> row{};

            for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
            {
                row.push_back(base + residuals.transitions[state][symbol]);
            }

            row.push_back(after ? dead : count + state);

            marked.transitions.push_back(std::move(row));
            marked.accepting.push_back(after && residuals.accepting[state]);
        }
    }

    marked.transitions.emplace_back(marker() + 1, dead);
    marked.accepting.push_back(false);

    return automata::minimize(marked);
}

Dfa Construction::erased(const Dfa& endpoints) const
{
    Nfa nfa{.symbol_count = marker(), .edges = {}, .epsilons = {}, .start = 0, .accepting = {}};

    std::vector<State_t> states{};

    for (std::size_t index{0}; index < endpoints.transitions.size(); ++index)
    {
        states.push_back(automata::add_state(nfa));
    }

    for (State_t state{0}; state < endpoints.transitions.size(); ++state)
    {
        for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
        {
            const auto next{endpoints.transitions[state][symbol]};

            nfa.edges[states[state]].push_back({.symbol = symbol, .to = states[next]});
        }

        const auto marked{endpoints.transitions[state][marker()]};

        nfa.epsilons[states[state]].push_back(states[marked]);

        if (endpoints.accepting[state])
        {
            nfa.accepting.push_back(states[state]);
        }
    }

    nfa.start = states[0];

    return automata::determinize(nfa);
}

std::optional<std::string> Construction::shortest_nonempty(const Dfa& residuals) const
{
    // The string that first reaches each state, the start reached by none until a symbol is read.
    std::vector<std::optional<std::string>> reached(residuals.transitions.size());

    std::deque<State_t> frontier{};

    const auto visit{[&](const State_t state, std::string word) {
        if (reached[state])
        {
            return;
        }

        reached[state] = std::move(word);
        frontier.push_back(state);
    }};

    for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
    {
        visit(residuals.transitions[0][symbol], std::string{symbols_[symbol]});
    }

    while (!frontier.empty())
    {
        const auto state{frontier.front()};

        frontier.pop_front();

        if (residuals.accepting[state])
        {
            return reached[state];
        }

        for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
        {
            auto longer{*reached[state] + symbols_[symbol]};

            visit(residuals.transitions[state][symbol], std::move(longer));
        }
    }

    return std::nullopt;
}

Dfa Construction::first_pieces(const Dfa& relation) const
{
    // A state before the first boundary is Q's state, one after it that plus Q's state count.
    const auto count{relation.transitions.size()};

    Dfa pieces{.symbol_count = 2 * marker(), .transitions = {}, .accepting = {}};

    for (const auto found : {false, true})
    {
        const auto base{found ? count : 0};

        for (State_t state{0}; state < count; ++state)
        {
            std::vector<State_t> row{};

            for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
            {
                const auto next{relation.transitions[state][symbol]};

                const auto ended{relation.transitions[next][marker()]};

                row.push_back(base + next);
                row.push_back(found ? count + next : count + ended);
            }

            const auto ends_here{relation.transitions[state][marker()]};

            const auto accepts{found ? relation.accepting[state] : relation.accepting[ends_here]};

            pieces.transitions.push_back(std::move(row));
            pieces.accepting.push_back(accepts);
        }
    }

    return automata::minimize(pieces);
}

Dfa Construction::nonempty() const
{
    const auto symbol_count{2 * marker()};

    return {.symbol_count = symbol_count,
            .transitions = std::vector(2, std::vector<State_t>(symbol_count, 1)),
            .accepting = {false, true}};
}

Dfa Construction::ending_in(const bool boundary) const
{
    std::vector<State_t> row{};

    for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
    {
        for (const auto bit : {false, true})
        {
            row.push_back(bit == boundary ? 0 : 1);
        }
    }

    return {.symbol_count = 2 * marker(), .transitions = {row, row}, .accepting = {true, false}};
}

munch::verifier::Verifier Construction::table(const Dfa& marked) const
{
    munch::verifier::Verifier::Transitions_t transitions{};

    munch::verifier::Verifier::Accept_states_t accept_states{};

    for (State_t state{0}; state < marked.transitions.size(); ++state)
    {
        for (Symbol_t symbol{0}; symbol < marker(); ++symbol)
        {
            for (const auto boundary : {false, true})
            {
                const munch::verifier::Marked read{
                        .byte = static_cast<unsigned char>(symbols_[symbol]),
                        .boundary_after = boundary};

                const auto marked_index{marked_symbol(symbol, boundary)};

                const auto next{marked.transitions[state][marked_index]};

                transitions.emplace(munch::verifier::Verifier::Key_t{state, read}, next);
            }
        }

        if (marked.accepting[state])
        {
            accept_states.insert(state);
        }
    }

    return munch::verifier::Verifier{0, std::move(transitions), std::move(accept_states)};
}

Symbol_t Construction::marked_symbol(const Symbol_t symbol, const bool boundary) noexcept
{
    return (2 * symbol) + (boundary ? 1 : 0);
}

} // namespace

munch::verifier::Verifier split_verifier(const Pattern& pattern)
{
    const Construction construction{pattern.alphabet.symbols};

    return construction.verifier(pattern.alternatives);
}

munch::core::Lexer factorization_lexer(const Pattern& pattern)
{
    munch::core::Builder builder{};

    for (std::size_t index{0}; index < pattern.alternatives.size(); ++index)
    {
        const auto& language{pattern.alternatives[index].language};

        builder.add_token(language, index, index);
    }

    return builder.build();
}

} // namespace matisse::pretok
