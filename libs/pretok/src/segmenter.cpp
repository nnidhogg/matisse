#include "matisse/pretok/segmenter.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <munch/core/builder.hpp>
#include <munch/dfa/builder.hpp>
#include <munch/dfa/label.hpp>
#include <munch/dfa/token.hpp>
#include <munch/pattern/parse.hpp>

namespace matisse::pretok
{
namespace
{
/**
 * @brief The byte after the last symbol of a class string as the alternatives read it: no class symbol, since the
 *        symbols are ASCII, and a symbol every lookahead holds, so that an accept at the end of the input passes.
 */
constexpr char sentinel{'\xFF'};

/**
 * @brief Returns the bracket of a gate's symbols and the sentinel in munch's pattern syntax, each written as a byte
 *        escape, `[\xHH...\xFF]`.
 * @param gate The class symbols that may follow an accept.
 * @return The bracket.
 */
[[nodiscard]] std::string lookahead_of(const std::string_view gate)
{
    std::string members{gate};

    members += sentinel;

    std::string bracket{"["};

    for (const auto symbol : members)
    {
        const auto value{static_cast<unsigned char>(symbol)};

        bracket += std::format(R"(\x{:02X})", value);
    }

    bracket += ']';

    return bracket;
}

/**
 * @brief Returns the DFA of an alternative's language followed by its lookahead, compiled by munch as a single token.
 * @param gated The language and its lookahead, in munch's pattern syntax.
 * @return The DFA.
 */
[[nodiscard]] munch::dfa::Simulator simulator_of(const std::string_view gated)
{
    munch::core::Builder builder{};

    builder.add_token(munch::pattern::parse(gated), std::size_t{0}, 0);

    const auto lexer{builder.build()};

    return lexer.simulator();
}

/**
 * @brief The state of an alternative that has stopped in a walk of the product: its automaton has no move on the symbol
 *        read, no accept is reachable from where it stands, or an alternative before it has accepted.
 */
constexpr std::size_t stopped{std::numeric_limits<std::size_t>::max()};

/**
 * @brief Returns the state a state of the product of the alternatives' automata moves to on a symbol, std::nullopt
 *        where every alternative that can still decide the piece has stopped.
 *
 * A state of the product is each alternative's state, then the index of the first alternative that has accepted, the
 * alternatives' count while none has. Each alternative steps on the symbol; the first that has accepted becomes the
 * least of the old one and every alternative accepting after the step, and every alternative after it stops, since a
 * later alternative never decides the piece once an earlier one has an accept.
 * @param gated The alternatives' automata, each its language followed by its lookahead.
 * @param state The state of the product.
 * @param symbol The class symbol or the sentinel read.
 * @return The state moved to, std::nullopt where every alternative that can still decide the piece has stopped.
 */
[[nodiscard]] std::optional<std::vector<std::size_t>> moved_on(
        const std::span<const munch::dfa::Simulator> gated, const std::span<const std::size_t> state, const char symbol)
{
    const auto count{gated.size()};

    std::vector moved(count + 1, stopped);

    auto first{state[count]};

    for (std::size_t alternative{0}; alternative < count; ++alternative)
    {
        const auto from{state[alternative]};

        if (from == stopped)
        {
            continue;
        }

        const auto& automaton{gated[alternative]};

        const auto to{automaton.step(from, static_cast<unsigned char>(symbol))};

        if (!to || !automaton.is_live(*to))
        {
            continue;
        }

        moved[alternative] = *to;

        if (automaton.is_accepting(*to) && alternative < first)
        {
            first = alternative;
        }
    }

    moved[count] = first;

    const auto deciding_count{std::min(first + 1, count)};

    const auto states{std::span{moved}.first(count)};

    const auto later{states.subspan(deciding_count)};

    std::ranges::fill(later, stopped);

    const auto deciding{states.first(deciding_count)};

    const auto all_stopped{std::ranges::all_of(
            deciding, [](const std::size_t alternative_state) { return alternative_state == stopped; })};

    if (all_stopped)
    {
        return std::nullopt;
    }

    return moved;
}

/**
 * @brief Returns whether a state of the product accepts: the first alternative that has accepted accepts in it.
 * @param gated The alternatives' automata.
 * @param state The state of the product.
 * @return True when it accepts.
 */
[[nodiscard]] bool accepts(const std::span<const munch::dfa::Simulator> gated, const std::span<const std::size_t> state)
{
    const auto first{state[gated.size()]};

    if (first == gated.size() || state[first] == stopped)
    {
        return false;
    }

    return gated[first].is_accepting(state[first]);
}

/**
 * @brief Returns how many bytes the UTF-8 sequence a lead byte begins holds, as its high bits say.
 * @param lead The lead byte.
 * @return One to four.
 */
[[nodiscard]] constexpr std::size_t sequence_length(const unsigned char lead) noexcept
{
    return lead < 0x80U ? 1 : lead >= 0xF0U ? 4 : lead >= 0xE0U ? 3 : 2;
}

/**
 * @brief Returns the byte offset a number of code points past an offset of a well-formed UTF-8 text.
 * @param text The text.
 * @param from The offset, at the start of a code point.
 * @param points The number of code points, at most those after the offset.
 * @return The offset after them.
 */
[[nodiscard]] std::size_t byte_after(
        const std::string_view text, const std::size_t from, const std::size_t points) noexcept
{
    auto byte{from};

    for (std::size_t point{0}; point < points; ++point)
    {
        byte += sequence_length(static_cast<unsigned char>(text[byte]));
    }

    return byte;
}

} // namespace

Segmenter::Segmenter(const Pattern& pattern) : classes_{pattern.alphabet}, automaton_{automaton_of(pattern)}
{}

munch::dfa::Simulator Segmenter::automaton_of(const Pattern& pattern)
{
    const auto& [alphabet, alternatives]{pattern};

    std::vector<munch::dfa::Simulator> gated{};

    for (const auto& [language, gate] : alternatives)
    {
        const auto lookahead{lookahead_of(gate)};

        const auto gated_language{std::format("({}){}", language, lookahead)};

        gated.push_back(simulator_of(gated_language));
    }

    std::vector<std::size_t> start{};

    for (const auto& automaton : gated)
    {
        start.push_back(automaton.init_state());
    }

    start.push_back(gated.size());

    munch::dfa::Builder builder{};

    // The states of the product by their number, the start the builder's initial state, zero.
    std::map<std::vector<std::size_t>, munch::dfa::Dfa::State_t> numbered{{start, builder.init_state()}};

    std::vector<std::vector<std::size_t>> sources{start};

    std::string symbols{alphabet.symbols};

    symbols += sentinel;

    for (munch::dfa::Dfa::State_t from{0}; from < sources.size(); ++from)
    {
        const auto state{sources[from]};

        if (accepts(gated, state))
        {
            builder.add_accept_state(from, munch::dfa::Token{0});
        }

        for (const auto symbol : symbols)
        {
            const auto moved{moved_on(gated, state, symbol)};

            if (!moved)
            {
                continue;
            }

            const auto [entry, inserted]{numbered.try_emplace(*moved, sources.size())};

            if (inserted)
            {
                sources.push_back(*moved);
            }

            builder.add_transition(from, munch::dfa::Label{symbol}, entry->second);
        }
    }

    const auto dfa{std::move(builder).build()};

    return munch::dfa::Simulator{dfa};
}

std::vector<std::size_t> Segmenter::boundaries(const std::string_view text) const
{
    auto symbols{classes_.symbols(text)};

    const auto points{symbols.size()};

    symbols += sentinel;

    std::vector<std::size_t> ends{};

    std::size_t byte{0};

    for (std::size_t at{0}; at < points;)
    {
        const auto length{piece_length(symbols, at)};

        byte = byte_after(text, byte, length);

        at += length;

        ends.push_back(byte);
    }

    return ends;
}

std::size_t Segmenter::piece_length(const std::string_view symbols, const std::size_t at) const
{
    const auto rest{symbols.substr(at)};

    // The match is the longest accept of the first alternative with one, followed by the symbol its gate passes on,
    // which the piece does not take.
    const auto [token, length]{automaton_.run(rest)};

    return length == 0 ? 0 : length - 1;
}

} // namespace matisse::pretok
