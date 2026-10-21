#include "matisse/pretok/segmenter.hpp"

#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <munch/core/builder.hpp>
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

Segmenter::Segmenter(const Pattern& pattern)
    : classes_{pattern.alphabet}, alternatives_{compiled_of(pattern.alternatives)}
{}

std::vector<Segmenter::Compiled> Segmenter::compiled_of(const std::span<const Alternative> alternatives)
{
    std::vector<Compiled> compiled{};

    for (const auto& [language, gate] : alternatives)
    {
        const auto lookahead{lookahead_of(gate)};

        const auto gated{std::format("({}){}", language, lookahead)};

        auto simulator{simulator_of(gated)};

        compiled.push_back({.simulator = std::move(simulator)});
    }

    return compiled;
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

    for (const auto& [simulator] : alternatives_)
    {
        // The match is the longest accept followed by the symbol its gate passes on, which the piece does not take.
        const auto [token, length]{simulator.run(rest)};

        if (length != 0)
        {
            return length - 1;
        }
    }

    return 0;
}

} // namespace matisse::pretok
