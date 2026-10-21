#include "matisse/pretok/segmenter.hpp"

#include <cstddef>
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
 * @brief Returns the DFA of one alternative's language, compiled by munch as a single token.
 * @param language The language, over the class symbols in munch's pattern syntax.
 * @return The DFA.
 */
[[nodiscard]] munch::dfa::Simulator simulator_of(const std::string_view language)
{
    munch::core::Builder builder{};

    builder.add_token(munch::pattern::parse(language), std::size_t{0}, 0);

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
        auto simulator{simulator_of(language)};

        compiled.push_back({.simulator = std::move(simulator), .gate = gate});
    }

    return compiled;
}

std::vector<std::size_t> Segmenter::boundaries(const std::string_view text) const
{
    const auto symbols{classes_.symbols(text)};

    std::vector<std::size_t> ends{};

    std::size_t byte{0};

    for (std::size_t at{0}; at < symbols.size();)
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
    for (const auto& alternative : alternatives_)
    {
        const auto longest{longest_accept(alternative, symbols, at)};

        if (longest != 0)
        {
            return longest;
        }
    }

    return 0;
}

std::size_t Segmenter::longest_accept(
        const Compiled& alternative, const std::string_view symbols, const std::size_t at) noexcept
{
    const auto& [simulator, gate]{alternative};

    std::size_t longest{0};

    auto state{simulator.init_state()};

    for (auto end{at}; end < symbols.size();)
    {
        const auto next{simulator.step(state, static_cast<unsigned char>(symbols[end]))};

        ++end;

        if (!next || !simulator.is_live(*next))
        {
            break;
        }

        state = *next;

        if (simulator.is_accepting(state) && passes(gate, symbols, end))
        {
            longest = end - at;
        }
    }

    return longest;
}

bool Segmenter::passes(const std::string_view gate, const std::string_view symbols, const std::size_t end) noexcept
{
    if (end == symbols.size())
    {
        return true;
    }

    return gate.contains(symbols[end]);
}

} // namespace matisse::pretok
