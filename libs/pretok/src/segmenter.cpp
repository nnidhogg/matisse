#include "matisse/pretok/segmenter.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <munch/core/builder.hpp>
#include <munch/pattern/parse.hpp>

#include "matisse/pretok/classes.hpp"

namespace matisse::pretok
{
namespace
{
/**
 * @brief One alternative of a published pattern as the Segmenter compiles it: its gate-erased language written over
 *        the class symbols in munch's pattern syntax, and its gate as the pattern spells it.
 */
struct Spelling
{
    /**
     * @brief The gate-erased language, over the class symbols.
     */
    std::string_view pattern{};

    /**
     * @brief The gate, the lookahead `(?!\S)`, the anchor `$`, or empty for none.
     */
    std::string_view gate{};
};

/**
 * @brief The lookahead gate as the patterns spell it.
 */
constexpr std::string_view lookahead{R"((?!\S))"};

/**
 * @brief The end anchor as cl100k's pattern spells it.
 */
constexpr std::string_view anchor{"$"};

/**
 * @brief GPT-2's pattern, `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, alternative by
 *        alternative over its classes: the letters are `A` and the contraction letters, the numbers `0`, the white
 *        space `\t` and the space, and everything else `!` and the apostrophe.
 */
constexpr std::array<Spelling, 12> gpt2_spellings{{
        {.pattern = "'s"},
        {.pattern = "'t"},
        {.pattern = "'re"},
        {.pattern = "'ve"},
        {.pattern = "'m"},
        {.pattern = "'ll"},
        {.pattern = "'d"},
        {.pattern = R"(\x20?[Adelmrstv]+)"},
        {.pattern = R"(\x20?0+)"},
        {.pattern = R"(\x20?[!']+)"},
        {.pattern = R"([\t\x20]+)", .gate = lookahead},
        {.pattern = R"([\t\x20]+)"},
}};

/**
 * @brief cl100k's pattern, `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|`
 *        `\s++$|\s*[\r\n]|\s+(?!\S)|\s`, alternative by alternative over its classes: the letters are `A` and the
 *        contraction letters, each holding both cases, the numbers `0`, the white space `\t`, `\n`, `\r` and the space,
 *        and everything else `!` and the apostrophe.
 */
constexpr std::array<Spelling, 8> cl100k_spellings{{
        {.pattern = "'([sdmt]|ll|ve|re)"},
        {.pattern = R"([\t\x20!']?[Adelmrstv]+)"},
        {.pattern = "0{1,3}"},
        {.pattern = R"(\x20?[!']+[\r\n]*)"},
        {.pattern = R"([\t\n\r\x20]+)", .gate = anchor},
        {.pattern = R"([\t\n\r\x20]*[\r\n])"},
        {.pattern = R"([\t\n\r\x20]+)", .gate = lookahead},
        {.pattern = R"([\t\n\r\x20])"},
}};

/**
 * @brief Returns the class symbols of a pattern's white space, `\s`.
 * @param pattern The pattern, GPT-2's or cl100k's.
 * @return The symbols.
 */
[[nodiscard]] std::string_view white_space_of(const load::Pattern pattern) noexcept
{
    return pattern == load::Pattern::gpt2 ? "\t " : "\t\n\r ";
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

} // namespace

Segmenter::Segmenter(const load::Pattern pattern)
    : pattern_{pattern}, alternatives_{alternatives_of(pattern)}, white_space_{white_space_of(pattern)}
{}

std::vector<Segmenter::Alternative> Segmenter::alternatives_of(const load::Pattern pattern)
{
    if (pattern == load::Pattern::none)
    {
        throw std::invalid_argument{"Segmenter: Pattern::none has no pre-tokenizer"};
    }

    const auto spellings{
            pattern == load::Pattern::gpt2 ? std::span<const Spelling>{gpt2_spellings} :
                                             std::span<const Spelling>{cl100k_spellings}};

    std::vector<Alternative> alternatives{};

    for (const auto& [spelling, gate] : spellings)
    {
        munch::core::Builder builder{};

        builder.add_token(munch::pattern::parse(spelling), std::size_t{0}, 0);

        const auto lexer{builder.build()};

        const auto kind{gate == lookahead ? Gate::next_white_space : gate == anchor ? Gate::end_of_input : Gate::none};

        alternatives.push_back({.simulator = lexer.simulator(), .gate = kind});
    }

    return alternatives;
}

std::vector<std::size_t> Segmenter::boundaries(const std::string_view text) const
{
    const auto classes{classes_of(pattern_, text)};

    const auto symbol_of_class{class_symbols(pattern_)};

    std::string symbols{};

    for (const auto symbol : classes)
    {
        symbols += symbol_of_class[symbol];
    }

    std::vector<std::size_t> ends{};

    std::size_t byte{0};

    for (std::size_t at{0}; at < symbols.size();)
    {
        const auto length{piece_length(symbols, at)};

        for (std::size_t point{0}; point < length; ++point)
        {
            byte += sequence_length(static_cast<unsigned char>(text[byte]));
        }

        at += length;

        ends.push_back(byte);
    }

    return ends;
}

std::size_t Segmenter::piece_length(const std::string_view symbols, const std::size_t at) const
{
    for (const auto& [simulator, gate] : alternatives_)
    {
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

        if (longest != 0)
        {
            return longest;
        }
    }

    return 0;
}

bool Segmenter::passes(const Gate gate, const std::string_view symbols, const std::size_t end) const noexcept
{
    if (gate == Gate::none || end == symbols.size())
    {
        return true;
    }

    return gate == Gate::next_white_space && white_space_.contains(symbols[end]);
}

} // namespace matisse::pretok
