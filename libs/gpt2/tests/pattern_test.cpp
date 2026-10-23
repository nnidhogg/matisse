#include "matisse/pretok/pattern.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <string_view>
#include <vector>

#include <munch/core/builder.hpp>
#include <munch/dfa/simulator.hpp>
#include <munch/pattern/parse.hpp>
#include <munch/regex/regex.hpp>

#include "matisse/gpt2/tokenizer.hpp"

namespace
{
/**
 * @brief GPT-2's alternatives spelled over its class symbols in munch's pattern syntax, in the order of the pattern,
 *        each the vendor's alternative read over the classes.
 */
constexpr std::array<std::string_view, 12> spellings{
        "'s",           "'t",
        "'re",          "'ve",
        "'m",           "'ll",
        "'d",           R"(\x20?[Adelmrstv]+)",
        R"(\x20?0+)",   R"(\x20?[!']+)",
        R"([\t\x20]+)", R"([\t\x20]+)",
};

/**
 * @brief Returns the minimal DFA of a language, compiled by munch as a single token.
 * @param language The language.
 * @return The DFA.
 */
munch::dfa::Simulator simulator_of(const munch::regex::Regex& language)
{
    munch::core::Builder builder{};

    builder.add_token(language, std::size_t{0}, 0);

    const auto lexer{builder.build()};

    return lexer.simulator();
}

/**
 * @brief Returns whether two DFAs are the same state for state: as many states each, and the states reachable from the
 *        starts paired one to one, the starts with each other, so that paired states agree on accepting and on being
 *        live and on every byte both move to a pair or neither moves.
 * @param left The one DFA.
 * @param right The other DFA.
 * @return Whether they are.
 */
bool same_state_for_state(const munch::dfa::Simulator& left, const munch::dfa::Simulator& right)
{
    if (left.state_count() != right.state_count())
    {
        return false;
    }

    std::vector<std::optional<std::size_t>> partner(left.state_count());

    std::vector<bool> paired(right.state_count());

    partner[left.init_state()] = right.init_state();

    paired[right.init_state()] = true;

    std::vector<std::size_t> pending{left.init_state()};

    while (!pending.empty())
    {
        const auto state{pending.back()};

        pending.pop_back();

        const auto other{*partner[state]};

        if (left.is_accepting(state) != right.is_accepting(other) || left.is_live(state) != right.is_live(other))
        {
            return false;
        }

        for (unsigned value{0}; value < 256; ++value)
        {
            const auto byte{static_cast<unsigned char>(value)};

            const auto moved{left.step(state, byte)};

            const auto other_moved{right.step(other, byte)};

            if (moved.has_value() != other_moved.has_value())
            {
                return false;
            }

            if (!moved || partner[*moved] == other_moved)
            {
                continue;
            }

            if (partner[*moved] || paired[*other_moved])
            {
                return false;
            }

            partner[*moved] = other_moved;

            paired[*other_moved] = true;

            pending.push_back(*moved);
        }
    }

    return true;
}

} // namespace

TEST(Pattern_test, Each_of_gpt2_s_alternatives_compiles_to_the_minimal_dfa_of_its_spelling_state_for_state)
{
    const auto pattern{matisse::gpt2::pattern()};

    ASSERT_EQ(pattern.alternatives.size(), spellings.size());

    for (std::size_t index{0}; index < spellings.size(); ++index)
    {
        const auto spelling{spellings[index]};

        const auto built{simulator_of(pattern.alternatives[index].language)};

        const auto parsed_language{munch::pattern::parse(spelling)};

        const auto parsed{simulator_of(parsed_language)};

        EXPECT_TRUE(same_state_for_state(built, parsed)) << spelling;
    }
}
