#include "matisse/pretok/automata.hpp"

#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <tuple>
#include <utility>
#include <vector>

using namespace matisse::pretok::automata;

namespace
{
/**
 * @brief A word over the alphabet of two symbols, zero for a and one for b.
 */
using Word_t = std::vector<Symbol_t>;

/**
 * @brief The longest word the tests enumerate.
 */
constexpr std::size_t longest_word{7};

/**
 * @brief Returns every word over the two symbols of lengths zero to longest_word, shorter first.
 * @return The words.
 */
std::vector<Word_t> words()
{
    std::vector<Word_t> all{{}};

    for (std::size_t from{0}; from < all.size(); ++from)
    {
        if (all[from].size() == longest_word)
        {
            continue;
        }

        for (const Symbol_t symbol : {0U, 1U})
        {
            auto longer{all[from]};

            longer.push_back(symbol);
            all.push_back(std::move(longer));
        }
    }

    return all;
}

/**
 * @brief Returns whether a DFA accepts a word.
 * @param dfa The DFA.
 * @param word The word.
 * @return True when the run from the start ends accepting.
 */
bool accepts(const Dfa& dfa, const Word_t& word)
{
    State_t state{0};

    for (const auto symbol : word)
    {
        state = dfa.transitions[state][symbol];
    }

    return dfa.accepting[state];
}

/**
 * @brief Returns the DFA of the words that end in b, with two states.
 * @return The DFA.
 */
Dfa ends_in_b()
{
    return {.symbol_count = 2, .transitions = {{0, 1}, {0, 1}}, .accepting = {false, true}};
}

/**
 * @brief Returns the DFA of the words of even length, with two states.
 * @return The DFA.
 */
Dfa even_length()
{
    return {.symbol_count = 2, .transitions = {{1, 1}, {0, 0}}, .accepting = {true, false}};
}

/**
 * @brief Returns the NFA of the words whose third symbol from the end is b, which guesses that b and reads two more.
 * @return The NFA.
 */
Nfa third_from_end_is_b()
{
    Nfa nfa{.symbol_count = 2, .edges = {}, .epsilons = {}, .start = 0, .accepting = {}};

    for (std::size_t index{0}; index < 4; ++index)
    {
        std::ignore = add_state(nfa);
    }

    nfa.edges[0] = {{.symbol = 0, .to = 0}, {.symbol = 1, .to = 0}, {.symbol = 1, .to = 1}};
    nfa.edges[1] = {{.symbol = 0, .to = 2}, {.symbol = 1, .to = 2}};
    nfa.edges[2] = {{.symbol = 0, .to = 3}, {.symbol = 1, .to = 3}};
    nfa.accepting = {3};

    return nfa;
}

/**
 * @brief Expects a DFA to accept exactly the words a predicate holds for, over every word the tests enumerate.
 * @tparam Language The predicate's type.
 * @param dfa The DFA.
 * @param language The predicate.
 */
template <std::predicate<const Word_t&> Language>
void expect_language(const Dfa& dfa, const Language& language)
{
    for (const auto& word : words())
    {
        EXPECT_EQ(accepts(dfa, word), language(word)) << "word of length " << word.size();
    }
}

} // namespace

TEST(Automata_test, Minimize_merges_the_states_no_word_tells_apart_and_keeps_the_rest)
{
    // ends_in_b() with each of its states doubled, the copies moving into each other.
    const Dfa doubled{
            .symbol_count = 2,
            .transitions = {{0, 3}, {2, 1}, {0, 3}, {2, 1}},
            .accepting = {false, true, false, true}};

    const auto minimal{minimize(doubled)};

    EXPECT_EQ(minimal.transitions, ends_in_b().transitions);
    EXPECT_EQ(minimal.accepting, ends_in_b().accepting);
}

TEST(Automata_test, Minimize_keeps_states_that_differ_only_after_a_word)
{
    const auto minimal{determinize(third_from_end_is_b())};

    // The minimal DFA remembers the last three symbols read.
    EXPECT_EQ(minimal.transitions.size(), 8U);

    expect_language(minimal, [](const Word_t& word) { return word.size() >= 3 && word[word.size() - 3] == 1; });
}

TEST(Automata_test, Minimize_drops_the_unreachable_states_and_numbers_the_start_zero)
{
    const Dfa with_island{.symbol_count = 2, .transitions = {{2, 2}, {1, 1}, {0, 0}}, .accepting = {true, true, false}};

    const auto minimal{minimize(with_island)};

    EXPECT_EQ(minimal.transitions, even_length().transitions);
    EXPECT_EQ(minimal.accepting, even_length().accepting);
}

TEST(Automata_test, The_complement_accepts_exactly_what_the_automaton_rejects)
{
    const auto rejected{complement(ends_in_b())};

    expect_language(rejected, [](const Word_t& word) { return word.empty() || word.back() == 0; });
}

TEST(Automata_test, A_product_accepts_as_its_operation_combines_the_two_languages)
{
    const auto ends{[](const Word_t& word) { return !word.empty() && word.back() == 1; }};

    const auto even{[](const Word_t& word) { return word.size() % 2 == 0; }};

    const auto both{product(ends_in_b(), even_length(), Accepts::both)};

    const auto either{product(ends_in_b(), even_length(), Accepts::either)};

    const auto first_only{product(ends_in_b(), even_length(), Accepts::first_only)};

    expect_language(both, [&](const Word_t& word) { return ends(word) && even(word); });
    expect_language(either, [&](const Word_t& word) { return ends(word) || even(word); });
    expect_language(first_only, [&](const Word_t& word) { return ends(word) && !even(word); });
}

TEST(Automata_test, The_concatenation_accepts_every_word_split_into_a_word_of_each)
{
    const auto concatenated{concatenation(ends_in_b(), even_length())};

    // Some prefix of the word ends in b and the rest of it has even length.
    expect_language(concatenated, [](const Word_t& word) {
        for (std::size_t end{1}; end <= word.size(); ++end)
        {
            if (word[end - 1] == 1 && (word.size() - end) % 2 == 0)
            {
                return true;
            }
        }

        return false;
    });
}

TEST(Automata_test, Embed_copies_a_dfa_into_an_nfa_after_its_states)
{
    Nfa nfa{.symbol_count = 2, .edges = {}, .epsilons = {}, .start = 0, .accepting = {}};

    std::ignore = add_state(nfa);

    const auto base{embed(nfa, even_length())};

    nfa.epsilons[0].push_back(base);
    nfa.accepting.push_back(base);

    EXPECT_EQ(base, 1U);
    expect_language(determinize(nfa), [](const Word_t& word) { return word.size() % 2 == 0; });
}
