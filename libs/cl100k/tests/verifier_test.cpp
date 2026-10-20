#include "matisse/pretok/verifier.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <munch/verifier/verifier_decisions.hpp>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/pretok/classes.hpp"
#include "matisse/pretok/segmenter.hpp"

using namespace matisse::pretok;

namespace
{
/**
 * @brief A boundary marking, one bit per class symbol, set when a piece ends after it.
 */
using Marking_t = std::vector<bool>;

/**
 * @brief What the verifier fixture holds, counted as it is checked.
 */
struct Fixture_counts
{
    /**
     * @brief The universe lines.
     */
    std::size_t universes{0};

    /**
     * @brief The states lines, one per verifier.
     */
    std::size_t states{0};

    /**
     * @brief The step lines, one per transition.
     */
    std::size_t steps{0};

    /**
     * @brief The accepting lines, one per verifier.
     */
    std::size_t accepting{0};

    /**
     * @brief The decision lines.
     */
    std::size_t decisions{0};
};

/**
 * @brief The verifier fixture under check: cl100k's split verifier, whether the current universe is cl100k's, and the
 *        counts of its lines so far.
 */
struct Fixture_check
{
    /**
     * @brief cl100k's split verifier.
     */
    munch::verifier::Verifier verifier{split_verifier(matisse::cl100k::pattern())};

    /**
     * @brief Whether the current universe is cl100k's, false before the first universe line.
     */
    bool inside{};

    /**
     * @brief The lines checked so far, by kind.
     */
    Fixture_counts counts{};

    /**
     * @brief The names of the decisions checked so far.
     */
    std::set<std::string> decided{};
};

/**
 * @brief Starts a universe line's universe, cl100k's when it names cl100k.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.inside = name == "cl100k";

    if (check.inside)
    {
        ++check.counts.universes;
    }
}

/**
 * @brief Checks a states line: cl100k's verifier has as many states as it says.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_states(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::size_t count{};

    fields >> count;

    EXPECT_EQ(check.verifier.state_count(), count) << context;

    ++check.counts.states;
}

/**
 * @brief Checks a step line: cl100k's verifier moves from its state on its byte, written in hex, and its boundary bit
 *        to its state.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_step(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::size_t from{};
    std::string byte{};
    unsigned mark{};
    std::size_t to{};

    fields >> from >> byte >> mark >> to;

    unsigned value{};

    std::ignore = std::from_chars(byte.data(), byte.data() + byte.size(), value, 16);

    const munch::verifier::Marked symbol{.byte = static_cast<unsigned char>(value), .boundary_after = mark == 1};

    EXPECT_EQ(check.verifier.step(from, symbol), std::optional{to}) << context;

    ++check.counts.steps;
}

/**
 * @brief Checks an accepting line: cl100k's verifier accepts at exactly its states.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_accepting(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    munch::verifier::Verifier::Accept_states_t states{};

    for (std::size_t state{}; fields >> state;)
    {
        states.insert(state);
    }

    EXPECT_EQ(check.verifier.accept_states(), states) << context;

    ++check.counts.accepting;
}

/**
 * @brief Checks a decision line: the reference decided that it holds, as the construction rests on M being functional
 *        and total and the pattern covering every nonempty residual.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_decision(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string name{};
    std::string verdict{};

    fields >> name >> verdict;

    EXPECT_EQ(verdict, "holds") << context;

    check.decided.insert(name);

    ++check.counts.decisions;
}

/**
 * @brief Checks one fixture line against cl100k's verifier, a comment or blank line checking nothing, a line of another
 *        universe skipped.
 * @param check The fixture under check.
 * @param line The line.
 */
void expect_fixture_line(Fixture_check& check, const std::string& line)
{
    std::istringstream fields{line};

    std::string kind{};

    if (!(fields >> kind) || kind.starts_with('#'))
    {
        return;
    }

    if (kind == "universe")
    {
        start_universe(check, fields);
    }
    else if (!check.inside)
    {
        return;
    }
    else if (kind == "states")
    {
        expect_states(check, fields, line);
    }
    else if (kind == "step")
    {
        expect_step(check, fields, line);
    }
    else if (kind == "accepting")
    {
        expect_accepting(check, fields, line);
    }
    else if (kind == "decision")
    {
        expect_decision(check, fields, line);
    }
    else
    {
        ADD_FAILURE() << "unknown fixture line: " << line;
    }
}

/**
 * @brief Returns the class string of a text under cl100k's alphabet, the symbol of every code point's class.
 * @param text The text.
 * @return The class string.
 */
std::string class_string(const std::string_view text)
{
    const auto alphabet{matisse::cl100k::pattern().alphabet};

    std::string symbols{};

    for (const auto symbol : classes_of(alphabet, text))
    {
        symbols += alphabet.symbols[symbol];
    }

    return symbols;
}

/**
 * @brief Returns whether a verifier accepts a class string under a marking.
 * @param verifier The verifier.
 * @param text The class string.
 * @param marking The marking, one bit per symbol.
 * @return True when the run from the start ends accepting.
 */
bool accepts(const munch::verifier::Verifier& verifier, const std::string_view text, const Marking_t& marking)
{
    auto state{std::optional{verifier.start()}};

    for (std::size_t index{0}; index < text.size() && state; ++index)
    {
        const munch::verifier::Marked symbol{
                .byte = static_cast<unsigned char>(text[index]),
                .boundary_after = marking[index]};

        state = verifier.step(*state, symbol);
    }

    return state && verifier.accepts(*state);
}

/**
 * @brief Expects cl100k's verifier to accept the marking cl100k's segmenter gives an ASCII text, read over its classes,
 *        and no other marking of it.
 * @param verifier cl100k's verifier.
 * @param text The text, ASCII, so that its byte offsets count code points.
 */
void expect_only_the_segmentation(const munch::verifier::Verifier& verifier, const std::string_view text)
{
    const auto symbols{class_string(text)};

    Marking_t segmented(symbols.size(), false);

    const Segmenter segmenter{matisse::cl100k::pattern()};

    const auto ends{segmenter.boundaries(text)};

    for (const auto end : ends)
    {
        if (end < symbols.size())
        {
            segmented[end - 1] = true;
        }
    }

    const auto marking_count{1UZ << (symbols.size() - 1)};

    for (std::size_t bits{0}; bits < marking_count; ++bits)
    {
        Marking_t marking(symbols.size(), false);

        for (std::size_t index{0}; index + 1 < symbols.size(); ++index)
        {
            marking[index] = ((bits >> index) & 1U) != 0;
        }

        EXPECT_EQ(accepts(verifier, symbols, marking), marking == segmented) << text;
    }
}

/**
 * @brief Returns every class string over cl100k's class symbols of a length.
 * @param length The length.
 * @return The strings.
 */
std::vector<std::string> windows_of(const std::size_t length)
{
    const auto symbols{matisse::cl100k::pattern().alphabet.symbols};

    std::vector<std::string> windows{{}};

    for (std::size_t width{0}; width < length; ++width)
    {
        std::vector<std::string> longer{};

        for (const auto& window : windows)
        {
            for (const auto symbol : symbols)
            {
                longer.push_back(window + symbol);
            }
        }

        windows = std::move(longer);
    }

    return windows;
}

/**
 * @brief Returns a seeded random text of the shapes the pattern meets: white space runs before letters and punctuation,
 *        newlines followed by tab runs, contractions in both cases, digits, and letters and emoji beyond ASCII.
 * @param random The random sequence.
 * @param fragments The number of fragments joined.
 * @return The text, well-formed UTF-8.
 */
std::string random_text(std::mt19937& random, const std::size_t fragments)
{
    constexpr std::array<std::string_view, 18> shapes{" ",    "  ",       "   a",   "\t\t!", "\t !",    "\n\n\t\t",
                                                      "\r\n", "Hello",    " world", "don't", "DON'T",   "'s",
                                                      "'LL",  " 1234567", "!!",     "héllo", " 日本語", "\U0001F642"};

    std::uniform_int_distribution<std::size_t> shape{0, shapes.size() - 1};

    std::string text{};

    for (std::size_t count{0}; count < fragments; ++count)
    {
        text += shapes[shape(random)];
    }

    return text;
}

/**
 * @brief Returns the number of code points of a UTF-8 text.
 * @param text The text.
 * @return The count.
 */
std::size_t code_points(const std::string_view text)
{
    const auto begins_one{[](const char byte) { return (static_cast<unsigned char>(byte) & 0xC0U) != 0x80U; }};

    return static_cast<std::size_t>(std::ranges::count_if(text, begins_one));
}

} // namespace

TEST(Verifier_test, Cl100k_s_split_verifier_is_the_fixture_s_table_state_for_state)
{
    std::ifstream file{std::string{SOURCE_DIR} + "/libs/pretok/tests/data/split_verifier_reference.txt"};

    ASSERT_TRUE(file.is_open());

    Fixture_check check{};

    for (std::string line{}; std::getline(file, line);)
    {
        expect_fixture_line(check, line);
    }

    EXPECT_EQ(check.counts.universes, 1U);
    EXPECT_EQ(check.counts.states, 1U);
    EXPECT_EQ(check.counts.steps, check.verifier.transitions().size());
    EXPECT_EQ(check.counts.accepting, 1U);
    EXPECT_EQ(check.counts.decisions, check.decided.size());
    EXPECT_EQ(check.decided, (std::set<std::string>{"covered", "functional", "total"}));
}

TEST(Verifier_test, Cl100k_s_split_verifier_accepts_the_segmenter_s_marking_of_a_text_and_no_other)
{
    const auto verifier{split_verifier(matisse::cl100k::pattern())};

    expect_only_the_segmentation(verifier, "a \n\t b\r\n");
    expect_only_the_segmentation(verifier, "don't DON'T 1234");
}

TEST(Verifier_test, Every_window_cl100k_s_factorization_lexer_certifies_is_policy_certified_at_its_origin)
{
    const auto lexer{factorization_lexer(matisse::cl100k::pattern())};

    const auto verifier{split_verifier(matisse::cl100k::pattern())};

    std::size_t certified{0};

    for (const std::size_t width : {2U, 3U})
    {
        for (const auto& window : windows_of(width))
        {
            const auto origin{lexer.is_split_window(window)};

            if (!origin)
            {
                continue;
            }

            EXPECT_FALSE(munch::verifier::miscovering(verifier, window, *origin).has_value()) << window;

            ++certified;
        }
    }

    EXPECT_GT(certified, 0U);
}

TEST(Verifier_test, Cl100k_s_split_verifier_accepts_the_segmenter_s_marking_of_seeded_random_texts)
{
    const auto verifier{split_verifier(matisse::cl100k::pattern())};

    const Segmenter segmenter{matisse::cl100k::pattern()};

    std::mt19937 random{29};

    for (std::size_t count{0}; count < 300; ++count)
    {
        const auto text{random_text(random, 40)};

        const auto symbols{class_string(text)};

        Marking_t marking(symbols.size(), false);

        for (const auto end : segmenter.boundaries(text))
        {
            const auto points{code_points(text.substr(0, end))};

            if (points < symbols.size())
            {
                marking[points - 1] = true;
            }
        }

        EXPECT_TRUE(accepts(verifier, symbols, marking)) << text;
    }
}
