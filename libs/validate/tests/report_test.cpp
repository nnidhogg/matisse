#include "matisse/validate/report.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "matisse/pretok/segmenter.hpp"

using namespace matisse::validate;
using matisse::pretok::Alphabet;
using matisse::pretok::Alternative;
using matisse::pretok::Pattern;

namespace
{
/**
 * @brief The piece starts of a segmentation of a text, ascending, the first zero.
 */
using Starts_t = std::vector<std::size_t>;

/**
 * @brief Texts, each with the piece starts of its segmentations.
 */
using Segmented_t = std::vector<std::pair<std::string, std::vector<Starts_t>>>;

/**
 * @brief The longest class string the brute force enumerates.
 */
constexpr std::size_t longest_text{8};

/**
 * @brief An alphabet of two classes, `a` a literal and `b` everything else.
 */
constexpr Alphabet two_symbols{.symbols = "ab", .literals = "a", .caseless = false, .sets = {}, .other = 'b'};

/**
 * @brief A pattern whose first alternative passes only before another `a`, as `\s+(?!\S)` passes only before white
 *        space, so that a run of `a` before a `b` leaves its last `a` to the second alternative.
 */
constexpr std::array<Alternative, 3> gated_alternatives{{
        {.language = "a+", .gate = "a"},
        {.language = "a+", .gate = "ab"},
        {.language = "b", .gate = "ab"},
}};

/**
 * @brief The gated pattern over the two classes.
 */
constexpr Pattern gated{.alphabet = two_symbols, .alternatives = gated_alternatives};

/**
 * @brief A pattern whose first alternative passes only at the end of the input, as `\s++$` does, and whose earlier
 *        alternative takes a word a later one would extend, so that the order of the alternatives decides the pieces.
 */
constexpr std::array<Alternative, 4> anchored_alternatives{{
        {.language = "a+", .gate = ""},
        {.language = "ab", .gate = "ab"},
        {.language = "a", .gate = "ab"},
        {.language = "b+", .gate = "ab"},
}};

/**
 * @brief The anchored pattern over the two classes.
 */
constexpr Pattern anchored{.alphabet = two_symbols, .alternatives = anchored_alternatives};

/**
 * @brief A pattern whose every piece is one symbol.
 */
constexpr std::array<Alternative, 2> single_alternatives{{
        {.language = "a", .gate = "ab"},
        {.language = "b", .gate = "ab"},
}};

/**
 * @brief The single-symbol pattern over the two classes.
 */
constexpr Pattern single{.alphabet = two_symbols, .alternatives = single_alternatives};

/**
 * @brief A pattern whose one piece is the whole text, so that no cut is ever certified.
 */
constexpr std::array<Alternative, 1> whole_alternatives{{
        {.language = "[ab]+", .gate = "ab"},
}};

/**
 * @brief The whole-text pattern over the two classes.
 */
constexpr Pattern whole{.alphabet = two_symbols, .alternatives = whole_alternatives};

/**
 * @brief Returns whether a class string is a word of the gated pattern's gate-erased union, a run of `a` or one `b`.
 * @param word The class string.
 * @return True when it is a word.
 */
bool gated_word(const std::string_view word)
{
    const auto run{!word.empty() && word.find_first_not_of('a') == std::string_view::npos};

    return run || word == "b";
}

/**
 * @brief Returns whether a class string is a word of the anchored pattern's gate-erased union, a run of `a`, `ab` or a
 *        run of `b`.
 * @param word The class string.
 * @return True when it is a word.
 */
bool anchored_word(const std::string_view word)
{
    const auto a_run{!word.empty() && word.find_first_not_of('a') == std::string_view::npos};

    const auto b_run{!word.empty() && word.find_first_not_of('b') == std::string_view::npos};

    return a_run || b_run || word == "ab";
}

/**
 * @brief Returns every string over the two classes of lengths one to a length, shorter first.
 * @param longest The longest length.
 * @return The strings.
 */
std::vector<std::string> texts_to(const std::size_t longest)
{
    std::vector<std::string> all{{}};

    for (std::size_t from{0}; from < all.size(); ++from)
    {
        if (all[from].size() == longest)
        {
            continue;
        }

        for (const auto symbol : two_symbols.symbols)
        {
            all.push_back(all[from] + symbol);
        }
    }

    all.erase(all.begin());

    return all;
}

/**
 * @brief Returns the piece starts of a segmenter's segmentation of a text.
 * @param segmenter The segmenter.
 * @param text The class string.
 * @return The starts.
 */
Starts_t segmented_starts(const matisse::pretok::Segmenter& segmenter, const std::string_view text)
{
    Starts_t starts{0};

    for (const auto end : segmenter.boundaries(text))
    {
        if (end < text.size())
        {
            starts.push_back(end);
        }
    }

    return starts;
}

/**
 * @brief Returns the piece starts of every complete factorization of a text into words.
 * @tparam Words The word predicate's type.
 * @param words The word predicate, whether a class string is a word.
 * @param text The class string.
 * @return The starts of each factorization.
 */
template <typename Words>
std::vector<Starts_t> factorizations(const Words& words, const std::string_view text)
{
    if (text.empty())
    {
        return {{}};
    }

    std::vector<Starts_t> all{};

    for (std::size_t length{1}; length <= text.size(); ++length)
    {
        if (!words(text.substr(0, length)))
        {
            continue;
        }

        for (auto rest : factorizations(words, text.substr(length)))
        {
            Starts_t starts{0};

            for (const auto start : rest)
            {
                starts.push_back(start + length);
            }

            all.push_back(std::move(starts));
        }
    }

    return all;
}

/**
 * @brief Returns the window certificates brute force finds over segmented texts: for every window of width two to
 *        four that occurs, the origin at which the piece holding its last symbol begins at every occurrence, or none.
 * @param segmentations The texts, each with the piece starts of its segmentations.
 * @return The certified origin of every window that occurs, std::nullopt for a refused one.
 */
std::map<std::string, std::optional<std::size_t>> brute_windows(const Segmented_t& segmentations)
{
    std::map<std::string, std::set<std::optional<std::size_t>>> seen{};

    for (const auto& [text, all] : segmentations)
    {
        for (const auto& starts : all)
        {
            for (std::size_t width{2}; width <= default_width; ++width)
            {
                for (std::size_t at{0}; at + width <= text.size(); ++at)
                {
                    const auto last{at + width - 1};

                    const auto after{std::ranges::upper_bound(starts, last)};

                    const auto covering{*std::ranges::prev(after)};

                    const auto origin{covering < at ? std::nullopt : std::optional{covering - at}};

                    const auto window{text.substr(at, width)};

                    seen[window].insert(origin);
                }
            }
        }
    }

    std::map<std::string, std::optional<std::size_t>> windows{};

    for (const auto& [window, origins] : seen)
    {
        windows[window] = origins.size() == 1 ? *origins.begin() : std::nullopt;
    }

    return windows;
}

/**
 * @brief Expects a window table to hold exactly the windows brute force certifies, at their origins.
 * @param table The window table.
 * @param brute The brute force's certified origin of every window, as brute_windows() gives it.
 */
void expect_brute_table(const Window_table& table, const std::map<std::string, std::optional<std::size_t>>& brute)
{
    std::map<std::string, std::size_t> certified{};

    for (const auto& [window, origin] : table.windows)
    {
        certified.emplace(window, origin);
    }

    std::size_t brute_certified{0};

    for (const auto& [window, origin] : brute)
    {
        const auto found{certified.find(window)};

        const auto decided{found == certified.end() ? std::nullopt : std::optional{found->second}};

        EXPECT_EQ(decided, origin) << window;

        brute_certified += origin ? 1 : 0;
    }

    EXPECT_EQ(table.windows.size(), brute_certified);
}

/**
 * @brief Expects a pattern's report to certify exactly what brute force over every text up to longest_text finds,
 *        under the policy certificate by the segmenter's pieces and under the factorization certificate by every
 *        complete factorization into the words.
 * @tparam Words The word predicate's type.
 * @param pattern The pattern.
 * @param words The word predicate of its gate-erased union, whether a class string is a word.
 */
template <typename Words>
void expect_brute_force(const Pattern& pattern, const Words& words)
{
    const auto report{validate(pattern)};

    const matisse::pretok::Segmenter segmenter{pattern};

    Segmented_t segmented{};

    Segmented_t factored{};

    for (const auto& text : texts_to(longest_text))
    {
        auto starts{segmented_starts(segmenter, text)};

        auto all{factorizations(words, text)};

        segmented.emplace_back(text, std::vector<Starts_t>{std::move(starts)});
        factored.emplace_back(text, std::move(all));
    }

    const auto policy_windows{brute_windows(segmented)};

    const auto factorization_windows{brute_windows(factored)};

    expect_brute_table(report.policy, policy_windows);
    expect_brute_table(report.factorization, factorization_windows);
}

/**
 * @brief Returns the ends of the pieces a segmenter gives a class string, up to an offset.
 * @param segmenter The segmenter.
 * @param text The class string, read as a whole input.
 * @param below The offset.
 * @return The piece ends at or below the offset, ascending.
 */
std::vector<std::size_t> ends_up_to(
        const matisse::pretok::Segmenter& segmenter, const std::string_view text, const std::size_t below)
{
    auto ends{segmenter.boundaries(text)};

    std::erase_if(ends, [&](const std::size_t end) { return end > below; });

    return ends;
}

/**
 * @brief Returns the right contexts that fail brute force over every text up to longest_text at a cut a table's
 *        windows certify, with the length of the shortest text each fails on: the segmenter's pieces up to the cut of
 *        the text cut short at the cut plus the context, read as a whole input, differ from the whole text's.
 * @param table The window table.
 * @param pattern The pattern.
 * @return The length of the shortest failing text, by context.
 */
std::map<std::size_t, std::size_t> brute_failing(const Window_table& table, const Pattern& pattern)
{
    const matisse::pretok::Segmenter segmenter{pattern};

    std::map<std::size_t, std::size_t> failing{};

    for (const auto& text : texts_to(longest_text))
    {
        for (const auto& [window, origin] : table.windows)
        {
            for (auto at{text.find(window)}; at != std::string::npos; at = text.find(window, at + 1))
            {
                const auto cut{at + origin};

                const auto whole{ends_up_to(segmenter, text, cut)};

                for (std::size_t context{0}; cut > 0 && cut + context < text.size(); ++context)
                {
                    const auto shortened{std::string_view{text}.substr(0, cut + context)};

                    const auto truncated{ends_up_to(segmenter, shortened, cut)};

                    if (truncated != whole)
                    {
                        failing.try_emplace(context, text.size());
                    }
                }
            }
        }
    }

    return failing;
}

/**
 * @brief Expects a window table's right context and failing contexts to be those brute force finds, each truncation
 *        as long as the shortest text brute force finds failing at its context, reaching past its cut by more than its
 *        context and failing there: the segmenter's pieces up to the cut of the truncation cut short at the cut plus
 *        the context differ from the whole truncation's.
 * @param table The window table.
 * @param pattern The pattern.
 */
void expect_brute_context(const Window_table& table, const Pattern& pattern)
{
    const auto brute{brute_failing(table, pattern)};

    const matisse::pretok::Segmenter segmenter{pattern};

    std::map<std::size_t, std::size_t> decided{};

    for (const auto& [segmentation, occurrence, cut, context] : table.failing)
    {
        const auto& bytes{segmentation.bytes};

        decided.emplace(context, bytes.size());

        ASSERT_LT(cut + context, bytes.size()) << bytes;

        const auto whole{ends_up_to(segmenter, bytes, cut)};

        const auto shortened{std::string_view{bytes}.substr(0, cut + context)};

        const auto truncated{ends_up_to(segmenter, shortened, cut)};

        EXPECT_NE(whole, truncated) << bytes;
    }

    const auto expected{brute.empty() ? 0 : brute.rbegin()->first + 1};

    EXPECT_EQ(decided, brute);
    EXPECT_EQ(table.right_context, expected);
}

} // namespace

TEST(Report_test, The_tables_certify_what_brute_force_over_the_segmenter_and_the_factorizations_finds)
{
    expect_brute_force(gated, gated_word);
    expect_brute_force(anchored, anchored_word);
}

TEST(Report_test, The_right_contexts_are_one_more_than_the_largest_context_brute_force_finds_failing)
{
    const auto gated_report{validate(gated)};

    expect_brute_context(gated_report.factorization, gated);
    expect_brute_context(gated_report.policy, gated);

    const auto anchored_report{validate(anchored)};

    expect_brute_context(anchored_report.factorization, anchored);
    expect_brute_context(anchored_report.policy, anchored);
}

TEST(Report_test, A_pattern_of_one_symbol_pieces_certifies_every_window_at_its_last_symbol_with_no_right_context)
{
    const auto [factorization, policy, gap]{validate(single)};

    EXPECT_EQ(factorization.windows.size(), 4U + 8U + 16U);
    EXPECT_EQ(policy.windows.size(), 4U + 8U + 16U);

    for (const auto& table : {factorization, policy})
    {
        for (const auto& [window, origin] : table.windows)
        {
            EXPECT_EQ(origin, window.size() - 1) << window;
        }

        EXPECT_EQ(table.right_context, 0U);
    }

    ASSERT_TRUE(std::holds_alternative<std::size_t>(gap));

    EXPECT_EQ(std::get<std::size_t>(gap), 1U);
}

TEST(Report_test, The_tables_carry_their_inventory_and_width)
{
    const auto report{validate(gated, 3)};

    EXPECT_EQ(report.factorization.inventory, Inventory::factorization);
    EXPECT_EQ(report.policy.inventory, Inventory::policy);
    EXPECT_EQ(report.factorization.width, 3U);
    EXPECT_EQ(report.policy.width, 3U);

    const auto widest{std::ranges::max(
            report.policy.windows, {}, [](const Certified_window& certified) { return certified.window.size(); })};

    EXPECT_EQ(widest.window.size(), 3U);

    // A run of a has no bound.
    EXPECT_TRUE(std::holds_alternative<munch::verifier::Lasso>(report.gap));
}

TEST(Report_test, The_text_reads_as_the_verdict_then_each_inventory_then_the_gap)
{
    // Every window of the single-symbol pattern is certified at its last symbol with no right context, and every
    // boundary is at most one symbol from the next.
    EXPECT_EQ(
            render(validate(single, 2)),
            "verdict                     windows certify a cut: 4 under the factorization certificate and 4 under the "
            "policy certificate, up to width 2\n"
            "factorization windows       4: 4 at width 2\n"
            "                            \"aa\" at 1\n"
            "                            \"ab\" at 1\n"
            "                            \"ba\" at 1\n"
            "                            \"bb\" at 1\n"
            "factorization context       0 code points read past the cut\n"
            "policy windows              4: 4 at width 2\n"
            "                            \"aa\" at 1\n"
            "                            \"ab\" at 1\n"
            "                            \"ba\" at 1\n"
            "                            \"bb\" at 1\n"
            "policy context              0 code points read past the cut\n"
            "boundary gap                1 code point\n");

    // The gated pattern leaves the last a of a run before a b to the second alternative, so "ab" is certified at the b
    // and "aa" nowhere; the cut in "aab" needs the b read past it, since "aa" alone is one piece, and a run of a
    // carries no boundary however long. Six examples are shown and the rest counted.
    EXPECT_EQ(
            render(validate(gated, 3)),
            "verdict                     windows certify a cut: 9 under the factorization certificate and 9 under the "
            "policy certificate, up to width 3\n"
            "factorization windows       9: 3 at width 2, 6 at width 3\n"
            "                            \"ab\" at 1\n"
            "                            \"ba\" at 1\n"
            "                            \"bb\" at 1\n"
            "                            \"aab\" at 2\n"
            "                            \"aba\" at 2\n"
            "                            \"abb\" at 2\n"
            "                            ... 3 more\n"
            "factorization context       1 code point read past the cut\n"
            "                            0 fails at cut 2 of \"aab\", boundaries 1 2\n"
            "policy windows              9: 3 at width 2, 6 at width 3\n"
            "                            \"ab\" at 1\n"
            "                            \"ba\" at 1\n"
            "                            \"bb\" at 1\n"
            "                            \"aab\" at 2\n"
            "                            \"aba\" at 2\n"
            "                            \"abb\" at 2\n"
            "                            ... 3 more\n"
            "policy context              1 code point read past the cut\n"
            "                            0 fails at cut 2 of \"aab\", boundaries 1 2\n"
            "boundary gap                unbounded: \"a\" repeats with no boundary\n");
}

TEST(Report_test, The_json_document_holds_every_figure_under_its_name)
{
    EXPECT_EQ(
            json(validate(single, 2)),
            R"({
  "verdict": "windows certify a cut: 4 under the factorization certificate and 4 under the policy certificate, up to width 2",
  "factorization": {"width": 2, "windows": [{"window": "aa", "origin": 1}, {"window": "ab", "origin": 1}, {"window": "ba", "origin": 1}, {"window": "bb", "origin": 1}], "right_context": 0, "failing": []},
  "policy": {"width": 2, "windows": [{"window": "aa", "origin": 1}, {"window": "ab", "origin": 1}, {"window": "ba", "origin": 1}, {"window": "bb", "origin": 1}], "right_context": 0, "failing": []},
  "boundary_gap": 1,
  "gap_lasso": null
})");

    EXPECT_EQ(
            json(validate(gated, 3)),
            R"({
  "verdict": "windows certify a cut: 9 under the factorization certificate and 9 under the policy certificate, up to width 3",
  "factorization": {"width": 3, "windows": [{"window": "ab", "origin": 1}, {"window": "ba", "origin": 1}, {"window": "bb", "origin": 1}, {"window": "aab", "origin": 2}, {"window": "aba", "origin": 2}, {"window": "abb", "origin": 2}, {"window": "bab", "origin": 2}, {"window": "bba", "origin": 2}, {"window": "bbb", "origin": 2}], "right_context": 1, "failing": [{"segmentation": {"text": "aab", "boundaries": [1, 2]}, "occurrence": 1, "cut": 2, "context": 0}]},
  "policy": {"width": 3, "windows": [{"window": "ab", "origin": 1}, {"window": "ba", "origin": 1}, {"window": "bb", "origin": 1}, {"window": "aab", "origin": 2}, {"window": "aba", "origin": 2}, {"window": "abb", "origin": 2}, {"window": "bab", "origin": 2}, {"window": "bba", "origin": 2}, {"window": "bbb", "origin": 2}], "right_context": 1, "failing": [{"segmentation": {"text": "aab", "boundaries": [1, 2]}, "occurrence": 1, "cut": 2, "context": 0}]},
  "boundary_gap": "unbounded",
  "gap_lasso": {"stem": {"text": "a", "boundaries": []}, "loop": {"text": "a", "boundaries": []}, "suffix": {"text": "", "boundaries": []}}
})");
}

TEST(Report_test, A_pattern_that_certifies_no_window_says_so_in_both_forms)
{
    const auto report{validate(whole)};

    const auto text{render(report)};

    EXPECT_TRUE(text.starts_with(
            "verdict                     no window up to width 4 certifies a cut under either certificate\n"
            "factorization windows       none\n"
            "factorization context       0 code points read past the cut\n"
            "policy windows              none\n"))
            << text;

    const auto document{json(report)};

    EXPECT_TRUE(document.contains(R"("verdict": "no window up to width 4 certifies a cut under either certificate")"))
            << document;
    EXPECT_TRUE(document.contains(R"("policy": {"width": 4, "windows": [], "right_context": 0, "failing": []})"))
            << document;
}

TEST(Report_test, A_json_string_escapes_the_quote_the_backslash_and_the_controls_and_passes_utf8_through)
{
    EXPECT_EQ(json_string(R"(say "hi"\)"), R"("say \"hi\"\\")");
    EXPECT_EQ(json_string("tab\tnewline\n"), R"("tab\u0009newline\u000a")");
    EXPECT_EQ(json_string("h\xC3\xA9llo"), "\"h\xC3\xA9llo\"");

    // A byte that begins no well-formed sequence is escaped as the code point of its value.
    EXPECT_EQ(json_string("a\xFF"), R"("a\u00ff")");
}
