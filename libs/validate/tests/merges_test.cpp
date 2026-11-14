#include "matisse/validate/merges.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <map>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <munch/verifier/merge_tower.hpp>
#include <munch/verifier/verifier_decisions.hpp>

using namespace matisse::validate;
using matisse::load::Definition;
using matisse::load::Merge_rule;
using munch::verifier::Merge;

namespace
{
/**
 * @brief The bytes the tables merge and the windows are drawn from; every other byte is a token alone that no merge
 *        reads, so a string that refutes a window over them is spelled by them alone.
 */
constexpr std::string_view alphabet{"abc"};

/**
 * @brief The widest window decided.
 */
constexpr std::size_t widest_window{3};

/**
 * @brief The longest product the seeded tables make.
 */
constexpr std::size_t longest_product{3};

/**
 * @brief How many seeded tables the brute force holds the decisions to.
 */
constexpr std::size_t seeded_tables{300};

/**
 * @brief How many of the seeded tables munch's merge tower decides as well, each of its decisions searching a product
 *        over every byte.
 */
constexpr std::size_t tower_tables{40};

/**
 * @brief The tables written out: none, one merge, a merge that extends its own product, a self-pair, two overlapping
 *        merges and their join, and a table whose products overlap both ways.
 */
const std::vector<std::vector<Merge>> written_tables{
        {},
        {{.left = "a", .right = "b"}},
        {{.left = "a", .right = "b"}, {.left = "ab", .right = "a"}},
        {{.left = "a", .right = "a"}, {.left = "aa", .right = "a"}},
        {{.left = "a", .right = "b"}, {.left = "b", .right = "c"}, {.left = "ab", .right = "c"}},
        {{.left = "b", .right = "a"},
         {.left = "a", .right = "b"},
         {.left = "ab", .right = "a"},
         {.left = "ba", .right = "ba"}},
};

/**
 * @brief Returns the definition of a merge table under the pair rule without the whole-piece lookup: the 256 bytes and
 *        then each merge's product as the ordinary tokens.
 * @param merges The merge table in rank order.
 * @return The definition.
 */
Definition definition_of(const std::vector<Merge>& merges)
{
    std::vector<std::string> spellings{};

    for (unsigned byte{0}; byte < 256; ++byte)
    {
        spellings.emplace_back(1, static_cast<char>(byte));
    }

    for (const auto& [left, right] : merges)
    {
        spellings.push_back(left + right);
    }

    return Definition{
            .merge_rule = Merge_rule::pairs,
            .whole_piece_lookup = false,
            .spellings = std::move(spellings),
            .merges = merges,
            .specials = {}};
}

/**
 * @brief Returns the encoding of a text under a table by its merges in rank order, each applied to every occurrence
 *        left to right before the next, which under distinct products is the merge of least rank at its leftmost
 *        occurrence, one at a time.
 * @param merges The merge table in rank order.
 * @param text The text.
 * @return The tokens.
 */
std::vector<std::string> encoded(const std::vector<Merge>& merges, const std::string_view text)
{
    std::vector<std::string> tokens{};

    for (const auto byte : text)
    {
        tokens.emplace_back(1, byte);
    }

    for (const auto& [left, right] : merges)
    {
        std::vector<std::string> merged{};

        for (const auto& token : tokens)
        {
            if (!merged.empty() && merged.back() == left && token == right)
            {
                merged.back() += token;

                continue;
            }

            merged.push_back(token);
        }

        tokens = std::move(merged);
    }

    return tokens;
}

/**
 * @brief Returns where the token covering a byte of a text begins in its encoding.
 * @param tokens The text's encoding.
 * @param byte The byte, an index of the text.
 * @return The index of the covering token's first byte.
 */
std::size_t covering_start(const std::vector<std::string>& tokens, const std::size_t byte)
{
    std::size_t start{0};

    for (const auto& token : tokens)
    {
        if (byte < start + token.size())
        {
            break;
        }

        start += token.size();
    }

    return start;
}

/**
 * @brief Returns every window over the alphabet of width one to the widest, narrower first.
 * @return The windows.
 */
std::vector<std::string> every_window()
{
    std::vector<std::string> windows{""};

    std::vector<std::string> all{};

    for (std::size_t width{1}; width <= widest_window; ++width)
    {
        std::vector<std::string> longer{};

        for (const auto& window : windows)
        {
            for (const auto byte : alphabet)
            {
                longer.push_back(window + byte);
            }
        }

        all.insert(all.end(), longer.begin(), longer.end());

        windows = std::move(longer);
    }

    return all;
}

/**
 * @brief Returns every string over the alphabet of one to a length of bytes.
 * @param longest The longest length.
 * @return The strings, shorter first.
 */
std::vector<std::string> every_string(const std::size_t longest)
{
    std::vector<std::string> strings{""};

    std::vector<std::string> all{};

    for (std::size_t length{1}; length <= longest; ++length)
    {
        std::vector<std::string> longer{};

        for (const auto& text : strings)
        {
            for (const auto byte : alphabet)
            {
                longer.push_back(text + byte);
            }
        }

        all.insert(all.end(), longer.begin(), longer.end());

        strings = std::move(longer);
    }

    return all;
}

/**
 * @brief Returns, for every window over the alphabet, where the token covering its last byte begins relative to the
 *        window's first byte, at every occurrence in every string up to a length, as a scan of the strings' encodings
 *        finds them.
 * @param merges The merge table in rank order.
 * @param longest The longest string scanned.
 * @return The relative starts seen of each window, a start before the window negative.
 */
std::map<std::string, std::set<std::ptrdiff_t>> scanned_starts(
        const std::vector<Merge>& merges, const std::size_t longest)
{
    std::map<std::string, std::set<std::ptrdiff_t>> starts{};

    for (const auto& text : every_string(longest))
    {
        const auto tokens{encoded(merges, text)};

        for (std::size_t width{1}; width <= widest_window; ++width)
        {
            for (std::size_t at{0}; at + width <= text.size(); ++at)
            {
                const auto start{covering_start(tokens, at + width - 1)};

                const auto relative{static_cast<std::ptrdiff_t>(start) - static_cast<std::ptrdiff_t>(at)};

                starts[text.substr(at, width)].insert(relative);
            }
        }
    }

    return starts;
}

/**
 * @brief Returns the longest product of a table, one when it has none.
 * @param merges The merge table.
 * @return The length.
 */
std::size_t longest_token(const std::vector<Merge>& merges)
{
    std::size_t longest{1};

    for (const auto& [left, right] : merges)
    {
        longest = std::max(longest, left.size() + right.size());
    }

    return longest;
}

/**
 * @brief Returns seeded random tables over the alphabet, each merge joining two tokens made so far into a product no
 *        longer than longest_product that no byte or earlier merge carries.
 * @param count How many tables.
 * @return The tables, the first of a larger count being the same.
 */
std::vector<std::vector<Merge>> seeded_random_tables(const std::size_t count)
{
    std::mt19937 random{20261111};

    std::uniform_int_distribution<std::size_t> merge_count{1, 6};

    std::vector<std::vector<Merge>> tables{};

    while (tables.size() < count)
    {
        std::vector<std::string> tokens{"a", "b", "c"};

        std::vector<Merge> merges{};

        const auto wanted{merge_count(random)};

        // A draw whose product is too long or already a token is skipped, so a table may hold fewer merges.
        for (std::size_t draw{0}; draw < 4 * wanted && merges.size() < wanted; ++draw)
        {
            std::uniform_int_distribution<std::size_t> pick{0, tokens.size() - 1};

            const auto left{tokens[pick(random)]};

            const auto right{tokens[pick(random)]};

            const auto product{left + right};

            if (product.size() > longest_product || std::ranges::contains(tokens, product))
            {
                continue;
            }

            merges.push_back(Merge{.left = left, .right = right});

            tokens.push_back(product);
        }

        tables.push_back(std::move(merges));
    }

    return tables;
}

/**
 * @brief Returns the written tables and seeded ones.
 * @param seeded_count How many seeded tables.
 * @return The tables.
 */
std::vector<std::vector<Merge>> tables_of(const std::size_t seeded_count)
{
    auto tables{written_tables};

    const auto seeded{seeded_random_tables(seeded_count)};

    tables.insert(tables.end(), seeded.begin(), seeded.end());

    return tables;
}

/**
 * @brief Expects every window over the alphabet to be decided as a scan of every string of at most 2L + 1 bytes finds
 *        it, L the longest token: certified exactly when the token covering its last byte begins at the origin at every
 *        occurrence there, the bound past which a refutation needs no longer string.
 * @param merges The merge table in rank order.
 */
void expect_scanned(const std::vector<Merge>& merges)
{
    const auto windows{every_window()};

    const auto decided{validate_merges(definition_of(merges), windows)};

    const auto starts{scanned_starts(merges, 2 * longest_token(merges) + widest_window - 2)};

    for (const auto& [window, origin, refutation] : decided)
    {
        const auto& seen{starts.at(window)};

        const auto certified{seen == std::set{static_cast<std::ptrdiff_t>(origin)}};

        EXPECT_EQ(!refutation, certified) << window << " at " << origin;

        // The window occurs in itself, where its own encoding begins the token covering its last byte at the origin.
        const auto own{encoded(merges, window)};

        EXPECT_EQ(origin, covering_start(own, window.size() - 1)) << window;
    }
}

/**
 * @brief Expects every refutation of every window over the alphabet to be the encoder's segmentation of its text, with
 *        the window at its occurrence and the token covering the window's last byte beginning elsewhere than the
 *        origin, within the bound of 2L + |window| - 2 bytes.
 * @param merges The merge table in rank order.
 */
void expect_refutations_replay(const std::vector<Merge>& merges)
{
    for (const auto& [window, origin, refutation] : validate_merges(definition_of(merges), every_window()))
    {
        if (!refutation)
        {
            continue;
        }

        const auto& [segmentation, occurrence]{*refutation};

        const auto& text{segmentation.bytes};

        const auto tokens{encoded(merges, text)};

        std::vector<bool> boundaries(text.size(), false);

        std::size_t end{0};

        // A boundary follows every token but the last, the string's end carrying none.
        for (const auto& token : tokens | std::views::take(tokens.size() - 1))
        {
            end += token.size();

            boundaries[end - 1] = true;
        }

        EXPECT_EQ(segmentation.boundaries, boundaries) << window << " in " << text;
        EXPECT_LE(text.size(), 2 * longest_token(merges) + window.size() - 2) << window << " in " << text;
        ASSERT_LE(occurrence + window.size(), text.size()) << window << " in " << text;
        EXPECT_EQ(text.substr(occurrence, window.size()), window) << window << " in " << text;
        EXPECT_NE(covering_start(tokens, occurrence + window.size() - 1), occurrence + origin)
                << window << " in " << text;
    }
}

} // namespace

TEST(Merges_test, Every_window_is_decided_as_a_scan_of_every_string_up_to_the_refutation_bound_decides_it)
{
    for (const auto& merges : tables_of(seeded_tables))
    {
        expect_scanned(merges);
    }
}

TEST(Merges_test, Every_window_is_decided_as_munch_s_exhaustive_decision_on_the_merge_tower_decides_it)
{
    for (const auto& merges : tables_of(tower_tables))
    {
        const auto tower{munch::verifier::merge_tower(merges)};

        for (const auto& [window, origin, refutation] : validate_merges(definition_of(merges), every_window()))
        {
            const auto exhaustive{munch::verifier::miscovering(tower, window, origin)};

            EXPECT_EQ(refutation.has_value(), exhaustive.has_value()) << window << " at " << origin;
        }
    }
}

TEST(Merges_test, Every_refutation_is_the_encoder_s_segmentation_of_its_text_covering_the_window_elsewhere)
{
    for (const auto& merges : tables_of(seeded_tables))
    {
        expect_refutations_replay(merges);
    }
}

TEST(Merges_test, Each_window_is_decided_where_its_own_encoding_begins_its_last_token_and_in_the_order_given)
{
    // Under (a, b) then (ab, a) aba is one token, so its own encoding begins its last token at 0, ba is b then a, and
    // every ab lies in a token beginning at its a, so ab is certified at 0.
    const std::vector<std::string> candidates{"aba", "ba", "ab", "a"};

    const auto decided{validate_merges(definition_of(written_tables[2]), candidates)};

    ASSERT_EQ(decided.size(), candidates.size());

    for (const auto [at, expected] : std::views::enumerate(candidates))
    {
        EXPECT_EQ(decided[static_cast<std::size_t>(at)].window, expected);
    }

    EXPECT_EQ(decided[0].origin, 0U);
    EXPECT_EQ(decided[1].origin, 1U);
    EXPECT_EQ(decided[2].origin, 0U);
    EXPECT_EQ(decided[3].origin, 0U);
    EXPECT_FALSE(decided[2].refutation.has_value());

    // The empty table cuts between every two bytes, so every window is certified at its last byte.
    for (const auto& [window, origin, refutation] : validate_merges(definition_of({}), every_window()))
    {
        EXPECT_EQ(origin, window.size() - 1) << window;
        EXPECT_FALSE(refutation.has_value()) << window;
    }
}

TEST(Merges_test, A_table_that_recreates_a_token_is_refused_as_munch_s_token_automaton_refuses_it)
{
    // ccca is one token under the encoder, and ccc then a under the merges applied in rank order, each to the end.
    const std::vector<Merge> merges{
            {.left = "c", .right = "c"},
            {.left = "c", .right = "cc"},
            {.left = "ccc", .right = "a"},
            {.left = "cc", .right = "c"}};

    const std::vector<std::string> candidates{"c"};

    EXPECT_THROW(
            {
                try
                {
                    std::ignore = validate_merges(definition_of(merges), candidates);
                }
                catch (const std::invalid_argument& error)
                {
                    EXPECT_STREQ(
                            error.what(),
                            R"(token_automaton: the merge ("cc", "c") at rank 3 recreates the token "ccc", which the )"
                            "alphabet or an earlier merge already carries");

                    throw;
                }
            },
            std::invalid_argument);
}

TEST(Merges_test, A_rank_rule_a_whole_piece_lookup_and_an_empty_candidate_are_refused)
{
    const std::vector<std::string> candidates{"ab", ""};

    const auto refusal{[](const Definition& definition, const std::vector<std::string>& windows) -> std::string {
        try
        {
            std::ignore = validate_merges(definition, windows);
        }
        catch (const std::invalid_argument& error)
        {
            return error.what();
        }

        return "";
    }};

    auto ranks{definition_of({})};

    ranks.merge_rule = Merge_rule::ranks;

    auto lookup{definition_of(written_tables[1])};

    lookup.whole_piece_lookup = true;

    EXPECT_EQ(
            refusal(ranks, {"ab"}),
            "validate_merges: the definition merges by the rank rule, which carries no merge table");
    EXPECT_EQ(
            refusal(lookup, {"ab"}),
            "validate_merges: the definition looks a whole piece up, which with no pre-tokenizer takes a whole text "
            "that is a token as that token");
    EXPECT_EQ(refusal(definition_of(written_tables[1]), candidates), "validate_merges: candidate 1 is empty");
    EXPECT_TRUE(validate_merges(definition_of(written_tables[1]), {}).empty());
}

TEST(Merges_test, The_text_reads_as_the_verdict_then_the_certified_and_the_refuted_windows)
{
    // Under (a, b) then (ab, a) every ab lies in a token beginning at its a, a b begins a token only when no a precedes
    // it, and an a after ab joins it into aba, one token, unless a b follows the a and takes it first.
    const std::vector<std::string> candidates{"ab", "b", "\n", "ba", "a", "x", "aa", "bb", "c"};

    const auto decided{validate_merges(definition_of(written_tables[2]), candidates)};

    const auto text{render_merges(decided)};

    EXPECT_EQ(
            text,
            "verdict                     candidate windows certify a cut: 6 of 9\n"
            "certified windows           6: 3 at width 1, 3 at width 2\n"
            "                            \"ab\" at 0\n"
            "                            \"x\" at 0\n"
            "                            \"aa\" at 1\n"
            "                            \"bb\" at 1\n"
            "                            \"c\" at 0\n"
            "                            \"\\n\" at 0\n"
            "refuted windows             3: 2 at width 1, 1 at width 2\n"
            "                            \"b\" at 0 fails at 1 of \"ab\", one token\n"
            "                            \"ba\" at 1 fails at 1 of \"aba\", one token\n"
            "                            \"a\" at 0 fails at 2 of \"aba\", one token\n");

    // Under (a, a) then (aa, a) a run of a is paired from its left, so no window of a's certifies a cut, and the
    // refutation of aaa is aaaa, cut between its two halves.
    const std::vector<std::string> runs{"aa", "a", "aaa"};

    const auto runs_decided{validate_merges(definition_of(written_tables[3]), runs)};

    EXPECT_EQ(
            render_merges(runs_decided),
            "verdict                     no window of the 3 candidates certifies a cut\n"
            "certified windows           none\n"
            "refuted windows             3: 1 at width 1, 1 at width 2, 1 at width 3\n"
            "                            \"aa\" at 0 fails at 1 of \"aaa\", one token\n"
            "                            \"a\" at 0 fails at 1 of \"aa\", one token\n"
            "                            \"aaa\" at 0 fails at 0 of \"aaaa\", boundaries 2\n");
}

TEST(Merges_test, The_json_document_holds_each_window_in_the_order_decided_with_its_refutation_or_null)
{
    const std::vector<std::string> candidates{"ab", "b", "\n"};

    const auto decided{validate_merges(definition_of(written_tables[2]), candidates)};

    EXPECT_EQ(
            merges_json(decided),
            R"({
  "verdict": "candidate windows certify a cut: 2 of 3",
  "windows": [{"window": "ab", "origin": 0, "refutation": null}, {"window": "b", "origin": 0, "refutation": {"segmentation": {"text": "ab", "boundaries": []}, "occurrence": 1}}, {"window": "\u000a", "origin": 0, "refutation": null}]
})");
}
