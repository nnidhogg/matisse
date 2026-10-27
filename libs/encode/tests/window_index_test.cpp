#include "matisse/encode/window_index.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <munch/regex/regex.hpp>
#include <munch/regex/set.hpp>
#include <munch/regex/utf8.hpp>

#include "matisse/pretok/classes.hpp"
#include "matisse/validate/report.hpp"

using namespace matisse::encode;
using matisse::pretok::Alphabet;
using matisse::pretok::Alternative;
using matisse::pretok::Class_table;
using matisse::pretok::Code_point_set;
using matisse::pretok::Pattern;
using matisse::validate::Inventory;
using matisse::validate::Window_table;
using munch::regex::any_of;
using munch::regex::concat;
using munch::regex::optional;
using munch::regex::plus;
using munch::regex::Set;
using munch::regex::text;
using munch::regex::utf8::Code_point_range;

namespace
{
/**
 * @brief The code point of the space.
 */
constexpr std::array<Code_point_range, 1> space_points{{{.first = U' ', .last = U' '}}};

/**
 * @brief The letters of the test pattern: Latin, Latin-1 and Latin Extended, Greek, Cyrillic, kana and CJK ideographs.
 */
constexpr std::array<Code_point_range, 7> letter_points{{
        {.first = U'A', .last = U'Z'},
        {.first = U'a', .last = U'z'},
        {.first = U'À', .last = U'ɏ'},
        {.first = U'Ͱ', .last = U'Ͽ'},
        {.first = U'Ѐ', .last = U'ӿ'},
        {.first = U'぀', .last = U'ヿ'},
        {.first = U'一', .last = U'鿿'},
}};

/**
 * @brief The sets the test alphabet reads, the space and the letters.
 */
const std::vector<Code_point_set> sets{
        {.symbol = ' ', .ranges = {space_points.begin(), space_points.end()}},
        {.symbol = 'a', .ranges = {letter_points.begin(), letter_points.end()}},
};

/**
 * @brief An alphabet of three classes: the space, the letters, and everything else.
 */
const Alphabet alphabet{.symbols = " !a", .sets = sets, .other = '!'};

/**
 * @brief A pattern in GPT-2's shape over the three classes: a letter run or a run of everything else, each after an
 *        optional space, a run of spaces that only a space or the end follows, as GPT-2's `\s+(?!\S)`, and a run of
 *        spaces.
 */
const std::vector<Alternative> alternatives{
        {.language = concat(optional(text(" ")), plus(text("a"))), .gate = " !a"},
        {.language = concat(optional(text(" ")), plus(any_of(Set{'!'}))), .gate = " !a"},
        {.language = plus(text(" ")), .gate = " "},
        {.language = plus(text(" ")), .gate = " !a"},
};

/**
 * @brief The test pattern.
 */
const Pattern pattern{.alphabet = alphabet, .alternatives = alternatives};

/**
 * @brief Returns the window table of an inventory of the test pattern.
 * @param inventory The inventory.
 * @return The table.
 */
Window_table table_of(const Inventory inventory)
{
    const auto report{matisse::validate::validate(pattern)};

    return inventory == Inventory::factorization ? report.factorization : report.policy;
}

/**
 * @brief Returns whether a byte continues a UTF-8 sequence rather than beginning one.
 * @param byte The byte.
 * @return True for a continuation byte.
 */
bool is_continuation(const char byte)
{
    return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

/**
 * @brief Returns every anchor of a text by brute force: the byte offset of t + o for every occurrence at t of every
 *        window of a table certified at o, the text's start left out.
 * @param table The window table.
 * @param text The text, well-formed UTF-8.
 * @return The anchors, ascending, each once.
 */
std::set<std::size_t> anchors_of(const Window_table& table, const std::string_view text)
{
    const auto classes{matisse::pretok::classes_of(alphabet, text)};

    std::string symbols{};

    for (const auto symbol : classes)
    {
        symbols += alphabet.symbols[symbol];
    }

    std::vector<std::size_t> offsets{};

    for (const auto [at, byte] : std::views::enumerate(text))
    {
        if (!is_continuation(byte))
        {
            offsets.push_back(static_cast<std::size_t>(at));
        }
    }

    offsets.push_back(text.size());

    std::set<std::size_t> anchors{};

    for (const auto& [window, origin] : table.windows)
    {
        for (auto at{symbols.find(window)}; at != std::string::npos; at = symbols.find(window, at + 1))
        {
            if (at + origin > 0)
            {
                anchors.insert(offsets[at + origin]);
            }
        }
    }

    return anchors;
}

/**
 * @brief Returns the least of a set of anchors at or after a byte.
 * @param anchors The anchors.
 * @param from The byte.
 * @return The anchor, std::nullopt when none stands at or after the byte.
 */
std::optional<std::size_t> least_at_or_after(const std::set<std::size_t>& anchors, const std::size_t from)
{
    const auto found{anchors.lower_bound(from)};

    if (found == anchors.end())
    {
        return std::nullopt;
    }

    return *found;
}

/**
 * @brief Returns a seeded random text of letter runs, runs of spaces, punctuation and letters beyond ASCII.
 * @param random The random sequence.
 * @param fragments The number of fragments joined.
 * @return The text, well-formed UTF-8.
 */
std::string random_text(std::mt19937& random, const std::size_t fragments)
{
    constexpr std::array<std::string_view, 14> shapes{"a",  "ab",   " ab",   "  ",     "   ",       "!",          "?!",
                                                      " !", "\t\n", "héllo", "日本語", " Ελληνικά", "\U0001F642", "12"};

    std::uniform_int_distribution<std::size_t> shape{0, shapes.size() - 1};

    std::string text{};

    for (std::size_t count{0}; count < fragments; ++count)
    {
        text += shapes[shape(random)];
    }

    return text;
}

/**
 * @brief Expects first_anchor() of a text at every code point, and at its end, to be the least anchor brute force finds
 *        at or after it.
 * @param index The index.
 * @param class_table The class table of the test alphabet.
 * @param table The table the index was built of.
 * @param text The text, well-formed UTF-8.
 */
void expect_brute_force(
        const Window_index& index, const Class_table& class_table, const Window_table& table,
        const std::string_view text)
{
    const auto anchors{anchors_of(table, text)};

    for (std::size_t from{0}; from <= text.size(); ++from)
    {
        if (from < text.size() && is_continuation(text[from]))
        {
            continue;
        }

        EXPECT_EQ(index.first_anchor(class_table, text, from), least_at_or_after(anchors, from))
                << text << " from " << from;
    }
}

} // namespace

TEST(Window_index_test, The_first_anchor_is_the_least_anchor_brute_force_finds_at_or_after_every_byte)
{
    for (const auto inventory : {Inventory::factorization, Inventory::policy})
    {
        const auto table{table_of(inventory)};

        const Window_index index{table};

        const Class_table class_table{alphabet};

        std::mt19937 random{13};

        for (std::size_t count{0}; count < 100; ++count)
        {
            expect_brute_force(index, class_table, table, random_text(random, 120));
        }
    }
}

TEST(Window_index_test, An_anchor_whose_window_crosses_the_edge_of_a_block_is_found)
{
    const auto table{table_of(Inventory::factorization)};

    const Window_index index{table};

    const Class_table class_table{alphabet};

    // The one anchor stands where the letters meet the punctuation, its window read around the first block's edge at
    // byte 256 or far beyond it; after the leading a, every two-byte letter begins at an odd byte, so the edge falls
    // inside a code point.
    for (std::size_t letters{240}; letters < 272; ++letters)
    {
        for (const std::string_view letter : {"a", "é"})
        {
            std::string text{"a"};

            while (text.size() < letters)
            {
                text += letter;
            }

            text += "!!";

            const auto anchors{anchors_of(table, text)};

            ASSERT_EQ(anchors.size(), 1U) << letters;
            EXPECT_EQ(index.first_anchor(class_table, text, 0), *anchors.begin()) << letters;
            EXPECT_EQ(index.first_anchor(class_table, text, *anchors.begin()), *anchors.begin()) << letters;
        }
    }

    const auto far{std::string(5000, 'a') + "!!"};

    EXPECT_EQ(index.first_anchor(class_table, far, 0), 5000U);
}

TEST(Window_index_test, A_stretch_that_holds_no_certified_window_has_no_anchor)
{
    const auto table{table_of(Inventory::factorization)};

    const Window_index index{table};

    const Class_table class_table{alphabet};

    const std::string letters(10000, 'a');

    EXPECT_FALSE(index.first_anchor(class_table, letters, 0).has_value());
    EXPECT_FALSE(index.first_anchor(class_table, "", 0).has_value());
    EXPECT_FALSE(index.first_anchor(class_table, "a!", 2).has_value());
}

TEST(Window_index_test, The_start_of_a_stretch_is_no_anchor_and_the_width_is_the_table_s)
{
    const auto table{table_of(Inventory::factorization)};

    const Window_index index{table};

    const Class_table class_table{alphabet};

    EXPECT_EQ(index.first_anchor(class_table, "a!", 0), 1U);
    EXPECT_EQ(index.first_anchor(class_table, "!a", 0), 1U);
    EXPECT_EQ(index.width(), table.width);
}
