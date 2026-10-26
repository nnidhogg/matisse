#include "matisse/validate/report.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include <munch/verifier/verifier_decisions.hpp>

#include "matisse/llama3/tokenizer.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/pretok/verifier.hpp"

using namespace matisse::validate;

namespace
{
/**
 * @brief The widest window the windows fixture holds, which its script decides to width four.
 */
constexpr std::size_t fixture_width{4};

/**
 * @brief The certified windows of a table with their origins, found by their class symbols.
 */
using Origins_t = std::map<std::string, std::size_t, std::less<>>;

/**
 * @brief What the windows and right-context fixtures hold, counted as they are checked.
 */
struct Fixture_counts
{
    /**
     * @brief The universe lines, one per fixture.
     */
    std::size_t universes{0};

    /**
     * @brief The certified lines of each inventory, by its name, one per certified window.
     */
    std::map<std::string, std::size_t> certified{};

    /**
     * @brief The refused lines, one per inventory and width.
     */
    std::size_t refused{0};

    /**
     * @brief The suffix lines.
     */
    std::size_t suffixes{0};

    /**
     * @brief The context lines of each inventory whose right context holds, by its name.
     */
    std::map<std::string, std::size_t> holding{};

    /**
     * @brief The context lines of each inventory whose right context fails, by its name.
     */
    std::map<std::string, std::size_t> failing{};

    /**
     * @brief The right lines of each inventory, by its name.
     */
    std::map<std::string, std::size_t> rights{};

    /**
     * @brief The base windows of each inventory, by its name, at which the suffix line is checked.
     */
    std::map<std::string, std::size_t> bases{};

    /**
     * @brief The base windows of each inventory, by its name, whose shortest refutation by dependence() is of the
     *        prefix alone, leaving the suffix side open there.
     */
    std::map<std::string, std::size_t> open{};
};

/**
 * @brief The fixtures under check: Llama 3's report, its split verifier and its segmenter, whether the current universe
 *        is Llama 3's, and the counts of their lines so far.
 */
struct Fixture_check
{
    /**
     * @brief Llama 3's report.
     */
    Report report{validate(matisse::llama3::pattern(), fixture_width)};

    /**
     * @brief Llama 3's split verifier.
     */
    munch::verifier::Verifier verifier{matisse::pretok::split_verifier(matisse::llama3::pattern())};

    /**
     * @brief Llama 3's segmenter, which replays a witness.
     */
    matisse::pretok::Segmenter segmenter{matisse::llama3::pattern()};

    /**
     * @brief Whether the current universe is Llama 3's, false before the first universe line.
     */
    bool inside{};

    /**
     * @brief The lines checked so far, by kind.
     */
    Fixture_counts counts{};
};

/**
 * @brief Starts a universe line's universe, Llama 3's when it names llama3.
 * @param check The fixtures under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.inside = name == "llama3";

    if (check.inside)
    {
        ++check.counts.universes;
    }
}

/**
 * @brief Returns the bytes a lowercase hex field spells, two digits per byte, "-" spelling the empty text.
 * @param field The hex digits.
 * @return The bytes.
 */
std::string bytes_of(const std::string_view field)
{
    std::string bytes{};

    if (field == "-")
    {
        return bytes;
    }

    for (std::size_t at{0}; at + 1 < field.size(); at += 2)
    {
        unsigned value{};

        std::ignore = std::from_chars(field.data() + at, field.data() + at + 2, value, 16);

        bytes += static_cast<char>(value);
    }

    return bytes;
}

/**
 * @brief Returns the window table a fixture line names, the policy table for policy and the factorization table for
 *        factorization, and fails for any other name.
 * @param report The report.
 * @param inventory The inventory's name.
 * @return The table.
 */
const Window_table& table_named(const Report& report, const std::string_view inventory)
{
    EXPECT_TRUE(inventory == "factorization" || inventory == "policy") << "unknown inventory: " << inventory;

    return inventory == "policy" ? report.policy : report.factorization;
}

/**
 * @brief Checks a certified line: the next window of its inventory's table is its window, written in hex, at its
 *        origin.
 * @param check The fixtures under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_certified(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string inventory{};
    std::string window{};
    std::size_t origin{};

    fields >> inventory >> window >> origin;

    const auto& windows{table_named(check.report, inventory).windows};

    auto& line{check.counts.certified[inventory]};

    ASSERT_LT(line, windows.size()) << context;

    EXPECT_EQ(windows[line].window, bytes_of(window)) << context;
    EXPECT_EQ(windows[line].origin, origin) << context;

    ++line;
}

/**
 * @brief Checks a refused line: its inventory's table certifies every window of its width but its count.
 * @param check The fixtures under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_refused(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string inventory{};
    std::size_t width{};
    std::size_t count{};

    fields >> inventory >> width >> count;

    const auto& windows{table_named(check.report, inventory).windows};

    const auto symbols{matisse::llama3::pattern().alphabet.symbols.size()};

    std::size_t all{1};

    for (std::size_t symbol{0}; symbol < width; ++symbol)
    {
        all *= symbols;
    }

    const auto certified{std::ranges::count_if(windows, [&](const Certified_window& certified_window) {
        return certified_window.window.size() == width;
    })};

    EXPECT_EQ(all - static_cast<std::size_t>(certified), count) << context;

    ++check.counts.refused;
}

/**
 * @brief Returns whether a certified window holds a shorter certified window whose origin lands on the same symbol, so
 *        that every cut of the window is a cut of the shorter one at the same place.
 * @param origins The certified windows with their origins.
 * @param window The window.
 * @param origin The window's origin.
 * @return True when it holds such a window.
 */
bool holds_shorter(const Origins_t& origins, const std::string_view window, const std::size_t origin)
{
    for (std::size_t length{2}; length < window.size(); ++length)
    {
        for (std::size_t from{0}; from <= origin && from + length <= window.size(); ++from)
        {
            const auto inner{window.substr(from, length)};

            const auto shorter{origins.find(inner)};

            if (shorter != origins.end() && from + shorter->second == origin)
            {
                return true;
            }
        }
    }

    return false;
}

/**
 * @brief Returns the base windows of a table, those that hold no shorter certified window whose origin lands on the
 *        same symbol, every cut of another window being a cut of a base window at the same place.
 * @param table The window table.
 * @return The base windows, in the table's order.
 */
std::vector<Certified_window> base_windows(const Window_table& table)
{
    Origins_t origins{};

    for (const auto& [window, origin] : table.windows)
    {
        origins.emplace(window, origin);
    }

    std::vector<Certified_window> bases{};

    for (const auto& certified : table.windows)
    {
        if (!holds_shorter(origins, certified.window, certified.origin))
        {
            bases.push_back(certified);
        }
    }

    return bases;
}

/**
 * @brief Checks a suffix line: the suffix side holds, so that the shortest refutation dependence() gives at a base
 *        window of either inventory, whose cuts are every cut of the inventory, is of the prefix alone.
 * @param check The fixtures under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_suffix(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string verdict{};

    fields >> verdict;

    EXPECT_EQ(verdict, "holds") << context;

    const auto expect_prefix_only{[&](const std::string& inventory) {
        const auto& table{table_named(check.report, inventory)};

        for (const auto& [window, origin] : base_windows(table))
        {
            ++check.counts.bases[inventory];

            const auto dependence{munch::verifier::dependence(check.verifier, window, origin)};

            if (!dependence)
            {
                continue;
            }

            EXPECT_FALSE(dependence->suffix_fails) << context << " at " << window;

            ++check.counts.open[inventory];
        }
    }};

    expect_prefix_only("factorization");
    expect_prefix_only("policy");

    ++check.counts.suffixes;
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
 * @brief Returns the offsets a comma-separated field lists, "-" listing none.
 * @param field The offsets.
 * @return The offsets, in the field's order.
 */
std::vector<std::size_t> offsets_of(const std::string_view field)
{
    std::vector<std::size_t> offsets{};

    if (field == "-")
    {
        return offsets;
    }

    for (std::size_t at{0}; at <= field.size();)
    {
        std::size_t offset{};

        const auto read{std::from_chars(field.data() + at, field.data() + field.size(), offset)};

        offsets.push_back(offset);

        at = static_cast<std::size_t>(read.ptr - field.data()) + 1;
    }

    return offsets;
}

/**
 * @brief Checks a failing context line: the table's truncation at its context is its text, written in hex, at its cut,
 *        with an occurrence of one of the table's windows whose origin lands on the cut; and Llama 3's segmenter ends
 *        the pieces at or below the cut at its offsets on the whole text and on the text cut short past the cut by the
 *        context, read as a whole input, which differ.
 * @param check The fixtures under check.
 * @param table The window table the line names.
 * @param right_context The context that fails.
 * @param fields The line after its verdict.
 * @param context The line.
 */
void expect_failing_context(
        const Fixture_check& check, const Window_table& table, const std::size_t right_context,
        std::istringstream& fields, const std::string& context)
{
    std::string text_field{};
    std::size_t cut{};
    std::string whole_field{};
    std::string truncated_field{};

    fields >> text_field >> cut >> whole_field >> truncated_field;

    const auto found{std::ranges::find(table.failing, right_context, &munch::verifier::Truncation::context)};

    ASSERT_NE(found, table.failing.end()) << context;

    const auto& truncation{*found};

    const auto text{bytes_of(text_field)};

    EXPECT_EQ(truncation.segmentation.bytes, text) << context;
    EXPECT_EQ(truncation.cut, cut) << context;

    const auto lands{[&](const Certified_window& certified) {
        return std::string_view{text}.substr(truncation.occurrence).starts_with(certified.window) &&
               truncation.occurrence + certified.origin == cut;
    }};

    EXPECT_TRUE(std::ranges::any_of(table.windows, lands)) << context;

    const auto whole{ends_up_to(check.segmenter, text, cut)};

    const auto shortened{std::string_view{text}.substr(0, cut + right_context)};

    const auto truncated{ends_up_to(check.segmenter, shortened, cut)};

    EXPECT_EQ(whole, offsets_of(whole_field)) << context;
    EXPECT_EQ(truncated, offsets_of(truncated_field)) << context;
    EXPECT_NE(whole, truncated) << context;
}

/**
 * @brief Checks a context line: a right context that holds fails at no cut of its inventory, and one that fails has the
 *        fixture's witness.
 * @param check The fixtures under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_context(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string inventory{};
    std::size_t right_context{};
    std::string verdict{};

    fields >> inventory >> right_context >> verdict;

    const auto& table{table_named(check.report, inventory)};

    if (verdict == "holds")
    {
        const auto fails_here{
                std::ranges::contains(table.failing, right_context, &munch::verifier::Truncation::context)};

        EXPECT_FALSE(fails_here) << context;

        ++check.counts.holding[inventory];

        return;
    }

    EXPECT_EQ(verdict, "fails") << context;

    expect_failing_context(check, table, right_context, fields, context);

    ++check.counts.failing[inventory];
}

/**
 * @brief Checks a right line: its inventory's table carries its right context.
 * @param check The fixtures under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_right(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string inventory{};
    std::size_t right_context{};

    fields >> inventory >> right_context;

    EXPECT_EQ(table_named(check.report, inventory).right_context, right_context) << context;

    ++check.counts.rights[inventory];
}

/**
 * @brief Checks one line of either fixture against Llama 3's report, a comment or blank line checking nothing, a line
 *        of another universe skipped.
 * @param check The fixtures under check.
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
    else if (kind == "certified")
    {
        expect_certified(check, fields, line);
    }
    else if (kind == "refused")
    {
        expect_refused(check, fields, line);
    }
    else if (kind == "suffix")
    {
        expect_suffix(check, fields, line);
    }
    else if (kind == "context")
    {
        expect_context(check, fields, line);
    }
    else if (kind == "right")
    {
        expect_right(check, fields, line);
    }
    else
    {
        ADD_FAILURE() << "unknown fixture line: " << line;
    }
}

} // namespace

TEST(Report_test, Llama3_s_report_holds_the_windows_and_right_context_fixtures_and_an_unbounded_boundary_gap)
{
    Fixture_check check{};

    for (const std::string_view name : {"windows_reference.txt", "right_context_reference.txt"})
    {
        std::ifstream file{std::string{SOURCE_DIR} + "/libs/validate/tests/data/" + std::string{name}};

        ASSERT_TRUE(file.is_open()) << name;

        check.inside = false;

        for (std::string line{}; std::getline(file, line);)
        {
            expect_fixture_line(check, line);
        }
    }

    const auto& [factorization, policy, gap, pattern]{check.report};

    EXPECT_EQ(check.counts.universes, 2U);
    EXPECT_EQ(check.counts.certified["factorization"], factorization.windows.size());
    EXPECT_EQ(check.counts.certified["policy"], policy.windows.size());
    EXPECT_EQ(check.counts.refused, 2 * (fixture_width - 1));
    EXPECT_EQ(check.counts.suffixes, 1U);
    EXPECT_EQ(check.counts.holding["factorization"], 1U);
    EXPECT_EQ(check.counts.holding["policy"], 1U);
    EXPECT_EQ(check.counts.failing["factorization"], factorization.failing.size());
    EXPECT_EQ(check.counts.failing["policy"], policy.failing.size());
    EXPECT_EQ(check.counts.rights["factorization"], 1U);
    EXPECT_EQ(check.counts.rights["policy"], 1U);
    EXPECT_EQ(check.counts.bases["factorization"], 112U);
    EXPECT_EQ(check.counts.bases["policy"], 678U);
    EXPECT_EQ(check.counts.open["factorization"], 4U);
    EXPECT_EQ(check.counts.open["policy"], 4U);

    // A letter run of any length is one piece.
    EXPECT_TRUE(std::holds_alternative<munch::verifier::Lasso>(gap));
}
