#include "matisse/pretok/classes.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <munch/regex/utf8.hpp>

#include "matisse/common/sha256.hpp"
#include "matisse/llama3/tokenizer.hpp"

using namespace matisse::pretok;

namespace
{
/**
 * @brief What the classes fixture holds, counted as it is checked.
 */
struct Fixture_counts
{
    /**
     * @brief The universe lines.
     */
    std::size_t universes{0};

    /**
     * @brief The class lines.
     */
    std::size_t classes{0};

    /**
     * @brief The ranges lines.
     */
    std::size_t ranges{0};

    /**
     * @brief The at lines, one per code point on either side of a change of class.
     */
    std::size_t points{0};
};

/**
 * @brief The classes fixture under check: whether the current universe is Llama 3's, and the counts of its lines so
 *        far.
 */
struct Fixture_check
{
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
 * @brief Returns the UTF-8 encoding of a code point.
 * @param point The code point, a scalar value.
 * @return Its one to four bytes.
 */
std::string encoded(const char32_t point)
{
    if (point < 0x80)
    {
        return {static_cast<char>(point)};
    }

    if (point < 0x800)
    {
        return {static_cast<char>(0xC0 | (point >> 6)), static_cast<char>(0x80 | (point & 0x3F))};
    }

    if (point < 0x10000)
    {
        return {static_cast<char>(0xE0 | (point >> 12)), static_cast<char>(0x80 | ((point >> 6) & 0x3F)),
                static_cast<char>(0x80 | (point & 0x3F))};
    }

    return {static_cast<char>(0xF0 | (point >> 18)), static_cast<char>(0x80 | ((point >> 12) & 0x3F)),
            static_cast<char>(0x80 | ((point >> 6) & 0x3F)), static_cast<char>(0x80 | (point & 0x3F))};
}

/**
 * @brief Returns the lines of a checked-in table as the fixture digests them, "FIRST LAST\n" per range in lowercase
 *        hex.
 * @param name The table's name in Llama 3's class_ranges.inc.
 * @return The lines, and how many ranges they hold.
 */
std::pair<std::string, std::size_t> table_lines(const std::string_view name)
{
    std::ifstream file{std::string{SOURCE_DIR} + "/libs/llama3/src/class_ranges.inc"};

    std::string lines{};

    std::size_t count{0};

    bool inside{};

    for (std::string line{}; std::getline(file, line);)
    {
        if (!inside)
        {
            inside = line.contains(std::format(" {}{{", name));

            continue;
        }

        unsigned long first{};

        unsigned long last{};

        const auto parsed{std::sscanf(line.c_str(), " {.first = 0x%lX, .last = 0x%lX}", &first, &last)};

        if (parsed != 2)
        {
            break;
        }

        lines += std::format("{:x} {:x}\n", first, last);

        ++count;
    }

    return {lines, count};
}

/**
 * @brief Checks a unicode line: Llama 3's tables are generated from the version it names.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_unicode(std::istringstream& fields, const std::string& context)
{
    std::string version{};

    fields >> version;

    EXPECT_EQ(matisse::llama3::pattern().alphabet.unicode_version, version) << context;
}

/**
 * @brief Starts a universe line's universe, Llama 3's when it names llama3.
 * @param check The fixture under check.
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
 * @brief Checks a class line: the class at its index has the symbol it names, written in hex.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_class(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::size_t index{};

    std::string symbol{};

    fields >> index >> symbol;

    EXPECT_EQ(std::string{matisse::llama3::pattern().alphabet.symbols.at(index)}, bytes_of(symbol)) << context;

    ++check.counts.classes;
}

/**
 * @brief Checks a ranges line: Llama 3's checked-in table of the set it names, letters, numbers or white space, holds
 *        its number of ranges and its digest.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_ranges(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string name{};

    std::size_t count{};

    std::string digest{};

    fields >> name >> count >> digest;

    const auto table{name == "letters" ? "letter_ranges" : name == "numbers" ? "number_ranges" : "white_space_ranges"};

    const auto [lines, ranges]{table_lines(table)};

    EXPECT_EQ(ranges, count) << context;
    EXPECT_EQ(matisse::common::sha256(lines), digest) << context;

    ++check.counts.ranges;
}

/**
 * @brief Checks an at line: the code point it names in hex is of the class at its index.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_point(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string point_field{};

    std::size_t index{};

    fields >> point_field >> index;

    std::uint32_t point{};

    std::ignore = std::from_chars(point_field.data(), point_field.data() + point_field.size(), point, 16);

    const auto classes{classes_of(matisse::llama3::pattern().alphabet, encoded(point))};

    ASSERT_EQ(classes.size(), 1U) << context;
    EXPECT_EQ(classes.front(), index) << context;

    ++check.counts.points;
}

/**
 * @brief Checks one fixture line against Llama 3's tables and classes, a comment or blank line checking nothing and a
 *        line of another universe skipped.
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

    if (kind == "unicode")
    {
        expect_unicode(fields, line);
    }
    else if (kind == "universe")
    {
        start_universe(check, fields);
    }
    else if (!check.inside)
    {
        return;
    }
    else if (kind == "class")
    {
        expect_class(check, fields, line);
    }
    else if (kind == "ranges")
    {
        expect_ranges(check, fields, line);
    }
    else if (kind == "at")
    {
        expect_point(check, fields, line);
    }
    else
    {
        ADD_FAILURE() << "unknown fixture line: " << line;
    }
}

/**
 * @brief Returns whether a table of ranges, ascending and apart, holds a code point.
 * @param ranges The table.
 * @param point The code point.
 * @return True when a range holds it.
 */
bool holds(const std::span<const munch::regex::utf8::Code_point_range> ranges, const char32_t point)
{
    const auto above{std::ranges::upper_bound(ranges, point, {}, &munch::regex::utf8::Code_point_range::first)};

    if (above == ranges.begin())
    {
        return false;
    }

    return point <= std::prev(above)->last;
}

/**
 * @brief Returns the symbol of a code point's class as the alphabet defines it, searching every set in turn: the symbol
 *        of the first set holding it, or the symbol of everything else.
 * @param alphabet The alphabet.
 * @param point The code point.
 * @return The symbol.
 */
char symbol_of(const Alphabet& alphabet, const char32_t point)
{
    for (const auto& [symbol, ranges] : alphabet.sets)
    {
        if (holds(ranges, point))
        {
            return symbol;
        }
    }

    return alphabet.other;
}

} // namespace

TEST(Classes_test, Llama3_has_its_class_count_and_symbols_and_a_symbol_is_a_member_of_its_class)
{
    const auto alphabet{matisse::llama3::pattern().alphabet};

    const auto symbols{alphabet.symbols};

    EXPECT_EQ(symbols.size(), 16U);
    EXPECT_EQ(symbols, "\t\n\r !'0Adelmrstv");

    const auto classes{classes_of(alphabet, symbols)};

    ASSERT_EQ(classes.size(), symbols.size());

    for (std::size_t index{0}; index < classes.size(); ++index)
    {
        EXPECT_EQ(classes[index], index) << symbols[index];
    }
}

TEST(Classes_test, The_contraction_letters_are_caseless_for_llama3)
{
    const auto alphabet{matisse::llama3::pattern().alphabet};

    const auto classes{classes_of(alphabet, "sS\u017F")};

    const auto s_letter{static_cast<Class_t>(alphabet.symbols.find('s'))};

    EXPECT_EQ(classes, (std::vector<Class_t>{s_letter, s_letter, s_letter}));
}

TEST(Classes_test, The_tables_and_the_classes_give_the_reference_answers_at_every_change_of_class_of_the_fixture)
{
    std::ifstream file{std::string{SOURCE_DIR} + "/libs/pretok/tests/data/classes_reference.txt"};

    ASSERT_TRUE(file.is_open());

    Fixture_check check{};

    for (std::string line{}; std::getline(file, line);)
    {
        expect_fixture_line(check, line);
    }

    EXPECT_EQ(check.counts.universes, 1U);
    EXPECT_EQ(check.counts.classes, 16U);
    EXPECT_EQ(check.counts.ranges, 3U);
    EXPECT_EQ(check.counts.points, 2772U);
}

TEST(Classes_test, The_class_table_gives_every_scalar_value_the_symbol_of_the_first_set_holding_it)
{
    const auto alphabet{matisse::llama3::pattern().alphabet};

    // Every scalar value once, in order, so that the table's ASCII path and its blocks are read at every code point.
    std::string text{};

    std::vector<char32_t> points{};

    for (char32_t point{0}; point < 0x110000; ++point)
    {
        if (point >= 0xD800 && point <= 0xDFFF)
        {
            continue;
        }

        text += encoded(point);

        points.push_back(point);
    }

    const Class_table table{alphabet};

    const auto symbols{table.symbols(text)};

    ASSERT_EQ(points.size(), 1112064U);
    ASSERT_EQ(symbols.size(), points.size());

    for (const auto [index, point] : std::views::enumerate(points))
    {
        const auto expected{symbol_of(alphabet, point)};

        ASSERT_EQ(symbols[static_cast<std::size_t>(index)], expected)
                << std::format("U+{:04X}", static_cast<std::uint32_t>(point));
    }
}
