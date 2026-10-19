#include "matisse/pretok/classes.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using namespace matisse::pretok;

using matisse::load::Pattern;

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
 * @brief The classes fixture under check: the current universe's pattern and the counts so far.
 */
struct Fixture_check
{
    /**
     * @brief The current universe's pattern, Pattern::none before the first universe line.
     */
    Pattern pattern{Pattern::none};

    /**
     * @brief The lines checked so far, by kind.
     */
    Fixture_counts counts{};
};

/**
 * @brief Returns a message padded as SHA-256 pads it: a one bit, zeros to 56 mod 64, and the bit length big-endian.
 * @param text The message.
 * @return The padded message, a whole number of 64-byte blocks.
 */
std::string padded_message(const std::string_view text)
{
    std::string padded{text};

    padded.push_back(static_cast<char>(0x80));

    while (padded.size() % 64 != 56)
    {
        padded.push_back('\0');
    }

    const auto bit_length{static_cast<std::uint64_t>(text.size()) * 8};

    for (int shift{56}; shift >= 0; shift -= 8)
    {
        padded.push_back(static_cast<char>((bit_length >> static_cast<unsigned>(shift)) & 0xFF));
    }

    return padded;
}

/**
 * @brief Returns the message schedule of one 64-byte block: its sixteen big-endian words and the 48 SHA-256 derives
 *        from them.
 * @param block The block.
 * @return The 64 words.
 */
std::array<std::uint32_t, 64> schedule_of(const std::string_view block)
{
    std::array<std::uint32_t, 64> schedule{};

    for (std::size_t word{0}; word < 16; ++word)
    {
        for (std::size_t byte{0}; byte < 4; ++byte)
        {
            schedule[word] = (schedule[word] << 8U) | static_cast<unsigned char>(block[(word * 4U) + byte]);
        }
    }

    for (std::size_t word{16}; word < 64; ++word)
    {
        const auto s0{
                std::rotr(schedule[word - 15], 7) ^ std::rotr(schedule[word - 15], 18) ^ (schedule[word - 15] >> 3)};

        const auto s1{
                std::rotr(schedule[word - 2], 17) ^ std::rotr(schedule[word - 2], 19) ^ (schedule[word - 2] >> 10)};

        schedule[word] = schedule[word - 16] + s0 + schedule[word - 7] + s1;
    }

    return schedule;
}

/**
 * @brief Returns a hash state after one block, its schedule run through SHA-256's 64 rounds and added to the state.
 * @param state The hash state before the block.
 * @param schedule The block's message schedule.
 * @return The hash state after it.
 */
std::array<std::uint32_t, 8> compressed(
        std::array<std::uint32_t, 8> state, const std::array<std::uint32_t, 64>& schedule)
{
    auto [a, b, c, d, e, f, g, h]{state};

    static constexpr std::array<std::uint32_t, 64> round_constants{
            0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
            0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
            0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
            0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
            0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
            0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
            0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
            0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};

    for (std::size_t round{0}; round < 64; ++round)
    {
        const auto s1{std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)};

        const auto choice{(e & f) ^ (~e & g)};

        const auto t1{h + s1 + choice + round_constants[round] + schedule[round]};

        const auto s0{std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)};

        const auto majority{(a & b) ^ (a & c) ^ (b & c)};

        const auto t2{s0 + majority};

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;

    return state;
}

/**
 * @brief Returns the SHA-256 of a byte string, as hexadecimal, the digest the fixture names each table by.
 * @param text The bytes.
 * @return The 64 hexadecimal digits.
 */
std::string sha256(const std::string_view text)
{
    std::array<std::uint32_t, 8> state{0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                                       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

    const auto padded{padded_message(text)};

    for (std::size_t block{0}; block < padded.size(); block += 64)
    {
        const auto schedule{schedule_of(std::string_view{padded}.substr(block, 64))};

        state = compressed(state, schedule);
    }

    std::string hex{};

    for (const auto word : state)
    {
        hex += std::format("{:08x}", word);
    }

    return hex;
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
 * @param name The table's name in class_ranges.inc.
 * @return The lines, and how many ranges they hold.
 */
std::pair<std::string, std::size_t> table_lines(const std::string_view name)
{
    std::ifstream file{std::string{SOURCE_DIR} + "/libs/pretok/src/class_ranges.inc"};

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
 * @brief Returns the refusal classes_of() makes of a text, empty when it makes none.
 * @param pattern The pattern.
 * @param text The text.
 * @return The refusal's message.
 */
std::string refusal_of(const Pattern pattern, const std::string_view text)
{
    try
    {
        std::ignore = classes_of(pattern, text);
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

/**
 * @brief Checks a unicode line: the tables are generated from the version it names.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_unicode(std::istringstream& fields, const std::string& context)
{
    std::string version{};

    fields >> version;

    EXPECT_EQ(unicode_version(), version) << context;
}

/**
 * @brief Starts a universe line's universe: the pattern it names, gpt2 or cl100k.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.pattern = name == "gpt2" ? Pattern::gpt2 : Pattern::cl100k;

    ++check.counts.universes;
}

/**
 * @brief Checks a class line: the class at its index has the symbol it names, written in hex.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_class(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    ASSERT_NE(check.pattern, Pattern::none) << context;

    std::size_t index{};

    std::string symbol{};

    fields >> index >> symbol;

    EXPECT_EQ(std::string{class_symbols(check.pattern).at(index)}, bytes_of(symbol)) << context;

    ++check.counts.classes;
}

/**
 * @brief Checks a ranges line: the checked-in table of the set it names, letters, numbers or white space, holds its
 *        number of ranges and its digest.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_ranges(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    ASSERT_NE(check.pattern, Pattern::none) << context;

    std::string name{};

    std::size_t count{};

    std::string digest{};

    fields >> name >> count >> digest;

    const auto table{name == "letters" ? "letter_ranges" : name == "numbers" ? "number_ranges" : "white_space_ranges"};

    const auto [lines, ranges]{table_lines(table)};

    EXPECT_EQ(ranges, count) << context;
    EXPECT_EQ(sha256(lines), digest) << context;

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
    ASSERT_NE(check.pattern, Pattern::none) << context;

    std::string point_field{};

    std::size_t index{};

    fields >> point_field >> index;

    std::uint32_t point{};

    std::ignore = std::from_chars(point_field.data(), point_field.data() + point_field.size(), point, 16);

    const auto classes{classes_of(check.pattern, encoded(point))};

    ASSERT_EQ(classes.size(), 1U) << context;
    EXPECT_EQ(classes.front(), index) << context;

    ++check.counts.points;
}

/**
 * @brief Checks one fixture line against the tables and the classes, a comment or blank line checking nothing.
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

} // namespace

TEST(Classes_test, Each_pattern_has_its_class_count_and_symbols_and_a_symbol_is_a_member_of_its_class)
{
    EXPECT_EQ(class_count(Pattern::gpt2), 14U);
    EXPECT_EQ(class_count(Pattern::cl100k), 16U);
    EXPECT_EQ(class_symbols(Pattern::gpt2), "\t !'0Adelmrstv");
    EXPECT_EQ(class_symbols(Pattern::cl100k), "\t\n\r !'0Adelmrstv");

    for (const auto pattern : {Pattern::gpt2, Pattern::cl100k})
    {
        const auto symbols{class_symbols(pattern)};

        const auto classes{classes_of(pattern, symbols)};

        ASSERT_EQ(classes.size(), symbols.size());

        for (std::size_t index{0}; index < classes.size(); ++index)
        {
            EXPECT_EQ(classes[index], index) << symbols[index];
        }
    }
}

TEST(Classes_test, The_contraction_letters_are_case_sensitive_for_gpt2_and_caseless_for_cl100k)
{
    const auto gpt2{classes_of(Pattern::gpt2, "sS\u017F")};
    const auto cl100k{classes_of(Pattern::cl100k, "sS\u017F")};

    const auto letter{static_cast<Class_t>(class_symbols(Pattern::gpt2).find('A'))};
    const auto gpt2_s{static_cast<Class_t>(class_symbols(Pattern::gpt2).find('s'))};
    const auto cl100k_s{static_cast<Class_t>(class_symbols(Pattern::cl100k).find('s'))};

    EXPECT_EQ(gpt2, (std::vector<Class_t>{gpt2_s, letter, letter}));
    EXPECT_EQ(cl100k, (std::vector<Class_t>{cl100k_s, cl100k_s, cl100k_s}));
}

TEST(Classes_test, Refuses_text_that_is_not_utf8_at_the_offset_of_its_first_ill_formed_sequence)
{
    EXPECT_TRUE(refusal_of(Pattern::gpt2, "ab\xC0\x80").contains("classes_of: the text is not UTF-8 at byte 2"));
    EXPECT_TRUE(refusal_of(Pattern::gpt2, "a\xE2\x82").contains("not UTF-8 at byte 1"));
    EXPECT_TRUE(refusal_of(Pattern::cl100k, "\xED\xA0\x80").contains("not UTF-8 at byte 0"));
    EXPECT_TRUE(refusal_of(Pattern::cl100k, "\xF4\x90\x80\x80").contains("not UTF-8 at byte 0"));
    EXPECT_TRUE(refusal_of(Pattern::cl100k, "").empty());
}

TEST(Classes_test, Refuses_a_definition_without_a_pattern)
{
    EXPECT_TRUE(refusal_of(Pattern::none, "a").contains("classes_of: Pattern::none has no class alphabet"));
    EXPECT_THROW(std::ignore = class_count(Pattern::none), std::invalid_argument);
    EXPECT_THROW(std::ignore = class_symbols(Pattern::none), std::invalid_argument);
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

    EXPECT_EQ(check.counts.universes, 2U);
    EXPECT_EQ(check.counts.classes, 30U);
    EXPECT_EQ(check.counts.ranges, 6U);
    EXPECT_EQ(check.counts.points, 5523U);
}
