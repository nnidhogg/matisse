#include "matisse/pretok/segmenter.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "matisse/o200k/tokenizer.hpp"

using namespace matisse::pretok;

namespace
{
/**
 * @brief What the segmentation fixture holds, counted as it is checked.
 */
struct Fixture_counts
{
    /**
     * @brief The universe lines.
     */
    std::size_t universes{0};

    /**
     * @brief The segment lines, one per class string.
     */
    std::size_t segments{0};

    /**
     * @brief The corpus lines, which the corpus probe checks.
     */
    std::size_t corpora{0};
};

/**
 * @brief The segmentation fixture under check: o200k's segmenter, whether the current universe is o200k's, and the
 *        counts of its lines so far.
 */
struct Fixture_check
{
    /**
     * @brief o200k's segmenter.
     */
    Segmenter segmenter{matisse::o200k::pattern()};

    /**
     * @brief Whether the current universe is o200k's, false before the first universe line.
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
 * @brief Returns the text a class string stands for: every symbol of a class with an ASCII member is one of its
 *        members, and each of the four symbols of the classes with none is replaced by its class's least code point.
 * @param symbols The class string.
 * @return The text, well-formed UTF-8.
 */
std::string text_of(const std::string_view symbols)
{
    constexpr std::array<std::pair<char, std::string_view>, 4> least{{
            {'#', "\u00AA"},
            {'$', "\u01C5"},
            {'%', "\u02B0"},
            {'&', "\u0300"},
    }};

    std::string text{};

    for (const auto symbol : symbols)
    {
        const auto found{std::ranges::find(least, symbol, &std::pair<char, std::string_view>::first)};

        if (found == least.end())
        {
            text += symbol;

            continue;
        }

        text += found->second;
    }

    return text;
}

/**
 * @brief Returns the code point offsets of byte offsets into a UTF-8 text.
 * @param text The text.
 * @param ends The byte offsets, each at the start of a code point or the text's end.
 * @return The offsets in code points.
 */
std::vector<std::size_t> points_of(const std::string_view text, const std::vector<std::size_t>& ends)
{
    const auto begins_one{[](const char byte) { return (static_cast<unsigned char>(byte) & 0xC0U) != 0x80U; }};

    std::vector<std::size_t> points{};

    for (const auto end : ends)
    {
        const auto prefix{text.substr(0, end)};

        points.push_back(static_cast<std::size_t>(std::ranges::count_if(prefix, begins_one)));
    }

    return points;
}

/**
 * @brief Starts a universe line's universe, o200k's when it names o200k.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.inside = name == "o200k";

    if (check.inside)
    {
        ++check.counts.universes;
    }
}

/**
 * @brief Checks a segment line: o200k's segmenter ends the pieces of the text its class string, written in hex, stands
 *        for at its offsets, counted in code points.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 * @param context The line.
 */
void expect_segment(Fixture_check& check, std::istringstream& fields, const std::string& context)
{
    std::string text{};

    fields >> text;

    std::vector<std::size_t> ends{};

    for (std::size_t end{}; fields >> end;)
    {
        ends.push_back(end);
    }

    const auto spelled{text_of(bytes_of(text))};

    EXPECT_EQ(points_of(spelled, check.segmenter.boundaries(spelled)), ends) << context;

    ++check.counts.segments;
}

/**
 * @brief Checks one fixture line against o200k's segmenter, a comment or blank line checking nothing, a line of another
 *        universe skipped and a corpus line only counted.
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
    else if (kind == "corpus")
    {
        ++check.counts.corpora;
    }
    else if (kind == "segment")
    {
        expect_segment(check, fields, line);
    }
    else
    {
        ADD_FAILURE() << "unknown fixture line: " << line;
    }
}

} // namespace

TEST(Segmenter_test, O200k_ends_the_pieces_of_a_text_where_its_engine_does)
{
    const Segmenter o200k{matisse::o200k::pattern()};

    // o200k ends a run of white space at its last line feed, keeps a contraction with its word in either case, reads
    // numbers in threes, takes slashes after a line break and splits a word where lower case turns upper.
    EXPECT_EQ(o200k.boundaries("a \n\t b"), (std::vector<std::size_t>{1, 3, 4, 6}));
    EXPECT_EQ(o200k.boundaries("don't DON'T"), (std::vector<std::size_t>{5, 11}));
    EXPECT_EQ(o200k.boundaries("1234567"), (std::vector<std::size_t>{3, 6, 7}));
    EXPECT_EQ(o200k.boundaries("x  "), (std::vector<std::size_t>{1, 3}));
    EXPECT_EQ(o200k.boundaries("a/\n/b"), (std::vector<std::size_t>{1, 4, 5}));
    EXPECT_EQ(o200k.boundaries("helloWorld \u01C5a"), (std::vector<std::size_t>{5, 10, 14}));
}

TEST(Segmenter_test, The_boundaries_are_byte_offsets_into_the_text)
{
    const Segmenter o200k{matisse::o200k::pattern()};

    EXPECT_EQ(o200k.boundaries("\u00A0\u00A0x"), (std::vector<std::size_t>{2, 5}));
}

TEST(Segmenter_test, The_segmenter_gives_the_engine_s_segmentation_of_every_class_string_of_the_fixture)
{
    std::ifstream file{std::string{SOURCE_DIR} + "/libs/pretok/tests/data/segmentation_reference.txt"};

    ASSERT_TRUE(file.is_open());

    Fixture_check check{};

    for (std::string line{}; std::getline(file, line);)
    {
        expect_fixture_line(check, line);
    }

    EXPECT_EQ(check.counts.universes, 1U);
    EXPECT_EQ(check.counts.segments, 27931U);
    EXPECT_EQ(check.counts.corpora, 3U);
}
