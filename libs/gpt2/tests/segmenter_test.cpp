#include "matisse/pretok/segmenter.hpp"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "matisse/gpt2/tokenizer.hpp"

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
 * @brief The segmentation fixture under check: GPT-2's segmenter, whether the current universe is GPT-2's, and the
 *        counts of its lines so far.
 */
struct Fixture_check
{
    /**
     * @brief GPT-2's segmenter.
     */
    Segmenter segmenter{matisse::gpt2::pattern()};

    /**
     * @brief Whether the current universe is GPT-2's, false before the first universe line.
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
 * @brief Starts a universe line's universe, GPT-2's when it names gpt2.
 * @param check The fixture under check.
 * @param fields The line after its first word.
 */
void start_universe(Fixture_check& check, std::istringstream& fields)
{
    std::string name{};

    fields >> name;

    check.inside = name == "gpt2";

    if (check.inside)
    {
        ++check.counts.universes;
    }
}

/**
 * @brief Checks a segment line: GPT-2's segmenter ends the pieces of its class string, written in hex, at its offsets.
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

    EXPECT_EQ(check.segmenter.boundaries(bytes_of(text)), ends) << context;

    ++check.counts.segments;
}

/**
 * @brief Checks one fixture line against GPT-2's segmenter, a comment or blank line checking nothing, a line of
 *        another universe skipped and a corpus line only counted.
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

TEST(Segmenter_test, Gpt2_ends_the_pieces_of_a_text_where_its_engine_does)
{
    const Segmenter gpt2{matisse::gpt2::pattern()};

    // GPT-2 leaves the last space of a run to the word after it, and keeps a run of line feeds whole.
    EXPECT_EQ(gpt2.boundaries("Hello  world\n\n"), (std::vector<std::size_t>{5, 6, 12, 14}));
    EXPECT_EQ(gpt2.boundaries("don't DON'T"), (std::vector<std::size_t>{3, 5, 9, 10, 11}));

    EXPECT_TRUE(gpt2.boundaries("").empty());
}

TEST(Segmenter_test, The_boundaries_are_byte_offsets_into_the_text)
{
    const Segmenter gpt2{matisse::gpt2::pattern()};

    EXPECT_EQ(gpt2.boundaries("h\u00E9llo w\u00F6rld"), (std::vector<std::size_t>{6, 13}));
    EXPECT_EQ(gpt2.boundaries("\u65E5\u672C\u8A9E"), (std::vector<std::size_t>{9}));
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
    EXPECT_EQ(check.counts.segments, 41371U);
    EXPECT_EQ(check.counts.corpora, 3U);
}
