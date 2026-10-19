#include "matisse/pretok/segmenter.hpp"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <gtest/gtest.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace matisse::pretok;

using matisse::load::Pattern;

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
 * @brief Returns the refusal a Segmenter makes of a pattern or a text, empty when it makes none.
 * @param pattern The pattern.
 * @param text The text.
 * @return The refusal's message.
 */
std::string refusal_of(const Pattern pattern, const std::string_view text)
{
    try
    {
        std::ignore = Segmenter{pattern}.boundaries(text);
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

} // namespace

TEST(Segmenter_test, The_patterns_end_the_pieces_of_a_text_where_their_engines_do)
{
    const Segmenter gpt2{Pattern::gpt2};
    const Segmenter cl100k{Pattern::cl100k};

    // GPT-2 leaves the last space of a run to the word after it, and keeps a run of line feeds whole.
    EXPECT_EQ(gpt2.boundaries("Hello  world\n\n"), (std::vector<std::size_t>{5, 6, 12, 14}));
    EXPECT_EQ(gpt2.boundaries("don't DON'T"), (std::vector<std::size_t>{3, 5, 9, 10, 11}));

    // cl100k ends a run of white space at its last line feed, reads contractions caselessly and numbers in threes.
    EXPECT_EQ(cl100k.boundaries("a \n\t b"), (std::vector<std::size_t>{1, 3, 4, 6}));
    EXPECT_EQ(cl100k.boundaries("don't DON'T"), (std::vector<std::size_t>{3, 5, 9, 11}));
    EXPECT_EQ(cl100k.boundaries("1234567"), (std::vector<std::size_t>{3, 6, 7}));
    EXPECT_EQ(cl100k.boundaries("x  "), (std::vector<std::size_t>{1, 3}));

    EXPECT_TRUE(gpt2.boundaries("").empty());
}

TEST(Segmenter_test, The_boundaries_are_byte_offsets_into_the_text)
{
    const Segmenter gpt2{Pattern::gpt2};

    EXPECT_EQ(gpt2.boundaries("h\u00E9llo w\u00F6rld"), (std::vector<std::size_t>{6, 13}));
    EXPECT_EQ(gpt2.boundaries("\u65E5\u672C\u8A9E"), (std::vector<std::size_t>{9}));
    EXPECT_EQ(Segmenter{Pattern::cl100k}.boundaries("\u00A0\u00A0x"), (std::vector<std::size_t>{2, 5}));
}

TEST(Segmenter_test, Refuses_a_definition_without_a_pattern_and_text_that_is_not_utf8)
{
    EXPECT_EQ(refusal_of(Pattern::none, "a"), "Segmenter: Pattern::none has no pre-tokenizer");
    EXPECT_TRUE(refusal_of(Pattern::gpt2, "ab\xFF").contains("the text is not UTF-8 at byte 2"));
}

TEST(Segmenter_test, The_segmenter_gives_the_engine_s_segmentation_of_every_class_string_of_the_fixture)
{
    std::ifstream file{std::string{SOURCE_DIR} + "/libs/pretok/tests/data/segmentation_reference.txt"};

    ASSERT_TRUE(file.is_open());

    std::optional<Segmenter> segmenter{};

    Fixture_counts counts{};

    for (std::string line{}; std::getline(file, line);)
    {
        std::istringstream fields{line};

        std::string kind{};

        if (!(fields >> kind) || kind.starts_with('#'))
        {
            continue;
        }

        if (kind == "universe")
        {
            std::string name{};

            fields >> name;

            segmenter.emplace(name == "gpt2" ? Pattern::gpt2 : Pattern::cl100k);

            ++counts.universes;

            continue;
        }

        if (kind == "corpus")
        {
            ++counts.corpora;

            continue;
        }

        ASSERT_EQ(kind, "segment") << line;
        ASSERT_TRUE(segmenter) << line;

        std::string text{};

        fields >> text;

        std::vector<std::size_t> ends{};

        for (std::size_t end{}; fields >> end;)
        {
            ends.push_back(end);
        }

        EXPECT_EQ(segmenter->boundaries(bytes_of(text)), ends) << line;

        ++counts.segments;
    }

    EXPECT_EQ(counts.universes, 2U);
    EXPECT_EQ(counts.segments, 41371U + 69905U);
    EXPECT_EQ(counts.corpora, 6U);
}
