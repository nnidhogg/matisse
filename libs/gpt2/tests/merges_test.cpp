#include "matisse/validate/merges.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "matisse/common/sha256.hpp"
#include "matisse/encode/merger.hpp"
#include "matisse/load/definition.hpp"

using matisse::encode::Merger;
using matisse::load::Definition;
using matisse::load::Token_id_t;
using matisse::validate::Merge_window;

namespace
{
/**
 * @brief The SHA-256 of GPT-2's encoder.json as pinned, gpt2-vocab.json in the vocabulary directory.
 */
constexpr std::string_view gpt2_encoder_digest{"196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"};

/**
 * @brief The SHA-256 of GPT-2's vocab.bpe as pinned, gpt2-merges.txt in the vocabulary directory.
 */
constexpr std::string_view gpt2_vocab_digest{"1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"};

/**
 * @brief The SHA-256 of twitter.json as pinned, the corpus the candidates are drawn from.
 */
constexpr std::string_view twitter_digest{"30721e496a8d73cfc50658923c34eb2c0fbe15ee6835005e43ee624d8dedf200"};

/**
 * @brief How many of the corpus's most frequent pairs and of its most frequent triples are drawn, as matisse-validate
 *        draws them unless told.
 */
constexpr std::size_t drawn_count{1000};

/**
 * @brief The values a byte takes, every candidate of width one.
 */
constexpr std::size_t byte_values{256};

/**
 * @brief Returns the whole of a file.
 * @param path The file's path.
 * @return Its bytes.
 */
std::string contents(const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};

    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

/**
 * @brief Returns the path of a file in a directory a build option names.
 * @param directory The directory, empty when the option names none.
 * @param name The file's name.
 * @return The path, std::nullopt when no directory is named or it does not hold the file.
 */
std::optional<std::filesystem::path> named(const std::filesystem::path& directory, const std::string_view name)
{
    if (directory.empty() || !std::filesystem::exists(directory / name))
    {
        return std::nullopt;
    }

    return directory / name;
}

/**
 * @brief Returns GPT-2's pinned files read, after checking each against its digest.
 * @return The definition, std::nullopt when the vocabulary directory does not hold the files.
 */
std::optional<Definition> pinned_gpt2()
{
    const auto encoder{named(VOCAB_DIR, "gpt2-vocab.json")};
    const auto vocab{named(VOCAB_DIR, "gpt2-merges.txt")};

    if (!encoder || !vocab)
    {
        return std::nullopt;
    }

    EXPECT_EQ(matisse::common::sha256(contents(*encoder)), gpt2_encoder_digest);
    EXPECT_EQ(matisse::common::sha256(contents(*vocab)), gpt2_vocab_digest);

    return matisse::load::read_gpt2(*encoder, *vocab);
}

/**
 * @brief Returns a text's windows of one width that occur most often, the most frequent first and a tie in the order
 *        of the bytes, as matisse-validate draws them.
 * @param text The text.
 * @param width The windows' width.
 * @param count How many are drawn at most.
 * @return The windows.
 */
std::vector<std::string> most_frequent(const std::string_view text, const std::size_t width, const std::size_t count)
{
    std::unordered_map<std::string_view, std::size_t> occurrences{};

    for (std::size_t at{0}; at + width <= text.size(); ++at)
    {
        ++occurrences[text.substr(at, width)];
    }

    std::vector<std::pair<std::string_view, std::size_t>> ranked{occurrences.begin(), occurrences.end()};

    const auto more_frequent{[]<typename Entry>(const Entry& left, const Entry& right) {
        const auto& [left_window, left_count]{left};

        const auto& [right_window, right_count]{right};

        return left_count != right_count ? left_count > right_count : left_window < right_window;
    }};

    std::ranges::sort(ranked, more_frequent);

    std::vector<std::string> windows{};

    for (const auto& [window, occurrence_count] : ranked | std::views::take(count))
    {
        windows.emplace_back(window);
    }

    return windows;
}

/**
 * @brief Returns where the token covering each byte of a text begins, by the ids of its encoding.
 * @param definition The definition, whose spellings give each id's length.
 * @param ids The text's ids.
 * @return The start of the covering token of every byte, by the byte's index.
 */
std::vector<std::size_t> covering_starts(const Definition& definition, const std::vector<Token_id_t>& ids)
{
    std::vector<std::size_t> starts{};

    for (const auto id : ids)
    {
        const auto start{starts.size()};

        starts.insert(starts.end(), definition.spellings[id].size(), start);
    }

    return starts;
}

/**
 * @brief Returns the boundaries of a text's encoding as a marked string carries them: one flag per byte, set after
 *        every token but the last.
 * @param definition The definition, whose spellings give each id's length.
 * @param ids The text's ids.
 * @return The flags.
 */
std::vector<bool> boundaries_of(const Definition& definition, const std::vector<Token_id_t>& ids)
{
    std::vector<bool> boundaries{};

    for (const auto id : ids)
    {
        boundaries.insert(boundaries.end(), definition.spellings[id].size(), false);

        boundaries.back() = true;
    }

    if (!boundaries.empty())
    {
        boundaries.back() = false;
    }

    return boundaries;
}

} // namespace

TEST(Merges_test, Gpt2_s_table_decides_twitter_json_s_frequent_windows_and_every_certified_cut_holds_on_its_encoding)
{
    const auto definition{pinned_gpt2()};
    const auto corpus{named(CORPUS_DIR, "twitter.json")};

    if (!definition || !corpus)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR and MATISSE_CORPUS_DIR name no directories holding GPT-2's files and "
                        "twitter.json";
    }

    const auto text{contents(*corpus)};

    EXPECT_EQ(matisse::common::sha256(text), twitter_digest);

    auto candidates{most_frequent(text, 1, byte_values)};

    for (const auto width : {2UZ, 3UZ})
    {
        const auto drawn{most_frequent(text, width, drawn_count)};

        candidates.insert(candidates.end(), drawn.begin(), drawn.end());
    }

    const auto decided{matisse::validate::validate_merges(*definition, candidates)};

    std::map<std::size_t, std::size_t> certified_per_width{};

    for (const auto& [window, origin, refutation] : decided)
    {
        if (!refutation)
        {
            ++certified_per_width[window.size()];
        }
    }

    EXPECT_EQ(candidates.size(), 2177U);
    EXPECT_EQ(certified_per_width, (std::map<std::size_t, std::size_t>{{1, 4}, {2, 488}, {3, 527}}));

    // The corpus encoded as one piece, as a merge table without a pre-tokenizer encodes it.
    const Merger merger{*definition};

    const auto starts{covering_starts(*definition, merger.merged(text))};

    ASSERT_EQ(starts.size(), text.size());

    for (const auto& [window, origin, refutation] : decided)
    {
        if (refutation)
        {
            const auto& [segmentation, occurrence]{*refutation};

            const auto& bytes{segmentation.bytes};

            const auto ids{merger.merged(bytes)};

            const auto last{occurrence + window.size() - 1};

            ASSERT_LE(occurrence + window.size(), bytes.size()) << window;
            EXPECT_EQ(segmentation.boundaries, boundaries_of(*definition, ids)) << window;
            EXPECT_EQ(bytes.substr(occurrence, window.size()), window);
            EXPECT_NE(covering_starts(*definition, ids)[last], occurrence + origin) << window;

            continue;
        }

        for (auto at{text.find(window)}; at != std::string::npos; at = text.find(window, at + 1))
        {
            ASSERT_EQ(starts[at + window.size() - 1], at + origin) << window << " at byte " << at;
        }
    }
}
