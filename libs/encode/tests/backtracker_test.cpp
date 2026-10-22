#include "matisse/encode/backtracker.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "matisse/common/sha256.hpp"
#include "matisse/encode/join_table.hpp"
#include "matisse/encode/merger.hpp"
#include "matisse/encode/spelling_table.hpp"

using namespace matisse::encode;
using matisse::load::Definition;
using matisse::load::Merge_rule;
using matisse::load::Token_id_t;

namespace
{
/**
 * @brief The rank of no join, standing for an absent pair.
 */
constexpr std::uint32_t no_rank{std::numeric_limits<std::uint32_t>::max()};

/**
 * @brief The SHA-256 of GPT-2's encoder.json as pinned, gpt2-vocab.json in the vocabulary directory.
 */
constexpr std::string_view gpt2_encoder_digest{"196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"};

/**
 * @brief The SHA-256 of GPT-2's vocab.bpe as pinned, gpt2-merges.txt in the vocabulary directory.
 */
constexpr std::string_view gpt2_vocab_digest{"1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"};

/**
 * @brief The SHA-256 of cl100k_base.tiktoken as pinned.
 */
constexpr std::string_view cl100k_digest{"223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"};

/**
 * @brief The SHA-256 of o200k_base.tiktoken as pinned.
 */
constexpr std::string_view o200k_digest{"446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"};

/**
 * @brief The SHA-256 of Llama 3's llama3.tiktoken as pinned.
 */
constexpr std::string_view llama3_digest{"82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"};

/**
 * @brief The tables a backtracking merge is decided with, built from a definition as the Merger builds them.
 */
struct Tables
{
    /**
     * @brief The id of every byte as a token of one byte.
     */
    std::array<Token_id_t, 256> byte_ids{};

    /**
     * @brief The join of every pair of parts the merge may join.
     */
    Join_table joins;
};

/**
 * @brief Returns the tables of a definition's merge as the Merger builds them: under the rank rule every split of a
 *        spelling into two spellings, joining at the spelling's id, the first id of a pair met kept; under the pair
 *        rule every merge, a pair listed twice at its last rank.
 * @param definition The definition, every byte one of its tokens and every merge's part and product one.
 * @return The tables.
 */
Tables tables_of(const Definition& definition)
{
    const auto& [merge_rule, whole_piece_lookup, spellings, merges, specials]{definition};

    const Spelling_table ids{spellings};

    std::array<Token_id_t, 256> byte_ids{};

    for (std::size_t value{0}; value < byte_ids.size(); ++value)
    {
        const std::string byte{static_cast<char>(value)};

        byte_ids[value] = *ids.find(byte);
    }

    Join_table::Joins_t joins{};

    if (merge_rule == Merge_rule::ranks)
    {
        for (std::size_t id{0}; id < spellings.size(); ++id)
        {
            const std::string_view whole{spellings[id]};

            const auto product{static_cast<Token_id_t>(id)};

            for (std::size_t split{1}; split < whole.size(); ++split)
            {
                const auto left{ids.find(whole.substr(0, split))};

                const auto right{ids.find(whole.substr(split))};

                if (!left || !right)
                {
                    continue;
                }

                joins.emplace(Join_table::key_of(*left, *right), Join_table::Join{.rank = product, .product = product});
            }
        }
    }

    for (std::size_t rank{0}; rank < merges.size(); ++rank)
    {
        const auto& [left, right]{merges[rank]};

        const auto key{Join_table::key_of(*ids.find(left), *ids.find(right))};

        const auto product{*ids.find(left + right)};

        joins.insert_or_assign(key, Join_table::Join{.rank = static_cast<std::uint32_t>(rank), .product = product});
    }

    return {.byte_ids = byte_ids, .joins = Join_table{joins}};
}

/**
 * @brief Returns the backtracking merge of a definition, or none where the definition does not meet its condition.
 * @param definition The definition.
 * @param tables The definition's tables.
 * @return The backtracking merge, std::nullopt when the definition does not meet the condition.
 */
std::optional<Backtracker> backtracker_of(const Definition& definition, const Tables& tables)
{
    return Backtracker::of(definition.spellings, tables.byte_ids, tables.joins);
}

/**
 * @brief Returns the ids of a piece under the plain loop over a merge's joins: join the adjacent pair of least rank,
 *        the leftmost on a tie, until no pair has a join.
 * @param piece The piece, at least one byte.
 * @param tables The definition's tables.
 * @return The ids.
 */
std::vector<Token_id_t> plain_merge(const std::string_view piece, const Tables& tables)
{
    std::vector<Token_id_t> parts{};

    for (const auto byte : piece)
    {
        parts.push_back(tables.byte_ids[static_cast<unsigned char>(byte)]);
    }

    while (true)
    {
        std::pair best{no_rank, std::size_t{0}};

        Token_id_t product{};

        for (std::size_t at{0}; at + 1 < parts.size(); ++at)
        {
            const auto join{tables.joins.find(parts[at], parts[at + 1])};

            if (join && join->rank < best.first)
            {
                best = {join->rank, at};

                product = join->product;
            }
        }

        const auto [rank, at]{best};

        if (rank == no_rank)
        {
            return parts;
        }

        parts[at] = product;

        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }
}

/**
 * @brief Returns every single byte as a spelling, by its value, the start of every definition the tests build.
 * @return The 256 spellings.
 */
std::vector<std::string> single_bytes()
{
    std::vector<std::string> spellings{};

    for (unsigned value{0}; value < 256; ++value)
    {
        spellings.emplace_back(1, static_cast<char>(value));
    }

    return spellings;
}

/**
 * @brief Returns a definition under the rank rule whose tokens are the single bytes and then the given tokens, the rank
 *        of each its id.
 * @param tokens The multi-byte tokens in rank order.
 * @return The definition.
 */
Definition ranks_definition(const std::vector<std::string>& tokens)
{
    auto spellings{single_bytes()};

    spellings.insert(spellings.end(), tokens.begin(), tokens.end());

    return {.merge_rule = Merge_rule::ranks,
            .whole_piece_lookup = true,
            .spellings = std::move(spellings),
            .merges = {},
            .specials = {}};
}

/**
 * @brief Returns a definition under the pair rule with the given merges, whose tokens are the single bytes and then
 *        every part and product of a merge that is not one already.
 * @param merges The merges in rank order.
 * @return The definition.
 */
Definition pairs_definition(const std::vector<munch::verifier::Merge>& merges)
{
    auto spellings{single_bytes()};

    std::set<std::string> known{spellings.begin(), spellings.end()};

    for (const auto& [left, right] : merges)
    {
        for (const auto& spelling : {left, right, left + right})
        {
            if (known.insert(spelling).second)
            {
                spellings.push_back(spelling);
            }
        }
    }

    return {.merge_rule = Merge_rule::pairs,
            .whole_piece_lookup = false,
            .spellings = std::move(spellings),
            .merges = merges,
            .specials = {}};
}

/**
 * @brief Returns the ids of a run of spellings under a definition.
 * @param spellings The spellings, each a token.
 * @param definition The definition.
 * @return The ids, the first id of each spelling.
 */
std::vector<Token_id_t> ids_of(const std::vector<std::string>& spellings, const Definition& definition)
{
    const Spelling_table ids{definition.spellings};

    std::vector<Token_id_t> out{};

    for (const auto& spelling : spellings)
    {
        out.push_back(*ids.find(spelling));
    }

    return out;
}

/**
 * @brief Returns a string of symbols drawn from an alphabet, of a length drawn between two bounds.
 * @param random The random sequence.
 * @param alphabet The symbols.
 * @param shortest The least length.
 * @param longest The greatest length.
 * @return The string.
 */
std::string random_string(
        std::mt19937& random, const std::string_view alphabet, const std::size_t shortest, const std::size_t longest)
{
    std::uniform_int_distribution<std::size_t> length{shortest, longest};

    std::uniform_int_distribution<std::size_t> symbol{0, alphabet.size() - 1};

    std::string out{};

    for (auto count{length(random)}; count > 0; --count)
    {
        out += alphabet[symbol(random)];
    }

    return out;
}

/**
 * @brief Returns seeded random pieces under a vocabulary: half uniform random bytes of 2 to 40, nearly half
 *        concatenations of 2 to 6 random tokens cut at 64 bytes, so that merges happen, and one in 32 a concatenation
 *        of 40 random tokens cut at 100 to 200 bytes.
 * @param spellings The vocabulary's spellings.
 * @param count The number of pieces.
 * @return The pieces.
 */
std::vector<std::string> random_pieces(const std::vector<std::string>& spellings, const std::size_t count)
{
    std::mt19937 random{20261022};

    std::uniform_int_distribution<int> byte{0, 255};

    std::uniform_int_distribution<std::size_t> length{2, 40};

    std::uniform_int_distribution<std::size_t> tokens{2, 6};

    std::uniform_int_distribution<std::size_t> token{0, spellings.size() - 1};

    std::uniform_int_distribution<std::size_t> long_length{100, 200};

    std::vector<std::string> pieces{};

    for (std::size_t index{0}; index < count; ++index)
    {
        std::string piece{};

        if (index % 2 == 0)
        {
            for (auto left{length(random)}; left > 0; --left)
            {
                piece += static_cast<char>(byte(random));
            }
        }
        else if (index % 32 != 1)
        {
            for (auto left{tokens(random)}; left > 0; --left)
            {
                piece += spellings[token(random)];
            }

            piece.resize(std::min<std::size_t>(piece.size(), 64));
        }
        else
        {
            for (std::size_t left{40}; left > 0; --left)
            {
                piece += spellings[token(random)];
            }

            const auto cut{long_length(random)};

            piece.resize(std::min(piece.size(), cut));
        }

        pieces.push_back(std::move(piece));
    }

    return pieces;
}

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
 * @brief Returns the path of a pinned vocabulary file in the directory MATISSE_VOCAB_DIR names.
 * @param name The file's name.
 * @return The path, std::nullopt when no directory is named or it does not hold the file.
 */
std::optional<std::filesystem::path> pinned(const std::string_view name)
{
    const std::filesystem::path directory{VOCAB_DIR};

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
    const auto encoder{pinned("gpt2-vocab.json")};
    const auto vocab{pinned("gpt2-merges.txt")};

    if (!encoder || !vocab)
    {
        return std::nullopt;
    }

    EXPECT_EQ(matisse::common::sha256(contents(*encoder)), gpt2_encoder_digest);
    EXPECT_EQ(matisse::common::sha256(contents(*vocab)), gpt2_vocab_digest);

    return matisse::load::read_gpt2(*encoder, *vocab);
}

/**
 * @brief Returns a pinned rank file read with no special tokens, after checking it against its digest.
 * @param name The file's name.
 * @param digest The file's SHA-256.
 * @return The definition, std::nullopt when the vocabulary directory does not hold the file.
 */
std::optional<Definition> pinned_tiktoken(const std::string_view name, const std::string_view digest)
{
    const auto path{pinned(name)};

    if (!path)
    {
        return std::nullopt;
    }

    EXPECT_EQ(matisse::common::sha256(contents(*path)), digest) << name;

    return matisse::load::read_tiktoken(*path, {});
}

/**
 * @brief What holding a vocabulary's backtracking merge to the plain loop found.
 */
struct Vocabulary_check
{
    /**
     * @brief The multi-byte tokens merged.
     */
    std::size_t tokens{0};

    /**
     * @brief The multi-byte tokens the merge does not form from their own bytes.
     */
    std::size_t not_their_own_merge{0};

    /**
     * @brief The random pieces merged.
     */
    std::size_t pieces{0};
};

/**
 * @brief Holds the backtracking merge of a definition that meets its condition to the plain loop over its joins on
 *        every multi-byte token and on seeded random pieces.
 * @param definition The definition.
 * @return What the check found.
 */
Vocabulary_check expect_vocabulary_backtracks_as_the_plain_loop(const Definition& definition)
{
    const auto tables{tables_of(definition)};

    const auto backtracker{backtracker_of(definition, tables)};

    EXPECT_TRUE(backtracker.has_value());

    if (!backtracker)
    {
        return {};
    }

    Vocabulary_check check{};

    for (std::size_t id{0}; id < definition.spellings.size(); ++id)
    {
        const auto& spelling{definition.spellings[id]};

        if (spelling.size() < 2)
        {
            continue;
        }

        const auto merged{backtracker->merged(spelling)};

        EXPECT_EQ(merged, plain_merge(spelling, tables)) << matisse::common::sha256(spelling);

        ++check.tokens;

        check.not_their_own_merge += merged != std::vector{static_cast<Token_id_t>(id)} ? 1 : 0;
    }

    for (const auto& piece : random_pieces(definition.spellings, 20000))
    {
        EXPECT_EQ(backtracker->merged(piece), plain_merge(piece, tables)) << matisse::common::sha256(piece);

        ++check.pieces;
    }

    return check;
}

} // namespace

TEST(Backtracker_test, Under_aa_aba_ab_aba_is_joined_from_the_later_ab_so_the_table_is_not_backtracked)
{
    // aba at 257 is joined last from ab, at 258, and a: abaa is ab aa by the rule, where the bpe crate, whose pair test
    // takes every token to be joined from two of lower rank, gives aba a.
    const auto definition{ranks_definition({"aa", "aba", "ab"})};

    EXPECT_FALSE(backtracker_of(definition, tables_of(definition)).has_value());
    EXPECT_EQ(Merger{definition}.merged("abaa"), ids_of({"ab", "aa"}, definition));
}

TEST(Backtracker_test, Under_ab_aaa_aa_aaa_is_joined_from_the_later_aa_so_the_table_is_not_backtracked)
{
    // aaa at 257 is joined last from aa, at 258, and a: aaab is aa ab by the rule, where the bpe crate gives aaa b.
    const auto definition{ranks_definition({"ab", "aaa", "aa"})};

    EXPECT_FALSE(backtracker_of(definition, tables_of(definition)).has_value());
    EXPECT_EQ(Merger{definition}.merged("aaab"), ids_of({"aa", "ab"}, definition));
}

TEST(Backtracker_test, Under_aab_aa_aab_is_joined_from_the_later_aa_so_the_table_is_not_backtracked)
{
    // aab at 256 is joined last from aa, at 257, and b, so the joins of aaaab do not come in order of rank: aa aab is
    // the rule's answer, and the pair test, which reads the run in that order, refuses aa beside aab.
    const auto definition{ranks_definition({"aab", "aa"})};

    EXPECT_FALSE(backtracker_of(definition, tables_of(definition)).has_value());
    EXPECT_EQ(Merger{definition}.merged("aaaab"), ids_of({"aa", "aab"}, definition));
}

TEST(Backtracker_test, A_token_that_does_not_encode_to_itself_is_never_laid)
{
    // aba is a token, but neither ab nor ba is one, so no merge forms it and aba merges to a b a.
    const auto definition{ranks_definition({"aba", "bb"})};

    const auto tables{tables_of(definition)};

    const auto backtracker{backtracker_of(definition, tables)};

    ASSERT_TRUE(backtracker.has_value());
    EXPECT_EQ(backtracker->merged("aba"), ids_of({"a", "b", "a"}, definition));
    EXPECT_EQ(backtracker->merged("abba"), ids_of({"a", "bb", "a"}, definition));
}

TEST(Backtracker_test, The_backtracking_merge_of_a_single_byte_is_its_byte_and_of_the_empty_piece_nothing)
{
    const auto definition{ranks_definition({"ab"})};

    const auto backtracker{backtracker_of(definition, tables_of(definition))};

    ASSERT_TRUE(backtracker.has_value());
    EXPECT_EQ(backtracker->merged("a"), std::vector<Token_id_t>{'a'});
    EXPECT_TRUE(backtracker->merged("").empty());
}

TEST(Backtracker_test, On_random_tables_over_three_symbols_that_meet_the_condition_backtracking_is_the_plain_loop)
{
    // Pair tables of up to 12 pairs of strings of one to three symbols, ranked at random, and rank tables of up to 20
    // strings of one to five symbols, each with 20 pieces of up to 12 symbols and one of 100 to 160: ties, overlaps and
    // tokens that do not encode to themselves among them.
    constexpr std::string_view alphabet{"abc"};

    constexpr std::size_t tables_drawn{3000};

    constexpr std::size_t pieces_per_table{20};

    std::mt19937 random{2};

    std::uniform_int_distribution<std::size_t> pair_count{1, 12};

    std::uniform_int_distribution<std::size_t> token_count{1, 20};

    std::size_t met{0};

    std::size_t not_met{0};

    for (std::size_t table{0}; table < tables_drawn; ++table)
    {
        std::set<std::pair<std::string, std::string>> drawn_pairs{};

        for (auto count{pair_count(random)}; count > 0; --count)
        {
            auto left{random_string(random, alphabet, 1, 3)};

            auto right{random_string(random, alphabet, 1, 3)};

            drawn_pairs.emplace(std::move(left), std::move(right));
        }

        std::vector<munch::verifier::Merge> merges{};

        for (const auto& [left, right] : drawn_pairs)
        {
            merges.push_back({.left = left, .right = right});
        }

        std::ranges::shuffle(merges, random);

        std::set<std::string> drawn_tokens{};

        for (auto count{token_count(random)}; count > 0; --count)
        {
            drawn_tokens.insert(random_string(random, alphabet, 2, 5));
        }

        std::vector<std::string> tokens{drawn_tokens.begin(), drawn_tokens.end()};

        std::ranges::shuffle(tokens, random);

        for (const auto& definition : {pairs_definition(merges), ranks_definition(tokens)})
        {
            const auto tables{tables_of(definition)};

            const auto backtracker{backtracker_of(definition, tables)};

            if (!backtracker)
            {
                ++not_met;

                continue;
            }

            ++met;

            for (std::size_t count{0}; count <= pieces_per_table; ++count)
            {
                const auto long_piece{count == pieces_per_table};

                const auto shortest{long_piece ? 100UZ : 1UZ};

                const auto longest{long_piece ? 160UZ : 12UZ};

                const auto piece{random_string(random, alphabet, shortest, longest)};

                EXPECT_EQ(backtracker->merged(piece), plain_merge(piece, tables)) << "table " << table << " " << piece;
            }
        }
    }

    EXPECT_EQ(met, 4429U);
    EXPECT_EQ(not_met, 1571U);
}

TEST(Backtracker_test, Every_multi_byte_token_of_gpt2_and_random_pieces_backtrack_as_the_plain_loop)
{
    const auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    const auto [tokens, not_their_own_merge, pieces]{expect_vocabulary_backtracks_as_the_plain_loop(*definition)};

    // <|endoftext|> is the one multi-byte token the merge does not form, which encoder.py reads as text.
    EXPECT_EQ(tokens, 50001U);
    EXPECT_EQ(not_their_own_merge, 1U);
    EXPECT_EQ(pieces, 20000U);
}

TEST(Backtracker_test, Every_multi_byte_token_of_cl100k_and_random_pieces_backtrack_as_the_plain_loop)
{
    const auto definition{pinned_tiktoken("cl100k_base.tiktoken", cl100k_digest)};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto [tokens, not_their_own_merge, pieces]{expect_vocabulary_backtracks_as_the_plain_loop(*definition)};

    EXPECT_EQ(tokens, 100000U);
    EXPECT_EQ(not_their_own_merge, 0U);
    EXPECT_EQ(pieces, 20000U);
}

TEST(Backtracker_test, Every_multi_byte_token_of_o200k_and_random_pieces_backtrack_as_the_plain_loop)
{
    const auto definition{pinned_tiktoken("o200k_base.tiktoken", o200k_digest)};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding o200k_base.tiktoken";
    }

    const auto [tokens, not_their_own_merge, pieces]{expect_vocabulary_backtracks_as_the_plain_loop(*definition)};

    EXPECT_EQ(tokens, 199742U);
    EXPECT_EQ(not_their_own_merge, 0U);
    EXPECT_EQ(pieces, 20000U);
}

TEST(Backtracker_test, Llama3_s_rank_file_does_not_meet_the_condition_since_a_split_part_follows_its_token)
{
    // .:.: at 100421 is joined last from .:., at 105051, and :, the first by rank of 81 such tokens.
    const auto definition{pinned_tiktoken("llama3.tiktoken", llama3_digest)};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding llama3.tiktoken";
    }

    EXPECT_FALSE(backtracker_of(*definition, tables_of(*definition)).has_value());
}
