#include "matisse/encode/merger.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "matisse/common/sha256.hpp"

using namespace matisse::encode;
using matisse::load::Definition;
using matisse::load::Merge_rule;
using matisse::load::Token_id_t;

namespace
{
/**
 * @brief The rank of every token by its bytes, a rank file's table, as tiktoken holds it.
 */
using Ranks_t = std::unordered_map<std::string, std::size_t>;

/**
 * @brief The rank of every pair of a merge table, as encoder.py and Hugging Face's BPE model hold it.
 */
using Pair_ranks_t = std::map<std::pair<std::string, std::string>, std::size_t>;

/**
 * @brief The parts a piece is merged into, each a token's bytes.
 */
using Parts_t = std::vector<std::string>;

/**
 * @brief The rank no table gives, standing for an absent pair.
 */
constexpr std::size_t absent{std::numeric_limits<std::size_t>::max()};

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
 * @brief Returns the parts of a piece under tiktoken's _byte_pair_merge, transcribed: each part carries the rank of the
 *        pair it begins, the least rank is joined, the first on a tie, and the two ranks beside the join are read
 *        again.
 * @param piece The piece, at least one byte.
 * @param ranks The rank of every token.
 * @return The parts.
 */
Parts_t tiktoken_scan(const std::string_view piece, const Ranks_t& ranks)
{
    const auto rank_of{[&ranks](const std::string_view bytes) {
        const auto found{ranks.find(std::string{bytes})};

        return found == ranks.end() ? absent : found->second;
    }};

    // Each part is where it starts and the rank of the pair it begins, the last two the ends.
    std::vector<std::pair<std::size_t, std::size_t>> parts{};

    for (std::size_t at{0}; at + 1 < piece.size(); ++at)
    {
        parts.emplace_back(at, rank_of(piece.substr(at, 2)));
    }

    parts.emplace_back(piece.size() - 1, absent);

    parts.emplace_back(piece.size(), absent);

    const auto rank_at{[&](const std::size_t index) {
        if (index + 3 >= parts.size())
        {
            return absent;
        }

        const auto start{parts[index].first};

        return rank_of(piece.substr(start, parts[index + 3].first - start));
    }};

    const auto least{[&parts] {
        std::pair minimum{absent, std::size_t{0}};

        for (std::size_t index{0}; index + 1 < parts.size(); ++index)
        {
            if (parts[index].second < minimum.first)
            {
                minimum = {parts[index].second, index};
            }
        }

        return minimum;
    }};

    for (auto [rank, index]{least()}; rank != absent; std::tie(rank, index) = least())
    {
        if (index > 0)
        {
            parts[index - 1].second = rank_at(index - 1);
        }

        parts[index].second = rank_at(index);

        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(index) + 1);
    }

    Parts_t out{};

    for (std::size_t index{0}; index + 1 < parts.size(); ++index)
    {
        const auto start{parts[index].first};

        out.emplace_back(piece.substr(start, parts[index + 1].first - start));
    }

    return out;
}

/**
 * @brief Returns the bytes of a piece, each a part of its own.
 * @param piece The piece.
 * @return The parts.
 */
Parts_t bytes_of(const std::string_view piece)
{
    Parts_t parts{};

    for (const auto byte : piece)
    {
        parts.emplace_back(1, byte);
    }

    return parts;
}

/**
 * @brief Returns the parts of a piece under the plain pair loop, Hugging Face's rule written as it reads: join the
 *        adjacent pair of least rank, the leftmost on a tie, until no pair is in the table.
 * @param piece The piece.
 * @param ranks The rank of every pair.
 * @return The parts.
 */
Parts_t pair_loop(const std::string_view piece, const Pair_ranks_t& ranks)
{
    auto parts{bytes_of(piece)};

    while (true)
    {
        std::pair best{absent, std::size_t{0}};

        for (std::size_t at{0}; at + 1 < parts.size(); ++at)
        {
            const auto found{ranks.find({parts[at], parts[at + 1]})};

            if (found != ranks.end() && found->second < best.first)
            {
                best = {found->second, at};
            }
        }

        const auto [rank, at]{best};

        if (rank == absent)
        {
            return parts;
        }

        parts[at] += parts[at + 1];

        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }
}

/**
 * @brief Returns the parts of a piece under encoder.py's bpe() loop: take the adjacent pair of least rank and join
 *        every occurrence of it in one pass from the left, until no pair is in the table.
 * @param piece The piece.
 * @param ranks The rank of every pair.
 * @return The parts.
 */
Parts_t pass_loop(const std::string_view piece, const Pair_ranks_t& ranks)
{
    auto word{bytes_of(piece)};

    while (word.size() > 1)
    {
        auto best{ranks.end()};

        for (std::size_t at{0}; at + 1 < word.size(); ++at)
        {
            const auto found{ranks.find({word[at], word[at + 1]})};

            if (found != ranks.end() && (best == ranks.end() || found->second < best->second))
            {
                best = found;
            }
        }

        if (best == ranks.end())
        {
            break;
        }

        const auto& [first, second]{best->first};

        Parts_t joined{};

        for (std::size_t at{0}; at < word.size();)
        {
            if (at + 1 < word.size() && word[at] == first && word[at + 1] == second)
            {
                joined.push_back(first + second);

                at += 2;

                continue;
            }

            joined.push_back(word[at]);

            ++at;
        }

        word = std::move(joined);
    }

    return word;
}

/**
 * @brief Returns the id of every spelling of a definition.
 * @param definition The definition.
 * @return The ids, by spelling.
 */
std::unordered_map<std::string, Token_id_t> ids_by_spelling(const Definition& definition)
{
    std::unordered_map<std::string, Token_id_t> ids{};

    for (std::size_t id{0}; id < definition.spellings.size(); ++id)
    {
        ids.emplace(definition.spellings[id], static_cast<Token_id_t>(id));
    }

    return ids;
}

/**
 * @brief Returns the ids of a run of parts.
 * @param parts The parts, each a token.
 * @param ids The id of every spelling.
 * @return The ids, std::numeric_limits<Token_id_t>::max() for a part that is no token.
 */
std::vector<Token_id_t> ids_of(const Parts_t& parts, const std::unordered_map<std::string, Token_id_t>& ids)
{
    std::vector<Token_id_t> out{};

    for (const auto& part : parts)
    {
        const auto found{ids.find(part)};

        out.push_back(found == ids.end() ? std::numeric_limits<Token_id_t>::max() : found->second);
    }

    return out;
}

/**
 * @brief Returns the rank table of a definition under the rank rule, the rank of each token its id.
 * @param definition The definition.
 * @return The ranks.
 */
Ranks_t ranks_of(const Definition& definition)
{
    Ranks_t ranks{};

    for (std::size_t id{0}; id < definition.spellings.size(); ++id)
    {
        ranks.emplace(definition.spellings[id], id);
    }

    return ranks;
}

/**
 * @brief Returns the pair table of a definition under the pair rule, a pair listed twice at its last rank.
 * @param definition The definition.
 * @return The ranks.
 */
Pair_ranks_t pair_ranks_of(const Definition& definition)
{
    Pair_ranks_t ranks{};

    for (std::size_t rank{0}; rank < definition.merges.size(); ++rank)
    {
        const auto& [left, right]{definition.merges[rank]};

        ranks.insert_or_assign({left, right}, rank);
    }

    return ranks;
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
 * @brief Returns the refusal a Merger makes of a definition, empty when it makes none.
 * @param definition The definition.
 * @return The refusal's message.
 */
std::string merger_refusal(const Definition& definition)
{
    try
    {
        std::ignore = Merger{definition};
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
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
 *        of 40 random tokens cut at 100 to 200 bytes, either side of the longest piece the merge scans.
 * @param spellings The vocabulary's spellings.
 * @param count The number of pieces.
 * @return The pieces.
 */
std::vector<std::string> random_pieces(const std::vector<std::string>& spellings, const std::size_t count)
{
    std::mt19937 random{20261020};

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
 * @brief What holding a vocabulary's merge to the plain references found.
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
 * @brief Holds the merge of a definition to its plain reference on every multi-byte token and on seeded random pieces,
 *        and under the pair rule to encoder.py's pass as well.
 * @param definition The definition.
 * @param random_count The number of random pieces.
 * @return What the check found.
 */
Vocabulary_check expect_vocabulary_merges_as_the_references(
        const Definition& definition, const std::size_t random_count)
{
    const Merger merger{definition};

    const auto ids{ids_by_spelling(definition)};

    const auto pairs{definition.merge_rule == Merge_rule::pairs};

    const auto ranks{pairs ? Ranks_t{} : ranks_of(definition)};

    const auto pair_ranks{pairs ? pair_ranks_of(definition) : Pair_ranks_t{}};

    const auto reference{[&](const std::string_view piece) {
        return pairs ? pair_loop(piece, pair_ranks) : tiktoken_scan(piece, ranks);
    }};

    const auto expect_merged{[&](const std::string_view piece) {
        const auto parts{reference(piece)};

        const auto merged{merger.merged(piece)};

        EXPECT_EQ(merged, ids_of(parts, ids)) << matisse::common::sha256(piece);

        if (pairs)
        {
            EXPECT_EQ(pass_loop(piece, pair_ranks), parts) << matisse::common::sha256(piece);
        }

        return merged;
    }};

    Vocabulary_check check{};

    for (std::size_t id{0}; id < definition.spellings.size(); ++id)
    {
        const auto& spelling{definition.spellings[id]};

        if (spelling.size() < 2)
        {
            continue;
        }

        const auto merged{expect_merged(spelling)};

        ++check.tokens;

        check.not_their_own_merge += merged != std::vector{static_cast<Token_id_t>(id)} ? 1UZ : 0UZ;
    }

    for (const auto& piece : random_pieces(definition.spellings, random_count))
    {
        std::ignore = expect_merged(piece);

        ++check.pieces;
    }

    return check;
}

} // namespace

TEST(Merger_test, Under_ab_a_then_a_b_the_pair_rule_joins_one_pair_at_a_time_where_encoder_py_s_pass_differs)
{
    // (ab, a) at rank 0 has the part ab that (a, b) forms at rank 1: one join at a time gives aba b on abab, since the
    // first join of (a, b) makes (ab, a), the least pair; the pass joins both occurrences of (a, b) first, ab ab.
    const auto definition{pairs_definition({{.left = "ab", .right = "a"}, {.left = "a", .right = "b"}})};

    const Merger merger{definition};

    const auto ids{ids_by_spelling(definition)};

    const auto pair_ranks{pair_ranks_of(definition)};

    EXPECT_EQ(merger.merged("abab"), ids_of({"aba", "b"}, ids));
    EXPECT_EQ(pair_loop("abab", pair_ranks), (Parts_t{"aba", "b"}));
    EXPECT_EQ(pass_loop("abab", pair_ranks), (Parts_t{"ab", "ab"}));
}

TEST(Merger_test, Under_the_rank_rule_a_join_that_makes_a_pair_of_lower_rank_than_its_own_is_followed_by_that_pair)
{
    // Llama 3's shape on ".:.:": ":." joins first, then ".:." at the higher rank, which makes ".:.:" of a lower rank
    // than its own, and that join comes next.
    const auto definition{ranks_definition({":.", ".:.:", ".:."})};

    const Merger merger{definition};

    const auto ids{ids_by_spelling(definition)};

    EXPECT_EQ(merger.merged(".:.:"), ids_of({".:.:"}, ids));
    EXPECT_EQ(tiktoken_scan(".:.:", ranks_of(definition)), (Parts_t{".:.:"}));
}

TEST(Merger_test, Under_the_pair_rule_a_join_that_makes_a_pair_of_lower_rank_than_its_own_is_followed_by_that_pair)
{
    // (a, b) joins at rank 1 and makes (ab, c) at rank 0, which joins next, before (c, d) at rank 2.
    const auto definition{
            pairs_definition({{.left = "ab", .right = "c"}, {.left = "a", .right = "b"}, {.left = "c", .right = "d"}})};

    const Merger merger{definition};

    EXPECT_EQ(merger.merged("abcd"), ids_of({"abc", "d"}, ids_by_spelling(definition)));
    EXPECT_EQ(pair_loop("abcd", pair_ranks_of(definition)), (Parts_t{"abc", "d"}));
}

TEST(Merger_test, Occurrences_of_one_pair_tie_and_the_leftmost_joins_first_under_both_rules)
{
    const auto ranks{ranks_definition({"aa"})};

    const auto pairs{pairs_definition({{.left = "a", .right = "a"}})};

    for (const auto& definition : {ranks, pairs})
    {
        const Merger merger{definition};

        const auto ids{ids_by_spelling(definition)};

        EXPECT_EQ(merger.merged("aaa"), ids_of({"aa", "a"}, ids));
        EXPECT_EQ(merger.merged("aaaa"), ids_of({"aa", "aa"}, ids));
        EXPECT_EQ(merger.merged("aaaaa"), ids_of({"aa", "aa", "a"}, ids));
    }
}

TEST(Merger_test, An_entry_whose_parts_a_join_has_changed_is_discarded_rather_than_joined)
{
    // On "abc" with bc before ab, the entry of ab is left with a right part that has grown into bc; with ab before bc,
    // the entry of bc is left with a left part that has been joined into ab. Neither may join.
    const auto bc_first{ranks_definition({"bc", "ab"})};

    const auto ab_first{ranks_definition({"ab", "bc"})};

    EXPECT_EQ(Merger{bc_first}.merged("abc"), ids_of({"a", "bc"}, ids_by_spelling(bc_first)));
    EXPECT_EQ(Merger{ab_first}.merged("abc"), ids_of({"ab", "c"}, ids_by_spelling(ab_first)));

    const auto pairs{pairs_definition({{.left = "b", .right = "c"}, {.left = "a", .right = "b"}})};

    EXPECT_EQ(Merger{pairs}.merged("abc"), ids_of({"a", "bc"}, ids_by_spelling(pairs)));
}

TEST(Merger_test, Under_the_pair_rule_a_pair_not_in_the_table_is_never_joined_whatever_its_concatenation_spells)
{
    // abc is a token, formed by (ab, c), but (a, bc) is no merge, so a piece that makes bc first stays a bc; under the
    // rank rule over the same tokens the concatenation is looked up and abc forms.
    const auto pairs{
            pairs_definition({{.left = "b", .right = "c"}, {.left = "ab", .right = "c"}, {.left = "a", .right = "b"}})};

    const auto ranks{ranks_definition({"bc", "abc", "ab"})};

    EXPECT_EQ(Merger{pairs}.merged("abc"), ids_of({"a", "bc"}, ids_by_spelling(pairs)));
    EXPECT_EQ(Merger{ranks}.merged("abc"), ids_of({"abc"}, ids_by_spelling(ranks)));
}

TEST(Merger_test, The_merge_of_a_single_byte_is_its_byte_and_of_the_empty_piece_nothing)
{
    const auto definition{ranks_definition({"ab"})};

    const Merger merger{definition};

    EXPECT_EQ(merger.merged("a"), std::vector<Token_id_t>{'a'});
    EXPECT_TRUE(merger.merged("").empty());
}

TEST(Merger_test, The_whole_piece_lookup_finds_a_token_the_merge_does_not_form_only_where_the_definition_says_so)
{
    // aba is a token, but neither ab nor ba is one, so the merge leaves a b a.
    auto definition{ranks_definition({"aba"})};

    const auto aba{static_cast<Token_id_t>(definition.spellings.size() - 1)};

    const Merger looking{definition};

    EXPECT_EQ(looking.whole_piece("aba"), aba);
    EXPECT_EQ(looking.whole_piece("b"), Token_id_t{'b'});
    EXPECT_FALSE(looking.whole_piece("ab").has_value());
    EXPECT_EQ(looking.merged("aba"), (std::vector<Token_id_t>{'a', 'b', 'a'}));

    definition.whole_piece_lookup = false;

    const Merger merging{definition};

    EXPECT_FALSE(merging.whole_piece("aba").has_value());
}

TEST(Merger_test, A_definition_with_a_byte_that_is_no_token_is_refused_naming_the_byte)
{
    auto definition{ranks_definition({})};

    definition.spellings[0x80] = "\x80\x80";

    const auto refusal{merger_refusal(definition)};

    EXPECT_TRUE(refusal.contains(R"(the byte "\x80" is not a token of the definition)")) << refusal;
}

TEST(Merger_test, A_merge_naming_a_part_or_a_product_that_is_no_token_is_refused_naming_the_merge_and_its_rank)
{
    auto definition{pairs_definition({{.left = "a", .right = "b"}, {.left = "ab", .right = "c"}})};

    std::erase(definition.spellings, "abc");

    const auto refusal{merger_refusal(definition)};

    EXPECT_TRUE(refusal.contains(R"(the merge ("ab", "c") at rank 1 names "abc", which is not a token)")) << refusal;
}

TEST(Merger_test, A_pair_listed_twice_takes_its_last_rank)
{
    // (a, b) at rank 0 and again at rank 2, after (b, c) at rank 1: at its last rank bc joins first on abc.
    const auto definition{
            pairs_definition({{.left = "a", .right = "b"}, {.left = "b", .right = "c"}, {.left = "a", .right = "b"}})};

    EXPECT_EQ(Merger{definition}.merged("abc"), ids_of({"a", "bc"}, ids_by_spelling(definition)));
}

TEST(Merger_test, On_random_tables_over_three_symbols_the_merge_is_the_plain_reference_under_both_rules)
{
    // Pair tables of up to 12 pairs of strings of one to three symbols, ranked at random, and rank tables of up to 20
    // strings of one to five symbols, each with 20 pieces of up to 12 symbols and every tenth with one of 100 to 160,
    // either side of the longest piece the merge scans: the generator makes ties, overlaps and joins that make pairs of
    // lower rank, and pair tables on which the pass and one join at a time differ.
    constexpr std::string_view alphabet{"abc"};

    constexpr std::size_t tables{3000};

    constexpr std::size_t pieces_per_table{20};

    std::mt19937 random{1};

    std::uniform_int_distribution<std::size_t> pair_count{1, 12};

    std::uniform_int_distribution<std::size_t> token_count{1, 20};

    std::size_t pass_differs{0};

    for (std::size_t table{0}; table < tables; ++table)
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

        const auto pairs{pairs_definition(merges)};

        const auto ranks{ranks_definition(tokens)};

        const Merger pairs_merger{pairs};

        const Merger ranks_merger{ranks};

        const auto pair_ids{ids_by_spelling(pairs)};

        const auto rank_ids{ids_by_spelling(ranks)};

        const auto pair_ranks{pair_ranks_of(pairs)};

        const auto rank_table{ranks_of(ranks)};

        const auto pieces{table % 10 == 0 ? pieces_per_table + 1 : pieces_per_table};

        for (std::size_t count{0}; count < pieces; ++count)
        {
            const auto long_piece{count == pieces_per_table};

            const auto shortest{long_piece ? 100UZ : 1UZ};

            const auto longest{long_piece ? 160UZ : 12UZ};

            const auto piece{random_string(random, alphabet, shortest, longest)};

            const auto one_at_a_time{pair_loop(piece, pair_ranks)};

            EXPECT_EQ(pairs_merger.merged(piece), ids_of(one_at_a_time, pair_ids)) << "table " << table << " " << piece;
            EXPECT_EQ(ranks_merger.merged(piece), ids_of(tiktoken_scan(piece, rank_table), rank_ids))
                    << "table " << table << " " << piece;

            pass_differs += pass_loop(piece, pair_ranks) != one_at_a_time ? 1UZ : 0UZ;
        }
    }

    EXPECT_GT(pass_differs, 0U);
}

TEST(Merger_test, Every_multi_byte_token_of_gpt2_and_random_pieces_merge_as_the_pair_loop_and_encoder_py_s_pass)
{
    const auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    const auto [tokens, not_their_own_merge, pieces]{expect_vocabulary_merges_as_the_references(*definition, 20000)};

    // <|endoftext|> is the one multi-byte token the merge does not form, which encoder.py reads as text.
    EXPECT_EQ(tokens, 50001U);
    EXPECT_EQ(not_their_own_merge, 1U);
    EXPECT_EQ(pieces, 20000U);
}

TEST(Merger_test, Every_multi_byte_token_of_cl100k_and_random_pieces_merge_as_tiktoken_s_scan)
{
    const auto definition{pinned_tiktoken("cl100k_base.tiktoken", cl100k_digest)};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    const auto [tokens, not_their_own_merge, pieces]{expect_vocabulary_merges_as_the_references(*definition, 20000)};

    EXPECT_EQ(tokens, 100000U);
    EXPECT_EQ(not_their_own_merge, 0U);
    EXPECT_EQ(pieces, 20000U);
}

TEST(Merger_test, Every_multi_byte_token_of_o200k_and_random_pieces_merge_as_tiktoken_s_scan)
{
    const auto definition{pinned_tiktoken("o200k_base.tiktoken", o200k_digest)};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding o200k_base.tiktoken";
    }

    const auto [tokens, not_their_own_merge, pieces]{expect_vocabulary_merges_as_the_references(*definition, 20000)};

    EXPECT_EQ(tokens, 199742U);
    EXPECT_EQ(not_their_own_merge, 0U);
    EXPECT_EQ(pieces, 20000U);
}

TEST(Merger_test, Every_multi_byte_token_of_llama3_and_random_pieces_merge_as_tiktoken_s_scan)
{
    const auto definition{pinned_tiktoken("llama3.tiktoken", llama3_digest)};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding llama3.tiktoken";
    }

    const auto [tokens, not_their_own_merge, pieces]{expect_vocabulary_merges_as_the_references(*definition, 20000)};

    // The 588 tokens the merge does not form are one id each only through the whole-piece lookup.
    EXPECT_EQ(tokens, 127744U);
    EXPECT_EQ(not_their_own_merge, 588U);
    EXPECT_EQ(pieces, 20000U);

    const Merger merger{*definition};

    const auto& spellings{definition->spellings};

    for (std::size_t id{0}; id < spellings.size(); ++id)
    {
        EXPECT_EQ(merger.whole_piece(spellings[id]), static_cast<Token_id_t>(id));
    }
}
