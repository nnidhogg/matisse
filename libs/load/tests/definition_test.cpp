#include "matisse/load/definition.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace matisse::load;

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
 * @brief The SHA-256 of cl100k_base.tiktoken as pinned.
 */
constexpr std::string_view cl100k_digest{"223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"};

/**
 * @brief What the known-answer fixture holds, counted as it is checked.
 */
struct Fixture_counts
{
    /**
     * @brief The universe lines.
     */
    std::size_t universes{0};

    /**
     * @brief The encode lines.
     */
    std::size_t encodings{0};

    /**
     * @brief The corpus lines, which the corpus probe checks.
     */
    std::size_t corpora{0};
};

/**
 * @brief Returns the directory a test writes its files into, one per test and process.
 * @return The directory, created.
 */
std::filesystem::path scratch()
{
    const std::string_view test{testing::UnitTest::GetInstance()->current_test_info()->name()};

    const auto name{std::format("matisse-load-{}-{}", test, ::getpid())};

    const auto dir{std::filesystem::temp_directory_path() / name};

    std::filesystem::create_directories(dir);

    return dir;
}

/**
 * @brief Writes a file into the test's directory.
 * @param name The file's name.
 * @param text The file's bytes.
 * @return The file's path.
 */
std::filesystem::path written(const std::string_view name, const std::string_view text)
{
    const auto path{scratch() / name};

    std::ofstream{path, std::ios::binary} << text;

    return path;
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
 * @brief Returns the SHA-256 of a byte string, as hexadecimal, the digest the pinned vocabularies are named by.
 * @param text The bytes.
 * @return The 64 hexadecimal digits.
 */
std::string sha256(const std::string_view text)
{
    std::array<std::uint32_t, 8> state{0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                                       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

    // The message, padded: a one bit, zeros to 56 mod 64, and the bit length big-endian.
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

    for (std::size_t block{0}; block < padded.size(); block += 64)
    {
        std::array<std::uint32_t, 64> schedule{};

        for (std::size_t word{0}; word < 16; ++word)
        {
            for (std::size_t byte{0}; byte < 4; ++byte)
            {
                schedule[word] =
                        (schedule[word] << 8U) | static_cast<unsigned char>(padded[block + (word * 4U) + byte]);
            }
        }

        for (std::size_t word{16}; word < 64; ++word)
        {
            const auto s0{
                    std::rotr(schedule[word - 15], 7) ^ std::rotr(schedule[word - 15], 18) ^
                    (schedule[word - 15] >> 3)};

            const auto s1{
                    std::rotr(schedule[word - 2], 17) ^ std::rotr(schedule[word - 2], 19) ^ (schedule[word - 2] >> 10)};

            schedule[word] = schedule[word - 16] + s0 + schedule[word - 7] + s1;
        }

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
    }

    std::string hex{};

    for (const auto word : state)
    {
        hex += std::format("{:08x}", word);
    }

    return hex;
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
 * @brief Returns the text a run of ordinary token ids spells under a definition.
 * @param definition The definition.
 * @param ids The ids.
 * @return The concatenated spellings.
 */
std::string spelled(const Definition& definition, const std::vector<Token_id_t>& ids)
{
    std::string text{};

    for (const auto id : ids)
    {
        text += definition.spellings.at(id);
    }

    return text;
}

/**
 * @brief Returns the refusal read_gpt2() makes of an encoder.json and a vocab.bpe, empty when it makes none.
 * @param encoder The bytes of encoder.json.
 * @param vocab The bytes of vocab.bpe.
 * @return The refusal's message.
 */
std::string gpt2_refusal(const std::string_view encoder, const std::string_view vocab)
{
    try
    {
        std::ignore = read_gpt2(written("encoder.json", encoder), written("vocab.bpe", vocab));
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
}

/**
 * @brief Returns the refusal read_tiktoken() makes of a rank file under a pattern, empty when it makes none.
 * @param text The bytes of the rank file.
 * @param pattern The pattern.
 * @return The refusal's message.
 */
std::string tiktoken_refusal(const std::string_view text, const Pattern pattern)
{
    try
    {
        std::ignore = read_tiktoken(written("ranks.tiktoken", text), pattern);
    }
    catch (const std::invalid_argument& refusal)
    {
        return refusal.what();
    }

    return {};
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

    EXPECT_EQ(sha256(contents(*encoder)), gpt2_encoder_digest);
    EXPECT_EQ(sha256(contents(*vocab)), gpt2_vocab_digest);

    return read_gpt2(*encoder, *vocab);
}

/**
 * @brief Returns cl100k_base's pinned rank file read, after checking it against its digest.
 * @return The definition, std::nullopt when the vocabulary directory does not hold the file.
 */
std::optional<Definition> pinned_cl100k()
{
    const auto path{pinned("cl100k_base.tiktoken")};

    if (!path)
    {
        return std::nullopt;
    }

    EXPECT_EQ(sha256(contents(*path)), cl100k_digest);

    return read_tiktoken(*path, Pattern::cl100k);
}

} // namespace

TEST(Definition_test, Read_gpt2_undoes_the_byte_map_in_the_spellings_and_the_merges)
{
    const auto encoder{written("encoder.json", R"({"a": 0, "Ġ": 1, "Ġa": 2, "\"": 3, "Ã©": 4})")};
    const auto vocab{written("vocab.bpe", "#version: 0.2\nĠ a\nÃ ©\n")};

    const auto [pattern, merge_rule, spellings, merges, specials]{read_gpt2(encoder, vocab)};

    EXPECT_EQ(pattern, Pattern::gpt2);
    EXPECT_EQ(merge_rule, Merge_rule::pairs);
    EXPECT_EQ(spellings, (std::vector<std::string>{"a", " ", " a", "\"", "\xC3\xA9"}));
    ASSERT_EQ(merges.size(), 2U);
    EXPECT_EQ(merges[0].left, " ");
    EXPECT_EQ(merges[0].right, "a");
    EXPECT_EQ(merges[1].left, "\xC3");
    EXPECT_EQ(merges[1].right, "\xA9");
    EXPECT_TRUE(specials.empty());
}

TEST(Definition_test, Read_gpt2_drops_the_text_after_the_last_newline_of_the_merges_as_encoder_py_does)
{
    const auto encoder{written("encoder.json", R"({"a": 0, "b": 1})")};
    const auto vocab{written("vocab.bpe", "#version: 0.2\na b\nb a")};

    const auto definition{read_gpt2(encoder, vocab)};

    ASSERT_EQ(definition.merges.size(), 1U);
    EXPECT_EQ(definition.merges[0].left, "a");
}

TEST(Definition_test, Read_gpt2_refuses_ids_that_are_not_dense_from_zero)
{
    const auto beyond{gpt2_refusal(R"({"a": 0, "b": 2})", "#version: 0.2\n")};
    const auto repeated{gpt2_refusal(R"({"a": 0, "b": 0})", "#version: 0.2\n")};

    EXPECT_TRUE(beyond.contains("are not dense from zero: 2 tokens, and the id 2")) << beyond;
    EXPECT_TRUE(repeated.contains("are not dense from zero: 2 tokens, and the id 0")) << repeated;
}

TEST(Definition_test, Read_gpt2_refuses_an_empty_token)
{
    const auto refusal{gpt2_refusal(R"({"a": 0, "": 1})", "#version: 0.2\n")};

    EXPECT_TRUE(refusal.contains("holds an empty token at byte 9")) << refusal;
}

TEST(Definition_test, Read_gpt2_refuses_a_character_the_byte_map_does_not_carry)
{
    const auto escaped{gpt2_refusal(R"({"ń": 0})", "#version: 0.2\n")};
    const auto raw{gpt2_refusal(R"({"a b": 0})", "#version: 0.2\n")};
    const auto merged{gpt2_refusal(R"({"a": 0})", "#version: 0.2\na \xC5\x84\n")};

    EXPECT_TRUE(escaped.contains("at byte 2 holds U+0144, which GPT-2's byte map does not carry")) << escaped;
    EXPECT_TRUE(raw.contains("at byte 3 holds U+0020")) << raw;
    EXPECT_TRUE(merged.contains("line 2 of")) << merged;
    EXPECT_TRUE(merged.contains("holds U+0144")) << merged;
}

TEST(Definition_test, Read_gpt2_refuses_a_merge_line_that_is_not_two_parts)
{
    const auto refusal{gpt2_refusal(R"({"a": 0})", "#version: 0.2\na a\na\n")};

    EXPECT_TRUE(refusal.contains("line 3 of")) << refusal;
    EXPECT_TRUE(refusal.contains("is not a merge, two parts separated by white space")) << refusal;
}

TEST(Definition_test, Read_gpt2_refuses_an_encoder_that_is_not_one_flat_object_of_spellings_and_ids)
{
    const auto colon{gpt2_refusal(R"({"a" 0})", "#version: 0.2\n")};
    const auto id{gpt2_refusal(R"({"a": "0"})", "#version: 0.2\n")};
    const auto escape{gpt2_refusal(R"({"\q": 0})", "#version: 0.2\n")};
    const auto trailing{gpt2_refusal(R"({"a": 0} {})", "#version: 0.2\n")};

    EXPECT_TRUE(colon.contains("has no colon at byte 5")) << colon;
    EXPECT_TRUE(id.contains("has no id at byte 6")) << id;
    EXPECT_TRUE(escape.contains("has no JSON escape at byte 3")) << escape;
    EXPECT_TRUE(trailing.contains("has no end of the object at byte 9")) << trailing;
}

TEST(Definition_test, Read_gpt2_reads_an_empty_object_and_a_version_line_alone_as_an_empty_definition)
{
    const auto definition{read_gpt2(written("encoder.json", " { } "), written("vocab.bpe", "#version: 0.2\n"))};

    EXPECT_TRUE(definition.spellings.empty());
    EXPECT_TRUE(definition.merges.empty());
}

TEST(Definition_test, The_readers_refuse_a_file_they_cannot_read)
{
    const auto missing{scratch() / "missing"};

    EXPECT_THROW(std::ignore = read_gpt2(missing, missing), std::runtime_error);
    EXPECT_THROW(std::ignore = read_tiktoken(missing, Pattern::cl100k), std::runtime_error);

    try
    {
        std::ignore = read_tiktoken(missing, Pattern::cl100k);
    }
    catch (const std::runtime_error& refusal)
    {
        EXPECT_EQ(std::string_view{refusal.what()}, std::format("cannot read {}", missing.string()));
    }
}

TEST(Definition_test, Read_tiktoken_reads_the_ranks_as_ids_and_supplies_the_vendor_s_special_tokens)
{
    const auto path{written("ranks.tiktoken", "Ig== 1\nIQ== 0\n\nIGE= 2\r\n")};

    const auto [pattern, merge_rule, spellings, merges, specials]{read_tiktoken(path, Pattern::cl100k)};

    EXPECT_EQ(pattern, Pattern::cl100k);
    EXPECT_EQ(merge_rule, Merge_rule::ranks);
    EXPECT_EQ(spellings, (std::vector<std::string>{"!", "\"", " a"}));
    EXPECT_TRUE(merges.empty());
    ASSERT_EQ(specials.size(), 5U);
    EXPECT_EQ(specials[0].spelling, "<|endoftext|>");
    EXPECT_EQ(specials[0].id, 100257U);
    EXPECT_EQ(specials[4].spelling, "<|endofprompt|>");
    EXPECT_EQ(specials[4].id, 100276U);

    const auto gpt2{read_tiktoken(path, Pattern::gpt2)};

    ASSERT_EQ(gpt2.specials.size(), 1U);
    EXPECT_EQ(gpt2.specials[0].spelling, "<|endoftext|>");
    EXPECT_EQ(gpt2.specials[0].id, 50256U);
}

TEST(Definition_test, Read_tiktoken_refuses_a_rank_file_read_under_no_pattern)
{
    const auto refusal{tiktoken_refusal("IQ== 0\n", Pattern::none)};

    EXPECT_TRUE(refusal.contains("is read under GPT-2's or cl100k's pattern, and the pattern given is none"))
            << refusal;
}

TEST(Definition_test, Read_tiktoken_refuses_a_line_that_is_not_a_base64_token_and_a_decimal_rank)
{
    for (const std::string_view line : {"IQ=", "I*== 1", "IQ==", "IQ== -1", "IQ== 1x", "IQ== 1 2", "==== 1"})
    {
        const auto refusal{tiktoken_refusal(std::format("Ig== 0\n{}\n", line), Pattern::cl100k)};

        EXPECT_TRUE(refusal.contains("line 2 of")) << line << ": " << refusal;
        EXPECT_TRUE(refusal.contains("is not a base64 token and a decimal rank")) << line << ": " << refusal;
    }
}

TEST(Definition_test, Read_tiktoken_refuses_ranks_that_are_not_dense_from_zero)
{
    const auto beyond{tiktoken_refusal("IQ== 0\nIg== 2\n", Pattern::cl100k)};
    const auto repeated{tiktoken_refusal("IQ== 0\nIg== 0\n", Pattern::cl100k)};

    EXPECT_TRUE(beyond.contains("are not dense from zero: 2 tokens, and the rank 2")) << beyond;
    EXPECT_TRUE(repeated.contains("are not dense from zero: 2 tokens, and the rank 0")) << repeated;
}

TEST(Definition_test, The_pinned_gpt2_files_read_as_its_vocabulary_and_spell_the_published_known_answer)
{
    const auto definition{pinned_gpt2()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding gpt2-vocab.json and gpt2-merges.txt";
    }

    EXPECT_EQ(definition->spellings.size(), 50257U);
    EXPECT_EQ(definition->merges.size(), 50000U);
    EXPECT_EQ(definition->spellings[50256], "<|endoftext|>");
    EXPECT_EQ(definition->spellings[220], " ");
    EXPECT_EQ(spelled(*definition, {15496, 995}), "Hello world");
}

TEST(Definition_test, The_pinned_cl100k_file_reads_as_its_vocabulary_and_spells_the_published_known_answer)
{
    const auto definition{pinned_cl100k()};

    if (!definition)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding cl100k_base.tiktoken";
    }

    EXPECT_EQ(definition->spellings.size(), 100256U);
    EXPECT_EQ(definition->specials.size(), 5U);
    EXPECT_EQ(spelled(*definition, {83, 1609, 5963, 374, 2294, 0}), "tiktoken is great!");
}

TEST(Definition_test, The_ids_of_every_known_answer_of_the_fixture_spell_its_text_under_the_pinned_vocabularies)
{
    const auto gpt2{pinned_gpt2()};
    const auto cl100k{pinned_cl100k()};

    if (!gpt2 || !cl100k)
    {
        GTEST_SKIP() << "MATISSE_VOCAB_DIR names no directory holding the GPT-2 and cl100k_base vocabularies";
    }

    std::ifstream file{std::string{SOURCE_DIR} + "/libs/load/tests/data/known_answers_reference.txt"};

    ASSERT_TRUE(file.is_open());

    std::optional<Definition> definition{};

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

            definition = name == "gpt2" ? gpt2 : cl100k;

            ++counts.universes;

            continue;
        }

        if (kind == "corpus")
        {
            ++counts.corpora;

            continue;
        }

        ASSERT_EQ(kind, "encode") << line;
        ASSERT_TRUE(definition) << line;

        std::string text{};

        fields >> text;

        std::vector<Token_id_t> ids{};

        for (Token_id_t id{}; fields >> id;)
        {
            ids.push_back(id);
        }

        EXPECT_EQ(spelled(*definition, ids), bytes_of(text)) << line;

        ++counts.encodings;
    }

    EXPECT_EQ(counts.universes, 2U);
    EXPECT_EQ(counts.encodings, 30U);
    EXPECT_EQ(counts.corpora, 6U);
}
