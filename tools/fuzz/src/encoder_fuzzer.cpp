#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/encode/encoder.hpp"
#include "matisse/gpt2/tokenizer.hpp"
#include "matisse/llama3/tokenizer.hpp"
#include "matisse/load/definition.hpp"
#include "matisse/o200k/tokenizer.hpp"
#include "matisse/pretok/pattern.hpp"
#include "matisse/tools/fuzz/harness.hpp"
#include "matisse/validate/report.hpp"

namespace
{
using matisse::encode::Encoder;
using matisse::load::Definition;
using matisse::load::Merge_rule;
using matisse::load::Special_token;
using matisse::tools::fuzz::require;
using matisse::validate::Inventory;

/**
 * @brief A tokenizer whose pattern the harness encodes under: its pattern and the vendor's special tokens, all of them
 *        allowed.
 */
struct Tokenizer
{
    /**
     * @brief The pattern.
     */
    matisse::pretok::Pattern pattern{};

    /**
     * @brief The vendor's special tokens.
     */
    std::vector<Special_token> specials{};
};

/**
 * @brief The widest window every report is decided to, three, as o200k's tests decide its: under the sanitizers the
 *        four reports take about a minute with GPT-2's, cl100k's and Llama 3's at the default of five, which a bounded
 *        run would spend before its first input, and seconds at three, which certifies cuts in every pattern; the
 *        claim holds at every width.
 */
constexpr std::size_t decided_width{3};

/**
 * @brief The bytes every multi-byte token of the synthetic table is spelled over: two letters, the space, the line feed
 *        and the two bytes of U+00E9, a letter outside ASCII.
 */
constexpr std::string_view token_bytes{"ab \n\xC3\xA9"};

/**
 * @brief The tokens of the synthetic table that its merges never form, since no pair of their bytes is a token, so
 *        that only the whole-piece lookup gives them: an ASCII word, a word after a space and two CJK ideographs.
 */
constexpr std::array<std::string_view, 3> unformed{"hello", " world", "\xE6\x97\xA5\xE6\x9C\xAC"};

/**
 * @brief The seed of the order the synthetic table ranks its multi-byte tokens in.
 */
constexpr std::mt19937::result_type rank_seed{17};

/**
 * @brief The bytes of U+FFFD, the replacement character.
 */
constexpr std::string_view replacement{"\xEF\xBF\xBD"};

/**
 * @brief The UTF-8 sequence a lead byte begins, as the Unicode Standard's table of well-formed byte sequences gives it.
 */
struct Sequence
{
    /**
     * @brief The sequence's length in bytes, zero when the byte begins none.
     */
    std::size_t length{};

    /**
     * @brief The least second byte the sequence takes.
     */
    unsigned char low{0x80};

    /**
     * @brief The greatest second byte the sequence takes.
     */
    unsigned char high{0xBF};
};

/**
 * @brief Returns the sequence a lead byte begins.
 * @param lead The byte.
 * @return Its length and the range of its second byte.
 */
Sequence sequence_of(const unsigned char lead)
{
    if (lead < 0x80)
    {
        return {.length = 1};
    }

    if (lead >= 0xC2 && lead <= 0xDF)
    {
        return {.length = 2};
    }

    if (lead == 0xE0)
    {
        return {.length = 3, .low = 0xA0};
    }

    if (lead == 0xED)
    {
        return {.length = 3, .high = 0x9F};
    }

    if (lead >= 0xE1 && lead <= 0xEF)
    {
        return {.length = 3};
    }

    if (lead == 0xF0)
    {
        return {.length = 4, .low = 0x90};
    }

    if (lead == 0xF4)
    {
        return {.length = 4, .high = 0x8F};
    }

    if (lead >= 0xF1 && lead <= 0xF3)
    {
        return {.length = 4};
    }

    return {};
}

/**
 * @brief Returns bytes made well-formed UTF-8: every well-formed sequence kept and every maximal subpart of an
 *        ill-formed one replaced by U+FFFD, as the Unicode Standard recommends, so that every input is a text the
 *        encoder takes and a text that is already well-formed is itself.
 * @param bytes The bytes.
 * @return The text.
 */
std::string well_formed(const std::string_view bytes)
{
    std::string text{};

    for (std::size_t at{0}; at < bytes.size();)
    {
        const auto [length, low, high]{sequence_of(static_cast<unsigned char>(bytes[at]))};

        std::size_t taken{1};

        while (taken < length && at + taken < bytes.size())
        {
            const auto next{static_cast<unsigned char>(bytes[at + taken])};

            const auto least{taken == 1 ? low : static_cast<unsigned char>(0x80)};

            const auto greatest{taken == 1 ? high : static_cast<unsigned char>(0xBF)};

            if (next < least || next > greatest)
            {
                break;
            }

            ++taken;
        }

        text += length != 0 && taken == length ? bytes.substr(at, length) : replacement;

        at += taken;
    }

    return text;
}

/**
 * @brief Returns the synthetic definition under a merge rule: the single bytes, then every string of two and three
 *        bytes of token_bytes ranked in a seeded random order, then the unformed tokens, with the vendor's special
 *        tokens; under the pair rule each multi-byte string of token_bytes is formed by the merge of all of it but its
 *        last byte with that byte, at its rank, and under the rank rule a piece is looked up whole first, as tiktoken
 *        looks it up.
 * @param merge_rule The merge rule.
 * @param specials The vendor's special tokens.
 * @return The definition.
 */
Definition synthetic(const Merge_rule merge_rule, std::vector<Special_token> specials)
{
    std::vector<std::string> merged{};

    for (const auto first : token_bytes)
    {
        for (const auto second : token_bytes)
        {
            merged.push_back(std::string{first} + second);

            for (const auto third : token_bytes)
            {
                merged.push_back(std::string{first} + second + third);
            }
        }
    }

    std::mt19937 random{rank_seed};

    std::ranges::shuffle(merged, random);

    std::vector<std::string> spellings{};

    for (unsigned value{0}; value < 256; ++value)
    {
        spellings.emplace_back(1, static_cast<char>(value));
    }

    spellings.insert(spellings.end(), merged.begin(), merged.end());

    spellings.insert(spellings.end(), unformed.begin(), unformed.end());

    std::vector<munch::verifier::Merge> merges{};

    if (merge_rule == Merge_rule::pairs)
    {
        for (const auto& spelling : merged)
        {
            merges.push_back(
                    {.left = spelling.substr(0, spelling.size() - 1), .right = spelling.substr(spelling.size() - 1)});
        }
    }

    return {.merge_rule = merge_rule,
            .whole_piece_lookup = merge_rule == Merge_rule::ranks,
            .spellings = std::move(spellings),
            .merges = std::move(merges),
            .specials = std::move(specials)};
}

/**
 * @brief Returns the encoders the harness draws from: for each of GPT-2's, cl100k's, o200k's and Llama 3's patterns,
 *        its report decided once, under each inventory and the synthetic definition of each merge rule, every vendor's
 *        special token allowed.
 * @return The sixteen encoders, the tokenizer varying fastest, then the inventory, then the merge rule.
 */
std::vector<Encoder> built_encoders()
{
    const std::array<Tokenizer, 4> tokenizers{{
            {.pattern = matisse::gpt2::pattern(), .specials = matisse::gpt2::specials()},
            {.pattern = matisse::cl100k::pattern(), .specials = matisse::cl100k::specials()},
            {.pattern = matisse::o200k::pattern(), .specials = matisse::o200k::specials()},
            {.pattern = matisse::llama3::pattern(), .specials = matisse::llama3::specials()},
    }};

    std::vector<matisse::validate::Report> reports{};

    for (const auto& tokenizer : tokenizers)
    {
        reports.push_back(matisse::validate::validate(tokenizer.pattern, decided_width));
    }

    std::vector<Encoder> built{};

    for (const auto merge_rule : {Merge_rule::ranks, Merge_rule::pairs})
    {
        for (const auto inventory : {Inventory::factorization, Inventory::policy})
        {
            for (std::size_t index{0}; index < tokenizers.size(); ++index)
            {
                const auto& specials{tokenizers[index].specials};

                std::vector<std::string> allowed{};

                for (const auto& [spelling, id] : specials)
                {
                    allowed.push_back(spelling);
                }

                built.emplace_back(synthetic(merge_rule, specials), reports[index], inventory, allowed);
            }
        }
    }

    return built;
}

/**
 * @brief Returns the encoders the harness draws from, built on the first call.
 * @return The encoders.
 */
const std::vector<Encoder>& encoders()
{
    static const auto built{built_encoders()};

    return built;
}

} // namespace

/**
 * @brief Builds the encoders before the first input, the fuzz initialization run once per process, so that the seconds
 *        their reports take are not counted against the first input's timeout.
 * @param argc The count of the command line's arguments, which the harness does not read.
 * @param argv The command line's arguments, which the harness does not read.
 * @return Zero.
 */
extern "C" int LLVMFuzzerInitialize([[maybe_unused]] int* const argc, [[maybe_unused]] char*** const argv)
{
    static_cast<void>(encoders());

    return 0;
}

/**
 * @brief Encodes the fuzz input both ways and holds the parallel encoding to the serial one, the fuzz entry point run
 *        once per input.
 *
 * The first byte chooses the encoder, the pattern, the inventory and the merge rule of a synthetic definition, the
 * second the chunk count, any from zero, which behaves as one, to 255, and the rest made well-formed UTF-8 is the text.
 * The encoder's one claim is the property: encode_parallel() returns the ids encode() returns, at the chunk count
 * drawn, and decode() spells the text back from them. The encoders are built once, before the first input, and no
 * vocabulary file is read: the definition is built here, the bytes first so that every text is encoded. Anything that
 * escapes, a violated property and any sanitizer report is a finding.
 * @param data The fuzz input.
 * @param size The input's size.
 * @return Zero; a violated invariant traps instead.
 */
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* const data, const std::size_t size)
{
    const auto& built{encoders()};

    const std::span input{data, size};

    const auto& encoder{built[input.empty() ? 0 : input[0] % built.size()]};

    const std::size_t chunks{input.size() < 2 ? 0UZ : input[1]};

    const auto rest{input.subspan(std::min<std::size_t>(input.size(), 2))};

    const auto text{well_formed({reinterpret_cast<const char*>(rest.data()), rest.size()})};

    const auto ids{encoder.encode(text)};

    require(encoder.encode_parallel(text, chunks) == ids);

    require(encoder.decode(ids) == text);

    return 0;
}
