#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "matisse/load/definition.hpp"

namespace matisse::load
{
namespace
{
/**
 * @brief The white space a line's token and rank are separated by, the ASCII white space Python's bytes.split() splits
 *        at.
 */
constexpr std::string_view separators{" \t\n\v\f\r"};

/**
 * @brief The digits of base64, the value of each its index.
 */
constexpr std::string_view base64_digits{"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};

/**
 * @brief Returns the whole of a file.
 * @param path The file's path.
 * @return Its bytes.
 * @throws std::runtime_error If it cannot be opened.
 */
[[nodiscard]] std::string contents(const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};

    if (!stream)
    {
        throw std::runtime_error{std::format("cannot read {}", path.string())};
    }

    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

/**
 * @brief Returns the vendor's special tokens of a pattern, quoted from tiktoken's openai_public.py: those of the gpt2
 *        and r50k_base encodings for GPT-2's pattern and those of cl100k_base for cl100k's.
 * @param pattern The pattern, GPT-2's or cl100k's.
 * @return The special tokens, by id ascending.
 */
[[nodiscard]] std::vector<Special_token> specials_of(const Pattern pattern)
{
    if (pattern == Pattern::gpt2)
    {
        return {{.spelling = "<|endoftext|>", .id = 50256}};
    }

    return {{.spelling = "<|endoftext|>", .id = 100257},
            {.spelling = "<|fim_prefix|>", .id = 100258},
            {.spelling = "<|fim_middle|>", .id = 100259},
            {.spelling = "<|fim_suffix|>", .id = 100260},
            {.spelling = "<|endofprompt|>", .id = 100276}};
}

/**
 * @brief Returns the bytes a base64 field spells, four digits for every three bytes, the last group padded with `=`.
 * @param field The field.
 * @return The bytes, std::nullopt when the field is not base64.
 */
[[nodiscard]] std::optional<std::string> decoded(const std::string_view field)
{
    constexpr std::size_t group{4};

    const auto padding{field.ends_with("==") ? 2UZ : field.ends_with('=') ? 1UZ : 0UZ};

    const auto digits{field.substr(0, field.size() - padding)};

    if (field.size() % group != 0 || digits.find_first_not_of(base64_digits) != std::string_view::npos)
    {
        return std::nullopt;
    }

    std::string bytes{};

    std::uint32_t bits{0};

    std::size_t held{0};

    for (const auto digit : digits)
    {
        const auto value{static_cast<std::uint32_t>(base64_digits.find(digit))};

        bits = (bits << 6U) | value;

        held += 6;

        if (held < 8)
        {
            continue;
        }

        held -= 8;

        bytes += static_cast<char>((bits >> held) & 0xFFU);
    }

    return bytes;
}

} // namespace

Definition read_tiktoken(const std::filesystem::path& path, const Pattern pattern)
{
    const auto name{path.string()};

    if (pattern == Pattern::none)
    {
        throw std::invalid_argument{std::format(
                "read_tiktoken: {} is read under GPT-2's or cl100k's pattern, and the pattern given is none", name)};
    }

    const auto text{contents(path)};

    std::vector<std::pair<std::string, std::size_t>> entries{};

    std::size_t number{0};

    for (std::size_t start{0}; start < text.size();)
    {
        const auto end{std::min(text.find('\n', start), text.size())};

        const std::string_view line{text.data() + start, end - start};

        start = end + 1;

        ++number;

        const auto token_start{line.find_first_not_of(separators)};

        if (token_start == std::string_view::npos)
        {
            continue;
        }

        const auto token_end{std::min(line.find_first_of(separators, token_start), line.size())};

        const auto rank_start{std::min(line.find_first_not_of(separators, token_end), line.size())};

        const auto rank_end{std::min(line.find_first_of(separators, rank_start), line.size())};

        const auto rest{line.find_first_not_of(separators, rank_end)};

        const auto rank_field{line.substr(rank_start, rank_end - rank_start)};

        auto bytes{decoded(line.substr(token_start, token_end - token_start))};

        std::size_t rank{};

        const auto [rank_stop, error]{std::from_chars(rank_field.data(), rank_field.data() + rank_field.size(), rank)};

        const auto whole_rank{error == std::errc{} && rank_stop == rank_field.data() + rank_field.size()};

        if (!bytes || rank_field.empty() || !whole_rank || rest != std::string_view::npos)
        {
            throw std::invalid_argument{
                    std::format("read_tiktoken: line {} of {} is not a base64 token and a decimal rank", number, name)};
        }

        entries.emplace_back(std::move(*bytes), rank);
    }

    std::vector<std::string> spellings(entries.size());

    std::vector<bool> placed(entries.size());

    for (auto& [bytes, rank] : entries)
    {
        if (rank >= entries.size() || placed[rank])
        {
            throw std::invalid_argument{std::format(
                    "read_tiktoken: the ranks of {} are not dense from zero: {} tokens, and the rank {} is beyond them "
                    "or repeats",
                    name, entries.size(), rank)};
        }

        placed[rank] = true;

        spellings[rank] = std::move(bytes);
    }

    return {.pattern = pattern,
            .merge_rule = Merge_rule::ranks,
            .spellings = std::move(spellings),
            .merges = {},
            .specials = specials_of(pattern)};
}

} // namespace matisse::load
