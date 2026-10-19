#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
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
 * @brief The lines of a rank file as read, each token's bytes with its rank, in the file's order.
 */
using Entries_t = std::vector<std::pair<std::string, std::size_t>>;

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
 * @brief Returns the bytes a base64 field spells, four digits for every three bytes, the last group padded with `=`.
 * @param field The field.
 * @return The bytes, std::nullopt when the field is not base64.
 */
[[nodiscard]] std::optional<std::string> decoded(const std::string_view field)
{
    const auto padding{field.ends_with("==") ? 2UZ : field.ends_with('=') ? 1UZ : 0UZ};

    const auto digits{field.substr(0, field.size() - padding)};

    constexpr std::size_t group{4};

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

/**
 * @brief Returns the rank a decimal field spells.
 * @param field The field.
 * @return The rank, std::nullopt when the field is empty or not decimal digits alone.
 */
[[nodiscard]] std::optional<std::size_t> rank_of(const std::string_view field) noexcept
{
    std::size_t rank{};

    const auto field_end{field.data() + field.size()};

    const auto [stop, error]{std::from_chars(field.data(), field_end, rank)};

    if (error != std::errc{} || stop != field_end)
    {
        return std::nullopt;
    }

    return rank;
}

/**
 * @brief Returns the token and the rank of a line of a rank file, its two fields separated by white space.
 * @param line The line, without its line feed.
 * @param number The line's number, from one, for the refusal.
 * @param name The file's path, for the refusal.
 * @return The token's bytes and its rank, std::nullopt for a line of white space alone, which tiktoken skips.
 * @throws std::invalid_argument If the line is not a base64 token and a decimal rank.
 */
[[nodiscard]] std::optional<std::pair<std::string, std::size_t>> entry_of(
        const std::string_view line, const std::size_t number, const std::string_view name)
{
    const auto token_start{line.find_first_not_of(separators)};

    if (token_start == std::string_view::npos)
    {
        return std::nullopt;
    }

    const auto token_end{std::min(line.find_first_of(separators, token_start), line.size())};

    const auto rank_start{std::min(line.find_first_not_of(separators, token_end), line.size())};

    const auto rank_end{std::min(line.find_first_of(separators, rank_start), line.size())};

    const auto rest{line.find_first_not_of(separators, rank_end)};

    const auto token_field{line.substr(token_start, token_end - token_start)};

    const auto rank_field{line.substr(rank_start, rank_end - rank_start)};

    auto bytes{decoded(token_field)};

    const auto rank{rank_of(rank_field)};

    if (!bytes || !rank || rest != std::string_view::npos)
    {
        throw std::invalid_argument{
                std::format("read_tiktoken: line {} of {} is not a base64 token and a decimal rank", number, name)};
    }

    return std::pair{std::move(*bytes), *rank};
}

/**
 * @brief Returns the entries of a rank file, its lines read in order and the lines of white space alone skipped.
 * @param text The file's bytes.
 * @param name The file's path, for the refusals.
 * @return Every token's bytes with its rank, in the file's order.
 * @throws std::invalid_argument If a line is not a base64 token and a decimal rank.
 */
[[nodiscard]] Entries_t entries_of(const std::string_view text, const std::string_view name)
{
    Entries_t entries{};

    std::size_t number{0};

    for (std::size_t start{0}; start < text.size();)
    {
        const auto end{std::min(text.find('\n', start), text.size())};

        const auto line{text.substr(start, end - start)};

        start = end + 1;

        ++number;

        auto entry{entry_of(line, number, name)};

        if (!entry)
        {
            continue;
        }

        entries.push_back(std::move(*entry));
    }

    return entries;
}

/**
 * @brief Returns the spellings of a rank file's entries by rank, the rank of a token being its id.
 * @param entries Every token's bytes with its rank.
 * @param name The file's path, for the refusal.
 * @return The bytes of every token, by id.
 * @throws std::invalid_argument If the ranks are not dense from zero.
 */
[[nodiscard]] std::vector<std::string> spellings_by_rank(Entries_t entries, const std::string_view name)
{
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

    return spellings;
}

} // namespace

Definition read_tiktoken(const std::filesystem::path& path, const std::span<const Special_token> specials)
{
    const auto name{path.string()};

    const auto text{contents(path)};

    auto entries{entries_of(text, name)};

    auto spellings{spellings_by_rank(std::move(entries), name)};

    std::vector<Special_token> vendor_specials(specials.begin(), specials.end());

    return {.merge_rule = Merge_rule::ranks,
            .spellings = std::move(spellings),
            .merges = {},
            .specials = std::move(vendor_specials)};
}

} // namespace matisse::load
