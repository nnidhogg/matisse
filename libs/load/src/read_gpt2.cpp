#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
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
 * @brief One past the last character of GPT-2's byte map: the 188 printable bytes stand for themselves and the other 68
 *        bytes, in order, for U+0100 to U+0143.
 */
constexpr char32_t map_end{0x144};

/**
 * @brief The byte each character of GPT-2's byte map stands for, by code point, std::nullopt where the map carries no
 *        character.
 */
using Byte_map_t = std::array<std::optional<unsigned char>, map_end>;

/**
 * @brief The white space read between encoder.json's tokens and between a merge line's two parts, the ASCII white space
 *        Python's str.split() splits at; no character of the byte map is white space.
 */
constexpr std::string_view separators{" \t\n\v\f\r"};

/**
 * @brief Returns whether a byte stands for itself in GPT-2's byte map: the printable bytes `!` to `~`, `¡` to `¬` and
 *        `®` to `ÿ`, as encoder.py's bytes_to_unicode() lists them.
 * @param byte The byte.
 * @return True when the byte's character is the byte itself.
 */
[[nodiscard]] constexpr bool is_printable(const unsigned byte) noexcept
{
    return (byte >= 0x21U && byte <= 0x7EU) || (byte >= 0xA1U && byte <= 0xACU) || (byte >= 0xAEU && byte <= 0xFFU);
}

/**
 * @brief Returns GPT-2's byte map inverted: encoder.py's bytes_to_unicode(), which gives every printable byte its own
 *        character and the others, in byte order, the characters from U+0100 on, read from character to byte.
 * @return The byte of every character the map carries.
 */
[[nodiscard]] constexpr Byte_map_t byte_map_of()
{
    constexpr unsigned byte_count{256};

    Byte_map_t map{};

    char32_t next{0x100};

    for (unsigned byte{0}; byte < byte_count; ++byte)
    {
        const auto value{static_cast<unsigned char>(byte)};

        if (is_printable(byte))
        {
            map[byte] = value;

            continue;
        }

        map[next] = value;

        ++next;
    }

    return map;
}

/**
 * @brief GPT-2's byte map inverted, by code point.
 */
constexpr Byte_map_t byte_map{byte_map_of()};

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
 * @brief Returns the code point of the UTF-8 sequence at an index and its length, as its lead byte's high bits give
 *        them; a character outside the byte map is refused by its caller, so no further check is made here.
 * @param text The text.
 * @param at The index of the lead byte, within the text.
 * @return The code point and the number of bytes the sequence takes, at most the bytes left.
 */
[[nodiscard]] std::pair<char32_t, std::size_t> character_at(const std::string_view text, const std::size_t at) noexcept
{
    const auto lead{static_cast<unsigned char>(text[at])};

    const auto length{lead < 0x80U ? 1UZ : lead >= 0xF0U ? 4UZ : lead >= 0xE0U ? 3UZ : 2UZ};

    const auto available{std::min(length, text.size() - at)};

    char32_t character{length == 1 ? lead : lead & (0xFFU >> (length + 1))};

    for (std::size_t index{1}; index < available; ++index)
    {
        const auto continuation{static_cast<unsigned char>(text[at + index])};

        character = (character << 6U) | (continuation & 0x3FU);
    }

    return {character, available};
}

/**
 * @brief Returns the byte a character of GPT-2's byte map stands for.
 * @param character The character.
 * @return The byte, std::nullopt when the map carries no such character.
 */
[[nodiscard]] std::optional<char> byte_of(const char32_t character) noexcept
{
    if (character >= map_end || !byte_map[character])
    {
        return std::nullopt;
    }

    return static_cast<char>(*byte_map[character]);
}

/**
 * @brief Returns the refusal of a character the byte map does not carry.
 * @param character The character.
 * @param where Where it stands: the file and the byte offset, or the line.
 * @return The refusal.
 */
[[nodiscard]] std::invalid_argument unmapped(const char32_t character, const std::string_view where)
{
    return std::invalid_argument{std::format(
            "read_gpt2: {} holds U+{:04X}, which GPT-2's byte map does not carry", where,
            static_cast<std::uint32_t>(character))};
}

/**
 * @brief Returns the spellings of encoder.json by id, its one flat object read with JSON's escapes, each spelling's
 *        byte map undone.
 * @param text The file's bytes.
 * @param name The file's path, for the refusals.
 * @return The bytes of every token, by id.
 * @throws std::invalid_argument If the text is not one object of quoted spellings and decimal ids, a spelling is empty
 *         or holds a character the byte map does not carry, or the ids are not dense from zero.
 */
[[nodiscard]] std::vector<std::string> spellings_of(const std::string_view text, const std::string& name)
{
    std::size_t at{0};

    const auto refuse{[&](const std::string_view expected) {
        throw std::invalid_argument{std::format("read_gpt2: {} has no {} at byte {}", name, expected, at)};
    }};

    const auto skip_space{[&] {
        while (at < text.size() && separators.contains(text[at]))
        {
            ++at;
        }
    }};

    const auto expect{[&](const char symbol, const std::string_view expected) {
        skip_space();

        if (at == text.size() || text[at] != symbol)
        {
            refuse(expected);
        }

        ++at;
    }};

    const auto consumed{[&](const char symbol) {
        skip_space();

        const auto found{at < text.size() && text[at] == symbol};

        at += found ? 1 : 0;

        return found;
    }};

    const auto escaped{[&]() -> char32_t {
        constexpr std::string_view simple{R"("\/bfnrt)"};
        constexpr std::string_view meant{"\"\\/\b\f\n\r\t"};

        const auto escape{at < text.size() ? text[at] : '\0'};

        if (const auto found{simple.find(escape)}; found != std::string_view::npos)
        {
            ++at;

            return static_cast<unsigned char>(meant[found]);
        }

        constexpr std::size_t hex_digits{4};

        const auto digits{escape == 'u' && at + 1 + hex_digits <= text.size() ? text.substr(at + 1, hex_digits) : ""};

        std::uint32_t value{};

        const auto [end, error]{std::from_chars(digits.data(), digits.data() + digits.size(), value, 16)};

        if (digits.empty() || error != std::errc{} || end != digits.data() + digits.size())
        {
            refuse("JSON escape");
        }

        at += 1 + hex_digits;

        return value;
    }};

    const auto spelling{[&] {
        expect('"', "quoted spelling");

        // The token is named by the offset of its opening quote.
        const auto start{at - 1};

        std::string bytes{};

        while (at < text.size() && text[at] != '"')
        {
            const auto character_start{at};

            const auto escape{text[at] == '\\'};

            const auto [character, length]{escape ? std::pair{char32_t{}, 1UZ} : character_at(text, at)};

            at += length;

            const auto code_point{escape ? escaped() : character};

            const auto byte{byte_of(code_point)};

            if (!byte)
            {
                throw unmapped(code_point, std::format("{} at byte {}", name, character_start));
            }

            bytes += *byte;
        }

        expect('"', "closing quote");

        if (bytes.empty())
        {
            throw std::invalid_argument{std::format("read_gpt2: {} holds an empty token at byte {}", name, start)};
        }

        return bytes;
    }};

    const auto id{[&] {
        skip_space();

        std::uint64_t value{};

        const auto [end, error]{std::from_chars(text.data() + at, text.data() + text.size(), value)};

        if (error != std::errc{} || value > std::numeric_limits<Token_id_t>::max())
        {
            refuse("id");
        }

        at = static_cast<std::size_t>(end - text.data());

        return static_cast<Token_id_t>(value);
    }};

    std::vector<std::pair<std::string, Token_id_t>> entries{};

    expect('{', "opening brace");

    if (!consumed('}'))
    {
        do
        {
            auto bytes{spelling()};

            expect(':', "colon");

            const auto token{id()};

            entries.emplace_back(std::move(bytes), token);
        } while (consumed(','));

        expect('}', "closing brace");
    }

    skip_space();

    if (at != text.size())
    {
        refuse("end of the object");
    }

    std::vector<std::string> spellings(entries.size());

    std::vector<bool> placed(entries.size());

    for (auto& [bytes, token] : entries)
    {
        if (token >= entries.size() || placed[token])
        {
            throw std::invalid_argument{std::format(
                    "read_gpt2: the ids of {} are not dense from zero: {} tokens, and the id {} is beyond them or "
                    "repeats",
                    name, entries.size(), token)};
        }

        placed[token] = true;

        spellings[token] = std::move(bytes);
    }

    return spellings;
}

/**
 * @brief Returns the merge table of vocab.bpe, as encoder.py reads it: the lines between the first newline and the
 *        last, each two parts separated by white space, each part's byte map undone.
 * @param text The file's bytes.
 * @param name The file's path, for the refusals.
 * @return The merges in rank order.
 * @throws std::invalid_argument If a line is not two parts, or a part holds a character the byte map does not carry.
 */
[[nodiscard]] std::vector<munch::verifier::Merge> merges_of(const std::string_view text, const std::string& name)
{
    std::vector<munch::verifier::Merge> merges{};

    auto start{text.find('\n')};

    for (std::size_t number{2}; start != std::string_view::npos; ++number)
    {
        const auto end{text.find('\n', start + 1)};

        if (end == std::string_view::npos)
        {
            break;
        }

        const auto line{text.substr(start + 1, end - start - 1)};

        start = end;

        const auto where{std::format("line {} of {}", number, name)};

        std::vector<std::string> parts{};

        for (std::size_t at{line.find_first_not_of(separators)}; at != std::string_view::npos;)
        {
            const auto stop{std::min(line.find_first_of(separators, at), line.size())};

            const auto field{line.substr(at, stop - at)};

            std::string bytes{};

            for (std::size_t index{0}; index < field.size();)
            {
                const auto [character, length]{character_at(field, index)};

                const auto byte{byte_of(character)};

                if (!byte)
                {
                    throw unmapped(character, where);
                }

                bytes += *byte;

                index += length;
            }

            parts.push_back(std::move(bytes));

            at = line.find_first_not_of(separators, stop);
        }

        if (parts.size() != 2)
        {
            throw std::invalid_argument{
                    std::format("read_gpt2: {} is not a merge, two parts separated by white space", where)};
        }

        merges.push_back({.left = std::move(parts.front()), .right = std::move(parts.back())});
    }

    return merges;
}

} // namespace

Definition read_gpt2(const std::filesystem::path& encoder, const std::filesystem::path& vocab)
{
    const auto encoder_text{contents(encoder)};

    const auto vocab_text{contents(vocab)};

    auto spellings{spellings_of(encoder_text, encoder.string())};

    auto merges{merges_of(vocab_text, vocab.string())};

    return {.pattern = Pattern::gpt2,
            .merge_rule = Merge_rule::pairs,
            .spellings = std::move(spellings),
            .merges = std::move(merges),
            .specials = {}};
}

} // namespace matisse::load
