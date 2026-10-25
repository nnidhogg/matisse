#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
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
 * @brief The line of vocab.bpe that holds the merge of rank zero, after the version line.
 */
constexpr std::size_t first_merge_line{2};

/**
 * @brief The members of encoder.json as read, each spelling's bytes with its id, in the file's order.
 */
using Entries_t = std::vector<std::pair<std::string, Token_id_t>>;

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

    char32_t character{static_cast<char32_t>(length == 1 ? lead : lead & (0xFFU >> (length + 1)))};

    for (std::size_t index{1}; index < available; ++index)
    {
        const auto continuation{static_cast<unsigned char>(text[at + index])};

        character = (character << 6U) | static_cast<char32_t>(continuation & 0x3FU);
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
 * @brief A reader over encoder.json, the one flat JSON object of GPT-2's spellings and ids, read with JSON's escapes
 *        and each spelling's byte map undone.
 *
 * A refusal names the file and the byte offset where the reading stopped: what the object's syntax expected there, the
 * opening quote of an empty token, or the start of a character the byte map does not carry.
 */
class Encoder_reader
{
public:
    /**
     * @brief Binds the reader to a file's bytes.
     * @param text The file's bytes.
     * @param name The file's path, for the refusals.
     */
    Encoder_reader(std::string_view text, std::string_view name) noexcept;

    /**
     * @brief Reads the whole object, its spellings placed by id.
     * @return The bytes of every token, by id.
     * @throws std::invalid_argument If the text is not one object of quoted spellings and decimal ids, a spelling is
     *         empty or holds a character the byte map does not carry, or the ids are not dense from zero.
     */
    [[nodiscard]] std::vector<std::string> read();

private:
    /**
     * @brief Consumes a symbol after white space or refuses, naming what the syntax expected.
     * @param symbol The symbol required.
     * @param expected What the syntax expected, for the refusal.
     * @throws std::invalid_argument If the symbol does not stand there.
     */
    void expect(char symbol, std::string_view expected);

    /**
     * @brief Skips the white space at the current position.
     */
    void skip_space() noexcept;

    /**
     * @brief Refuses the file at the current position.
     * @param expected What the syntax expected there.
     * @throws std::invalid_argument Always.
     */
    [[noreturn]] void fail(std::string_view expected) const;

    /**
     * @brief Consumes a symbol after white space when it stands there.
     * @param symbol The symbol.
     * @return True when it stood there and was consumed.
     */
    [[nodiscard]] bool accept(char symbol) noexcept;

    /**
     * @brief Reads the object's members up to its closing brace: quoted spellings and their ids, separated by commas.
     * @return Every spelling with its id, in the file's order.
     * @throws std::invalid_argument If a member is not a quoted spelling, a colon and an id, a spelling is empty or
     *         holds a character the byte map does not carry, or the members are not closed by a brace.
     */
    [[nodiscard]] Entries_t members();

    /**
     * @brief Reads a quoted spelling, each character's byte map undone.
     * @return The spelling's bytes, at least one.
     * @throws std::invalid_argument If no quote opens or closes it, an escape is not JSON's, a character is one the
     *         byte map does not carry, or it is empty.
     */
    [[nodiscard]] std::string spelling();

    /**
     * @brief Reads one character of a spelling, raw as UTF-8 or as a JSON escape.
     * @return The character's code point.
     * @throws std::invalid_argument If an escape is not JSON's.
     */
    [[nodiscard]] char32_t character();

    /**
     * @brief Reads a JSON escape after its backslash.
     * @return The code point the escape stands for.
     * @throws std::invalid_argument If the escape is not JSON's.
     */
    [[nodiscard]] char32_t escaped();

    /**
     * @brief Reads a `\uXXXX` escape after its backslash, four hexadecimal digits after the `u`.
     * @return The code point the escape stands for.
     * @throws std::invalid_argument If the escape is not `u` and four hexadecimal digits.
     */
    [[nodiscard]] char32_t hex_escape();

    /**
     * @brief Reads a decimal id after white space.
     * @return The id.
     * @throws std::invalid_argument If no decimal number stands there or it is beyond the 32 bits of an id.
     */
    [[nodiscard]] Token_id_t id();

    /**
     * @brief Returns the spellings placed by id.
     * @param entries Every spelling with its id.
     * @return The bytes of every token, by id.
     * @throws std::invalid_argument If the ids are not dense from zero.
     */
    [[nodiscard]] std::vector<std::string> by_id(Entries_t entries) const;

    /**
     * @brief The file's bytes.
     */
    std::string_view text_;

    /**
     * @brief The file's path, for the refusals.
     */
    std::string_view name_;

    /**
     * @brief The current position, a byte offset into the text.
     */
    std::size_t at_{0};
};

Encoder_reader::Encoder_reader(const std::string_view text, const std::string_view name) noexcept
    : text_{text}, name_{name}
{}

std::vector<std::string> Encoder_reader::read()
{
    expect('{', "opening brace");

    auto entries{accept('}') ? Entries_t{} : members()};

    skip_space();

    if (at_ != text_.size())
    {
        fail("end of the object");
    }

    return by_id(std::move(entries));
}

void Encoder_reader::expect(const char symbol, const std::string_view expected)
{
    skip_space();

    if (at_ == text_.size() || text_[at_] != symbol)
    {
        fail(expected);
    }

    ++at_;
}

void Encoder_reader::skip_space() noexcept
{
    while (at_ < text_.size() && separators.contains(text_[at_]))
    {
        ++at_;
    }
}

void Encoder_reader::fail(const std::string_view expected) const
{
    throw std::invalid_argument{std::format("read_gpt2: {} has no {} at byte {}", name_, expected, at_)};
}

bool Encoder_reader::accept(const char symbol) noexcept
{
    skip_space();

    const auto found{at_ < text_.size() && text_[at_] == symbol};

    at_ += found ? 1 : 0;

    return found;
}

Entries_t Encoder_reader::members()
{
    Entries_t entries{};

    do
    {
        auto bytes{spelling()};

        expect(':', "colon");

        const auto token{id()};

        entries.emplace_back(std::move(bytes), token);
    } while (accept(','));

    expect('}', "closing brace");

    return entries;
}

std::string Encoder_reader::spelling()
{
    expect('"', "quoted spelling");

    // The token is named by the offset of its opening quote.
    const auto start{at_ - 1};

    std::string bytes{};

    while (at_ < text_.size() && text_[at_] != '"')
    {
        const auto character_start{at_};

        const auto code_point{character()};

        const auto byte{byte_of(code_point)};

        if (!byte)
        {
            throw unmapped(code_point, std::format("{} at byte {}", name_, character_start));
        }

        bytes += *byte;
    }

    expect('"', "closing quote");

    if (bytes.empty())
    {
        throw std::invalid_argument{std::format("read_gpt2: {} holds an empty token at byte {}", name_, start)};
    }

    return bytes;
}

char32_t Encoder_reader::character()
{
    if (text_[at_] == '\\')
    {
        ++at_;

        return escaped();
    }

    const auto [code_point, length]{character_at(text_, at_)};

    at_ += length;

    return code_point;
}

char32_t Encoder_reader::escaped()
{
    constexpr std::string_view simple{R"("\/bfnrt)"};
    constexpr std::string_view meant{"\"\\/\b\f\n\r\t"};

    const auto escape{at_ < text_.size() ? text_[at_] : '\0'};

    const auto found{simple.find(escape)};

    if (found == std::string_view::npos)
    {
        return hex_escape();
    }

    ++at_;

    return static_cast<unsigned char>(meant[found]);
}

char32_t Encoder_reader::hex_escape()
{
    constexpr std::size_t hex_digits{4};

    const auto spelled{at_ < text_.size() && text_[at_] == 'u' && at_ + 1 + hex_digits <= text_.size()};

    const auto digits{spelled ? text_.substr(at_ + 1, hex_digits) : ""};

    std::uint32_t value{};

    const auto digits_end{digits.data() + digits.size()};

    const auto [end, error]{std::from_chars(digits.data(), digits_end, value, 16)};

    if (digits.empty() || error != std::errc{} || end != digits_end)
    {
        fail("JSON escape");
    }

    at_ += 1 + hex_digits;

    return value;
}

Token_id_t Encoder_reader::id()
{
    skip_space();

    std::uint64_t value{};

    const auto [end, error]{std::from_chars(text_.data() + at_, text_.data() + text_.size(), value)};

    if (error != std::errc{} || value > std::numeric_limits<Token_id_t>::max())
    {
        fail("id");
    }

    at_ = static_cast<std::size_t>(end - text_.data());

    return static_cast<Token_id_t>(value);
}

std::vector<std::string> Encoder_reader::by_id(Entries_t entries) const
{
    std::vector<std::string> spellings(entries.size());

    std::vector<bool> placed(entries.size());

    for (auto& [bytes, token] : entries)
    {
        if (token >= entries.size() || placed[token])
        {
            throw std::invalid_argument{std::format(
                    "read_gpt2: the ids of {} are not dense from zero: {} tokens, and the id {} is beyond them or "
                    "repeats",
                    name_, entries.size(), token)};
        }

        placed[token] = true;

        spellings[token] = std::move(bytes);
    }

    return spellings;
}

/**
 * @brief Returns the bytes a part of a merge line stands for, each of its characters' byte map undone.
 * @param part The part.
 * @param where The line and the file, for the refusal.
 * @return The bytes.
 * @throws std::invalid_argument If the part holds a character the byte map does not carry.
 */
[[nodiscard]] std::string part_bytes(const std::string_view part, const std::string_view where)
{
    std::string bytes{};

    for (std::size_t index{0}; index < part.size();)
    {
        const auto [character, length]{character_at(part, index)};

        const auto byte{byte_of(character)};

        if (!byte)
        {
            throw unmapped(character, where);
        }

        bytes += *byte;

        index += length;
    }

    return bytes;
}

/**
 * @brief Returns the merge a line of vocab.bpe spells, its two parts separated by white space.
 * @param line The line, without its line feed.
 * @param where The line and the file, for the refusals.
 * @return The merge, each part's byte map undone.
 * @throws std::invalid_argument If the line is not two parts, or a part holds a character the byte map does not carry.
 */
[[nodiscard]] munch::verifier::Merge merge_of(const std::string_view line, const std::string_view where)
{
    std::vector<std::string> parts{};

    for (std::size_t at{line.find_first_not_of(separators)}; at != std::string_view::npos;)
    {
        const auto stop{std::min(line.find_first_of(separators, at), line.size())};

        const auto part{line.substr(at, stop - at)};

        parts.push_back(part_bytes(part, where));

        at = line.find_first_not_of(separators, stop);
    }

    if (parts.size() != 2)
    {
        throw std::invalid_argument{
                std::format("read_gpt2: {} is not a merge, two parts separated by white space", where)};
    }

    return {.left = std::move(parts.front()), .right = std::move(parts.back())};
}

/**
 * @brief Returns the merge table of vocab.bpe, as encoder.py reads it: the lines between the first newline and the
 *        last, each two parts separated by white space, each part's byte map undone.
 * @param text The file's bytes.
 * @param name The file's path, for the refusals.
 * @return The merges in rank order.
 * @throws std::invalid_argument If a line is not two parts, or a part holds a character the byte map does not carry.
 */
[[nodiscard]] std::vector<munch::verifier::Merge> merges_of(const std::string_view text, const std::string_view name)
{
    std::vector<munch::verifier::Merge> merges{};

    auto start{text.find('\n')};

    for (auto number{first_merge_line}; start != std::string_view::npos; ++number)
    {
        const auto end{text.find('\n', start + 1)};

        if (end == std::string_view::npos)
        {
            break;
        }

        const auto line{text.substr(start + 1, end - start - 1)};

        start = end;

        const auto where{std::format("line {} of {}", number, name)};

        merges.push_back(merge_of(line, where));
    }

    return merges;
}

/**
 * @brief Returns a part of a merge as a refusal quotes it: a printable ASCII byte as itself, the quote and the
 *        backslash escaped, and every other byte as `\xHH`, so that a part outside ASCII reads the same in every
 *        terminal.
 * @param part The part's bytes.
 * @return The part, quoted.
 */
[[nodiscard]] std::string quoted_part(const std::string_view part)
{
    constexpr unsigned char first_printable{0x20};
    constexpr unsigned char last_printable{0x7E};

    std::string out{'"'};

    for (const auto byte : part)
    {
        const auto value{static_cast<unsigned char>(byte)};

        const auto printable{value >= first_printable && value <= last_printable};

        if (byte == '"' || byte == '\\')
        {
            out += std::format(R"(\{})", byte);
        }
        else if (printable)
        {
            out += byte;
        }
        else
        {
            out += std::format(R"(\x{:02X})", value);
        }
    }

    out += '"';

    return out;
}

/**
 * @brief Requires a merge table on which the pair rule of Merge_rule::pairs, one join at a time, is encoder.py's loop:
 *        no pair listed twice, and no part of a merge formed by a merge of a later rank.
 *
 * encoder.py joins every occurrence of the least-ranked pair in one pass. Where every part is formed before the merges
 * that join it, a join makes no pair of lower rank than its own, so each pass is a run of single joins and the two
 * rules give the same tokens on every piece; where a part is formed later they can differ, (ab, a) then (a, b) giving
 * ab ab on abab under the pass and aba b one join at a time.
 * @param merges The merges in rank order, the first on line 2 of the file.
 * @param name The file's path, for the refusals.
 * @throws std::invalid_argument If a pair is listed twice, or a part of a merge is formed by a merge of a later rank;
 *         the message names the merge, its rank and its line, and the other merge's rank.
 */
void require_parts_formed_first(const std::span<const munch::verifier::Merge> merges, const std::string_view name)
{
    // The pairs listed so far by rank, and the latest rank at which each product is formed.
    std::map<std::pair<std::string_view, std::string_view>, std::size_t> ranks{};

    std::map<std::string, std::size_t, std::less<>> formed{};

    for (std::size_t rank{0}; rank < merges.size(); ++rank)
    {
        const auto& [left, right]{merges[rank]};

        const auto [listed, inserted]{ranks.try_emplace({left, right}, rank)};

        if (!inserted)
        {
            const auto line{rank + first_merge_line};

            throw std::invalid_argument{std::format(
                    "read_gpt2: the merge ({}, {}) at rank {}, line {} of {}, repeats the merge at rank {}",
                    quoted_part(left), quoted_part(right), rank, line, name, listed->second)};
        }

        auto product{left + right};

        formed.insert_or_assign(std::move(product), rank);
    }

    for (std::size_t rank{0}; rank < merges.size(); ++rank)
    {
        const auto& [left, right]{merges[rank]};

        for (const auto part : {std::string_view{left}, std::string_view{right}})
        {
            const auto forming{formed.find(part)};

            if (forming == formed.end() || forming->second < rank)
            {
                continue;
            }

            const auto line{rank + first_merge_line};

            throw std::invalid_argument{std::format(
                    "read_gpt2: the merge ({}, {}) at rank {}, line {} of {}, has the part {}, which the merge at "
                    "rank {} forms",
                    quoted_part(left), quoted_part(right), rank, line, name, quoted_part(part), forming->second)};
        }
    }
}

} // namespace

Definition read_gpt2(const std::filesystem::path& encoder, const std::filesystem::path& vocab)
{
    const auto encoder_text{contents(encoder)};

    const auto vocab_text{contents(vocab)};

    const auto encoder_name{encoder.string()};

    Encoder_reader reader{encoder_text, encoder_name};

    auto spellings{reader.read()};

    const auto vocab_name{vocab.string()};

    auto merges{merges_of(vocab_text, vocab_name)};

    require_parts_formed_first(merges, vocab_name);

    return {.merge_rule = Merge_rule::pairs,
            .whole_piece_lookup = false,
            .spellings = std::move(spellings),
            .merges = std::move(merges),
            .specials = {}};
}

} // namespace matisse::load
