#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <hopper/json/parser.hpp>
#include <hopper/json/value.hpp>
#include <hopper/parse/parse_error.hpp>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "matisse/load/definition.hpp"

namespace matisse::load
{
namespace
{
/**
 * @brief The pattern a ByteLevel pre-tokenizer applies with its regex, GPT-2's as openai/gpt-2's src/encoder.py spells
 *        it, which Hugging Face's tokenizers spells alike.
 */
constexpr std::string_view byte_level_spelling{
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"};

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
 * @brief A member of a JSON object, std::nullopt when the object has none of its name.
 */
using Member_t = std::optional<std::reference_wrapper<const hopper::json::Value>>;

/**
 * @brief The vocabulary's ordinary tokens as read, each spelling's bytes with its id, in the file's order.
 */
using Entries_t = std::vector<std::pair<std::string, Token_id_t>>;

/**
 * @brief The pre-tokenizer of a tokenizer.json as recognized: the published spelling of the pattern it runs, the member
 *        that spells it and what that member holds, as a refusal describes it.
 */
struct Recognized
{
    /**
     * @brief The published spelling, empty for a ByteLevel without its regex.
     */
    std::string spelling{};

    /**
     * @brief The member that spells the pattern.
     */
    std::string member{};

    /**
     * @brief What the member holds.
     */
    std::string found{};
};

/**
 * @brief A tokenizer.json parsed whole, read member by member into a definition.
 *
 * A refusal names the file, the member, as a path of names and indices from the root such as
 * `pre_tokenizer.pretokenizers[1].use_regex`, what it holds and what the reader takes there.
 */
class Tokenizer_json
{
public:
    /**
     * @brief Reads and parses a file.
     * @param path The file's path.
     * @throws std::runtime_error If the file cannot be read.
     * @throws std::invalid_argument If the file is not JSON or not an object.
     */
    explicit Tokenizer_json(const std::filesystem::path& path);

    /**
     * @brief Returns the pre-tokenizer as recognized.
     * @return The published spelling of its pattern, the member that spells it and what that member holds.
     * @throws std::invalid_argument If the pre-tokenizer is none of the three the reader recognizes, or its pattern
     *         holds an interval followed by a plus.
     */
    [[nodiscard]] Recognized recognized() const;

    /**
     * @brief Returns the definition the file holds, read under a published spelling.
     * @param published The published spelling of the pattern, empty for a pre-tokenizer that splits nothing.
     * @return The definition.
     * @throws std::invalid_argument If a member is not one the reader takes, or the vocabulary or the merges are not
     *         read exactly.
     */
    [[nodiscard]] Definition definition(std::string_view published) const;

private:
    /**
     * @brief Returns the root object.
     * @return The object.
     */
    [[nodiscard]] const hopper::json::Object& root() const noexcept;

    /**
     * @brief Refuses the file at a member.
     * @param member The member.
     * @param found What it holds.
     * @param taken What the reader takes there.
     * @throws std::invalid_argument Always.
     */
    [[noreturn]] void refuse(std::string_view member, std::string_view found, std::string_view taken) const;

    /**
     * @brief Returns whether a ByteLevel pre-tokenizer applies its regex, after requiring it to add no prefix space.
     * @param byte_level The ByteLevel's object.
     * @param member The ByteLevel's member, for the refusals.
     * @return Its `use_regex`, true when absent, as Hugging Face's default is.
     * @throws std::invalid_argument If `add_prefix_space` is not false or `use_regex` not a boolean.
     */
    [[nodiscard]] bool applies_regex(const hopper::json::Object& byte_level, std::string_view member) const;

    /**
     * @brief Returns the pattern of a Sequence of a Split and a ByteLevel without its regex, as recognized.
     * @param sequence The Sequence's object.
     * @return The Split's pattern, its member and what the member holds.
     * @throws std::invalid_argument If the Sequence is not a Split and such a ByteLevel, the Split's pattern is not a
     *         regex, it is not isolated and not inverted or removed and inverted, or the pattern holds an interval
     *         followed by a plus.
     */
    [[nodiscard]] Recognized split(const hopper::json::Object& sequence) const;

    /**
     * @brief Requires the pre-tokenizer recognized to be the published spelling's.
     * @param recognized The pre-tokenizer as recognized.
     * @param published The published spelling.
     * @throws std::invalid_argument If the spellings differ; the message names what the file holds and what the
     *         spelling takes.
     */
    void require_published(const Recognized& recognized, std::string_view published) const;

    /**
     * @brief Requires a member of the root to be null or absent.
     * @param member The member's name.
     * @throws std::invalid_argument If it holds anything else.
     */
    void require_null(std::string_view member) const;

    /**
     * @brief Requires the post-processor to add tokens only under `add_special_tokens`: null, ByteLevel,
     *        TemplateProcessing, or a Sequence of the two.
     * @throws std::invalid_argument If it is another.
     */
    void require_post_processor() const;

    /**
     * @brief Returns the BPE model, after requiring it to be one the reader takes.
     * @return The model's object.
     * @throws std::invalid_argument If it is not a BPE model, or has dropout, a continuing-subword prefix, an
     *         end-of-word suffix or byte fallback.
     */
    [[nodiscard]] const hopper::json::Object& model() const;

    /**
     * @brief Returns whether the model looks a whole piece up before merging.
     * @param model The model's object.
     * @return Its `ignore_merges`, false when absent.
     * @throws std::invalid_argument If it is not a boolean.
     */
    [[nodiscard]] bool ignores_merges(const hopper::json::Object& model) const;

    /**
     * @brief Returns the vocabulary, an object of spellings and their ids.
     * @param model The model's object.
     * @return The vocabulary's object.
     * @throws std::invalid_argument If the model's `vocab` is not an object.
     */
    [[nodiscard]] const hopper::json::Object& vocabulary(const hopper::json::Object& model) const;

    /**
     * @brief Returns the special tokens, the `added_tokens` all marked special, by id ascending.
     * @param vocab The vocabulary's object.
     * @return The special tokens.
     * @throws std::invalid_argument If an added token is not special, strips white space, matches single words, is
     *         empty, is not an object of a content and an id, or has another id than Hugging Face gives it.
     */
    [[nodiscard]] std::vector<Special_token> special_tokens(const hopper::json::Object& vocab) const;

    /**
     * @brief Refuses the file at a member that is not an id.
     * @param member The member.
     * @param value What it holds.
     * @throws std::invalid_argument Always.
     */
    [[noreturn]] void refuse_id(std::string_view member, const Member_t& value) const;

    /**
     * @brief Requires every special token to have the id Hugging Face gives an added token, which reads no added
     *        token's own id: a token whose spelling is in the vocabulary gets its id there, and any other the
     *        vocabulary's size, or one past the greatest id given before it when that is no less, in the file's order.
     * @param listed The special tokens in the file's order.
     * @param vocab The vocabulary's object.
     * @throws std::invalid_argument If a token's id is another; the message names it and the id Hugging Face gives.
     */
    void require_assigned_ids(std::span<const Special_token> listed, const hopper::json::Object& vocab) const;

    /**
     * @brief Returns the ordinary tokens, the vocabulary's entries but the special tokens', placed by id.
     * @param vocab The vocabulary's object.
     * @param specials The special tokens, by id ascending.
     * @return The bytes of every ordinary token, by id.
     * @throws std::invalid_argument If a spelling is empty or holds a character the byte map does not carry, an id is
     *         no id, the ids are not dense from zero, or two tokens have one spelling.
     */
    [[nodiscard]] std::vector<std::string> ordinary(
            const hopper::json::Object& vocab, std::span<const Special_token> specials) const;

    /**
     * @brief Refuses the file at a member that holds what the reader cannot read.
     * @param member The member.
     * @param held What it holds.
     * @throws std::invalid_argument Always.
     */
    [[noreturn]] void refuse_holding(std::string_view member, std::string_view held) const;

    /**
     * @brief Returns the merges in rank order, each part's byte map undone.
     * @param model The model's object.
     * @return The merges.
     * @throws std::invalid_argument If the merges are not an array of merges, each a string of two nonempty parts
     *         separated by a space or an array of two nonempty strings, a part holds a character the byte map does not
     *         carry, or a pair is listed twice.
     */
    [[nodiscard]] std::vector<munch::verifier::Merge> merges(const hopper::json::Object& model) const;

    /**
     * @brief The file's path, for the refusals.
     */
    std::string name_;

    /**
     * @brief The file's root value.
     */
    hopper::json::Value root_;
};

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
 * @brief Returns the value a JSON file holds, parsed by hopper's RFC 8259 parser.
 * @param path The file's path.
 * @param name The file's path as the refusal names it.
 * @return The value.
 * @throws std::runtime_error If the file cannot be read.
 * @throws std::invalid_argument If the file is not JSON; the message carries the parser's, with its line and column.
 */
[[nodiscard]] hopper::json::Value parsed(const std::filesystem::path& path, const std::string_view name)
{
    const auto text{contents(path)};

    try
    {
        hopper::json::Parser parser{text};

        return parser.parse();
    }
    catch (const hopper::parse::Parse_error& error)
    {
        throw std::invalid_argument{std::format("read_tokenizer_json: {} is not JSON: {}", name, error.what())};
    }
}

/**
 * @brief Returns the last member of an object carrying a name, as Hugging Face's reader keeps the last.
 * @param object The object.
 * @param name The member's name.
 * @return The member's value, std::nullopt when no member carries the name.
 */
[[nodiscard]] Member_t member_of(const hopper::json::Object& object, const std::string_view name)
{
    const auto index{object.find(name)};

    if (!index)
    {
        return std::nullopt;
    }

    return std::cref(object.members[*index].value);
}

/**
 * @brief Returns whether a member is absent or null.
 * @param member The member.
 * @return True when it is.
 */
[[nodiscard]] bool is_null(const Member_t& member) noexcept
{
    return !member || member->get().is_null();
}

/**
 * @brief Returns whether a value is an object.
 * @param value The value.
 * @return True when it is.
 */
[[nodiscard]] bool is_object(const hopper::json::Value& value) noexcept
{
    return std::holds_alternative<hopper::json::Object>(value.node());
}

/**
 * @brief Returns whether a value is an array.
 * @param value The value.
 * @return True when it is.
 */
[[nodiscard]] bool is_array(const hopper::json::Value& value) noexcept
{
    return std::holds_alternative<hopper::json::Array>(value.node());
}

/**
 * @brief Returns whether a value is a string.
 * @param value The value.
 * @return True when it is.
 */
[[nodiscard]] bool is_string(const hopper::json::Value& value) noexcept
{
    return std::holds_alternative<std::string>(value.node());
}

/**
 * @brief Returns whether a value is a boolean.
 * @param value The value.
 * @return True when it is.
 */
[[nodiscard]] bool is_bool(const hopper::json::Value& value) noexcept
{
    return std::holds_alternative<bool>(value.node());
}

/**
 * @brief Returns whether a member is the boolean false.
 * @param member The member.
 * @return True when it is present and false.
 */
[[nodiscard]] bool is_false(const Member_t& member) noexcept
{
    return member && is_bool(*member) && !member->get().as_bool();
}

/**
 * @brief Returns whether a member is absent or the boolean false, so that a flag that is absent is read as false.
 * @param member The member.
 * @return True when it is.
 */
[[nodiscard]] bool is_unset(const Member_t& member) noexcept
{
    return !member || is_false(member);
}

/**
 * @brief Returns the type an object names in its member `type`, as Hugging Face's components name theirs.
 * @param value The value.
 * @return The type, empty when the value is not an object or names none.
 */
[[nodiscard]] std::string_view type_of(const hopper::json::Value& value)
{
    if (!is_object(value))
    {
        return {};
    }

    const auto type{member_of(value.as_object(), "type")};

    if (!type || !is_string(*type))
    {
        return {};
    }

    return type->get().as_string();
}

/**
 * @brief Returns a text as a refusal quotes it: a printable ASCII byte as itself, the quote and the backslash escaped,
 *        and every other byte as `\xHH`, so that a text outside ASCII reads the same in every terminal.
 * @param text The text's bytes.
 * @return The text, quoted.
 */
[[nodiscard]] std::string quoted_text(const std::string_view text)
{
    constexpr unsigned char first_printable{0x20};
    constexpr unsigned char last_printable{0x7E};

    std::string out{'"'};

    for (const auto byte : text)
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
 * @brief Returns what a member holds, as a refusal describes it: absent, null, a boolean or a number as spelled, a
 *        string quoted, an array, or an object by the type it names.
 * @param member The member.
 * @return The description.
 */
[[nodiscard]] std::string described(const Member_t& member)
{
    if (!member)
    {
        return "absent";
    }

    const auto& value{member->get()};

    if (value.is_null())
    {
        return "null";
    }

    if (is_bool(value))
    {
        return value.as_bool() ? "true" : "false";
    }

    if (std::holds_alternative<hopper::json::Number>(value.node()))
    {
        return value.as_number().text;
    }

    if (is_string(value))
    {
        return quoted_text(value.as_string());
    }

    if (is_array(value))
    {
        return std::format("an array of {}", value.as_array().elements.size());
    }

    const auto type{type_of(value)};

    if (type.empty())
    {
        return "an object";
    }

    return std::format("a {} object", quoted_text(type));
}

/**
 * @brief Returns the first interval of a pattern's spelling that is followed by a plus, `{n}+`, `{n,}+`, `{,m}+` or
 *        `{n,m}+` after a character that is not a backslash, which Oniguruma, Hugging Face's engine, reads under its
 *        Ruby syntax as the interval repeated, since that syntax has no possessive interval.
 * @param spelling The spelling.
 * @return The interval with its plus, empty when the spelling holds none.
 */
[[nodiscard]] std::string_view repeated_interval(const std::string_view spelling) noexcept
{
    for (std::size_t open{spelling.find('{')}; open != std::string_view::npos; open = spelling.find('{', open + 1))
    {
        if (open > 0 && spelling[open - 1] == '\\')
        {
            continue;
        }

        const auto close{spelling.find('}', open)};

        if (close == std::string_view::npos || close + 1 >= spelling.size() || spelling[close + 1] != '+')
        {
            continue;
        }

        const auto bounds{spelling.substr(open + 1, close - open - 1)};

        const auto is_bound{
                [](const char character) { return character == ',' || (character >= '0' && character <= '9'); }};

        const auto commas{std::ranges::count(bounds, ',')};

        const auto digits{static_cast<std::ptrdiff_t>(bounds.size()) - commas};

        if (!std::ranges::all_of(bounds, is_bound) || commas > 1 || digits == 0)
        {
            continue;
        }

        return spelling.substr(open, close - open + 2);
    }

    return {};
}

/**
 * @brief Returns the id a value spells.
 * @param value The value.
 * @return The id, std::nullopt when the value is not a JSON integer from zero to the largest 32-bit id.
 */
[[nodiscard]] std::optional<Token_id_t> id_of(const hopper::json::Value& value)
{
    if (!std::holds_alternative<hopper::json::Number>(value.node()))
    {
        return std::nullopt;
    }

    const std::string_view text{value.as_number().text};

    const auto text_end{text.data() + text.size()};

    std::uint64_t id{};

    const auto [end, error]{std::from_chars(text.data(), text_end, id)};

    if (error != std::errc{} || end != text_end || id > std::numeric_limits<Token_id_t>::max())
    {
        return std::nullopt;
    }

    return static_cast<Token_id_t>(id);
}

/**
 * @brief Returns whether a byte continues a UTF-8 sequence rather than beginning one.
 * @param byte The byte.
 * @return True for a continuation byte, `10xxxxxx`.
 */
[[nodiscard]] constexpr bool is_continuation(const unsigned char byte) noexcept
{
    return (byte & 0xC0U) == 0x80U;
}

/**
 * @brief Returns the length of the well-formed UTF-8 sequence at an index, zero when none stands there: an ASCII byte,
 *        or a lead byte of the shape 110xxxxx, 1110xxxx or 11110xxx followed by as many continuation bytes as its shape
 *        says, the overlong, surrogate and beyond-U+10FFFF encodings excluded as the standard excludes them, which the
 *        second byte's narrower range after a lead of E0, ED, F0 or F4 rules out.
 * @param text The text.
 * @param at The index, within the text.
 * @return One to four, or zero.
 */
[[nodiscard]] constexpr std::size_t well_formed_length(const std::string_view text, const std::size_t at) noexcept
{
    const auto lead{static_cast<unsigned char>(text[at])};

    const auto length{
            lead < 0x80U                  ? 1UZ :
            lead >= 0xC2U && lead < 0xE0U ? 2UZ :
            lead >= 0xE0U && lead < 0xF0U ? 3UZ :
            lead >= 0xF0U && lead < 0xF5U ? 4UZ :
                                            0UZ};

    if (length == 0 || at + length > text.size())
    {
        return 0;
    }

    for (std::size_t index{1}; index < length; ++index)
    {
        if (!is_continuation(static_cast<unsigned char>(text[at + index])))
        {
            return 0;
        }
    }

    const auto second{static_cast<unsigned char>(text[at + (length > 1 ? 1 : 0)])};

    const auto malformed{
            (lead == 0xE0U && second < 0xA0U) || (lead == 0xEDU && second >= 0xA0U) ||
            (lead == 0xF0U && second < 0x90U) || (lead == 0xF4U && second >= 0x90U)};

    return malformed ? 0 : length;
}

/**
 * @brief Returns the code point of the well-formed UTF-8 sequence at an index and its length.
 * @param text The text.
 * @param at The index of the lead byte, within the text.
 * @return The code point and the number of bytes the sequence takes, std::nullopt when the bytes there are not UTF-8.
 */
[[nodiscard]] std::optional<std::pair<char32_t, std::size_t>> character_at(
        const std::string_view text, const std::size_t at) noexcept
{
    const auto length{well_formed_length(text, at)};

    if (length == 0)
    {
        return std::nullopt;
    }

    const auto lead{static_cast<unsigned char>(text[at])};

    char32_t character{static_cast<char32_t>(length == 1 ? lead : lead & (0xFFU >> (length + 1)))};

    for (std::size_t index{1}; index < length; ++index)
    {
        const auto continuation{static_cast<unsigned char>(text[at + index])};

        character = (character << 6U) | static_cast<char32_t>(continuation & 0x3FU);
    }

    return std::pair{character, length};
}

/**
 * @brief Returns the bytes of a text's characters with GPT-2's byte map undone.
 * @param text The text, its characters those of the byte map.
 * @return The bytes, or what the text holds instead: bytes that are not UTF-8, or a character the map does not carry.
 */
[[nodiscard]] std::expected<std::string, std::string> unmapped(const std::string_view text)
{
    std::string bytes{};

    for (std::size_t at{0}; at < text.size();)
    {
        const auto decoded{character_at(text, at)};

        if (!decoded)
        {
            return std::unexpected{std::string{"bytes that are not UTF-8"}};
        }

        const auto [character, length]{*decoded};

        if (character >= map_end || !byte_map[character])
        {
            return std::unexpected{std::format(
                    "U+{:04X}, which GPT-2's byte map does not carry", static_cast<std::uint32_t>(character))};
        }

        bytes += static_cast<char>(*byte_map[character]);

        at += length;
    }

    return bytes;
}

Tokenizer_json::Tokenizer_json(const std::filesystem::path& path) : name_{path.string()}, root_{parsed(path, name_)}
{
    if (!is_object(root_))
    {
        const auto found{described(std::cref(root_))};

        throw std::invalid_argument{
                std::format("read_tokenizer_json: {} is {}, where the reader takes an object", name_, found)};
    }
}

Recognized Tokenizer_json::recognized() const
{
    const auto pre_tokenizer{member_of(root(), "pre_tokenizer")};

    const auto type{is_null(pre_tokenizer) ? std::string_view{} : type_of(*pre_tokenizer)};

    if (type == "ByteLevel")
    {
        const auto regex{applies_regex(pre_tokenizer->get().as_object(), "pre_tokenizer")};

        if (regex)
        {
            return {.spelling = std::string{byte_level_spelling},
                    .member = "pre_tokenizer",
                    .found = "a ByteLevel with its regex, GPT-2's pattern"};
        }

        return {.spelling = {},
                .member = "pre_tokenizer",
                .found = "a ByteLevel without its regex, which splits nothing"};
    }

    if (type != "Sequence")
    {
        refuse("pre_tokenizer", described(pre_tokenizer),
               "a ByteLevel, or a Sequence of a Split and a ByteLevel without its regex");
    }

    return split(pre_tokenizer->get().as_object());
}

Definition Tokenizer_json::definition(const std::string_view published) const
{
    require_published(recognized(), published);

    require_null("normalizer");

    require_null("truncation");

    require_null("padding");

    require_post_processor();

    const auto& bpe{model()};

    const auto lookup{ignores_merges(bpe)};

    const auto& vocab{vocabulary(bpe)};

    auto specials{special_tokens(vocab)};

    auto spellings{ordinary(vocab, specials)};

    auto table{merges(bpe)};

    return {.merge_rule = Merge_rule::pairs,
            .whole_piece_lookup = lookup,
            .spellings = std::move(spellings),
            .merges = std::move(table),
            .specials = std::move(specials)};
}

const hopper::json::Object& Tokenizer_json::root() const noexcept
{
    return std::get<hopper::json::Object>(root_.node());
}

void Tokenizer_json::refuse(
        const std::string_view member, const std::string_view found, const std::string_view taken) const
{
    throw std::invalid_argument{
            std::format("read_tokenizer_json: {} of {} is {}, where the reader takes {}", member, name_, found, taken)};
}

bool Tokenizer_json::applies_regex(const hopper::json::Object& byte_level, const std::string_view member) const
{
    const auto prefix{member_of(byte_level, "add_prefix_space")};

    if (!is_false(prefix))
    {
        refuse(std::format("{}.add_prefix_space", member), described(prefix), "false");
    }

    const auto regex{member_of(byte_level, "use_regex")};

    if (!regex)
    {
        return true;
    }

    if (!is_bool(*regex))
    {
        refuse(std::format("{}.use_regex", member), described(regex), "true or false");
    }

    return regex->get().as_bool();
}

Recognized Tokenizer_json::split(const hopper::json::Object& sequence) const
{
    const auto steps{member_of(sequence, "pretokenizers")};

    constexpr std::string_view steps_taken{"an array of a Split and a ByteLevel without its regex"};

    if (!steps || !is_array(*steps))
    {
        refuse("pre_tokenizer.pretokenizers", described(steps), steps_taken);
    }

    const auto& elements{steps->get().as_array().elements};

    if (elements.size() != 2)
    {
        refuse("pre_tokenizer.pretokenizers", described(steps), steps_taken);
    }

    const auto& [split_step, byte_level_step]{[&] {
        return std::pair<const hopper::json::Value&, const hopper::json::Value&>{elements.front(), elements.back()};
    }()};

    if (type_of(split_step) != "Split")
    {
        const auto found{described(std::cref(split_step))};

        refuse("pre_tokenizer.pretokenizers[0]", found, "a Split");
    }

    if (type_of(byte_level_step) != "ByteLevel")
    {
        const auto found{described(std::cref(byte_level_step))};

        refuse("pre_tokenizer.pretokenizers[1]", found, "a ByteLevel without its regex");
    }

    const auto& byte_level_object{byte_level_step.as_object()};

    if (applies_regex(byte_level_object, "pre_tokenizer.pretokenizers[1]"))
    {
        const auto use_regex{member_of(byte_level_object, "use_regex")};

        refuse("pre_tokenizer.pretokenizers[1].use_regex", described(use_regex), "false");
    }

    const auto& split_object{split_step.as_object()};

    const auto pattern{member_of(split_object, "pattern")};

    constexpr std::string_view pattern_taken{R"(an object of a string under "Regex")"};

    if (!pattern || !is_object(*pattern))
    {
        refuse("pre_tokenizer.pretokenizers[0].pattern", described(pattern), pattern_taken);
    }

    const auto regex{member_of(pattern->get().as_object(), "Regex")};

    constexpr std::string_view member{"pre_tokenizer.pretokenizers[0].pattern.Regex"};

    if (!regex || !is_string(*regex))
    {
        refuse(member, described(regex), "a string");
    }

    const auto behavior{member_of(split_object, "behavior")};

    const auto invert{member_of(split_object, "invert")};

    const auto behavior_name{behavior && is_string(*behavior) ? std::string_view{behavior->get().as_string()} : ""};

    const auto inverted{invert && is_bool(*invert) && invert->get().as_bool()};

    const auto isolated{behavior_name == "Isolated" && is_unset(invert)};

    const auto removed{behavior_name == "Removed" && inverted};

    if (!isolated && !removed)
    {
        const auto found{
                std::format("a Split with the behavior {} and the invert {}", described(behavior), described(invert))};

        refuse("pre_tokenizer.pretokenizers[0]", found, "a Split isolated and not inverted, or removed and inverted");
    }

    const auto& spelling{regex->get().as_string()};

    // A Split on the empty pattern parts a text into its characters, which the empty spelling does not mean.
    if (spelling.empty())
    {
        refuse(member, R"("")", "a nonempty pattern");
    }

    const auto interval{repeated_interval(spelling)};

    if (!interval.empty())
    {
        const auto found{std::format(
                "{}, whose {} Oniguruma, Hugging Face's engine, reads as the interval repeated and not as possessive",
                quoted_text(spelling), quoted_text(interval))};

        refuse(member, found, "no pattern with an interval followed by a plus");
    }

    return {.spelling = spelling, .member = std::string{member}, .found = quoted_text(spelling)};
}

void Tokenizer_json::require_published(const Recognized& recognized, const std::string_view published) const
{
    if (recognized.spelling == published)
    {
        return;
    }

    const auto split_on{std::format("a Split on {} followed by a ByteLevel without its regex", quoted_text(published))};

    const auto taken{
            published.empty()                ? std::string{"a ByteLevel without its regex, for the empty spelling"} :
            published == byte_level_spelling ? std::format("a ByteLevel with its regex, or {}", split_on) :
                                               split_on};

    refuse(recognized.member, recognized.found, taken);
}

void Tokenizer_json::require_null(const std::string_view member) const
{
    const auto value{member_of(root(), member)};

    if (!is_null(value))
    {
        refuse(member, described(value), "null");
    }
}

void Tokenizer_json::require_post_processor() const
{
    const auto post_processor{member_of(root(), "post_processor")};

    if (is_null(post_processor))
    {
        return;
    }

    const auto adds_only_special{[](const hopper::json::Value& processor) {
        const auto type{type_of(processor)};

        return type == "ByteLevel" || type == "TemplateProcessing";
    }};

    if (adds_only_special(*post_processor))
    {
        return;
    }

    constexpr std::string_view taken{
            "null, a ByteLevel, a TemplateProcessing or a Sequence of them, which add tokens only under "
            "add_special_tokens"};

    if (type_of(*post_processor) != "Sequence")
    {
        refuse("post_processor", described(post_processor), taken);
    }

    const auto processors{member_of(post_processor->get().as_object(), "processors")};

    if (!processors || !is_array(*processors))
    {
        refuse("post_processor.processors", described(processors), "an array of a ByteLevel and a TemplateProcessing");
    }

    for (const auto& [index, processor] : std::views::enumerate(processors->get().as_array().elements))
    {
        if (!adds_only_special(processor))
        {
            const auto found{described(std::cref(processor))};

            refuse(std::format("post_processor.processors[{}]", index), found, taken);
        }
    }
}

const hopper::json::Object& Tokenizer_json::model() const
{
    const auto model{member_of(root(), "model")};

    if (!model || !is_object(*model))
    {
        refuse("model", described(model), "a BPE model");
    }

    const auto& bpe{model->get().as_object()};

    const auto type{member_of(bpe, "type")};

    if (type && type_of(*model) != "BPE")
    {
        refuse("model.type", described(type), R"("BPE")");
    }

    const auto dropout{member_of(bpe, "dropout")};

    if (!is_null(dropout))
    {
        refuse("model.dropout", described(dropout), "null");
    }

    constexpr std::array<std::string_view, 2> affixes{"continuing_subword_prefix", "end_of_word_suffix"};

    for (const auto affix : affixes)
    {
        const auto value{member_of(bpe, affix)};

        if (is_null(value))
        {
            continue;
        }

        const auto& held{value->get()};

        if (!is_string(held) || !held.as_string().empty())
        {
            refuse(std::format("model.{}", affix), described(value), R"(null or "")");
        }
    }

    const auto fallback{member_of(bpe, "byte_fallback")};

    if (!is_unset(fallback))
    {
        refuse("model.byte_fallback", described(fallback), "false");
    }

    return bpe;
}

bool Tokenizer_json::ignores_merges(const hopper::json::Object& model) const
{
    const auto ignore{member_of(model, "ignore_merges")};

    if (!ignore)
    {
        return false;
    }

    if (!is_bool(*ignore))
    {
        refuse("model.ignore_merges", described(ignore), "true or false");
    }

    return ignore->get().as_bool();
}

const hopper::json::Object& Tokenizer_json::vocabulary(const hopper::json::Object& model) const
{
    const auto vocab{member_of(model, "vocab")};

    if (!vocab || !is_object(*vocab))
    {
        refuse("model.vocab", described(vocab), "an object of spellings and their ids");
    }

    return vocab->get().as_object();
}

std::vector<Special_token> Tokenizer_json::special_tokens(const hopper::json::Object& vocab) const
{
    const auto added{member_of(root(), "added_tokens")};

    if (is_null(added))
    {
        return {};
    }

    if (!is_array(*added))
    {
        refuse("added_tokens", described(added), "an array of added tokens");
    }

    constexpr std::array<std::string_view, 3> flags{"lstrip", "rstrip", "single_word"};

    std::vector<Special_token> specials{};

    for (const auto& [index, token] : std::views::enumerate(added->get().as_array().elements))
    {
        const auto member{std::format("added_tokens[{}]", index)};

        if (!is_object(token))
        {
            const auto found{described(std::cref(token))};

            refuse(member, found, "an object of a content and an id");
        }

        const auto& fields{token.as_object()};

        const auto content{member_of(fields, "content")};

        if (!content || !is_string(*content))
        {
            refuse(std::format("{}.content", member), described(content), "a nonempty string");
        }

        const auto& spelling{content->get().as_string()};

        if (spelling.empty())
        {
            refuse(std::format("{}.content", member), described(content), "a nonempty string");
        }

        const auto special{member_of(fields, "special")};

        if (is_unset(special))
        {
            const auto found{std::format("{}, which is not special", quoted_text(spelling))};

            refuse(member, found,
                   "special added tokens alone, since Hugging Face matches every added token in the text before its "
                   "pre-tokenizer");
        }

        for (const auto flag : flags)
        {
            const auto set{member_of(fields, flag)};

            if (!is_unset(set))
            {
                refuse(std::format("{}.{}", member, flag), described(set), "false");
            }
        }

        const auto id{member_of(fields, "id")};

        if (!id)
        {
            refuse_id(std::format("{}.id", member), id);
        }

        const auto token_id{id_of(*id)};

        if (!token_id)
        {
            refuse_id(std::format("{}.id", member), id);
        }

        specials.push_back({.spelling = spelling, .id = *token_id});
    }

    require_assigned_ids(specials, vocab);

    std::ranges::stable_sort(specials, {}, &Special_token::id);

    return specials;
}

void Tokenizer_json::refuse_id(const std::string_view member, const Member_t& value) const
{
    const auto taken{std::format("an id from 0 to {}", std::numeric_limits<Token_id_t>::max())};

    refuse(member, described(value), taken);
}

void Tokenizer_json::require_assigned_ids(
        const std::span<const Special_token> listed, const hopper::json::Object& vocab) const
{
    // The id the vocabulary gives each special token's spelling, the last of its members as Hugging Face's map keeps.
    std::map<std::string_view, std::optional<Token_id_t>> in_vocabulary{};

    for (const auto& [spelling, id] : listed)
    {
        in_vocabulary.try_emplace(spelling);
    }

    for (const auto& [characters, value] : vocab.members)
    {
        const auto found{in_vocabulary.find(characters)};

        if (found == in_vocabulary.end())
        {
            continue;
        }

        const auto id{id_of(value)};

        if (!id)
        {
            const auto member{std::format("model.vocab[{}]", quoted_text(characters))};

            refuse_id(member, std::cref(value));
        }

        auto& [spelling, given]{*found};

        given = *id;
    }

    const auto size{vocab.members.size()};

    std::optional<std::size_t> greatest{};

    for (const auto& [index, special] : std::views::enumerate(listed))
    {
        const auto& [spelling, id]{special};

        const auto vocabulary_id{in_vocabulary.at(spelling)};

        const auto next{!greatest || *greatest < size ? size : *greatest + 1};

        const auto given{vocabulary_id ? std::size_t{*vocabulary_id} : next};

        if (id != given)
        {
            refuse(std::format("added_tokens[{}].id", index), std::format("{}", id),
                   std::format("{}, the id Hugging Face gives the token", given));
        }

        greatest = std::max(greatest.value_or(given), given);
    }
}

std::vector<std::string> Tokenizer_json::ordinary(
        const hopper::json::Object& vocab, const std::span<const Special_token> specials) const
{
    Entries_t entries{};

    for (const auto& [characters, value] : vocab.members)
    {
        const auto member{[&] { return std::format("model.vocab[{}]", quoted_text(characters)); }};

        if (characters.empty())
        {
            refuse_holding("model.vocab", "an empty token");
        }

        const auto id{id_of(value)};

        if (!id)
        {
            refuse_id(member(), std::cref(value));
        }

        if (std::ranges::binary_search(specials, *id, {}, &Special_token::id))
        {
            continue;
        }

        auto bytes{unmapped(characters)};

        if (!bytes)
        {
            refuse_holding(member(), bytes.error());
        }

        entries.emplace_back(std::move(*bytes), *id);
    }

    std::vector<std::string> spellings(entries.size());

    std::vector<bool> placed(entries.size());

    for (auto& [bytes, token] : entries)
    {
        if (token >= entries.size() || placed[token])
        {
            throw std::invalid_argument{std::format(
                    "read_tokenizer_json: the ordinary ids of {} are not dense from zero: {} tokens, and the id {} is "
                    "beyond them or repeats",
                    name_, entries.size(), token)};
        }

        placed[token] = true;

        spellings[token] = std::move(bytes);
    }

    // The ids are dense, so each spelling is seen once with the id it is placed at.
    std::map<std::string_view, std::size_t> ids{};

    for (const auto& [id, spelling] : std::views::enumerate(spellings))
    {
        const auto [listed, inserted]{ids.try_emplace(spelling, id)};

        if (!inserted)
        {
            const auto& [listed_spelling, listed_id]{*listed};

            throw std::invalid_argument{std::format(
                    "read_tokenizer_json: the token at id {} of {} repeats the spelling of the token at id {}", id,
                    name_, listed_id)};
        }
    }

    return spellings;
}

void Tokenizer_json::refuse_holding(const std::string_view member, const std::string_view held) const
{
    throw std::invalid_argument{std::format("read_tokenizer_json: {} of {} holds {}", member, name_, held)};
}

std::vector<munch::verifier::Merge> Tokenizer_json::merges(const hopper::json::Object& model) const
{
    const auto listed{member_of(model, "merges")};

    if (!listed || !is_array(*listed))
    {
        refuse("model.merges", described(listed), "an array of merges");
    }

    std::vector<munch::verifier::Merge> table{};

    std::map<std::pair<std::string, std::string>, std::size_t> ranks{};

    // A merge is a string of two parts at its first space, or an array of the two.
    const auto parts_of{
            [](const hopper::json::Value& merge) -> std::optional<std::pair<std::string_view, std::string_view>> {
                if (is_string(merge))
                {
                    const std::string_view text{merge.as_string()};

                    const auto space{text.find(' ')};

                    if (space == std::string_view::npos)
                    {
                        return std::nullopt;
                    }

                    return std::pair{text.substr(0, space), text.substr(space + 1)};
                }

                const auto two_strings{
                        is_array(merge) && merge.as_array().elements.size() == 2 &&
                        std::ranges::all_of(merge.as_array().elements, is_string)};

                if (!two_strings)
                {
                    return std::nullopt;
                }

                const auto& elements{merge.as_array().elements};

                return std::pair{
                        std::string_view{elements.front().as_string()}, std::string_view{elements.back().as_string()}};
            }};

    for (const auto& [rank, merge] : std::views::enumerate(listed->get().as_array().elements))
    {
        const auto member{[rank] { return std::format("model.merges[{}]", rank); }};

        const auto parts{parts_of(merge)};

        constexpr std::string_view taken{
                "two nonempty parts separated by a space, or an array of two nonempty strings"};

        if (!parts)
        {
            const auto found{described(std::cref(merge))};

            refuse(member(), found, taken);
        }

        const auto [left_characters, right_characters]{*parts};

        if (left_characters.empty() || right_characters.empty())
        {
            const auto found{described(std::cref(merge))};

            refuse(member(), found, taken);
        }

        auto left{unmapped(left_characters)};

        auto right{unmapped(right_characters)};

        if (!left || !right)
        {
            refuse_holding(member(), left ? right.error() : left.error());
        }

        const auto [repeated, inserted]{ranks.try_emplace({*left, *right}, rank)};

        if (!inserted)
        {
            const auto& [pair, repeated_rank]{*repeated};

            throw std::invalid_argument{std::format(
                    "read_tokenizer_json: {} of {}, the merge ({}, {}), repeats the merge at rank {}", member(), name_,
                    quoted_text(*left), quoted_text(*right), repeated_rank)};
        }

        table.push_back({.left = std::move(*left), .right = std::move(*right)});
    }

    return table;
}

} // namespace

Definition read_tokenizer_json(const std::filesystem::path& path, const std::string_view published)
{
    const Tokenizer_json file{path};

    return file.definition(published);
}

std::string tokenizer_json_spelling(const std::filesystem::path& path)
{
    const Tokenizer_json file{path};

    auto [spelling, member, found]{file.recognized()};

    return std::move(spelling);
}

} // namespace matisse::load
