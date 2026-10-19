#include <algorithm>
#include <cstddef>
#include <format>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <munch/verifier/verifier_decisions.hpp>

#include "matisse/validate/report.hpp"

namespace matisse::validate
{
namespace
{
/**
 * @brief How many examples a list in the text report shows before it counts the rest.
 */
constexpr std::size_t shown_examples{6};

/**
 * @brief The first byte past ASCII's printable range, the delete control.
 */
constexpr unsigned char delete_control{0x7F};

/**
 * @brief Returns the report's answer in one sentence: how many windows certify a cut under each certificate, up to the
 *        width decided.
 * @param report The report.
 * @return The sentence.
 */
[[nodiscard]] std::string verdict(const Report& report)
{
    const auto& [factorization, policy, gap]{report};

    if (factorization.windows.empty() && policy.windows.empty())
    {
        return std::format("no window up to width {} certifies a cut under either certificate", factorization.width);
    }

    return std::format(
            "windows certify a cut: {} under the factorization certificate and {} under the policy certificate, up "
            "to width {}",
            factorization.windows.size(), policy.windows.size(), factorization.width);
}

/**
 * @brief Returns one row of the text report: its label, padded to the column the values stand in, and its value.
 * @param label The label, empty for a row continuing the one before.
 * @param value The value.
 * @return The row, its newline included.
 */
[[nodiscard]] std::string row(const std::string_view label, const std::string& value)
{
    return std::format("{:<28}{}\n", label, value);
}

/**
 * @brief Returns the items written one after another with a separator between every two.
 * @tparam Items The type of the items' range.
 * @tparam One The type of what writes one item.
 * @param items The items.
 * @param separator What stands between two items.
 * @param one What an item is written as.
 * @return The text, empty for no item.
 */
template <typename Items, typename One>
[[nodiscard]] std::string joined(const Items& items, const std::string_view separator, const One& one)
{
    std::string text{};

    for (const auto& item : items)
    {
        if (!text.empty())
        {
            text += separator;
        }

        text += one(item);
    }

    return text;
}

/**
 * @brief Returns an inventory's name, as the text and the JSON spell it.
 * @param inventory The inventory.
 * @return The name.
 */
[[nodiscard]] std::string_view inventory_name(const Inventory inventory) noexcept
{
    switch (inventory)
    {
    case Inventory::factorization:
        return "factorization";
    case Inventory::policy:
        break;
    }

    return "policy";
}

/**
 * @brief Returns whether a byte is printable ASCII, the space among them.
 * @param byte The byte.
 * @return True for the space through `~`.
 */
[[nodiscard]] constexpr bool is_printable(const unsigned char byte) noexcept
{
    return byte >= ' ' && byte < delete_control;
}

/**
 * @brief Returns the escape a byte of a class string is written as inside the quotes the report shows it in, when it
 *        has one: a common control, the backslash, and the quote.
 * @param byte The byte.
 * @return The escape, or std::nullopt for a byte written as itself or in hex.
 */
[[nodiscard]] std::optional<std::string_view> quoted_escape(const unsigned char byte)
{
    switch (byte)
    {
    case '\n':
        return R"(\n)";
    case '\t':
        return R"(\t)";
    case '\r':
        return R"(\r)";
    case '\\':
        return R"(\\)";
    case '"':
        return R"(\")";
    default:
        break;
    }

    return std::nullopt;
}

/**
 * @brief Returns a class string as the report prints it, each byte the character when it is printable, an escape when
 *        it is a common control, and the escape `\xHH` otherwise.
 * @param bytes The class string.
 * @return The rendering, quoted.
 */
[[nodiscard]] std::string shown(const std::string_view bytes)
{
    std::string out{'"'};

    for (const auto byte : bytes)
    {
        const auto value{static_cast<unsigned char>(byte)};

        if (const auto escape{quoted_escape(value)})
        {
            out += *escape;
        }
        else if (is_printable(value))
        {
            out += byte;
        }
        else
        {
            out += std::format(R"(\x{:02X})", value);
        }
    }

    return out + '"';
}

/**
 * @brief Returns the windows' rows of one table: the count, by width ascending, and the first few as examples,
 *        printable windows first, since a reader recognises those.
 * @param table The window table.
 * @return The rows.
 */
[[nodiscard]] std::string window_rows(const Window_table& table)
{
    const auto& [inventory, width, windows, right_context, failing]{table};

    std::map<std::size_t, std::size_t> per_width{};

    for (const auto& [window, origin] : windows)
    {
        ++per_width[window.size()];
    }

    const auto at_width{[]<typename Entry>(const Entry& entry) {
        const auto& [length, count]{entry};

        return std::format("{} at width {}", count, length);
    }};

    const auto summary{joined(per_width, ", ", at_width)};

    const auto label{std::format("{} windows", inventory_name(inventory))};

    const auto value{windows.empty() ? std::string{"none"} : std::format("{}: {}", windows.size(), summary)};

    auto out{row(label, value)};

    // A window of printable ASCII makes the better example.
    const auto printable_window{[](const Certified_window& certified) {
        const auto& [window, origin]{certified};

        const auto printable{[](const char byte) { return is_printable(static_cast<unsigned char>(byte)); }};

        return std::ranges::all_of(window, printable);
    }};

    auto examples{windows};

    std::ranges::stable_partition(examples, printable_window);

    for (const auto& [window, origin] : examples | std::views::take(shown_examples))
    {
        out += row("", std::format("{} at {}", shown(window), origin));
    }

    if (windows.size() > shown_examples)
    {
        out += row("", std::format("... {} more", windows.size() - shown_examples));
    }

    return out;
}

/**
 * @brief Returns where a marked string's boundaries stand, each the offset after a symbol a boundary follows.
 * @param marked The marked string.
 * @return The offsets, ascending.
 */
[[nodiscard]] std::vector<std::size_t> boundaries_of(const munch::verifier::Marked_string& marked)
{
    std::vector<std::size_t> offsets{};

    for (const auto [at, boundary] : std::views::enumerate(marked.boundaries))
    {
        if (boundary)
        {
            offsets.push_back(static_cast<std::size_t>(at) + 1);
        }
    }

    return offsets;
}

/**
 * @brief Returns the context rows of one table: the right context its cuts need, then each context that fails with its
 *        shortest truncation, the text, its cut and the boundaries the split verifier accepts it with.
 * @param table The window table.
 * @return The rows.
 */
[[nodiscard]] std::string context_rows(const Window_table& table)
{
    const auto& [inventory, width, windows, right_context, failing]{table};

    const auto label{std::format("{} context", inventory_name(inventory))};

    const auto value{std::format("{} code point{} read past the cut", right_context, plural(right_context))};

    auto out{row(label, value)};

    const auto number{[](const std::size_t offset) { return std::to_string(offset); }};

    for (const auto& [segmentation, occurrence, cut, context] : failing)
    {
        const auto boundaries{joined(boundaries_of(segmentation), " ", number)};

        const auto text{shown(segmentation.bytes)};

        out += row("", std::format("{} fails at cut {} of {}, boundaries {}", context, cut, text, boundaries));
    }

    return out;
}

/**
 * @brief Returns the boundary gap's row: the supremum, or the loop whose repetition carries no boundary.
 * @param gap The boundary gap.
 * @return The row.
 */
[[nodiscard]] std::string gap_row(const munch::verifier::Boundary_gap& gap)
{
    if (std::holds_alternative<std::size_t>(gap))
    {
        const auto supremum{std::get<std::size_t>(gap)};

        return row("boundary gap", std::format("{} code point{}", supremum, plural(supremum)));
    }

    const auto& [stem, loop, suffix]{std::get<munch::verifier::Lasso>(gap)};

    return row("boundary gap", std::format("unbounded: {} repeats with no boundary", shown(loop.bytes)));
}

/**
 * @brief Writes one character of a JSON string: a quote or a backslash escaped, a byte that may not stand bare as the
 *        escape of its value, and any other as written.
 * @param out The JSON text, the character added.
 * @param byte The character's first byte.
 * @param bare Whether the character may stand as written.
 * @param written The character as written.
 */
void put_json_character(std::string& out, const char byte, const bool bare, const std::string_view written)
{
    if (byte == '"' || byte == '\\')
    {
        out += std::format(R"(\{})", byte);

        return;
    }

    if (!bare)
    {
        out += std::format(R"(\u{:04x})", static_cast<unsigned char>(byte));

        return;
    }

    out += written;
}

/**
 * @brief Returns a byte string as a JSON string, each byte the code point of its value.
 * @param bytes The bytes.
 * @return The JSON text, quotes included.
 */
[[nodiscard]] std::string json_byte_string(const std::string_view bytes)
{
    std::string out{'"'};

    for (std::size_t at{0}; at < bytes.size(); ++at)
    {
        const auto byte{bytes[at]};

        const auto bare{is_printable(static_cast<unsigned char>(byte))};

        const auto written{bytes.substr(at, 1)};

        put_json_character(out, byte, bare, written);
    }

    return out + '"';
}

/**
 * @brief Returns a list as a JSON array, each item written by the function given.
 * @tparam Items The type of the items' range.
 * @tparam One The type of what writes one item.
 * @param items The items.
 * @param one What an item is written as.
 * @return The JSON text.
 */
template <typename Items, typename One>
[[nodiscard]] std::string json_list(const Items& items, const One& one)
{
    return std::format("[{}]", joined(items, ", ", one));
}

/**
 * @brief Returns a marked string as JSON: its text and where its boundaries stand.
 * @param marked The marked string.
 * @return The JSON text.
 */
[[nodiscard]] std::string json_marked(const munch::verifier::Marked_string& marked)
{
    const auto number{[](const std::size_t offset) { return std::to_string(offset); }};

    const auto boundaries{json_list(boundaries_of(marked), number)};

    return std::format(R"({{"text": {}, "boundaries": {}}})", json_byte_string(marked.bytes), boundaries);
}

/**
 * @brief Returns one window table as JSON: the width, the windows with their origins, the right context and each
 *        failing context's truncation.
 * @param table The window table.
 * @return The JSON text, one line.
 */
[[nodiscard]] std::string json_table(const Window_table& table)
{
    const auto& [inventory, width, windows, right_context, failing]{table};

    const auto window_json{[](const Certified_window& certified) {
        const auto& [window, origin]{certified};

        return std::format(R"({{"window": {}, "origin": {}}})", json_byte_string(window), origin);
    }};

    const auto truncation_json{[](const munch::verifier::Truncation& truncation) {
        const auto& [segmentation, occurrence, cut, context]{truncation};

        return std::format(
                R"({{"segmentation": {}, "occurrence": {}, "cut": {}, "context": {}}})", json_marked(segmentation),
                occurrence, cut, context);
    }};

    const auto windows_json{json_list(windows, window_json)};

    const auto failing_json{json_list(failing, truncation_json)};

    return std::format(
            R"({{"width": {}, "windows": {}, "right_context": {}, "failing": {}}})", width, windows_json, right_context,
            failing_json);
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

} // namespace

std::string render(const Report& report)
{
    const auto& [factorization, policy, gap]{report};

    auto out{row("verdict", verdict(report))};

    out += window_rows(factorization);

    out += context_rows(factorization);

    out += window_rows(policy);

    out += context_rows(policy);

    out += gap_row(gap);

    return out;
}

std::string json(const Report& report)
{
    const auto& [factorization, policy, gap]{report};

    std::string out{"{\n"};

    auto first_member{true};

    const auto member{[&out, &first_member](const std::string_view key, const std::string& value) {
        const std::string_view separator{first_member ? "" : ",\n"};

        first_member = false;

        out += std::format(R"({}  "{}": {})", separator, key, value);
    }};

    member("verdict", json_byte_string(verdict(report)));

    member("factorization", json_table(factorization));

    member("policy", json_table(policy));

    const auto bounded{std::holds_alternative<std::size_t>(gap)};

    const auto gap_json{bounded ? std::to_string(std::get<std::size_t>(gap)) : std::string{R"("unbounded")"}};

    member("boundary_gap", gap_json);

    const auto lasso_json{[&gap, bounded]() -> std::string {
        if (bounded)
        {
            return "null";
        }

        const auto& [stem, loop, suffix]{std::get<munch::verifier::Lasso>(gap)};

        return std::format(
                R"({{"stem": {}, "loop": {}, "suffix": {}}})", json_marked(stem), json_marked(loop),
                json_marked(suffix));
    }()};

    member("gap_lasso", lasso_json);

    return std::format("{}\n}}", out);
}

std::string json_string(const std::string_view text)
{
    std::string out{'"'};

    for (std::size_t at{0}; at < text.size();)
    {
        const auto byte{text[at]};

        const auto value{static_cast<unsigned char>(byte)};

        const auto length{well_formed_length(text, at)};

        put_json_character(out, byte, value >= ' ' && length != 0, text.substr(at, length));

        at += std::max(length, 1UZ);
    }

    return out + '"';
}

} // namespace matisse::validate
