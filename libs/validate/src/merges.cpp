#include "matisse/validate/merges.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <munch/verifier/token_automaton.hpp>

#include "matisse/validate/report.hpp"

namespace matisse::validate
{
namespace
{
/**
 * @brief The rank of every merge of a table by its two parts' spellings.
 */
using Ranks_t = std::map<std::pair<std::string, std::string>, std::size_t>;

/**
 * @brief How many examples a list in the text shows before it counts the rest.
 */
constexpr std::size_t shown_examples{6};

/**
 * @brief The first byte past ASCII's printable range, the delete control.
 */
constexpr unsigned char delete_control{0x7F};

/**
 * @brief Returns the rank of every merge of a table by its two parts.
 * @param merges The merge table in rank order.
 * @return The ranks.
 */
[[nodiscard]] Ranks_t ranks_of(const std::span<const munch::verifier::Merge> merges)
{
    Ranks_t ranks{};

    for (const auto [rank, merge] : std::views::enumerate(merges))
    {
        const auto& [left, right]{merge};

        ranks.emplace(std::pair{left, right}, static_cast<std::size_t>(rank));
    }

    return ranks;
}

/**
 * @brief Returns where the token covering a window's last byte begins in the window's own encoding: at every step the
 *        merge of least rank present among the adjacent tokens, applied at its leftmost occurrence, until none applies.
 * @param ranks The rank of every merge by its parts.
 * @param window The window, nonempty.
 * @return The index of the covering token's first byte.
 */
[[nodiscard]] std::size_t own_origin(const Ranks_t& ranks, const std::string_view window)
{
    std::vector<std::string> tokens{};

    for (const auto byte : window)
    {
        tokens.emplace_back(1, byte);
    }

    while (true)
    {
        std::optional<std::size_t> least_at{};

        std::size_t least_rank{};

        for (std::size_t at{0}; at + 1 < tokens.size(); ++at)
        {
            const auto found{ranks.find(std::pair{tokens[at], tokens[at + 1]})};

            if (found == ranks.end())
            {
                continue;
            }

            const auto& [parts, rank]{*found};

            // The leftmost occurrence wins a tie, so only a strictly lower rank replaces the least found.
            if (least_at && least_rank <= rank)
            {
                continue;
            }

            least_at = at;

            least_rank = rank;
        }

        if (!least_at)
        {
            break;
        }

        const auto at{*least_at};

        tokens[at] += tokens[at + 1];

        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }

    return window.size() - tokens.back().size();
}

/**
 * @brief Returns the verdict in one sentence: how many of the candidates certify a cut.
 * @param certified The certified windows' count.
 * @param candidates The candidates' count.
 * @return The sentence.
 */
[[nodiscard]] std::string verdict(const std::size_t certified, const std::size_t candidates)
{
    if (certified == 0)
    {
        return std::format("no window of the {} candidate{} certifies a cut", candidates, plural(candidates));
    }

    return std::format("candidate windows certify a cut: {} of {}", certified, candidates);
}

/**
 * @brief Returns one row of the text: its label, padded to the column the values stand in, and its value.
 * @param label The label, empty for a row continuing the one before.
 * @param value The value.
 * @return The row, its newline included.
 */
[[nodiscard]] std::string row(const std::string_view label, const std::string& value)
{
    return std::format("{:<28}{}\n", label, value);
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
 * @brief Returns the escape a byte is written as inside the quotes the text shows it in, when it has one: a common
 *        control, the backslash, and the quote.
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
 * @brief Returns bytes as the text prints them, each byte the character when it is printable, an escape when it is a
 *        common control, and the escape `\xHH` otherwise.
 * @param bytes The bytes.
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
 * @brief Returns where a marked string's boundaries stand, each the offset after a byte a boundary follows.
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
 * @brief Returns offsets written one after another with a separator between every two.
 * @param offsets The offsets.
 * @param separator What stands between two offsets.
 * @return The text, empty for no offset.
 */
[[nodiscard]] std::string joined(const std::vector<std::size_t>& offsets, const std::string_view separator)
{
    std::string text{};

    for (const auto offset : offsets)
    {
        if (!text.empty())
        {
            text += separator;
        }

        text += std::to_string(offset);
    }

    return text;
}

/**
 * @brief Returns one kind of decided window's rows: the count, by width ascending, and the first few as examples,
 *        printable windows first, since a reader recognizes those.
 * @param label The rows' label.
 * @param windows The windows of the kind, in the order decided.
 * @return The rows.
 */
[[nodiscard]] std::string window_rows(const std::string_view label, const std::vector<Merge_window>& windows)
{
    std::map<std::size_t, std::size_t> per_width{};

    for (const auto& [window, origin, refutation] : windows)
    {
        ++per_width[window.size()];
    }

    std::string summary{};

    for (const auto& [width, count] : per_width)
    {
        const std::string_view separator{summary.empty() ? "" : ", "};

        summary += std::format("{}{} at width {}", separator, count, width);
    }

    const auto value{windows.empty() ? std::string{"none"} : std::format("{}: {}", windows.size(), summary)};

    auto out{row(label, value)};

    // A window of printable ASCII makes the better example.
    const auto printable_window{[](const Merge_window& decided) {
        const auto& [window, origin, refutation]{decided};

        const auto printable{[](const char byte) { return is_printable(static_cast<unsigned char>(byte)); }};

        return std::ranges::all_of(window, printable);
    }};

    auto examples{windows};

    std::ranges::stable_partition(examples, printable_window);

    for (const auto& [window, origin, refutation] : examples | std::views::take(shown_examples))
    {
        auto example{std::format("{} at {}", shown(window), origin)};

        if (refutation)
        {
            const auto& [segmentation, occurrence]{*refutation};

            const auto offsets{boundaries_of(segmentation)};

            const auto boundaries{offsets.empty() ? std::string{"one token"} : "boundaries " + joined(offsets, " ")};

            example += std::format(" fails at {} of {}, {}", occurrence, shown(segmentation.bytes), boundaries);
        }

        out += row("", example);
    }

    if (windows.size() > shown_examples)
    {
        const auto more{std::format("... {} more", windows.size() - shown_examples)};

        out += row("", more);
    }

    return out;
}

/**
 * @brief Returns a byte string as a JSON string, each byte the code point of its value: a quote or a backslash
 *        escaped, a byte that is not printable ASCII as the escape of its value, and any other as written.
 * @param bytes The bytes.
 * @return The JSON text, quotes included.
 */
[[nodiscard]] std::string json_byte_string(const std::string_view bytes)
{
    std::string out{'"'};

    for (const auto byte : bytes)
    {
        const auto value{static_cast<unsigned char>(byte)};

        if (byte == '"' || byte == '\\')
        {
            out += std::format(R"(\{})", byte);
        }
        else if (is_printable(value))
        {
            out += byte;
        }
        else
        {
            out += std::format(R"(\u{:04x})", value);
        }
    }

    return out + '"';
}

/**
 * @brief Returns one decided window as JSON: its bytes, its origin, and its refutation, the segmentation's text and
 *        boundaries and the occurrence, or null.
 * @param decided The window decided.
 * @return The JSON text, one line.
 */
[[nodiscard]] std::string json_window(const Merge_window& decided)
{
    const auto& [window, origin, refutation]{decided};

    const auto refutation_json{[&refutation]() -> std::string {
        if (!refutation)
        {
            return "null";
        }

        const auto& [segmentation, occurrence]{*refutation};

        const auto boundaries{joined(boundaries_of(segmentation), ", ")};

        return std::format(
                R"({{"segmentation": {{"text": {}, "boundaries": [{}]}}, "occurrence": {}}})",
                json_byte_string(segmentation.bytes), boundaries, occurrence);
    }()};

    return std::format(
            R"({{"window": {}, "origin": {}, "refutation": {}}})", json_byte_string(window), origin, refutation_json);
}

} // namespace

std::vector<Merge_window> validate_merges(
        const load::Definition& definition, const std::span<const std::string> candidates)
{
    if (definition.merge_rule != load::Merge_rule::pairs)
    {
        throw std::invalid_argument{
                "validate_merges: the definition merges by the rank rule, which carries no merge table"};
    }

    if (definition.whole_piece_lookup)
    {
        throw std::invalid_argument{
                "validate_merges: the definition looks a whole piece up, which with no pre-tokenizer takes a whole "
                "text that is a token as that token"};
    }

    const auto empty{std::ranges::find_if(candidates, std::ranges::empty)};

    if (empty != candidates.end())
    {
        const auto index{std::ranges::distance(candidates.begin(), empty)};

        throw std::invalid_argument{std::format("validate_merges: candidate {} is empty", index)};
    }

    const auto automaton{munch::verifier::token_automaton(definition.merges)};

    const auto ranks{ranks_of(definition.merges)};

    std::vector<Merge_window> decided{};

    decided.reserve(candidates.size());

    for (const auto& window : candidates)
    {
        const auto origin{own_origin(ranks, window)};

        auto refutation{munch::verifier::miscovering(automaton, window, origin)};

        decided.push_back(Merge_window{.window = window, .origin = origin, .refutation = std::move(refutation)});
    }

    return decided;
}

std::string render_merges(const std::span<const Merge_window> windows)
{
    std::vector<Merge_window> certified{};

    std::vector<Merge_window> refuted{};

    for (const auto& decided : windows)
    {
        auto& kind{decided.refutation ? refuted : certified};

        kind.push_back(decided);
    }

    const auto sentence{verdict(certified.size(), windows.size())};

    auto out{row("verdict", sentence)};

    out += window_rows("certified windows", certified);

    out += window_rows("refuted windows", refuted);

    return out;
}

std::string merges_json(const std::span<const Merge_window> windows)
{
    const auto is_certified{[](const Merge_window& decided) { return !decided.refutation; }};

    const auto certified{static_cast<std::size_t>(std::ranges::count_if(windows, is_certified))};

    const auto sentence{verdict(certified, windows.size())};

    const auto verdict_json{json_byte_string(sentence)};

    std::string windows_json{};

    for (const auto& decided : windows)
    {
        const std::string_view separator{windows_json.empty() ? "" : ", "};

        windows_json += std::format("{}{}", separator, json_window(decided));
    }

    return std::format("{{\n  \"verdict\": {},\n  \"windows\": [{}]\n}}", verdict_json, windows_json);
}

} // namespace matisse::validate
