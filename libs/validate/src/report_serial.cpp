#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <munch/verifier/verifier_decisions.hpp>

#include "matisse/validate/report.hpp"

namespace matisse::validate
{
namespace
{
/**
 * @brief The first line of the text, naming its form and the version of it.
 */
constexpr std::string_view header{"matisse report 1"};

/**
 * @brief The widest table the text holds, since an origin is written as one digit.
 */
constexpr std::size_t widest_table{9};

/**
 * @brief What stands for an empty field, and in a line of windows for a window that is not certified.
 */
constexpr char absent{'-'};

/**
 * @brief The digits of a byte written in hex, lower case.
 */
constexpr std::string_view hex_digits{"0123456789abcdef"};

/**
 * @brief The class symbols' places in the alphabet's order, by byte, the digits a window's index is written in.
 */
using Ranks_t = std::array<std::size_t, 256>;

/**
 * @brief Returns an inventory's name, as the text spells it.
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
 * @brief Returns the place of every class symbol in the alphabet's order, by byte.
 * @param symbols The class symbols, ascending.
 * @return The places, zero for a byte that is no symbol.
 */
[[nodiscard]] Ranks_t ranks_of(const std::string_view symbols) noexcept
{
    Ranks_t ranks{};

    for (std::size_t rank{0}; rank < symbols.size(); ++rank)
    {
        ranks[static_cast<unsigned char>(symbols[rank])] = rank;
    }

    return ranks;
}

/**
 * @brief Returns the number of windows of a width over a number of class symbols.
 * @param symbols The number of class symbols.
 * @param width The width.
 * @return The number, the symbols to the power of the width.
 */
[[nodiscard]] std::size_t window_count(const std::size_t symbols, const std::size_t width) noexcept
{
    std::size_t count{1};

    for (std::size_t symbol{0}; symbol < width; ++symbol)
    {
        count *= symbols;
    }

    return count;
}

/**
 * @brief Returns bytes written in hex, or `-` for none.
 * @param bytes The bytes.
 * @return The hex.
 */
[[nodiscard]] std::string hex_of(const std::string_view bytes)
{
    if (bytes.empty())
    {
        return std::string{absent};
    }

    std::string hex{};

    hex.reserve(2 * bytes.size());

    for (const auto byte : bytes)
    {
        const auto value{static_cast<unsigned char>(byte)};

        hex += hex_digits[value >> 4U];
        hex += hex_digits[value & 0x0FU];
    }

    return hex;
}

/**
 * @brief Returns a marked string's two fields: its bytes in hex and its boundaries as `0` and `1`, each `-` when
 *        empty.
 * @param marked The marked string.
 * @return The two fields, a space between them.
 */
[[nodiscard]] std::string marked_fields(const munch::verifier::Marked_string& marked)
{
    const auto& [bytes, boundaries]{marked};

    std::string bits{};

    for (const auto boundary : boundaries)
    {
        bits += boundary ? '1' : '0';
    }

    if (bits.empty())
    {
        bits += absent;
    }

    return std::format("{} {}", hex_of(bytes), bits);
}

/**
 * @brief Returns the gap's line: its bound, or `unbounded` with the stem, the loop and the suffix of its lasso.
 * @param gap The boundary gap.
 * @return The line, its newline included.
 */
[[nodiscard]] std::string gap_line(const munch::verifier::Boundary_gap& gap)
{
    if (const auto* const bound{std::get_if<std::size_t>(&gap)})
    {
        return std::format("gap {}\n", *bound);
    }

    const auto& [stem, loop, suffix]{std::get<munch::verifier::Lasso>(gap)};

    return std::format("gap unbounded {} {} {}\n", marked_fields(stem), marked_fields(loop), marked_fields(suffix));
}

/**
 * @brief Returns a window table's lines: the table's inventory, width and right context, a line for each failing
 *        truncation, and a line of the windows of each width from two to the table's.
 * @param table The window table.
 * @param symbols The class symbols, ascending.
 * @return The lines, a newline after every one.
 * @throws std::invalid_argument If the table is wider than nine symbols.
 */
[[nodiscard]] std::string table_lines(const Window_table& table, const std::string_view symbols)
{
    const auto& [inventory, width, windows, right_context, failing]{table};

    if (width > widest_table)
    {
        throw std::invalid_argument{std::format(
                "serialize: the {} table is {} symbols wide, wider than the {} an origin's one digit holds",
                inventory_name(inventory), width, widest_table)};
    }

    auto lines{std::format("table {} {} {}\n", inventory_name(inventory), width, right_context)};

    for (const auto& [segmentation, occurrence, cut, context] : failing)
    {
        lines += std::format("failing {} {} {} {}\n", marked_fields(segmentation), occurrence, cut, context);
    }

    const auto ranks{ranks_of(symbols)};

    // One line per width, every window of it a character in the order the class symbols give.
    std::vector<std::string> widths{};

    for (std::size_t window_width{2}; window_width <= width; ++window_width)
    {
        widths.emplace_back(window_count(symbols.size(), window_width), absent);
    }

    for (const auto& [window, origin] : windows)
    {
        std::size_t index{0};

        for (const auto symbol : window)
        {
            index = index * symbols.size() + ranks[static_cast<unsigned char>(symbol)];
        }

        widths[window.size() - 2][index] = static_cast<char>('0' + origin);
    }

    for (std::size_t window_width{2}; window_width <= width; ++window_width)
    {
        lines += std::format("windows {} {}\n", window_width, widths[window_width - 2]);
    }

    return lines;
}

/**
 * @brief A reader of the text serialize() writes, line by line, each line's fields separated by one space.
 */
class Serial_reader
{
public:
    /**
     * @brief Constructs the reader at the text's first line.
     * @param text The text.
     */
    explicit Serial_reader(const std::string_view text) noexcept : text_{text} {}

    /**
     * @brief Returns whether the next line begins with a keyword.
     * @param keyword The keyword.
     * @return True when a next line stands and its first field is the keyword.
     */
    [[nodiscard]] bool at(const std::string_view keyword) const noexcept
    {
        const auto end{std::min(text_.find('\n', start_), text_.size())};

        const auto line{text_.substr(start_, end - start_)};

        return start_ < text_.size() && line.substr(0, line.find(' ')) == keyword;
    }

    /**
     * @brief Returns whether every line has been read.
     * @return True at the text's end.
     */
    [[nodiscard]] bool done() const noexcept { return start_ >= text_.size(); }

    /**
     * @brief Reads the next line, which must end in a newline, and returns it whole.
     * @return The line, without its newline.
     * @throws std::invalid_argument If no line is left or the last does not end in a newline.
     */
    [[nodiscard]] std::string_view line()
    {
        const auto end{text_.find('\n', start_)};

        ++number_;

        if (start_ >= text_.size() || end == std::string_view::npos)
        {
            refuse("is missing or does not end in a newline");
        }

        const auto read{text_.substr(start_, end - start_)};

        start_ = end + 1;

        return read;
    }

    /**
     * @brief Reads the next line and returns its fields after its keyword, which must be as many as given.
     * @param keyword The keyword the line begins with.
     * @param count The number of fields after the keyword.
     * @return The fields.
     * @throws std::invalid_argument If the line does not begin with the keyword or has another number of fields.
     */
    [[nodiscard]] std::vector<std::string_view> fields(const std::string_view keyword, const std::size_t count)
    {
        auto split{split_of(line())};

        if (split.front() != keyword || split.size() != count + 1)
        {
            refuse(std::format("is not a {} line of {} fields", keyword, count));
        }

        split.erase(split.begin());

        return split;
    }

    /**
     * @brief Returns a line's fields, separated by one space.
     * @param line The line.
     * @return The fields, at least one.
     */
    [[nodiscard]] static std::vector<std::string_view> split_of(const std::string_view line)
    {
        std::vector<std::string_view> split{};

        for (std::size_t start{0}; start <= line.size();)
        {
            const auto end{std::min(line.find(' ', start), line.size())};

            split.push_back(line.substr(start, end - start));

            start = end + 1;
        }

        return split;
    }

    /**
     * @brief Returns a field read as a decimal number.
     * @param field The field.
     * @return The number.
     * @throws std::invalid_argument If the field is not digits alone.
     */
    [[nodiscard]] std::size_t number(const std::string_view field) const
    {
        std::size_t value{};

        const auto field_end{field.data() + field.size()};

        const auto [stop, error]{std::from_chars(field.data(), field_end, value)};

        if (field.empty() || error != std::errc{} || stop != field_end)
        {
            refuse(std::format("holds '{}' where a number stands", field));
        }

        return value;
    }

    /**
     * @brief Returns a marked string read from its two fields, its bytes in hex and its boundaries as `0` and `1`.
     * @param hex The bytes' field.
     * @param bits The boundaries' field.
     * @return The marked string.
     * @throws std::invalid_argument If a field is not as marked_fields() writes it.
     */
    [[nodiscard]] munch::verifier::Marked_string marked(const std::string_view hex, const std::string_view bits) const
    {
        return {.bytes = bytes_of(hex), .boundaries = boundaries_of(bits)};
    }

    /**
     * @brief Refuses the line last read.
     * @param what What is wrong with it.
     * @throws std::invalid_argument Always, naming the line's number.
     */
    [[noreturn]] void refuse(const std::string_view what) const
    {
        throw std::invalid_argument{std::format("deserialize: line {} {}", number_, what)};
    }

private:
    /**
     * @brief Returns bytes read from hex, none for `-`.
     * @param hex The hex.
     * @return The bytes.
     * @throws std::invalid_argument If the field is not hex digits in pairs.
     */
    [[nodiscard]] std::string bytes_of(const std::string_view hex) const
    {
        std::string bytes{};

        if (hex == std::string_view{&absent, 1})
        {
            return bytes;
        }

        if (hex.size() % 2 != 0)
        {
            refuse(std::format("holds '{}' where bytes in hex stand", hex));
        }

        for (std::size_t at{0}; at < hex.size(); at += 2)
        {
            const auto high{hex_digits.find(hex[at])};

            const auto low{hex_digits.find(hex[at + 1])};

            if (high == std::string_view::npos || low == std::string_view::npos)
            {
                refuse(std::format("holds '{}' where bytes in hex stand", hex));
            }

            bytes += static_cast<char>(high << 4U | low);
        }

        return bytes;
    }

    /**
     * @brief Returns boundaries read from `0` and `1`, none for `-`.
     * @param bits The field.
     * @return The boundaries.
     * @throws std::invalid_argument If the field holds another character.
     */
    [[nodiscard]] std::vector<bool> boundaries_of(const std::string_view bits) const
    {
        std::vector<bool> boundaries{};

        if (bits == std::string_view{&absent, 1})
        {
            return boundaries;
        }

        for (const auto bit : bits)
        {
            if (bit != '0' && bit != '1')
            {
                refuse(std::format("holds '{}' where boundaries stand", bits));
            }

            boundaries.push_back(bit == '1');
        }

        return boundaries;
    }

    /**
     * @brief The text.
     */
    std::string_view text_;

    /**
     * @brief Where the next line begins.
     */
    std::size_t start_{0};

    /**
     * @brief The number of the line last read, from one.
     */
    std::size_t number_{0};
};

/**
 * @brief Reads the boundary gap's line.
 * @param reader The reader, before the line.
 * @return The gap.
 * @throws std::invalid_argument If the line is not a bound or `unbounded` with a lasso's three marked strings.
 */
[[nodiscard]] munch::verifier::Boundary_gap gap_of(Serial_reader& reader)
{
    const auto fields{Serial_reader::split_of(reader.line())};

    if (fields.size() == 8 && fields[0] == "gap" && fields[1] == "unbounded")
    {
        return munch::verifier::Lasso{
                .stem = reader.marked(fields[2], fields[3]),
                .loop = reader.marked(fields[4], fields[5]),
                .suffix = reader.marked(fields[6], fields[7]),
        };
    }

    if (fields.size() != 2 || fields[0] != "gap")
    {
        reader.refuse("is not the gap line");
    }

    return reader.number(fields[1]);
}

/**
 * @brief Reads a window table's lines.
 * @param reader The reader, before the table's line.
 * @param inventory The inventory the table must hold.
 * @param symbols The class symbols, ascending.
 * @return The table.
 * @throws std::invalid_argument If a line is not as table_lines() writes it for the inventory, the table is wider than
 *         nine symbols, or a line of windows holds another number of windows than its width has or an origin outside
 *         its width.
 */
[[nodiscard]] Window_table table_of(Serial_reader& reader, const Inventory inventory, const std::string_view symbols)
{
    const auto head{reader.fields("table", 3)};

    if (head[0] != inventory_name(inventory))
    {
        reader.refuse(std::format("is not the {} table's line", inventory_name(inventory)));
    }

    const auto width{reader.number(head[1])};

    if (width > widest_table)
    {
        reader.refuse(std::format("names a table wider than {} symbols", widest_table));
    }

    Window_table table{.inventory = inventory, .width = width, .right_context = reader.number(head[2])};

    while (reader.at("failing"))
    {
        const auto fields{reader.fields("failing", 5)};

        table.failing.push_back({
                .segmentation = reader.marked(fields[0], fields[1]),
                .occurrence = reader.number(fields[2]),
                .cut = reader.number(fields[3]),
                .context = reader.number(fields[4]),
        });
    }

    for (std::size_t window_width{2}; window_width <= width; ++window_width)
    {
        const auto fields{reader.fields("windows", 2)};

        const auto& dense{fields[1]};

        if (reader.number(fields[0]) != window_width || dense.size() != window_count(symbols.size(), window_width))
        {
            reader.refuse(std::format("is not the line of every window of width {}", window_width));
        }

        // The windows of the width in the order the class symbols give, a window's places the digits of its index.
        for (std::size_t index{0}; index < dense.size(); ++index)
        {
            const auto mark{dense[index]};

            if (mark == absent)
            {
                continue;
            }

            const auto origin{static_cast<std::size_t>(mark - '0')};

            if (mark < '0' || origin >= window_width)
            {
                reader.refuse(std::format("holds '{}' where an origin of width {} stands", mark, window_width));
            }

            std::string window(window_width, absent);

            auto rest{index};

            for (auto place{window_width}; place > 0; --place)
            {
                window[place - 1] = symbols[rest % symbols.size()];

                rest /= symbols.size();
            }

            table.windows.push_back({.window = std::move(window), .origin = origin});
        }
    }

    return table;
}

} // namespace

std::string serialize(const Report& report)
{
    const auto& [factorization, policy, gap, pattern]{report};

    const auto& symbols{pattern.alphabet.symbols};

    auto text{std::format("{}\nalphabet {}\n", header, hex_of(symbols))};

    text += gap_line(gap);
    text += table_lines(factorization, symbols);
    text += table_lines(policy, symbols);

    return text;
}

Report deserialize(const std::string_view text, const pretok::Pattern& pattern)
{
    Serial_reader reader{text};

    if (reader.line() != header)
    {
        reader.refuse(std::format("is not '{}'", header));
    }

    const auto& symbols{pattern.alphabet.symbols};

    const auto alphabet{reader.fields("alphabet", 1)};

    if (alphabet[0] != hex_of(symbols))
    {
        reader.refuse(std::format("names the class symbols {}, not the pattern's {}", alphabet[0], hex_of(symbols)));
    }

    auto gap{gap_of(reader)};

    auto factorization{table_of(reader, Inventory::factorization, symbols)};

    auto policy{table_of(reader, Inventory::policy, symbols)};

    if (!reader.done())
    {
        std::ignore = reader.line();

        reader.refuse("follows the policy table");
    }

    return Report{std::move(factorization), std::move(policy), std::move(gap), pattern};
}

} // namespace matisse::validate
