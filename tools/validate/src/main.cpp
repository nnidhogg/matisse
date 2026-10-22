#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "matisse/load/definition.hpp"
#include "matisse/lookup/tokenizers.hpp"
#include "matisse/validate/report.hpp"
#include "matisse/validate/supply.hpp"

namespace
{
/**
 * @brief The narrowest window `--windows` takes; one symbol is no window.
 */
constexpr std::size_t narrowest_width{2};

/**
 * @brief The widest window `--windows` takes.
 */
constexpr std::size_t widest_width{5};

/**
 * @brief The indent the report's own lines take where they sit in the JSON document.
 */
constexpr std::string_view report_indent{"  "};

/**
 * @brief The exit status on a command-line error, or on a file or an input the command cannot read.
 */
constexpr int exit_command_error{2};

/**
 * @brief What every diagnostic the command writes to the standard error opens with.
 */
constexpr std::string_view diagnostic_prefix{"matisse-validate: "};

/**
 * @brief The usage text: the command line the validator takes, what each option does, and what each exit status means.
 */
constexpr std::string_view usage_text{R"(usage: matisse-validate [options] FILE

Reads a tokenizer definition, a Hugging Face tokenizer.json or a tiktoken rank file, and reports which windows of its
pre-tokenizer's classes certify a cut, at which origin and with which right context, under the factorization and the
policy certificate.

  --tiktoken PATTERN    read FILE as a tiktoken rank file whose pre-tokenizer is PATTERN, gpt2, cl100k, o200k or
                        llama3, since the file names none; a tokenizer.json, read without it, has no reader yet
  --windows N           the widest window decided, 2 to 5, 5 unless given
  --input FILE          measure each inventory's certified-anchor supply on this file: the anchors, per kibibyte, the
                        gaps between them, the share of the pre-token boundaries they reach and the longest piece
  --json                one JSON document instead of text

Exit status is 0 when the definition was read and reported, 1 when it was refused, and 2 on a command-line error or on
a file or an input the command cannot read.
)"};

/**
 * @brief What the command line asked for.
 */
struct Options
{
    /**
     * @brief The definition's file.
     */
    std::string file{};

    /**
     * @brief The name of the pattern a tiktoken rank file is read under, none when empty.
     */
    std::string tiktoken{};

    /**
     * @brief The widest window decided.
     */
    std::size_t width{matisse::validate::default_width};

    /**
     * @brief The path of the input the certified-anchor supply is measured on, none when empty.
     */
    std::string input{};

    /**
     * @brief Whether the output is JSON rather than text.
     */
    bool json{false};
};

/**
 * @brief The certified-anchor supply of both inventories of a report on the input.
 */
struct Measured
{
    /**
     * @brief The factorization inventory's supply.
     */
    matisse::validate::Supply factorization{};

    /**
     * @brief The policy inventory's supply.
     */
    matisse::validate::Supply policy{};
};

/**
 * @brief Returns the pattern name `--tiktoken` gives.
 * @param text The option's value.
 * @return The name.
 * @throws std::invalid_argument If no tokenizer of the lookup has the name.
 */
[[nodiscard]] std::string parse_tiktoken(const std::string_view text)
{
    if (matisse::lookup::tokenizer(text))
    {
        return std::string{text};
    }

    const auto described{matisse::lookup::tokenizers()};

    const auto names{
            described | std::views::transform(&matisse::lookup::Tokenizer::name) |
            std::views::join_with(std::string_view{", "})};

    std::string named{};

    std::ranges::copy(names, std::back_inserter(named));

    throw std::invalid_argument{std::format("--tiktoken takes one of {}, not '{}'", named, text)};
}

/**
 * @brief Returns the widest window `--windows` gives.
 * @param text The option's value, a decimal.
 * @return The width.
 * @throws std::invalid_argument If the text is no decimal from narrowest_width to widest_width.
 */
[[nodiscard]] std::size_t parse_width(const std::string_view text)
{
    std::size_t width{};

    const auto text_end{text.data() + text.size()};

    const auto [end, error]{std::from_chars(text.data(), text_end, width)};

    if (error != std::errc{} || end != text_end || width < narrowest_width || width > widest_width)
    {
        const auto message{std::format("--windows takes {} to {}, not '{}'", narrowest_width, widest_width, text)};

        throw std::invalid_argument{message};
    }

    return width;
}

/**
 * @brief Reads the command line.
 * @param arguments The arguments after the program's name.
 * @return The options.
 * @throws std::invalid_argument If an option is unknown or lacks its value, no file or a second one is named, or the
 *         file is not read as a rank file by `--tiktoken`, since a tokenizer.json has no reader yet.
 */
[[nodiscard]] Options parse_options(const std::span<const std::string_view> arguments)
{
    Options options{};

    for (std::size_t at{0}; at < arguments.size(); ++at)
    {
        const auto argument{arguments[at]};

        const auto value{[&]() -> std::string_view {
            if (at + 1 >= arguments.size())
            {
                throw std::invalid_argument{std::format("{} needs a value", argument)};
            }

            return arguments[++at];
        }};

        if (argument == "--tiktoken")
        {
            const auto text{value()};

            options.tiktoken = parse_tiktoken(text);
        }
        else if (argument == "--windows")
        {
            const auto text{value()};

            options.width = parse_width(text);
        }
        else if (argument == "--input")
        {
            options.input = value();
        }
        else if (argument == "--json")
        {
            options.json = true;
        }
        else if (argument.starts_with("--"))
        {
            throw std::invalid_argument{std::format("unknown option {}", argument)};
        }
        else if (!options.file.empty())
        {
            throw std::invalid_argument{std::format("one file only, and {} is a second", argument)};
        }
        else
        {
            options.file = argument;
        }
    }

    if (options.file.empty())
    {
        throw std::invalid_argument{"no file named"};
    }

    if (options.tiktoken.empty())
    {
        throw std::invalid_argument{std::format(
                "{} is read as a rank file with --tiktoken PATTERN; a tokenizer.json has no reader yet", options.file)};
    }

    return options;
}

/**
 * @brief Returns the whole of a file.
 * @param path The file's path.
 * @return Its text.
 * @throws std::runtime_error If it cannot be opened.
 */
[[nodiscard]] std::string contents(const std::string& path)
{
    std::ifstream stream{path, std::ios::binary};

    if (!stream)
    {
        throw std::runtime_error{std::format("cannot read {}", path)};
    }

    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

/**
 * @brief Writes a refused definition: its file's heading and the refusal as text, or the JSON document with the refusal
 *        and no report.
 * @param out The stream.
 * @param options The command line.
 * @param refused Why the definition was refused.
 */
void write_refused(std::ostream& out, const Options& options, const std::string_view refused)
{
    using matisse::validate::json_string;

    if (!options.json)
    {
        out << std::format("== {}\nrefused: {}\n", options.file, refused);

        return;
    }

    out << std::format(
            "{{\n  \"path\": {},\n  \"tokenizer\": {},\n  \"refused\": {},\n  \"tokens\": null,\n"
            "  \"special_tokens\": null,\n  \"report\": null\n}}\n",
            json_string(options.file), json_string(options.tiktoken), json_string(refused));
}

/**
 * @brief Measures the certified-anchor supply of both inventories of a report on the input.
 * @param report The report.
 * @param pattern The pattern the report was decided of.
 * @param input The input's text.
 * @param path The input's path, as given.
 * @return The supply of each inventory.
 * @throws std::runtime_error If the input is not well-formed UTF-8, the message naming the input and the byte offset.
 */
[[nodiscard]] Measured measured(
        const matisse::validate::Report& report, const matisse::pretok::Pattern& pattern, const std::string_view input,
        const std::string_view path)
{
    using matisse::validate::Inventory;

    try
    {
        auto factorization{matisse::validate::supply(report, Inventory::factorization, pattern, input)};

        auto policy{matisse::validate::supply(report, Inventory::policy, pattern, input)};

        return Measured{.factorization = std::move(factorization), .policy = std::move(policy)};
    }
    catch (const std::invalid_argument& error)
    {
        throw std::runtime_error{std::format("cannot read {}: {}", path, error.what())};
    }
}

/**
 * @brief Writes a report as the JSON document: the file's path, the tokenizer it was read as, its token counts, the
 *        report, and the supply when an input was measured.
 * @param out The stream.
 * @param options The command line.
 * @param definition The definition read.
 * @param report The report.
 * @param supplies The supply of each inventory on the input, none when none was given.
 */
void write_json(
        std::ostream& out, const Options& options, const matisse::load::Definition& definition,
        const matisse::validate::Report& report, const std::optional<Measured>& supplies)
{
    using matisse::validate::json_string;

    out << std::format(
            "{{\n  \"path\": {},\n  \"tokenizer\": {},\n  \"refused\": null,\n  \"tokens\": {},\n"
            "  \"special_tokens\": {},\n",
            json_string(options.file), json_string(options.tiktoken), definition.spellings.size(),
            definition.specials.size());

    auto document{matisse::validate::json(report)};

    // The report's own lines, indented to where they sit in the document, each search resuming past the indent.
    for (auto at{document.find('\n')}; at != std::string::npos; at = document.find('\n', at + 1 + report_indent.size()))
    {
        document.insert(at + 1, report_indent);
    }

    out << R"(  "report": )" << document;

    if (supplies)
    {
        const auto& [factorization, policy]{*supplies};

        out << ",\n" << R"(  "supply": )" << matisse::validate::supply_json(factorization, policy, options.input);
    }

    out << "\n}\n";
}

/**
 * @brief Writes a report as text: the file's heading, the tokenizer it was read as with its token counts, the report,
 *        and the supply section when an input was measured.
 * @param out The stream.
 * @param options The command line.
 * @param definition The definition read.
 * @param report The report.
 * @param supplies The supply of each inventory on the input, none when none was given.
 */
void write_text(
        std::ostream& out, const Options& options, const matisse::load::Definition& definition,
        const matisse::validate::Report& report, const std::optional<Measured>& supplies)
{
    using matisse::validate::plural;

    const auto tokens{definition.spellings.size()};

    const auto specials{definition.specials.size()};

    out << std::format(
            "== {}\n-- tokenizer {}: {} token{}, {} special token{}\n", options.file, options.tiktoken, tokens,
            plural(tokens), specials, plural(specials));

    out << matisse::validate::render(report);

    if (supplies)
    {
        const auto& [factorization, policy]{*supplies};

        out << matisse::validate::supply_section(factorization, policy, options.input);
    }
}

/**
 * @brief Reads the definition, decides its pattern's report, measures the supply on the input when one was given, and
 *        writes what was found.
 * @param options The command line.
 * @param out The stream.
 * @return Whether the definition was read, false when it was refused.
 * @throws std::runtime_error If the file or the input cannot be read.
 */
[[nodiscard]] bool run(const Options& options, std::ostream& out)
{
    // An empty input is measured like any other: it holds no anchor, which the supply says.
    std::optional<std::string> input{};

    if (!options.input.empty())
    {
        input = contents(options.input);
    }

    // The name was checked against the lookup when the command line was read.
    const auto [name, pattern, specials]{*matisse::lookup::tokenizer(options.tiktoken)};

    std::optional<matisse::load::Definition> definition{};

    try
    {
        definition = matisse::load::read_tiktoken(options.file, specials);
    }
    catch (const std::invalid_argument& error)
    {
        write_refused(out, options, error.what());

        return false;
    }

    const auto report{matisse::validate::validate(pattern, options.width)};

    std::optional<Measured> supplies{};

    if (input)
    {
        supplies = measured(report, pattern, *input, options.input);
    }

    if (options.json)
    {
        write_json(out, options, *definition, report, supplies);
    }
    else
    {
        write_text(out, options, *definition, report, supplies);
    }

    return true;
}

} // namespace

int main(const int argc, char** argv)
{
    const std::vector<std::string_view> arguments{argv + 1, argv + argc};

    try
    {
        const auto options{parse_options(arguments)};

        // Written whole once the report is decided, so that an error leaves no part of a document behind.
        std::ostringstream out{};

        const auto read{run(options, out)};

        std::cout << out.str();

        return read ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::invalid_argument& error)
    {
        std::cerr << std::format("{}{}\n\n{}", diagnostic_prefix, error.what(), usage_text);

        return exit_command_error;
    }
    catch (const std::exception& error)
    {
        std::cerr << std::format("{}{}\n", diagnostic_prefix, error.what());

        return exit_command_error;
    }
}
