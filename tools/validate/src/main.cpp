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
#include <unordered_map>
#include <utility>
#include <vector>

#include "matisse/load/definition.hpp"
#include "matisse/lookup/tokenizers.hpp"
#include "matisse/validate/merges.hpp"
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
 * @brief How many of an input's most frequent pairs and of its most frequent triples are drawn as candidates for a
 *        merge table without a pre-tokenizer unless `--candidates` gives another count.
 *
 * Under GPT-2's table it draws 2,177 candidates from twitter.json, which the command decides in 0.7 s, where all 19,434
 * of the input's windows of one to three bytes take 7.4 s.
 */
constexpr std::size_t default_candidates{1000};

/**
 * @brief The values a byte takes, every candidate of width one.
 */
constexpr std::size_t byte_values{256};

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
policy certificate. A tokenizer.json whose pre-tokenizer is a ByteLevel without its regex splits nothing, and the
windows of its merge table are decided instead, on candidates drawn from --input.

  --tiktoken PATTERN    read FILE as a tiktoken rank file whose pre-tokenizer is PATTERN, gpt2, cl100k, o200k or
                        llama3, since the file names none; without it FILE is a tokenizer.json, read under the
                        tokenizer whose pattern its pre-tokenizer runs
  --windows N           the widest window decided, 2 to 5, 5 unless given; not read for a merge table
  --input FILE          measure each inventory's certified-anchor supply on this file: the anchors, per kibibyte, the
                        gaps between them, the share of the pre-token boundaries they reach and the longest piece; for
                        a merge table without a pre-tokenizer, draw the candidate windows from it, every byte it holds
                        and its most frequent pairs and triples
  --candidates N        for a merge table, how many of the input's most frequent pairs and of its most frequent
                        triples are drawn, 1000 unless given
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
     * @brief The name of the pattern a tiktoken rank file is read under, empty for a tokenizer.json, which names its
     *        own.
     */
    std::string tiktoken{};

    /**
     * @brief The widest window decided.
     */
    std::size_t width{matisse::validate::default_width};

    /**
     * @brief The path of the input the certified-anchor supply is measured on, or the candidate windows of a merge
     *        table without a pre-tokenizer are drawn from, none when empty.
     */
    std::string input{};

    /**
     * @brief How many of the input's most frequent pairs and of its most frequent triples are drawn as candidates.
     */
    std::size_t candidates{default_candidates};

    /**
     * @brief Whether the output is JSON rather than text.
     */
    bool json{false};
};

/**
 * @brief A definition read, with the tokenizer whose pattern it is read under.
 */
struct Read
{
    /**
     * @brief The tokenizer, std::nullopt for a tokenizer.json whose pre-tokenizer splits nothing.
     */
    std::optional<matisse::lookup::Tokenizer> tokenizer{};

    /**
     * @brief The definition.
     */
    matisse::load::Definition definition{};
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
 * @brief The candidate windows drawn from an input for a merge table without a pre-tokenizer.
 */
struct Drawn
{
    /**
     * @brief The windows: every byte the input holds, then its most frequent pairs, then its most frequent triples,
     *        each width the most frequent first.
     */
    std::vector<std::string> windows{};

    /**
     * @brief How many bytes are drawn.
     */
    std::size_t bytes{};

    /**
     * @brief How many pairs are drawn.
     */
    std::size_t pairs{};

    /**
     * @brief How many triples are drawn.
     */
    std::size_t triples{};
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
 * @brief Returns the count `--candidates` gives.
 * @param text The option's value, a decimal.
 * @return The count.
 * @throws std::invalid_argument If the text is no decimal of at least one.
 */
[[nodiscard]] std::size_t parse_candidates(const std::string_view text)
{
    std::size_t count{};

    const auto text_end{text.data() + text.size()};

    const auto [end, error]{std::from_chars(text.data(), text_end, count)};

    if (error != std::errc{} || end != text_end || count == 0)
    {
        throw std::invalid_argument{std::format("--candidates takes a count of at least 1, not '{}'", text)};
    }

    return count;
}

/**
 * @brief Reads the command line.
 * @param arguments The arguments after the program's name.
 * @return The options.
 * @throws std::invalid_argument If an option is unknown or lacks its value, or no file or a second one is named.
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
        else if (argument == "--candidates")
        {
            const auto text{value()};

            options.candidates = parse_candidates(text);
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
 * @brief Reads the definition: a rank file under the tokenizer `--tiktoken` names, or a tokenizer.json under the
 *        tokenizer whose pattern its pre-tokenizer runs, found by the pattern's published spelling, or under none when
 *        its pre-tokenizer splits nothing.
 * @param options The command line.
 * @return The definition and its tokenizer.
 * @throws std::invalid_argument If the loader refuses the file, or no tokenizer matisse describes has the pattern a
 *         tokenizer.json's pre-tokenizer runs.
 * @throws std::runtime_error If the file cannot be read.
 */
[[nodiscard]] Read read_definition(const Options& options)
{
    if (!options.tiktoken.empty())
    {
        // The name was checked against the lookup when the command line was read.
        auto tokenizer{*matisse::lookup::tokenizer(options.tiktoken)};

        auto definition{matisse::load::read_tiktoken(options.file, tokenizer.specials)};

        return {.tokenizer = std::move(tokenizer), .definition = std::move(definition)};
    }

    const auto spelling{matisse::load::tokenizer_json_spelling(options.file)};

    if (spelling.empty())
    {
        auto definition{matisse::load::read_tokenizer_json(options.file, spelling)};

        return {.tokenizer = std::nullopt, .definition = std::move(definition)};
    }

    auto described{matisse::lookup::tokenizers()};

    const auto found{std::ranges::find(described, spelling, &matisse::lookup::Tokenizer::spelling)};

    if (found == described.end())
    {
        throw std::invalid_argument{std::format(
                "the pre-tokenizer of {} runs the pattern \"{}\", and no tokenizer matisse describes has that pattern",
                options.file, spelling)};
    }

    auto definition{matisse::load::read_tokenizer_json(options.file, spelling)};

    return {.tokenizer = std::move(*found), .definition = std::move(definition)};
}

/**
 * @brief Writes a refused definition: its file's heading and the refusal as text, or the JSON document with the refusal
 *        and no report, its tokenizer the one `--tiktoken` names and null for a tokenizer.json.
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

    const auto tokenizer{options.tiktoken.empty() ? std::string{"null"} : json_string(options.tiktoken)};

    out << std::format(
            "{{\n  \"path\": {},\n  \"tokenizer\": {},\n  \"refused\": {},\n  \"tokens\": null,\n"
            "  \"special_tokens\": null,\n  \"report\": null\n}}\n",
            json_string(options.file), tokenizer, json_string(refused));
}

/**
 * @brief Measures the certified-anchor supply of both inventories of a report on the input.
 * @param report The report.
 * @param input The input's text.
 * @param path The input's path, as given.
 * @return The supply of each inventory.
 * @throws std::runtime_error If the input is not well-formed UTF-8, the message naming the input and the byte offset.
 */
[[nodiscard]] Measured measured(
        const matisse::validate::Report& report, const std::string_view input, const std::string_view path)
{
    using matisse::validate::Inventory;

    try
    {
        auto factorization{matisse::validate::supply(report, Inventory::factorization, input)};

        auto policy{matisse::validate::supply(report, Inventory::policy, input)};

        return Measured{.factorization = std::move(factorization), .policy = std::move(policy)};
    }
    catch (const std::invalid_argument& error)
    {
        throw std::runtime_error{std::format("cannot read {}: {}", path, error.what())};
    }
}

/**
 * @brief Returns a JSON object of several lines indented to where it sits as a member of the document, every line but
 *        the first taking the indent.
 * @param object The object.
 * @return The object, indented.
 */
[[nodiscard]] std::string indented(std::string object)
{
    // Each search resumes past the indent just inserted.
    for (auto at{object.find('\n')}; at != std::string::npos; at = object.find('\n', at + 1 + report_indent.size()))
    {
        object.insert(at + 1, report_indent);
    }

    return object;
}

/**
 * @brief Writes a report as the JSON document: the file's path, the tokenizer it was read as, its token counts, the
 *        report, and the supply when an input was measured.
 * @param out The stream.
 * @param options The command line.
 * @param read The definition read and its tokenizer.
 * @param report The report.
 * @param supplies The supply of each inventory on the input, none when none was given.
 */
void write_json(
        std::ostream& out, const Options& options, const Read& read, const matisse::validate::Report& report,
        const std::optional<Measured>& supplies)
{
    using matisse::validate::json_string;

    const auto& [tokenizer, definition]{read};

    out << std::format(
            "{{\n  \"path\": {},\n  \"tokenizer\": {},\n  \"refused\": null,\n  \"tokens\": {},\n"
            "  \"special_tokens\": {},\n",
            json_string(options.file), json_string(tokenizer->name), definition.spellings.size(),
            definition.specials.size());

    const auto document{indented(matisse::validate::json(report))};

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
 * @param read The definition read and its tokenizer.
 * @param report The report.
 * @param supplies The supply of each inventory on the input, none when none was given.
 */
void write_text(
        std::ostream& out, const Options& options, const Read& read, const matisse::validate::Report& report,
        const std::optional<Measured>& supplies)
{
    using matisse::validate::plural;

    const auto& [tokenizer, definition]{read};

    const auto tokens{definition.spellings.size()};

    const auto specials{definition.specials.size()};

    out << std::format(
            "== {}\n-- tokenizer {}: {} token{}, {} special token{}\n", options.file, tokenizer->name, tokens,
            plural(tokens), specials, plural(specials));

    out << matisse::validate::render(report);

    if (supplies)
    {
        const auto& [factorization, policy]{*supplies};

        out << matisse::validate::supply_section(factorization, policy, options.input);
    }
}

/**
 * @brief Returns an input's windows of one width that occur most often, the most frequent first and a tie in the order
 *        of the bytes, so that every run draws the same.
 * @param input The input.
 * @param width The windows' width.
 * @param count How many are drawn at most.
 * @return The windows.
 */
[[nodiscard]] std::vector<std::string> most_frequent(
        const std::string_view input, const std::size_t width, const std::size_t count)
{
    std::unordered_map<std::string_view, std::size_t> occurrences{};

    for (std::size_t at{0}; at + width <= input.size(); ++at)
    {
        ++occurrences[input.substr(at, width)];
    }

    std::vector<std::pair<std::string_view, std::size_t>> ranked{occurrences.begin(), occurrences.end()};

    const auto more_frequent{[]<typename Entry>(const Entry& left, const Entry& right) {
        const auto& [left_window, left_count]{left};

        const auto& [right_window, right_count]{right};

        return left_count != right_count ? left_count > right_count : left_window < right_window;
    }};

    std::ranges::sort(ranked, more_frequent);

    std::vector<std::string> windows{};

    for (const auto& [window, occurrence_count] : ranked | std::views::take(count))
    {
        windows.emplace_back(window);
    }

    return windows;
}

/**
 * @brief Draws the candidate windows of a merge table without a pre-tokenizer from an input: every byte it holds and
 *        its most frequent pairs and triples.
 * @param input The input.
 * @param count How many pairs and how many triples are drawn at most.
 * @return The windows and how many of each width.
 */
[[nodiscard]] Drawn drawn_candidates(const std::string_view input, const std::size_t count)
{
    const auto bytes{most_frequent(input, 1, byte_values)};

    const auto pairs{most_frequent(input, 2, count)};

    const auto triples{most_frequent(input, 3, count)};

    std::vector<std::string> windows{bytes};

    windows.insert(windows.end(), pairs.begin(), pairs.end());

    windows.insert(windows.end(), triples.begin(), triples.end());

    return {.windows = std::move(windows), .bytes = bytes.size(), .pairs = pairs.size(), .triples = triples.size()};
}

/**
 * @brief Writes the windows decided of a merge table as the JSON document: the file's path, a null tokenizer, its token
 *        counts, a null report, the candidates drawn and the windows decided.
 * @param out The stream.
 * @param options The command line.
 * @param definition The definition read.
 * @param drawn The candidates drawn.
 * @param windows The windows decided.
 */
void write_merges_json(
        std::ostream& out, const Options& options, const matisse::load::Definition& definition, const Drawn& drawn,
        const std::span<const matisse::validate::Merge_window> windows)
{
    using matisse::validate::json_string;

    const auto& [candidates, bytes, pairs, triples]{drawn};

    out << std::format(
            "{{\n  \"path\": {},\n  \"tokenizer\": null,\n  \"refused\": null,\n  \"tokens\": {},\n"
            "  \"special_tokens\": {},\n  \"report\": null,\n",
            json_string(options.file), definition.spellings.size(), definition.specials.size());

    out << std::format(
            "  \"candidates\": {{\"input\": {}, \"bytes\": {}, \"pairs\": {}, \"triples\": {}}},\n",
            json_string(options.input), bytes, pairs, triples);

    const auto document{indented(matisse::validate::merges_json(windows))};

    out << R"(  "merges": )" << document << "\n}\n";
}

/**
 * @brief Writes the windows decided of a merge table as text: the file's heading, the table's token counts, the
 *        candidates drawn, and the windows decided.
 * @param out The stream.
 * @param options The command line.
 * @param definition The definition read.
 * @param drawn The candidates drawn.
 * @param windows The windows decided.
 */
void write_merges_text(
        std::ostream& out, const Options& options, const matisse::load::Definition& definition, const Drawn& drawn,
        const std::span<const matisse::validate::Merge_window> windows)
{
    using matisse::validate::plural;

    const auto& [candidates, bytes, pairs, triples]{drawn};

    const auto tokens{definition.spellings.size()};

    const auto specials{definition.specials.size()};

    out << std::format(
            "== {}\n-- merge table without a pre-tokenizer: {} token{}, {} special token{}\n", options.file, tokens,
            plural(tokens), specials, plural(specials));

    out << std::format(
            "-- candidates from {}: {} byte{}, {} pair{} and {} triple{}\n", options.input, bytes, plural(bytes), pairs,
            plural(pairs), triples, plural(triples));

    out << matisse::validate::render_merges(windows);
}

/**
 * @brief Decides the windows of a merge table without a pre-tokenizer on the candidates drawn from the input, and
 *        writes them, or the refusal of the table in their place.
 * @param options The command line.
 * @param definition The definition read.
 * @param input The input's text, none when none was given.
 * @param out The stream.
 * @return Whether the table was decided, false when it was refused.
 * @throws std::invalid_argument If no input was given, which the candidates are drawn from.
 */
[[nodiscard]] bool run_merges(
        const Options& options, const matisse::load::Definition& definition, const std::optional<std::string>& input,
        std::ostream& out)
{
    if (!input)
    {
        throw std::invalid_argument{std::format(
                "the merge table of {} has no pre-tokenizer, and its candidate windows are drawn from --input, "
                "which is not given",
                options.file)};
    }

    const auto drawn{drawn_candidates(*input, options.candidates)};

    std::vector<matisse::validate::Merge_window> windows{};

    try
    {
        windows = matisse::validate::validate_merges(definition, drawn.windows);
    }
    catch (const std::invalid_argument& error)
    {
        write_refused(out, options, error.what());

        return false;
    }

    if (options.json)
    {
        write_merges_json(out, options, definition, drawn, windows);
    }
    else
    {
        write_merges_text(out, options, definition, drawn, windows);
    }

    return true;
}

/**
 * @brief Reads the definition, decides its pattern's report, measures the supply on the input when one was given, and
 *        writes what was found; a merge table without a pre-tokenizer has its windows decided instead.
 * @param options The command line.
 * @param out The stream.
 * @return Whether the definition was read, false when it was refused.
 * @throws std::invalid_argument If a merge table without a pre-tokenizer is read and no input was given.
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

    std::optional<Read> read{};

    try
    {
        read = read_definition(options);
    }
    catch (const std::invalid_argument& error)
    {
        write_refused(out, options, error.what());

        return false;
    }

    if (!read->tokenizer)
    {
        return run_merges(options, read->definition, input, out);
    }

    const auto report{matisse::validate::validate(read->tokenizer->pattern, options.width)};

    std::optional<Measured> supplies{};

    if (input)
    {
        supplies = measured(report, *input, options.input);
    }

    if (options.json)
    {
        write_json(out, options, *read, report, supplies);
    }
    else
    {
        write_text(out, options, *read, report, supplies);
    }

    return true;
}

} // namespace

/**
 * @brief Runs the validator: reads the definition, decides its report and writes it to the standard output whole, the
 *        refusal in its place when the loader refuses the definition, or a diagnostic to the standard error.
 * @param argc The argument count.
 * @param argv The arguments: the options and the definition's file, as the usage text gives them.
 * @return EXIT_SUCCESS when the definition was read and reported, EXIT_FAILURE when it was refused, and 2 on a
 *         command-line error or on a file or an input the command cannot read.
 */
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
