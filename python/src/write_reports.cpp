// Writes the reports matisse-tokenizer ships, one file per pinned tokenizer, each validate::serialize() of the report
// validate() decides of the tokenizer's pattern at its width, so that a process reading a pinned vocabulary reads the
// report back with validate::deserialize() and skips the decision.
//
//     matisse_python_reports DIRECTORY NAME=WIDTH...
//
// NAME is one of gpt2, cl100k, o200k and llama3, as lookup::tokenizer() knows them, and each report is written to
// DIRECTORY/NAME.report. The exit status is 0 when every report was written and 1 otherwise, the error on stderr.

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "matisse/lookup/tokenizers.hpp"
#include "matisse/validate/report.hpp"

namespace
{
/**
 * @brief Returns the width of an argument `NAME=WIDTH`.
 * @param argument The argument.
 * @param separator Where its `=` stands.
 * @return The width.
 * @throws std::invalid_argument If what follows the `=` is not a decimal number.
 */
[[nodiscard]] std::size_t width_of(const std::string_view argument, const std::size_t separator)
{
    const auto field{argument.substr(separator + 1)};

    const auto field_end{field.data() + field.size()};

    std::size_t width{};

    const auto [stop, error]{std::from_chars(field.data(), field_end, width)};

    if (field.empty() || error != std::errc{} || stop != field_end)
    {
        throw std::invalid_argument{std::format("the argument '{}' is not NAME=WIDTH", argument)};
    }

    return width;
}

/**
 * @brief Decides the report of one tokenizer and writes its text to the directory.
 * @param directory The directory.
 * @param argument The argument naming the tokenizer and the width, `NAME=WIDTH`.
 * @throws std::invalid_argument If the argument is not NAME=WIDTH or names no tokenizer, or validate() refuses the
 *         width.
 * @throws std::runtime_error If the file cannot be written.
 */
void write_report(const std::filesystem::path& directory, const std::string_view argument)
{
    const auto separator{argument.find('=')};

    if (separator == std::string_view::npos)
    {
        throw std::invalid_argument{std::format("the argument '{}' is not NAME=WIDTH", argument)};
    }

    const auto name{argument.substr(0, separator)};

    const auto width{width_of(argument, separator)};

    const auto tokenizer{matisse::lookup::tokenizer(name)};

    if (!tokenizer)
    {
        throw std::invalid_argument{std::format("no tokenizer is named '{}'", name)};
    }

    const auto report{matisse::validate::validate(tokenizer->pattern, width)};

    const auto text{matisse::validate::serialize(report)};

    const auto path{directory / std::format("{}.report", name)};

    std::ofstream file{path, std::ios::binary};

    file << text;

    file.close();

    if (!file)
    {
        throw std::runtime_error{std::format("{} cannot be written", path.string())};
    }
}

} // namespace

/**
 * @brief Writes the report of each tokenizer named to the directory, or the error to the standard error.
 * @param argc The argument count.
 * @param argv The arguments: the directory, then one `NAME=WIDTH` for each tokenizer, as the usage line gives them.
 * @return EXIT_SUCCESS when every report was written, EXIT_FAILURE otherwise.
 */
int main(const int argc, char** argv)
{
    const std::vector<std::string_view> arguments{argv + 1, argv + argc};

    try
    {
        if (arguments.size() < 2)
        {
            throw std::invalid_argument{"usage: matisse_python_reports DIRECTORY NAME=WIDTH..."};
        }

        for (const auto argument : arguments | std::views::drop(1))
        {
            write_report(arguments.front(), argument);
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << std::format("matisse_python_reports: {}\n", error.what());

        return EXIT_FAILURE;
    }
}
