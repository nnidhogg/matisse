#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace
{
/**
 * @brief The rank file the tests read, from the repository's root: the 256 single bytes at their own ranks, a
 *        definition every pattern takes.
 */
constexpr std::string_view bytes_file{"tools/validate/tests/data/bytes.tiktoken"};

/**
 * @brief The input the tests measure the supply on, from the repository's root: a few lines of words, contractions,
 *        digits, white space runs and letters beyond ASCII.
 */
constexpr std::string_view sample_file{"tools/validate/tests/data/sample.txt"};

/**
 * @brief What one run of the command left behind.
 */
struct Run
{
    /**
     * @brief The exit status.
     */
    int status{};

    /**
     * @brief What it wrote to standard output.
     */
    std::string out{};

    /**
     * @brief What it wrote to standard error.
     */
    std::string err{};
};

/**
 * @brief Returns the directory a test's runs write their streams and files into, one per test and process.
 * @return The directory, created.
 */
std::filesystem::path scratch()
{
    const std::string_view test{testing::UnitTest::GetInstance()->current_test_info()->name()};

    const auto name{std::format("matisse-validate-{}-{}", test, ::getpid())};

    const auto dir{std::filesystem::temp_directory_path() / name};

    std::filesystem::create_directories(dir);

    return dir;
}

/**
 * @brief Returns the whole of a file.
 * @param path The file's path.
 * @return Its text.
 */
std::string contents(const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};

    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

/**
 * @brief Returns the path of a file beside the tests.
 * @param path The file's path from the repository's root.
 * @return The path.
 */
std::string data(const std::string_view path)
{
    return std::format("{}/{}", SOURCE_DIR, path);
}

/**
 * @brief Runs matisse-validate through the shell, each argument quoted so that it reaches the command as written.
 * @param arguments The arguments after the program's name.
 * @return The exit status and the two streams.
 */
Run run(const std::vector<std::string>& arguments)
{
    const auto quoted{[](const std::string_view text) {
        std::string out{'\''};

        for (const auto byte : text)
        {
            if (byte == '\'')
            {
                out += R"('\'')";

                continue;
            }

            out += byte;
        }

        return out + '\'';
    }};

    const auto dir{scratch()};

    auto command{quoted(MATISSE_VALIDATE)};

    for (const auto& argument : arguments)
    {
        command += ' ' + quoted(argument);
    }

    const auto out_path{dir / "out"};

    const auto err_path{dir / "err"};

    const auto out_quoted{quoted(out_path.string())};

    const auto err_quoted{quoted(err_path.string())};

    command += std::format(" >{} 2>{}", out_quoted, err_quoted);

    const auto status{std::system(command.c_str())};

    const auto exit_status{WIFEXITED(status) ? WEXITSTATUS(status) : -1};

    Run ran{.status = exit_status, .out = contents(out_path), .err = contents(err_path)};

    std::filesystem::remove(out_path);

    std::filesystem::remove(err_path);

    // The directory goes with the streams unless the test keeps a file of its own in it.
    std::error_code kept{};

    std::filesystem::remove(dir, kept);

    return ran;
}

/**
 * @brief Returns what a fixture beside the tests holds after its first line, which says how it was written: the
 *        command's output as it printed it.
 * @param name The fixture's file name.
 * @return The output.
 */
std::string fixture(const std::string_view name)
{
    const auto path{std::format("tools/validate/tests/data/{}", name)};

    const auto text{contents(data(path))};

    return text.substr(text.find('\n') + 1);
}

/**
 * @brief Expects a run to be a command-line error: exit status 2, nothing on standard output, and the diagnostic then
 *        the usage on standard error.
 * @param ran The run.
 * @param diagnostic The diagnostic, after the command's name.
 */
void expect_command_error(const Run& ran, const std::string_view diagnostic)
{
    const auto& [status, out, err]{ran};

    EXPECT_EQ(status, 2);
    EXPECT_EQ(out, "");
    EXPECT_TRUE(err.starts_with(std::format("matisse-validate: {}\n\nusage: matisse-validate", diagnostic))) << err;
}

/**
 * @brief Llama 3's pattern as a JSON string spells it, each backslash doubled.
 */
constexpr std::string_view llama3_json_spelling{
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|)"
        R"(\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)"};

/**
 * @brief Writes a tokenizer.json of the 256 single bytes, each spelled under GPT-2's byte map at its own id, and
 *        `<|endoftext|>` as a special token, the definition bytes.tiktoken is under gpt2.
 * @param path The file's path.
 * @param pre_tokenizer The pre-tokenizer, as JSON text.
 * @param normalizer The normalizer, as JSON text.
 */
void write_bytes_tokenizer_json(
        const std::filesystem::path& path, const std::string_view pre_tokenizer, const std::string_view normalizer)
{
    constexpr unsigned byte_count{256};

    std::string vocab{};

    unsigned next{0x100};

    for (unsigned byte{0}; byte < byte_count; ++byte)
    {
        const auto printable{(byte >= 0x21U && byte <= 0x7EU) || (byte >= 0xA1U && byte <= 0xACU) || byte >= 0xAEU};

        const auto character{printable ? byte : next++};

        vocab += std::format(R"({}"\u{:04X}": {})", byte == 0 ? "" : ", ", character, byte);
    }

    std::ofstream{path, std::ios::binary} << std::format(
            R"({{"normalizer": {}, "pre_tokenizer": {}, "post_processor": null,
                 "added_tokens": [{{"id": 256, "content": "<|endoftext|>", "special": true}}],
                 "model": {{"type": "BPE", "vocab": {{{}}}, "merges": []}}}})",
            normalizer, pre_tokenizer, vocab);
}

/**
 * @brief Returns a command's text output after its first line, the heading that names the file.
 * @param out The output.
 * @return The output after the heading.
 */
std::string after_heading(const std::string_view out)
{
    return std::string{out.substr(out.find('\n') + 1)};
}

} // namespace

TEST(Output_test, Each_pattern_prints_the_report_and_the_supply_the_fixture_holds)
{
    // Run from the repository's root, so the paths the report names are the fixture's.
    const auto here{std::filesystem::current_path()};

    std::filesystem::current_path(SOURCE_DIR);

    for (const std::string pattern : {"gpt2", "cl100k"})
    {
        const auto [status, out, err]{
                run({"--tiktoken", pattern, std::string{bytes_file}, "--input", std::string{sample_file}})};

        EXPECT_EQ(status, 0) << err;
        EXPECT_EQ(err, "");
        EXPECT_EQ(out, fixture(std::format("{}_report_reference.txt", pattern))) << pattern;
    }

    std::filesystem::current_path(here);
}

TEST(Output_test, Each_pattern_prints_the_json_document_the_fixture_holds)
{
    // Run from the repository's root, so the paths the document names are the fixture's.
    const auto here{std::filesystem::current_path()};

    std::filesystem::current_path(SOURCE_DIR);

    for (const std::string pattern : {"gpt2", "cl100k"})
    {
        const auto [status, out, err]{
                run({"--tiktoken", pattern, std::string{bytes_file}, "--windows", "2", "--input",
                     std::string{sample_file}, "--json"})};

        EXPECT_EQ(status, 0) << err;
        EXPECT_EQ(err, "");
        EXPECT_EQ(out, fixture(std::format("{}_json_reference.txt", pattern))) << pattern;
    }

    std::filesystem::current_path(here);
}

TEST(Output_test, A_run_without_input_is_the_report_alone)
{
    const auto file{data(bytes_file)};

    const std::vector<std::string> narrow{"--tiktoken", "gpt2", file, "--windows", "2"};

    auto measured_arguments{narrow};

    measured_arguments.insert(measured_arguments.end(), {"--input", data(sample_file)});

    const auto [status, out, err]{run(narrow)};

    const auto [measured_status, measured_out, measured_err]{run(measured_arguments)};

    // The supply section stands after the report, its blank line opening it.
    const auto section{measured_out.find("\n\ncertified-anchor supply on ")};

    ASSERT_NE(section, std::string::npos) << measured_out;
    EXPECT_EQ(status, 0);
    EXPECT_EQ(out, measured_out.substr(0, section + 1));
    EXPECT_TRUE(out.starts_with(std::format("== {}\n-- tokenizer gpt2: 256 tokens, 1 special token\nverdict", file)))
            << out;

    auto json_arguments{narrow};

    json_arguments.emplace_back("--json");

    const auto [json_status, json_out, json_err]{run(json_arguments)};

    EXPECT_EQ(json_status, 0);
    EXPECT_FALSE(json_out.contains(R"("supply")")) << json_out;
    EXPECT_TRUE(json_out.ends_with("}\n}\n")) << json_out;
}

TEST(Command_line_test, A_refused_definition_exits_one_with_the_refusal_in_place_of_the_report)
{
    const auto dir{scratch()};

    const auto refused{(dir / "refused.tiktoken").string()};

    std::ofstream{refused} << "not-base64 0\n";

    const auto [status, out, err]{run({"--tiktoken", "cl100k", refused})};

    const auto refusal{std::format("read_tiktoken: line 1 of {} is not a base64 token and a decimal rank", refused)};

    EXPECT_EQ(status, 1);
    EXPECT_EQ(err, "");
    EXPECT_EQ(out, std::format("== {}\nrefused: {}\n", refused, refusal));

    const auto [json_status, json_out, json_err]{run({"--tiktoken", "cl100k", refused, "--json"})};

    EXPECT_EQ(json_status, 1);
    EXPECT_EQ(
            json_out, std::format(
                              "{{\n  \"path\": \"{}\",\n  \"tokenizer\": \"cl100k\",\n  \"refused\": \"{}\",\n"
                              "  \"tokens\": null,\n  \"special_tokens\": null,\n  \"report\": null\n}}\n",
                              refused, refusal));

    std::filesystem::remove_all(dir);
}

TEST(Command_line_test, A_file_or_an_input_the_command_cannot_read_exits_two_without_the_usage)
{
    const auto dir{scratch()};

    // An input that is not UTF-8 is outside the model the classes are defined on. It is written before the first run,
    // which removes the directory it leaves empty.
    const auto malformed{(dir / "malformed.txt").string()};

    std::ofstream{malformed} << "ab\xFF";

    const auto missing{(dir / "missing.tiktoken").string()};

    const auto [missing_status, missing_out, missing_err]{run({"--tiktoken", "gpt2", missing})};

    EXPECT_EQ(missing_status, 2);
    EXPECT_EQ(missing_out, "");
    EXPECT_EQ(missing_err, std::format("matisse-validate: cannot read {}\n", missing));

    const auto absent{(dir / "absent.txt").string()};

    const auto [absent_status, absent_out, absent_err]{
            run({"--tiktoken", "gpt2", data(bytes_file), "--input", absent})};

    EXPECT_EQ(absent_status, 2);
    EXPECT_EQ(absent_err, std::format("matisse-validate: cannot read {}\n", absent));

    const auto [malformed_status, malformed_out, malformed_err]{
            run({"--tiktoken", "gpt2", data(bytes_file), "--windows", "2", "--input", malformed})};

    EXPECT_EQ(malformed_status, 2);
    EXPECT_EQ(malformed_out, "");
    EXPECT_EQ(
            malformed_err,
            std::format("matisse-validate: cannot read {}: classes_of: the text is not UTF-8 at byte 2\n", malformed));

    std::filesystem::remove_all(dir);
}

TEST(Command_line_test, A_command_line_error_exits_two_with_the_usage)
{
    const auto file{data(bytes_file)};

    expect_command_error(run({}), "no file named");
    expect_command_error(
            run({"--tiktoken", "gpt2", file, file}), std::format("one file only, and {} is a second", file));
    expect_command_error(run({"--tiktoken", "gpt2", file, "--trace"}), "unknown option --trace");
    expect_command_error(run({file, "--tiktoken"}), "--tiktoken needs a value");
    expect_command_error(
            run({"--tiktoken", "o200k_base", file}),
            "--tiktoken takes one of gpt2, cl100k, o200k, llama3, not 'o200k_base'");

    for (const std::string width : {"1", "6", "4x", ""})
    {
        expect_command_error(
                run({"--tiktoken", "gpt2", file, "--windows", width}),
                std::format("--windows takes 2 to 5, not '{}'", width));
    }
}

TEST(Command_line_test, A_tokenizer_json_is_read_under_the_tokenizer_whose_pattern_its_pre_tokenizer_runs)
{
    const auto dir{scratch()};

    const auto gpt2_file{(dir / "gpt2.tokenizer.json").string()};
    const auto llama3_file{(dir / "llama3.tokenizer.json").string()};

    write_bytes_tokenizer_json(gpt2_file, R"({"type": "ByteLevel", "add_prefix_space": false})", "null");

    write_bytes_tokenizer_json(
            llama3_file,
            std::format(
                    R"({{"type": "Sequence", "pretokenizers": [
                           {{"type": "Split", "pattern": {{"Regex": "{}"}}, "behavior": "Isolated", "invert": false}},
                           {{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}}]}})",
                    llama3_json_spelling),
            "null");

    const auto [gpt2_status, gpt2_out, gpt2_err]{run({gpt2_file, "--windows", "2"})};
    const auto [ranks_status, ranks_out, ranks_err]{run({"--tiktoken", "gpt2", data(bytes_file), "--windows", "2"})};

    EXPECT_EQ(gpt2_status, 0) << gpt2_err;
    EXPECT_TRUE(gpt2_out.starts_with(std::format("== {}\n-- tokenizer gpt2: 256 tokens, 1 special token\n", gpt2_file)))
            << gpt2_out;
    EXPECT_EQ(after_heading(gpt2_out), after_heading(ranks_out));

    const auto [llama3_status, llama3_out, llama3_err]{run({llama3_file, "--windows", "2", "--json"})};

    EXPECT_EQ(llama3_status, 0) << llama3_err;
    EXPECT_TRUE(llama3_out.contains(R"("tokenizer": "llama3",)")) << llama3_out;
    EXPECT_TRUE(llama3_out.contains(R"("tokens": 256,)")) << llama3_out;

    std::filesystem::remove_all(dir);
}

TEST(Command_line_test, A_refused_tokenizer_json_exits_one_with_the_refusal_and_no_tokenizer)
{
    const auto dir{scratch()};

    const auto normalized{(dir / "normalized.tokenizer.json").string()};
    const auto unknown{(dir / "unknown.tokenizer.json").string()};

    write_bytes_tokenizer_json(normalized, R"({"type": "ByteLevel", "add_prefix_space": false})", R"({"type": "NFC"})");

    write_bytes_tokenizer_json(
            unknown,
            R"({"type": "Sequence", "pretokenizers": [
                   {"type": "Split", "pattern": {"Regex": "\\p{L}+"}, "behavior": "Isolated", "invert": false},
                   {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}]})",
            "null");

    const auto [status, out, err]{run({normalized})};

    const auto refusal{std::format(
            R"(read_tokenizer_json: normalizer of {} is a "NFC" object, where the reader takes null)", normalized)};

    EXPECT_EQ(status, 1);
    EXPECT_EQ(err, "");
    EXPECT_EQ(out, std::format("== {}\nrefused: {}\n", normalized, refusal));

    const auto [json_status, json_out, json_err]{run({unknown, "--json"})};

    EXPECT_EQ(json_status, 1);
    EXPECT_TRUE(json_out.contains(R"("tokenizer": null,)")) << json_out;
    EXPECT_TRUE(
            json_out.contains(R"(runs the pattern \"\\p{L}+\", and no tokenizer matisse describes has that pattern)"))
            << json_out;

    std::filesystem::remove_all(dir);
}
