// Times matisse's encoder on GPT-2's, cl100k's, o200k's and Llama 3's pinned vocabularies, every figure taken after the
// result it times is held to a reference, so that a row measures a proved-identical encoding: the encoding with no
// cache, encode_parallel(), the batch and the prompts with a kept cache to encode()'s ids, and encode() itself to the
// digest it prints, which the fixtures' and the vendors' digests are compared by.
//
// The modes. single times, on one corpus, encode(), the same encoding with no cache, each piece looked up whole where
// the definition says so and otherwise merged alone, and the segmentation alone, the pattern's Segmenter over the whole
// text. prompts times encode() called once per prompt, the prompts of a file separated by NUL bytes, each call with a
// piece cache of its own and then with one cache kept across every call, and reports the latency of a call beside the
// throughput of a pass. parallel times encode_parallel() on one long text under the factorization inventory, the
// default, at each worker count given, beside encode() in the same run. batch times encode() over the documents of a
// file separated by NUL bytes, spread over a pool of threads that each take the next document from a shared counter and
// keep one piece cache across the documents they take, at each thread count given. Every mode prints the ids line, `ids
// TOKENIZER NAME COUNT DIGEST`, the SHA-256 of the ids as little-endian 32-bit words, the prompts' and the documents'
// ids concatenated in order.
//
// Every row runs one pass first, which warms the caches and gives the result every timed pass must reproduce, then
// PASSES timed passes over input already in memory, and reports the best, median and worst pass in MiB/s and the best
// and median in millions of tokens per second. A first pass that differs from its reference, encode()'s ids or the
// serial segmentation, or a timed pass that differs from the first, exits 1, as does a file that holds no prompt. The
// provenance block heads the output: the commit and whether the tree was dirty, the compiler, the system and the
// processors the process may run on, since a figure is quotable only beside the tree and the machine that produced it.
//
// Usage: matisse_bench MODE VOCAB_DIR TOKENIZERS PASSES FILE [COUNT...]
// MODE is single, prompts, parallel or batch, TOKENIZERS is one or more of gpt2, cl100k, o200k and llama3 separated by
// commas, and the counts are the worker counts of parallel (1, 2, 4, 8, 12, 16 and 24 unless given) or the thread
// counts of batch (1 and 24). Each report is decided to validate()'s default width, o200k's to three.

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <ranges>
#include <sched.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/utsname.h>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/common/sha256.hpp"
#include "matisse/encode/encoder.hpp"
#include "matisse/encode/merger.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/gpt2/tokenizer.hpp"
#include "matisse/llama3/tokenizer.hpp"
#include "matisse/load/definition.hpp"
#include "matisse/o200k/tokenizer.hpp"
#include "matisse/pretok/segmenter.hpp"
#include "matisse/tools/probes/provenance.hpp"
#include "matisse/validate/report.hpp"

namespace
{
using matisse::encode::Encoder;
using matisse::load::Definition;
using matisse::load::Token_id_t;
using matisse::validate::Inventory;

/**
 * @brief The ids of a text.
 */
using Ids = std::vector<Token_id_t>;

/**
 * @brief The ids of each document of a batch, in the documents' order.
 */
using Batch_ids = std::vector<Ids>;

/**
 * @brief The SHA-256 of GPT-2's encoder.json as pinned, gpt2-vocab.json in the vocabulary directory.
 */
constexpr std::string_view gpt2_encoder_digest{"196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"};

/**
 * @brief The SHA-256 of GPT-2's vocab.bpe as pinned, gpt2-merges.txt in the vocabulary directory.
 */
constexpr std::string_view gpt2_vocab_digest{"1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"};

/**
 * @brief The SHA-256 of cl100k_base.tiktoken as pinned.
 */
constexpr std::string_view cl100k_digest{"223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"};

/**
 * @brief The SHA-256 of o200k_base.tiktoken as pinned.
 */
constexpr std::string_view o200k_digest{"446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"};

/**
 * @brief The SHA-256 of Llama 3's llama3.tiktoken as pinned.
 */
constexpr std::string_view llama3_digest{"82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"};

/**
 * @brief The widest window of o200k's report: its 30 classes take validate()'s default of five to 245 seconds and 2.9
 *        GB, and four, at 16 seconds, adds to three's factorization inventory, the one timed, 23 anchors on munch's C++
 *        sources and none on twitter.json or wiki-zh.
 */
constexpr std::size_t o200k_width{3};

/**
 * @brief The bytes in a mebibyte, the unit every throughput is reported in.
 */
constexpr double bytes_per_mebibyte{1024.0 * 1024.0};

/**
 * @brief The worker counts parallel times unless given.
 */
constexpr std::array<std::size_t, 7> default_workers{1, 2, 4, 8, 12, 16, 24};

/**
 * @brief The thread counts batch times unless given.
 */
constexpr std::array<std::size_t, 2> default_threads{1, 24};

/**
 * @brief The modes the bench runs.
 */
constexpr std::array<std::string_view, 4> modes{"single", "prompts", "parallel", "batch"};

/**
 * @brief A tokenizer under measure: its name, its pattern, the widest window its report is decided to and its
 *        definition.
 */
struct Tokenizer
{
    /**
     * @brief The name, gpt2, cl100k, o200k or llama3.
     */
    std::string_view name{};

    /**
     * @brief The pattern, whose views point into the tokenizer library's constants.
     */
    matisse::pretok::Pattern pattern{};

    /**
     * @brief The widest window of the pattern's report.
     */
    std::size_t width{};

    /**
     * @brief The definition read from the pinned vocabulary.
     */
    Definition definition{};
};

/**
 * @brief What the command line asked for.
 */
struct Options
{
    /**
     * @brief The mode: single, prompts, parallel or batch.
     */
    std::string mode{};

    /**
     * @brief The vocabulary directory.
     */
    std::filesystem::path vocabularies{};

    /**
     * @brief The tokenizers measured, in the order given.
     */
    std::vector<std::string> tokenizers{};

    /**
     * @brief The timed passes of every row.
     */
    std::size_t passes{};

    /**
     * @brief The file measured.
     */
    std::filesystem::path file{};

    /**
     * @brief The worker or thread counts.
     */
    std::vector<std::size_t> counts{};
};

/**
 * @brief The pass times of a row and the count its result holds.
 */
struct Timing
{
    /**
     * @brief The timed passes in seconds, ascending.
     */
    std::vector<double> seconds{};

    /**
     * @brief The tokens, or the pieces, of the result.
     */
    std::size_t count{};
};

/**
 * @brief The timing of encode() called once per prompt: the ids of the first pass, and the seconds of every timed pass
 *        and of every timed call.
 */
struct Prompt_timing
{
    /**
     * @brief The ids of each prompt in the first pass, which every timed pass reproduces.
     */
    Batch_ids first{};

    /**
     * @brief The timed passes in seconds, each the sum of its calls, ascending.
     */
    std::vector<double> seconds{};

    /**
     * @brief Every timed call in seconds, ascending.
     */
    std::vector<double> latencies{};
};

/**
 * @brief Returns the positive whole number an argument spells.
 * @param text The argument.
 * @param what What the number is, named in the refusal.
 * @return The number.
 * @throws std::invalid_argument If the argument is not a positive whole number.
 */
std::size_t positive(const std::string_view text, const std::string_view what)
{
    std::size_t value{};

    const auto text_end{text.data() + text.size()};

    const auto [end, error]{std::from_chars(text.data(), text_end, value)};

    if (error != std::errc{} || end != text_end || value == 0)
    {
        throw std::invalid_argument{std::format("{} is a positive whole number, not '{}'", what, text)};
    }

    return value;
}

/**
 * @brief Reads the command line.
 * @param arguments The arguments after the program's name.
 * @return The options, the counts defaulted by the mode.
 * @throws std::invalid_argument If an argument is missing, the mode is unknown or a number is not one.
 */
Options parse_options(const std::span<const std::string_view> arguments)
{
    if (arguments.size() < 5)
    {
        throw std::invalid_argument{"usage: matisse_bench MODE VOCAB_DIR TOKENIZERS PASSES FILE [COUNT...]"};
    }

    const auto mode{arguments[0]};

    if (!std::ranges::contains(modes, mode))
    {
        throw std::invalid_argument{std::format("the modes are single, prompts, parallel and batch, not '{}'", mode)};
    }

    Options options{
            .mode = std::string{mode},
            .vocabularies = arguments[1],
            .tokenizers = {},
            .passes = positive(arguments[3], "PASSES"),
            .file = arguments[4],
            .counts = {}};

    for (const auto name : arguments[2] | std::views::split(','))
    {
        options.tokenizers.emplace_back(name.begin(), name.end());
    }

    for (const auto count : arguments.subspan(5))
    {
        options.counts.push_back(positive(count, "COUNT"));
    }

    if (!options.counts.empty())
    {
        return options;
    }

    if (mode == "parallel")
    {
        options.counts.assign(default_workers.begin(), default_workers.end());
    }

    if (mode == "batch")
    {
        options.counts.assign(default_threads.begin(), default_threads.end());
    }

    return options;
}

/**
 * @brief Prints what this run may be cited as: the commit built from, the compiler, the machine and the invocation.
 * @param arguments The command line after the program's name.
 */
void print_provenance(const std::span<const std::string_view> arguments)
{
    const auto now{std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};

    std::tm utc{};

    gmtime_r(&now, &utc);

    std::array<char, 32> stamp{};

    std::strftime(stamp.data(), stamp.size(), "%Y-%m-%d %H:%M:%S UTC", &utc);

    const auto commit{matisse::tools::probes::commit};

    const std::string_view dirty{matisse::tools::probes::dirty ? " (uncommitted changes present)" : ""};

    std::printf("matisse_bench\n");
    std::printf(
            "  commit      %.*s%.*s\n", static_cast<int>(commit.size()), commit.data(), static_cast<int>(dirty.size()),
            dirty.data());
    std::printf("  collected   %s\n", stamp.data());
    std::printf("  compiler    %s\n", __VERSION__);

    // The kernel and architecture without the node name.
    utsname system{};

    if (uname(&system) == 0)
    {
        std::printf("  system      %s %s %s\n", system.sysname, system.release, system.machine);
    }

    cpu_set_t allowed{};

    const auto read{sched_getaffinity(0, sizeof(allowed), &allowed) == 0};

    const auto affinity{read ? CPU_COUNT(&allowed) : 0};

    std::printf("  processors  %u visible, %d allowed\n", std::thread::hardware_concurrency(), affinity);

    // Every path named by its file name alone, so that the record names no machine.
    std::string invocation{};

    for (const auto argument : arguments)
    {
        const auto shown{std::filesystem::path{argument}.filename().string()};

        invocation += std::format(" {}", shown);
    }

    std::printf("  invocation  matisse_bench%s\n", invocation.c_str());
}

/**
 * @brief Returns the whole of a file.
 * @param path The file's path.
 * @return Its bytes.
 * @throws std::runtime_error If the file cannot be read.
 */
std::string contents(const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};

    if (!stream)
    {
        throw std::runtime_error{std::format("cannot read {}", path.string())};
    }

    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

/**
 * @brief Returns the parts of a text separated by NUL bytes.
 * @param text The text.
 * @return The parts, views into the text.
 */
std::vector<std::string_view> parts_of(const std::string_view text)
{
    std::vector<std::string_view> parts{};

    for (const auto part : text | std::views::split('\0'))
    {
        parts.emplace_back(part.begin(), part.end());
    }

    return parts;
}

/**
 * @brief Returns the path of a pinned vocabulary file after holding its bytes to the SHA-256 it is pinned by.
 * @param directory The vocabulary directory.
 * @param name The file's name.
 * @param digest The pinned SHA-256.
 * @return The path.
 * @throws std::runtime_error If the file cannot be read or its digest is another.
 */
std::filesystem::path pinned(
        const std::filesystem::path& directory, const std::string_view name, const std::string_view digest)
{
    const auto path{directory / name};

    const auto bytes{contents(path)};

    const auto actual{matisse::common::sha256(bytes)};

    if (actual != digest)
    {
        throw std::runtime_error{
                std::format("{} has the SHA-256 {}, not the pinned {}", path.string(), actual, digest)};
    }

    return path;
}

/**
 * @brief Returns a tokenizer read from the pinned files of a vocabulary directory, each held to its digest.
 * @param directory The vocabulary directory.
 * @param name The tokenizer, gpt2, cl100k, o200k or llama3.
 * @return The tokenizer.
 * @throws std::runtime_error If a file cannot be read or is not the pinned one.
 * @throws std::invalid_argument If the name is none of the four.
 */
Tokenizer tokenizer_of(const std::filesystem::path& directory, const std::string_view name)
{
    if (name == "gpt2")
    {
        const auto encoder{pinned(directory, "gpt2-vocab.json", gpt2_encoder_digest)};

        const auto vocab{pinned(directory, "gpt2-merges.txt", gpt2_vocab_digest)};

        return {.name = "gpt2",
                .pattern = matisse::gpt2::pattern(),
                .width = matisse::validate::default_width,
                .definition = matisse::load::read_gpt2(encoder, vocab)};
    }

    if (name == "cl100k")
    {
        const auto ranks{pinned(directory, "cl100k_base.tiktoken", cl100k_digest)};

        return {.name = "cl100k",
                .pattern = matisse::cl100k::pattern(),
                .width = matisse::validate::default_width,
                .definition = matisse::load::read_tiktoken(ranks, matisse::cl100k::specials())};
    }

    if (name == "o200k")
    {
        const auto ranks{pinned(directory, "o200k_base.tiktoken", o200k_digest)};

        return {.name = "o200k",
                .pattern = matisse::o200k::pattern(),
                .width = o200k_width,
                .definition = matisse::load::read_tiktoken(ranks, matisse::o200k::specials())};
    }

    if (name == "llama3")
    {
        const auto ranks{pinned(directory, "llama3.tiktoken", llama3_digest)};

        return {.name = "llama3",
                .pattern = matisse::llama3::pattern(),
                .width = matisse::validate::default_width,
                .definition = matisse::load::read_tiktoken(ranks, matisse::llama3::specials())};
    }

    throw std::invalid_argument{std::format("the tokenizers are gpt2, cl100k, o200k and llama3, not '{}'", name)};
}

/**
 * @brief Returns the median of a sorted sample: the mean of its two middle values, one value for an odd count.
 * @param sorted The sample, ascending and not empty.
 * @return The median.
 */
double median_of(const std::span<const double> sorted)
{
    const auto middle{sorted.size() / 2};

    if (sorted.size() % 2 == 1)
    {
        return sorted[middle];
    }

    return (sorted[middle - 1] + sorted[middle]) / 2.0;
}

/**
 * @brief Times a pass: a first pass held to the reference, then the timed passes, each held to the first.
 * @tparam Pass Callable running one pass and returning a result comparable with the reference.
 * @tparam Result The result's type.
 * @param passes The timed passes.
 * @param pass The pass.
 * @param reference The result every pass must give.
 * @param count The count the result holds, reported with the timing.
 * @return The pass times and the count.
 * @throws std::runtime_error If the first pass gives another result than the reference or a timed pass another than
 *         the first.
 */
template <typename Pass, typename Result>
Timing timed(const std::size_t passes, const Pass& pass, const Result& reference, const std::size_t count)
{
    const auto first{pass()};

    if (first != reference)
    {
        throw std::runtime_error{"the result differs from its reference"};
    }

    std::vector<double> seconds{};

    for (std::size_t index{0}; index < passes; ++index)
    {
        const auto start{std::chrono::steady_clock::now()};

        const auto result{pass()};

        const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - start};

        if (result != first)
        {
            throw std::runtime_error{"the result changed between passes"};
        }

        seconds.push_back(elapsed.count());
    }

    std::ranges::sort(seconds);

    return {.seconds = std::move(seconds), .count = count};
}

/**
 * @brief Prints a row: the tokenizer, the row and the input's names, the input's size, the count and the passes, the
 *        best, median and worst throughput and the best and median rate of the count.
 * @param tokenizer The tokenizer's name.
 * @param row The row's name.
 * @param input The input's name.
 * @param bytes The bytes a pass consumes.
 * @param timing The row's timing.
 * @param unit What the count counts, tokens or pieces.
 */
void print_row(
        const std::string_view tokenizer, const std::string_view row, const std::string_view input,
        const std::size_t bytes, const Timing& timing, const std::string_view unit)
{
    const auto& [seconds, count]{timing};

    const auto mib{static_cast<double>(bytes) / bytes_per_mebibyte};

    const auto median{median_of(seconds)};

    const auto millions{static_cast<double>(count) / 1e6};

    const auto best_rate{mib / seconds.front()};

    const auto median_rate{mib / median};

    const auto worst_rate{mib / seconds.back()};

    const auto best_count_rate{millions / seconds.front()};

    const auto median_count_rate{millions / median};

    std::printf(
            "%-7.*s %-15.*s %-24.*s %7.1f MiB, %10zu %.*s, %zu passes: best %7.1f, median %7.1f, worst %7.1f MiB/s; "
            "best %6.2f, median %6.2f M %.*s/s\n",
            static_cast<int>(tokenizer.size()), tokenizer.data(), static_cast<int>(row.size()), row.data(),
            static_cast<int>(input.size()), input.data(), mib, count, static_cast<int>(unit.size()), unit.data(),
            seconds.size(), best_rate, median_rate, worst_rate, best_count_rate, median_count_rate,
            static_cast<int>(unit.size()), unit.data());
}

/**
 * @brief Returns the SHA-256 of a run of ids written as little-endian 32-bit words, the fixtures' digest of ids.
 * @param ids The ids.
 * @return The 64 hexadecimal digits.
 */
std::string digest_of(const std::span<const Token_id_t> ids)
{
    std::string words{};

    words.reserve(ids.size() * 4);

    for (const auto id : ids)
    {
        for (unsigned shift{0}; shift < 32; shift += 8)
        {
            words += static_cast<char>((id >> shift) & 0xFFU);
        }
    }

    return matisse::common::sha256(words);
}

/**
 * @brief Prints the ids line of an input: the tokenizer and the input's names, the count of the ids and their digest.
 * @param tokenizer The tokenizer's name.
 * @param input The input's name.
 * @param ids The ids.
 */
void print_ids(const std::string_view tokenizer, const std::string_view input, const std::span<const Token_id_t> ids)
{
    const auto digest{digest_of(ids)};

    std::printf(
            "ids %.*s %.*s %zu %s\n", static_cast<int>(tokenizer.size()), tokenizer.data(),
            static_cast<int>(input.size()), input.data(), ids.size(), digest.c_str());
}

/**
 * @brief Returns the ids of a text with no cache: every piece the segmenter cuts looked up whole where the definition
 *        says so, and otherwise merged alone.
 * @param merger The definition's merger.
 * @param segmenter The pattern's segmenter.
 * @param text The text.
 * @return The ids.
 */
Ids uncached_ids(
        const matisse::encode::Merger& merger, const matisse::pretok::Segmenter& segmenter, const std::string_view text)
{
    Ids ids{};

    std::size_t start{0};

    for (const auto end : segmenter.boundaries(text))
    {
        const auto piece{text.substr(start, end - start)};

        start = end;

        if (const auto whole{merger.whole_piece(piece)})
        {
            ids.push_back(*whole);

            continue;
        }

        const auto merged{merger.merged(piece)};

        ids.insert(ids.end(), merged.begin(), merged.end());
    }

    return ids;
}

/**
 * @brief Times encode(), the encoding with no cache and the segmentation alone on one corpus.
 * @param tokenizer The tokenizer.
 * @param encoder The tokenizer's encoder.
 * @param name The corpus's name.
 * @param text The corpus.
 * @param passes The timed passes.
 */
void run_single(
        const Tokenizer& tokenizer, const Encoder& encoder, const std::string_view name, const std::string_view text,
        const std::size_t passes)
{
    const auto& [tokenizer_name, pattern, width, definition]{tokenizer};

    const auto ids{encoder.encode(text)};

    print_ids(tokenizer_name, name, ids);

    const auto encoding{timed(passes, [&] { return encoder.encode(text); }, ids, ids.size())};

    print_row(tokenizer_name, "encode", name, text.size(), encoding, "tokens");

    const matisse::encode::Merger merger{definition};

    const matisse::pretok::Segmenter segmenter{pattern};

    const auto uncached{timed(passes, [&] { return uncached_ids(merger, segmenter, text); }, ids, ids.size())};

    print_row(tokenizer_name, "encode no cache", name, text.size(), uncached, "tokens");

    const auto ends{segmenter.boundaries(text)};

    const auto segmentation{timed(passes, [&] { return segmenter.boundaries(text); }, ends, ends.size())};

    print_row(tokenizer_name, "segment", name, text.size(), segmentation, "pieces");
}

/**
 * @brief Returns the bytes of the parts of a text.
 * @param parts The parts.
 * @return Their sizes summed.
 */
std::size_t bytes_of(const std::span<const std::string_view> parts)
{
    const auto sizes{parts | std::views::transform(&std::string_view::size)};

    return std::ranges::fold_left(sizes, 0UZ, std::plus{});
}

/**
 * @brief Returns the ids of a batch concatenated in the documents' order.
 * @param encoded The ids of each document.
 * @return The ids.
 */
Ids concatenated(const std::span<const Ids> encoded)
{
    Ids all{};

    for (const auto& document_ids : encoded)
    {
        all.insert(all.end(), document_ids.begin(), document_ids.end());
    }

    return all;
}

/**
 * @brief Times one way of calling encode() once per prompt: a first pass, then the timed passes, each call timed alone.
 * @tparam Encode Callable returning the ids of one prompt.
 * @param prompts The prompts.
 * @param passes The timed passes.
 * @param encode_one The call.
 * @return The timing.
 * @throws std::runtime_error If a timed pass gives other ids than the first.
 */
template <typename Encode>
Prompt_timing time_prompts(
        const std::span<const std::string_view> prompts, const std::size_t passes, const Encode& encode_one)
{
    std::vector<double> latencies{};

    const auto encode_all{[&](const bool timed_pass) {
        Batch_ids encoded{};

        encoded.reserve(prompts.size());

        for (const auto prompt : prompts)
        {
            const auto start{std::chrono::steady_clock::now()};

            auto ids{encode_one(prompt)};

            const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - start};

            if (timed_pass)
            {
                latencies.push_back(elapsed.count());
            }

            encoded.push_back(std::move(ids));
        }

        return encoded;
    }};

    const auto first{encode_all(false)};

    std::vector<double> seconds{};

    for (std::size_t pass{0}; pass < passes; ++pass)
    {
        const auto before{latencies.size()};

        const auto result{encode_all(true)};

        if (result != first)
        {
            throw std::runtime_error{"the prompts' ids changed between passes"};
        }

        const auto pass_latencies{std::span{latencies}.subspan(before)};

        const auto pass_seconds{std::ranges::fold_left(pass_latencies, 0.0, std::plus{})};

        seconds.push_back(pass_seconds);
    }

    std::ranges::sort(seconds);

    std::ranges::sort(latencies);

    return {.first = first, .seconds = std::move(seconds), .latencies = std::move(latencies)};
}

/**
 * @brief Prints the row of a prompt timing and the latency of a call beside it.
 * @param tokenizer_name The tokenizer's name.
 * @param row The row's name.
 * @param latency_row The name of the latency line.
 * @param name The prompt file's name.
 * @param prompts The prompts.
 * @param timing The timing.
 */
void print_prompt_rows(
        const std::string_view tokenizer_name, const std::string_view row, const std::string_view latency_row,
        const std::string_view name, const std::span<const std::string_view> prompts, const Prompt_timing& timing)
{
    const auto& [first, seconds, latencies]{timing};

    const auto sizes{first | std::views::transform(&Ids::size)};

    const auto tokens{std::ranges::fold_left(sizes, 0UZ, std::plus{})};

    const auto bytes{bytes_of(prompts)};

    print_row(tokenizer_name, row, name, bytes, {.seconds = seconds, .count = tokens}, "tokens");

    // The order statistic at floor(q (n - 1)) of the n latencies ascending, as the vendors' harnesses take it.
    const auto microseconds{[&](const double quantile) {
        const auto index{static_cast<std::size_t>(quantile * static_cast<double>(latencies.size() - 1))};

        return latencies[index] * 1e6;
    }};

    const auto calls{static_cast<double>(prompts.size())};

    const auto mean{median_of(seconds) / calls * 1e6};

    const auto average_bytes{static_cast<double>(bytes) / calls};

    const auto p50{microseconds(0.5)};

    const auto p90{microseconds(0.9)};

    const auto p99{microseconds(0.99)};

    std::printf(
            "%-7.*s %-15.*s %-24.*s %zu calls of %.0f bytes on average: per call mean %.2f, p50 %.2f, p90 %.2f, p99 "
            "%.2f us\n",
            static_cast<int>(tokenizer_name.size()), tokenizer_name.data(), static_cast<int>(latency_row.size()),
            latency_row.data(), static_cast<int>(name.size()), name.data(), prompts.size(), average_bytes, mean, p50,
            p90, p99);
}

/**
 * @brief Times encode() called once per prompt, each call with a cache of its own and then with one cache kept across
 *        every call and pass, which the timed passes find full, and prints the latency of a call beside the throughput
 *        of a pass.
 * @param tokenizer_name The tokenizer's name.
 * @param encoder The tokenizer's encoder.
 * @param name The prompt file's name.
 * @param prompts The prompts.
 * @param passes The timed passes.
 * @throws std::invalid_argument If there is no prompt.
 * @throws std::runtime_error If a timed pass gives other ids than the first, or the kept cache other ids than encode().
 */
void run_prompts(
        const std::string_view tokenizer_name, const Encoder& encoder, const std::string_view name,
        const std::span<const std::string_view> prompts, const std::size_t passes)
{
    if (prompts.empty() || bytes_of(prompts) == 0)
    {
        throw std::invalid_argument{std::format("{} holds no prompt", name)};
    }

    const auto encode_alone{[&](const std::string_view prompt) { return encoder.encode(prompt); }};

    const auto alone{time_prompts(prompts, passes, encode_alone)};

    const auto all{concatenated(alone.first)};

    print_ids(tokenizer_name, name, all);

    print_prompt_rows(tokenizer_name, "prompts", "latency", name, prompts, alone);

    matisse::encode::Piece_cache cache{matisse::encode::piece_cache_capacity};

    const auto encode_kept{[&](const std::string_view prompt) { return encoder.encode(prompt, cache); }};

    const auto kept{time_prompts(prompts, passes, encode_kept)};

    if (kept.first != alone.first)
    {
        throw std::runtime_error{"the prompts' ids with a kept cache differ from encode()'s"};
    }

    print_prompt_rows(tokenizer_name, "prompts kept", "latency kept", name, prompts, kept);
}

/**
 * @brief Times encode_parallel() at each worker count on one long text, beside encode().
 * @param tokenizer_name The tokenizer's name.
 * @param encoder The tokenizer's encoder.
 * @param name The text's name.
 * @param text The text.
 * @param passes The timed passes.
 * @param workers The worker counts.
 */
void run_parallel(
        const std::string_view tokenizer_name, const Encoder& encoder, const std::string_view name,
        const std::string_view text, const std::size_t passes, const std::span<const std::size_t> workers)
{
    const auto ids{encoder.encode(text)};

    print_ids(tokenizer_name, name, ids);

    const auto serial{timed(passes, [&] { return encoder.encode(text); }, ids, ids.size())};

    print_row(tokenizer_name, "encode", name, text.size(), serial, "tokens");

    const auto serial_median{median_of(serial.seconds)};

    for (const auto count : workers)
    {
        const auto boundaries{encoder.chunk_boundaries(text, count)};

        const auto chunks{boundaries.size() - 1};

        const auto parallel{timed(passes, [&] { return encoder.encode_parallel(text, count); }, ids, ids.size())};

        const auto row{std::format("parallel x{}", count)};

        print_row(tokenizer_name, row, name, text.size(), parallel, "tokens");

        const auto parallel_median{median_of(parallel.seconds)};

        const auto speedup{serial_median / parallel_median};

        std::printf(
                "%-7.*s %-15s %-24.*s %zu chunks, median speedup %.2f over encode()\n",
                static_cast<int>(tokenizer_name.size()), tokenizer_name.data(), row.c_str(),
                static_cast<int>(name.size()), name.data(), chunks, speedup);
    }
}

/**
 * @brief Returns the ids of every document, encode() called on each by a pool of threads that take the next document
 *        from a shared counter, the calling thread one of them, each thread keeping one piece cache across the
 *        documents it takes.
 * @param encoder The encoder.
 * @param documents The documents.
 * @param threads The threads of the pool.
 * @return The ids of each document, in the documents' order.
 * @throws std::invalid_argument If a document is not well-formed UTF-8, the first refusal rethrown once every thread
 *         has joined.
 */
Batch_ids encode_batch(
        const Encoder& encoder, const std::span<const std::string_view> documents, const std::size_t threads)
{
    Batch_ids encoded(documents.size());

    std::atomic<std::size_t> next{0};

    std::mutex failure_mutex{};

    std::exception_ptr failure{};

    const auto work{[&] {
        try
        {
            matisse::encode::Piece_cache cache{matisse::encode::piece_cache_capacity};

            for (auto index{next.fetch_add(1)}; index < documents.size(); index = next.fetch_add(1))
            {
                encoded[index] = encoder.encode(documents[index], cache);
            }
        }
        catch (...)
        {
            const std::scoped_lock lock{failure_mutex};

            if (!failure)
            {
                failure = std::current_exception();
            }
        }
    }};

    {
        std::vector<std::jthread> workers{};

        for (std::size_t worker{1}; worker < threads; ++worker)
        {
            workers.emplace_back(work);
        }

        work();
    }

    if (failure)
    {
        std::rethrow_exception(failure);
    }

    return encoded;
}

/**
 * @brief Times encode() over the documents of a batch at each thread count, every pass held to encode() called on each
 *        document with a cache of its own, and prints the batch's ids line.
 * @param tokenizer_name The tokenizer's name.
 * @param encoder The tokenizer's encoder.
 * @param name The batch's name.
 * @param documents The documents.
 * @param passes The timed passes.
 * @param threads The thread counts.
 */
void run_batch(
        const std::string_view tokenizer_name, const Encoder& encoder, const std::string_view name,
        const std::span<const std::string_view> documents, const std::size_t passes,
        const std::span<const std::size_t> threads)
{
    Batch_ids serial{};

    serial.reserve(documents.size());

    for (const auto document : documents)
    {
        serial.push_back(encoder.encode(document));
    }

    const auto all{concatenated(serial)};

    print_ids(tokenizer_name, name, all);

    std::printf(
            "%-7.*s %-15s %-24.*s %zu documents\n", static_cast<int>(tokenizer_name.size()), tokenizer_name.data(),
            "batch", static_cast<int>(name.size()), name.data(), documents.size());

    const auto bytes{bytes_of(documents)};

    for (const auto count : threads)
    {
        const auto timing{timed(passes, [&] { return encode_batch(encoder, documents, count); }, serial, all.size())};

        const auto row{std::format("batch x{}", count)};

        print_row(tokenizer_name, row, name, bytes, timing, "tokens");
    }
}

/**
 * @brief Runs the mode asked for, for every tokenizer named.
 * @param options The options.
 * @throws std::runtime_error If a file cannot be read or a result differs from its reference.
 * @throws std::invalid_argument If a tokenizer is unknown or the input is not well-formed UTF-8.
 */
void run(const Options& options)
{
    const auto& [mode, vocabularies, tokenizer_names, passes, file, counts]{options};

    const auto text{contents(file)};

    const auto name{file.filename().string()};

    const auto parts{parts_of(text)};

    for (const auto& tokenizer_name : tokenizer_names)
    {
        const auto tokenizer{tokenizer_of(vocabularies, tokenizer_name)};

        const auto& [read_name, pattern, width, definition]{tokenizer};

        const auto report{matisse::validate::validate(pattern, width)};

        const Encoder encoder{definition, report, Inventory::factorization, pattern, {}};

        if (mode == "single")
        {
            run_single(tokenizer, encoder, name, text, passes);
        }
        else if (mode == "prompts")
        {
            run_prompts(read_name, encoder, name, parts, passes);
        }
        else if (mode == "parallel")
        {
            run_parallel(read_name, encoder, name, text, passes, counts);
        }
        else
        {
            run_batch(read_name, encoder, name, parts, passes, counts);
        }

        std::fflush(stdout);
    }
}

} // namespace

/**
 * @brief Runs the bench.
 * @param argc The argument count.
 * @param argv The arguments: the mode, the vocabulary directory, the tokenizers, the passes, the file and the counts.
 * @return EXIT_SUCCESS when every row was held to its reference and timed, EXIT_FAILURE otherwise.
 */
int main(const int argc, char** argv)
{
    const std::vector<std::string_view> arguments{argv + 1, argv + argc};

    try
    {
        const auto options{parse_options(arguments)};

        print_provenance(arguments);

        run(options);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "matisse_bench: %s\n", error.what());

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
