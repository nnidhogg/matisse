#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "matisse/load/definition.hpp"
#include "matisse/tools/fuzz/harness.hpp"

using matisse::load::Merge_rule;
using matisse::tools::fuzz::require;
using matisse::tools::fuzz::require_spelled_apart;
using matisse::tools::fuzz::written;

/**
 * @brief Reads the fuzz input as GPT-2's encoder.json and vocab.bpe, the fuzz entry point run once per input.
 *
 * The input is cut at its first NUL byte, which no encoder.json the reader takes holds, since JSON has no raw control
 * character, so that every pair the reader takes is an input: the bytes before it are written to the file read as
 * encoder.json and the bytes after it to the file read as vocab.bpe, and an input with no NUL is encoder.json alone
 * with an empty vocab.bpe. A std::invalid_argument is the reader refusing a pair it cannot take exactly, which is its
 * contract; anything else that escapes, and any sanitizer report, is a finding. A definition read is held to what the
 * reader promises of it: the pair rule with no whole-piece lookup and no special tokens, its tokens spelled apart, and
 * no pair listed twice.
 * @param data The fuzz input.
 * @param size The input's size.
 * @return Zero; a violated invariant traps instead.
 */
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* const data, const std::size_t size)
{
    const std::string_view input{reinterpret_cast<const char*>(data), size};

    const auto cut{input.find('\0')};

    const auto encoder{written("encoder.json", input.substr(0, cut))};

    const auto vocab{written("vocab.bpe", cut == std::string_view::npos ? std::string_view{} : input.substr(cut + 1))};

    try
    {
        const auto definition{matisse::load::read_gpt2(encoder, vocab)};

        require(definition.merge_rule == Merge_rule::pairs);

        require(!definition.whole_piece_lookup);

        require(definition.specials.empty());

        require_spelled_apart(definition.spellings);

        std::set<std::pair<std::string_view, std::string_view>> pairs{};

        for (const auto& [left, right] : definition.merges)
        {
            require(pairs.emplace(left, right).second);
        }
    }
    catch (const std::invalid_argument&)
    {}

    return 0;
}
