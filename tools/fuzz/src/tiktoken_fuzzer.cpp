#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "matisse/load/definition.hpp"
#include "matisse/tools/fuzz/harness.hpp"

using matisse::load::Merge_rule;
using matisse::tools::fuzz::require;
using matisse::tools::fuzz::require_spelled_apart;
using matisse::tools::fuzz::written;

/**
 * @brief Reads the fuzz input as a tiktoken rank file, the fuzz entry point run once per input.
 *
 * The input is written whole to the file read_tiktoken() reads, with no special tokens, which the reader carries over
 * without reading the file for them. A std::invalid_argument is the reader refusing a file it cannot take exactly,
 * which is its contract; anything else that escapes, and any sanitizer report, is a finding. A definition read is held
 * to what the reader promises of it: the rank rule with the whole-piece lookup, no merge table, and its tokens spelled
 * apart.
 * @param data The fuzz input.
 * @param size The input's size.
 * @return Zero; a violated invariant traps instead.
 */
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* const data, const std::size_t size)
{
    const std::string_view input{reinterpret_cast<const char*>(data), size};

    const auto path{written("ranks.tiktoken", input)};

    try
    {
        const auto definition{matisse::load::read_tiktoken(path, {})};

        require(definition.merge_rule == Merge_rule::ranks);

        require(definition.whole_piece_lookup);

        require(definition.merges.empty());

        require(definition.specials.empty());

        require_spelled_apart(definition.spellings);
    }
    catch (const std::invalid_argument&)
    {}

    return 0;
}
