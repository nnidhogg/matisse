#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "matisse/load/definition.hpp"
#include "matisse/tools/fuzz/harness.hpp"

using matisse::load::Merge_rule;
using matisse::load::Special_token;
using matisse::tools::fuzz::require;
using matisse::tools::fuzz::require_spelled_apart;
using matisse::tools::fuzz::written;

/**
 * @brief Reads the fuzz input as a Hugging Face tokenizer.json, the fuzz entry point run once per input.
 *
 * The input is written whole to the file the reader reads, its pattern's spelling found by tokenizer_json_spelling()
 * and the file read under it by read_tokenizer_json(). A std::invalid_argument is the reader refusing a file it cannot
 * take exactly, which is its contract; anything else that escapes, and any sanitizer report, is a finding. A definition
 * read is held to what the reader promises of it: the pair rule, its tokens spelled apart, no pair listed twice, and
 * its special tokens nonempty, by id ascending and past every ordinary id; and the file is refused under any other
 * spelling.
 * @param data The fuzz input.
 * @param size The input's size.
 * @return Zero; a violated invariant traps instead.
 */
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* const data, const std::size_t size)
{
    const std::string_view input{reinterpret_cast<const char*>(data), size};

    const auto path{written("tokenizer.json", input)};

    try
    {
        const auto spelling{matisse::load::tokenizer_json_spelling(path)};

        const auto other{spelling + "|"};

        try
        {
            std::ignore = matisse::load::read_tokenizer_json(path, other);

            require(false);
        }
        catch (const std::invalid_argument&)
        {}

        const auto definition{matisse::load::read_tokenizer_json(path, spelling)};

        require(definition.merge_rule == Merge_rule::pairs);

        require_spelled_apart(definition.spellings);

        std::set<std::pair<std::string_view, std::string_view>> pairs{};

        for (const auto& [left, right] : definition.merges)
        {
            const auto [position, inserted]{pairs.emplace(left, right)};

            require(inserted);
        }

        require(std::ranges::is_sorted(definition.specials, {}, &Special_token::id));

        for (const auto& [text, id, normalized] : definition.specials)
        {
            require(!text.empty());

            require(id >= definition.spellings.size());
        }
    }
    catch (const std::invalid_argument&)
    {}

    return 0;
}
