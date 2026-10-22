#include "matisse/lookup/tokenizers.hpp"

#include <algorithm>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "matisse/cl100k/tokenizer.hpp"
#include "matisse/gpt2/tokenizer.hpp"
#include "matisse/o200k/tokenizer.hpp"

namespace matisse::lookup
{
std::vector<Tokenizer> tokenizers()
{
    return {{.name = "gpt2", .pattern = gpt2::pattern(), .specials = gpt2::specials()},
            {.name = "cl100k", .pattern = cl100k::pattern(), .specials = cl100k::specials()},
            {.name = "o200k", .pattern = o200k::pattern(), .specials = o200k::specials()}};
}

std::optional<Tokenizer> tokenizer(const std::string_view name)
{
    auto described{tokenizers()};

    const auto found{std::ranges::find(described, name, &Tokenizer::name)};

    if (found == described.end())
    {
        return std::nullopt;
    }

    return std::move(*found);
}

} // namespace matisse::lookup
