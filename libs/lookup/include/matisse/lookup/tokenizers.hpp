#ifndef MATISSE_LIBS_LOOKUP_INCLUDE_MATISSE_LOOKUP_TOKENIZERS_HPP
#define MATISSE_LIBS_LOOKUP_INCLUDE_MATISSE_LOOKUP_TOKENIZERS_HPP

#include <optional>
#include <string_view>
#include <vector>

#include "matisse/load/definition.hpp"
#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The lookup by name over matisse's tokenizer libraries, tokenizers() and tokenizer(), for the command line and
 *        any caller choosing a tokenizer at run time.
 *
 * The list names every tokenizer library explicitly, so the lookup links them all; a caller that needs one tokenizer
 * links that library alone and calls it directly.
 */
namespace matisse::lookup
{
/**
 * @brief A tokenizer as a tokenizer library describes it: its name, its pre-tokenizer pattern, and the vendor's special
 *        tokens of a rank file read under it.
 */
struct Tokenizer
{
    /**
     * @brief The name, as the command line spells it.
     */
    std::string_view name{};

    /**
     * @brief The pre-tokenizer pattern.
     */
    pretok::Pattern pattern{};

    /**
     * @brief The vendor's special tokens, by id ascending.
     */
    std::vector<load::Special_token> specials{};
};

/**
 * @brief Returns every tokenizer matisse describes, gpt2, cl100k, o200k and llama3.
 * @return The tokenizers, in that order.
 */
[[nodiscard]] std::vector<Tokenizer> tokenizers();

/**
 * @brief Returns the tokenizer of a name.
 * @param name The name.
 * @return The tokenizer, std::nullopt when no tokenizer has the name.
 */
[[nodiscard]] std::optional<Tokenizer> tokenizer(std::string_view name);

} // namespace matisse::lookup

#endif // MATISSE_LIBS_LOOKUP_INCLUDE_MATISSE_LOOKUP_TOKENIZERS_HPP
