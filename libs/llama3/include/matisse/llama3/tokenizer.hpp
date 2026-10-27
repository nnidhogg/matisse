#ifndef MATISSE_LIBS_LLAMA3_INCLUDE_MATISSE_LLAMA3_TOKENIZER_HPP
#define MATISSE_LIBS_LLAMA3_INCLUDE_MATISSE_LLAMA3_TOKENIZER_HPP

#include <vector>

#include "matisse/load/definition.hpp"
#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief Llama 3's tokenizer as matisse describes it: its pre-tokenizer pattern, pattern(), and the vendor's special
 *        tokens of its rank file, specials().
 *
 * llama3.tiktoken is read by load::read_tiktoken() with specials(); the stages take pattern() beside the definition.
 */
namespace matisse::llama3
{
/**
 * @brief Returns Llama 3's pattern, `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|`
 *        ` ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, as meta-llama/llama3's llama/tokenizer.py publishes
 *        it, described over its 16 classes.
 *
 * The pattern is cl100k's as tiktoken shipped it before the possessive rewrite, so its classes are cl100k's, the
 * symbols `\t\n\r !'0Adelmrstv`, read from tables pinned to Unicode 15.0.0, the version of the Python reference the
 * fixtures are written by; it has no alternative anchored at the end of the input, so white space after a line break at
 * the end of a text is a piece of its own.
 * @return The pattern, its languages built with munch's combinators.
 */
[[nodiscard]] pretok::Pattern pattern();

/**
 * @brief Returns the vendor's special tokens of llama3.tiktoken, as meta-llama/llama3's llama/tokenizer.py numbers them
 *        from 128000, the count of the file's ordinary tokens: `<|begin_of_text|>`, `<|end_of_text|>`,
 *        `<|reserved_special_token_0|>` to `<|reserved_special_token_3|>`, `<|start_header_id|>`, `<|end_header_id|>`,
 *        `<|reserved_special_token_4|>` and `<|eot_id|>`, then `<|reserved_special_token_5|>` to
 *        `<|reserved_special_token_250|>`, 256 in all.
 * @return The special tokens, by id ascending.
 */
[[nodiscard]] std::vector<load::Special_token> specials();

} // namespace matisse::llama3

#endif // MATISSE_LIBS_LLAMA3_INCLUDE_MATISSE_LLAMA3_TOKENIZER_HPP
