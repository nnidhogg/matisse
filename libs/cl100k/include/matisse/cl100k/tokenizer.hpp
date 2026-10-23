#ifndef MATISSE_LIBS_CL100K_INCLUDE_MATISSE_CL100K_TOKENIZER_HPP
#define MATISSE_LIBS_CL100K_INCLUDE_MATISSE_CL100K_TOKENIZER_HPP

#include <vector>

#include "matisse/load/definition.hpp"
#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief cl100k_base's tokenizer as matisse describes it: its pre-tokenizer pattern, pattern(), and the vendor's
 *        special tokens of its rank file, specials().
 *
 * cl100k_base.tiktoken is read by load::read_tiktoken() with specials(); the stages take pattern() beside the
 * definition.
 */
namespace matisse::cl100k
{
/**
 * @brief Returns cl100k's pattern, `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|`
 *        ` ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s`, as openai/tiktoken's tiktoken_ext/openai_public.py
 *        publishes it, described over its 16 classes.
 *
 * The classes are GPT-2's with the carriage return and the line feed its own besides, and its contractions are
 * caseless, so each contraction letter's class holds both cases and that of `s` also U+017F, which matches `s`
 * caselessly; the symbols are `\t\n\r !'0Adelmrstv`, read from tables pinned to Unicode 15.0.0, the version of the
 * Python reference the fixtures are written by.
 * @return The pattern, its languages built with munch's combinators and its views and spans static.
 */
[[nodiscard]] pretok::Pattern pattern();

/**
 * @brief Returns the vendor's special tokens of cl100k_base's rank file, quoted from tiktoken's openai_public.py:
 *        `<|endoftext|>` 100257, `<|fim_prefix|>` 100258, `<|fim_middle|>` 100259, `<|fim_suffix|>` 100260 and
 *        `<|endofprompt|>` 100276.
 * @return The special tokens, by id ascending.
 */
[[nodiscard]] std::vector<load::Special_token> specials();

} // namespace matisse::cl100k

#endif // MATISSE_LIBS_CL100K_INCLUDE_MATISSE_CL100K_TOKENIZER_HPP
