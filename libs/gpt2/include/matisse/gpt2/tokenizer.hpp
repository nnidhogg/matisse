#ifndef MATISSE_LIBS_GPT2_INCLUDE_MATISSE_GPT2_TOKENIZER_HPP
#define MATISSE_LIBS_GPT2_INCLUDE_MATISSE_GPT2_TOKENIZER_HPP

#include <vector>

#include "matisse/load/definition.hpp"
#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief GPT-2's tokenizer as matisse describes it: its pre-tokenizer pattern, pattern(), and the vendor's special
 *        tokens of a rank file read under it, specials().
 *
 * GPT-2's files, encoder.json and vocab.bpe, are read by load::read_gpt2() and a tiktoken rank file of the gpt2 or
 * r50k_base encoding by load::read_tiktoken() with specials(); the stages take pattern() beside the definition.
 */
namespace matisse::gpt2
{
/**
 * @brief Returns GPT-2's pattern, `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, as
 *        openai/gpt-2's src/encoder.py publishes it, described over its 14 classes.
 *
 * The classes are the space, the apostrophe and the letters of the contractions, `d`, `e`, `l`, `m`, `r`, `s`, `t` and
 * `v`, each its own and case sensitive, then the rest of the white space, of the letters and of the numbers, and
 * everything else, with the symbols `\t !'0Adelmrstv`, read from tables pinned to Unicode 15.0.0, the version of the
 * Python reference the fixtures are written by.
 * @return The pattern, its languages built with munch's combinators.
 */
[[nodiscard]] pretok::Pattern pattern();

/**
 * @brief Returns the vendor's special tokens of a rank file read under GPT-2's pattern, those of the gpt2 and r50k_base
 *        encodings quoted from tiktoken's openai_public.py: `<|endoftext|>` 50256.
 * @return The special tokens, by id ascending.
 */
[[nodiscard]] std::vector<load::Special_token> specials();

} // namespace matisse::gpt2

#endif // MATISSE_LIBS_GPT2_INCLUDE_MATISSE_GPT2_TOKENIZER_HPP
