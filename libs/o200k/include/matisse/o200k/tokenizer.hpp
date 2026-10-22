#ifndef MATISSE_LIBS_O200K_INCLUDE_MATISSE_O200K_TOKENIZER_HPP
#define MATISSE_LIBS_O200K_INCLUDE_MATISSE_O200K_TOKENIZER_HPP

#include <vector>

#include "matisse/load/definition.hpp"
#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief o200k_base's tokenizer as matisse describes it: its pre-tokenizer pattern, pattern(), and the vendor's special
 *        tokens of its rank file, specials().
 *
 * o200k_base.tiktoken is read by load::read_tiktoken() with specials(); the stages take pattern() beside the
 * definition.
 */
namespace matisse::o200k
{
/**
 * @brief Returns o200k's pattern, the seven alternatives
 *        `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
 *        `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
 *        `\p{N}{1,3}`, ` ?[^\s\p{L}\p{N}]+[\r\n/]*`, `\s*[\r\n]+`, `\s+(?!\S)` and `\s+` joined by `|`, as
 *        openai/tiktoken's tiktoken_ext/openai_public.py publishes it, described over its 30 classes.
 *
 * The classes are cl100k's literals with the slash besides, the general categories Lu, Ll, Lt, Lm and Lo of the letters
 * apart, since the pattern reads upper and lower case apart, and the marks, M; each contraction letter, matched
 * caselessly, is split into its upper and its lower case, and U+017F, the long s, which `s` matches caselessly and
 * which is a lower case letter, is in the class of `s`. A class with an ASCII member is written by one, and the four
 * with none, the other, title case and modifier letters and the marks, by `#`, `$`, `%` and `&`, bytes of the class of
 * everything else, `!`; the symbols are `\t\n\r !#$%&'/0ADELMRSTVadelmrstv`, read from tables pinned to Unicode 15.0.0,
 * the version of the Python reference the fixtures are written by.
 * @return The pattern, whose views and spans are static.
 */
[[nodiscard]] pretok::Pattern pattern() noexcept;

/**
 * @brief Returns the vendor's special tokens of o200k_base's rank file, quoted from tiktoken's openai_public.py:
 *        `<|endoftext|>` 199999 and `<|endofprompt|>` 200018.
 * @return The special tokens, by id ascending.
 */
[[nodiscard]] std::vector<load::Special_token> specials();

} // namespace matisse::o200k

#endif // MATISSE_LIBS_O200K_INCLUDE_MATISSE_O200K_TOKENIZER_HPP
