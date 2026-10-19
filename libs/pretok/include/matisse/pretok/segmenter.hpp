#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP

/**
 * @file
 * @brief The vendor pre-tokenizers of GPT-2 and cl100k_base: the runtime Segmenter, split_verifier() building the
 *        pattern's verifier, and factorization_lexer() compiling the gate-erased union of its alternatives.
 *
 * The Segmenter commits at each boundary the longest gate-passing accept of the first alternative that has one, each
 * alternative a small DFA over the classes of classes.hpp, and returns the piece ends as byte offsets. split_verifier()
 * is the minimized verifier of the pattern's one-step relation, handed to munch as a Verifier table over the class
 * symbols, 22 states for GPT-2 and 33 for cl100k, and factorization_lexer() is the munch token set on which
 * is_split_window() decides the factorization certificate. Each refuses a definition without a pattern. Nothing is
 * declared here yet; docs/design.md sketches the declarations.
 */
namespace matisse::pretok
{
} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_SEGMENTER_HPP
