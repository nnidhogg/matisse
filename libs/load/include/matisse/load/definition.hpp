#ifndef MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP
#define MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP

/**
 * @file
 * @brief The loaders, read_tokenizer_json(), read_gpt2() and read_tiktoken(), each reading a real tokenizer's files
 *        into one Definition: its pre-tokenizer pattern, its merge rule, the bytes of every ordinary token by id, the
 *        merge table in rank order and the special tokens.
 *
 * A definition is held in raw bytes throughout, with GPT-2's byte-to-character map undone by the loader. A
 * pre-tokenizer is recognized only by the exact published spelling of its pattern, GPT-2's or cl100k's, and content a
 * loader cannot take exactly is refused with std::invalid_argument naming the member and where it is: a model other
 * than BPE, a normalizer other than none, any other pattern, byte fallback, a post-processor that adds tokens, ids that
 * are not dense from zero and an empty token. A file that cannot be read throws std::runtime_error. A tiktoken rank
 * file names neither its pattern nor its special tokens, so the caller names the pattern and the vendor's special
 * tokens come from a table of matisse's own. Nothing is declared here yet; docs/design.md sketches the declarations.
 */
namespace matisse::load
{
} // namespace matisse::load

#endif // MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP
