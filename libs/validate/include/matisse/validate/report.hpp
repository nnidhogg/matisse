#ifndef MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP
#define MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP

/**
 * @file
 * @brief The validator, validate(), supply(), validate_merges(), render() and json(): which class windows of a
 *        definition's pattern certify a cut, at which origin and with which right context, how many anchors they
 *        supply on an input, and which candidate windows of a merge table without a pre-tokenizer certify.
 *
 * validate() decides every class window of width two to four under both certificates, the policy certificate by
 * munch::verifier::miscovering() on the split verifier and the factorization certificate by is_split_window() on the
 * factorization lexer; each window table carries the right context its inventory needs, and the report carries
 * boundary_gap() of the split verifier. validate_merges() builds munch::verifier::token_automaton() once and decides
 * each candidate window by miscovering() on it, the lazy certification. supply() measures one inventory's anchors on an
 * input as munch-audit's supply does, with the share of the serial pre-token boundaries that are anchors and the
 * longest piece beside them, and render() and json() follow munch-audit's report renderings. Nothing is declared here
 * yet; docs/design.md sketches the declarations.
 */
namespace matisse::validate
{
} // namespace matisse::validate

#endif // MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP
