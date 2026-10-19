#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_PATTERN_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_PATTERN_HPP

#include <span>
#include <string_view>

#include <munch/regex/utf8.hpp>

/**
 * @file
 * @brief The description of a pre-tokenizer pattern as data, Pattern: its class alphabet, Alphabet, with the code point
 *        sets the alphabet reads, Code_point_set, and its ordered gated alternatives, Alternative.
 *
 * A pattern is an ordered gated pattern over its class alphabet: a list of regular languages of nonempty class strings,
 * each with a gate, the class symbols that may follow its accepts, the end of the input always passing. Its discipline
 * commits, on a nonempty residual, the longest prefix in an alternative's language whose next symbol passes that
 * alternative's gate, for the first alternative that has one. An engine running the published pattern realizes the
 * description when no alternative reads left of where its match begins, each alternative matches the longest
 * gate-passing prefix of its language and fails only where there is none, the pattern is total, and the engine reads
 * code points only through the classes. Each tokenizer library gives its own description; the stages read it and name
 * no tokenizer.
 */
namespace matisse::pretok
{
/**
 * @brief A set of code points an alphabet reads, such as the general category L, and the symbol of its class.
 */
struct Code_point_set
{
    /**
     * @brief The symbol of the set's class.
     */
    char symbol{};

    /**
     * @brief The set's code points, ranges ascending and apart.
     */
    std::span<const munch::regex::utf8::Code_point_range> ranges{};
};

/**
 * @brief The class alphabet of a pattern: the partition of the code points into classes that agree on every set the
 *        pattern names and on every literal it spells, each class written as one ASCII symbol.
 *
 * A code point is of a literal's class when it is that literal, or when the alphabet is caseless and its case folding
 * is; otherwise of the class of the first set holding it; otherwise of the other class.
 */
struct Alphabet
{
    /**
     * @brief The symbol of every class, ascending, so a class is its symbol's place among them.
     */
    std::string_view symbols{};

    /**
     * @brief The symbols that are literals of the pattern, each a class of that code point alone, or of its case
     *        foldings when the alphabet is caseless.
     */
    std::string_view literals{};

    /**
     * @brief Whether the pattern's literal letters match caselessly, as cl100k's `(?i:...)` matches its contraction
     *        letters: A to Z fold to a to z and U+017F, the long s, to s, the simple case folding both the regex crate
     *        and Python's re apply.
     */
    bool caseless{};

    /**
     * @brief The sets the alphabet reads, in the order a code point is classed by them.
     */
    std::span<const Code_point_set> sets{};

    /**
     * @brief The symbol of every code point in no set and no literal.
     */
    char other{};

    /**
     * @brief The Unicode version the sets' tables were generated from.
     */
    std::string_view unicode_version{};
};

/**
 * @brief One alternative of a pattern: its consuming language and its gate.
 */
struct Alternative
{
    /**
     * @brief The alternative's language, nonempty class strings written over the class symbols in munch's pattern
     *        syntax, with the published pattern's possessive quantifiers read as greedy.
     */
    std::string_view language{};

    /**
     * @brief The class symbols that may follow an accept, the end of the input always passing: every symbol for an
     *        ungated alternative, the white space for `(?!\S)`, and none for an alternative anchored at the end.
     */
    std::string_view gate{};
};

/**
 * @brief A pre-tokenizer pattern, an ordered gated pattern over its class alphabet.
 */
struct Pattern
{
    /**
     * @brief The class alphabet.
     */
    Alphabet alphabet{};

    /**
     * @brief The alternatives, in the order of the published pattern.
     */
    std::span<const Alternative> alternatives{};
};

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_PATTERN_HPP
