#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP

#include <cstddef>
#include <string_view>
#include <vector>

#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The class alphabets of the GPT-2 and cl100k patterns: classes_of(), class_count(), class_symbols() and
 *        unicode_version().
 *
 * Two code points in the same class agree on membership in every set a pattern names, the general categories L and N
 * and the White_Space property, and on every literal it spells, so the pattern segments a text exactly as it segments
 * the text with every code point replaced by its class's symbol. The symbol is a member of the class, an ASCII byte,
 * which makes a class string a text of the pattern again. The sets are read from Unicode tables generated into the tree
 * by tools/unicode/generate_classes.py, pinned to the version unicode_version() gives.
 */
namespace matisse::pretok
{
/**
 * @brief A class of a pattern's alphabet, numbered from zero in the order of the class symbols.
 */
using Class_t = unsigned char;

/**
 * @brief Returns the class of every code point of a text, the text decoded as UTF-8.
 *
 * GPT-2's pattern has 14 classes: the space, the apostrophe and the letters of its contractions, `d`, `e`, `l`, `m`,
 * `r`, `s`, `t` and `v`, each its own, then the rest of the white space, of the letters and of the numbers, and
 * everything else. cl100k's has 16, the carriage return and the line feed its own besides, and its contractions are
 * caseless, so each contraction letter's class holds both cases and that of `s` also U+017F, which matches `s`
 * caselessly.
 * @param pattern The pattern, GPT-2's or cl100k's.
 * @param text The text, well-formed UTF-8.
 * @return The classes, one per code point.
 * @throws std::invalid_argument If the pattern is Pattern::none, or the text is not well-formed UTF-8, the message
 *         naming the byte offset of the first ill-formed sequence.
 */
[[nodiscard]] std::vector<Class_t> classes_of(load::Pattern pattern, std::string_view text);

/**
 * @brief Returns the number of classes of a pattern's alphabet.
 * @param pattern The pattern, GPT-2's or cl100k's.
 * @return 14 for GPT-2's, 16 for cl100k's.
 * @throws std::invalid_argument If the pattern is Pattern::none.
 */
[[nodiscard]] std::size_t class_count(load::Pattern pattern);

/**
 * @brief Returns the symbol of every class of a pattern's alphabet, by class, each an ASCII byte of the class.
 *
 * The symbols ascend, so a class is its symbol's place among them; a class string written in them is a text the pattern
 * segments as it segments every text of those classes.
 * @param pattern The pattern, GPT-2's or cl100k's.
 * @return The symbols, `\t !'0Adelmrstv` for GPT-2's pattern and `\t\n\r !'0Adelmrstv` for cl100k's.
 * @throws std::invalid_argument If the pattern is Pattern::none.
 */
[[nodiscard]] std::string_view class_symbols(load::Pattern pattern);

/**
 * @brief Returns the Unicode version the class tables were generated from, "15.0.0", the version of the Python
 *        reference the fixtures are written by.
 * @return The version string.
 */
[[nodiscard]] std::string_view unicode_version() noexcept;

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP
