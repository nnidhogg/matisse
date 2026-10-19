#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP

#include <string_view>
#include <vector>

#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The classes of a text under a pattern's class alphabet, classes_of().
 *
 * Two code points in the same class agree on membership in every set a pattern names and on every literal it spells, so
 * the pattern segments a text exactly as it segments the text with every code point replaced by its class's symbol. The
 * symbol is a member of the class, an ASCII byte, which makes a class string a text of the pattern again. The sets are
 * read from the Unicode tables of the tokenizer library that describes the pattern, generated into its tree by
 * tools/unicode/generate_classes.py and pinned to the version its alphabet names.
 */
namespace matisse::pretok
{
/**
 * @brief A class of an alphabet, numbered from zero in the order of the class symbols.
 */
using Class_t = unsigned char;

/**
 * @brief Returns the class of every code point of a text, the text decoded as UTF-8.
 * @param alphabet The class alphabet.
 * @param text The text, well-formed UTF-8.
 * @return The classes, one per code point.
 * @throws std::invalid_argument If the text is not well-formed UTF-8, the message naming the byte offset of the first
 *         ill-formed sequence.
 */
[[nodiscard]] std::vector<Class_t> classes_of(const Alphabet& alphabet, std::string_view text);

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP
