#ifndef MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP
#define MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The classes of a text under a pattern's class alphabet: the table of every code point's class, Class_table,
 *        and classes_of().
 *
 * Two code points in the same class agree on membership in every set a pattern names and on every literal it spells, so
 * the pattern segments a text exactly as it segments the text with every code point replaced by another member of its
 * class. Each class is written as one ASCII symbol, so a class string is a text over bytes. In GPT-2's and cl100k's
 * alphabets every symbol is also a member of its class, which makes a class string a text of the pattern again; that is
 * a property of those alphabets, not of the type, and a class without an ASCII member has a symbol outside it. The
 * Unicode sets are read from the tables of the tokenizer library that describes the pattern, generated into its tree by
 * tools/unicode/generate_classes.py and pinned to the version its alphabet names.
 */
namespace matisse::pretok
{
/**
 * @brief A class of an alphabet, numbered from zero in the order of the class symbols.
 */
using Class_t = unsigned char;

/**
 * @brief The class symbol of every code point under an alphabet: a table of the ASCII bytes, read in one load, and for
 *        every other code point the index of its block of 256 among the distinct blocks, each kept once, and its cell
 *        in that block, read in two.
 *
 * A code point's symbol is the symbol of the first set of the alphabet holding it, or the alphabet's other symbol.
 * GPT-2's and cl100k's alphabets have 127 and 128 distinct blocks, so an index is a byte and the table is 37 KB; it is
 * built in about a millisecond, so a caller that classes many texts holds one.
 */
class Class_table
{
public:
    /**
     * @brief Builds the table of an alphabet.
     * @param alphabet The alphabet.
     * @throws std::invalid_argument If the alphabet has more distinct blocks than a byte indexes; the message names the
     *         count.
     */
    explicit Class_table(const Alphabet& alphabet);

    /**
     * @brief Returns the class string of a text: the symbol of every code point's class, the text decoded as UTF-8.
     * @param text The text, well-formed UTF-8.
     * @return The class string, one symbol per code point.
     * @throws std::invalid_argument If the text is not well-formed UTF-8, the message naming the byte offset of the
     *         first ill-formed sequence.
     */
    [[nodiscard]] std::string symbols(std::string_view text) const;

    /**
     * @brief Returns the class of every code point of a text, the text decoded as UTF-8.
     * @param text The text, well-formed UTF-8.
     * @return The classes, one per code point.
     * @throws std::invalid_argument If the text is not well-formed UTF-8, as symbols() refuses it.
     */
    [[nodiscard]] std::vector<Class_t> classes(std::string_view text) const;

private:
    /**
     * @brief The code points of a block, the low byte of a code point naming its place in its block.
     */
    static constexpr std::size_t block_points{256};

    /**
     * @brief Returns the symbol of a code point's class.
     * @param point The code point, a scalar value.
     * @return The symbol.
     */
    [[nodiscard]] char symbol(char32_t point) const noexcept;

    /**
     * @brief The symbol of every ASCII byte.
     */
    std::array<char, 128> ascii_;

    /**
     * @brief The index among the distinct blocks of every block of code points, in the order of the code points.
     */
    std::vector<std::uint8_t> block_of_;

    /**
     * @brief The symbols of the distinct blocks, each block once, block_points symbols each.
     */
    std::vector<char> blocks_;

    /**
     * @brief The class of every class symbol, by the symbol's byte.
     */
    std::array<Class_t, 256> class_of_symbol_{};
};

/**
 * @brief Returns the class of every code point of a text, the text decoded as UTF-8, by the alphabet's Class_table
 *        built for the one text.
 * @param alphabet The class alphabet.
 * @param text The text, well-formed UTF-8.
 * @return The classes, one per code point.
 * @throws std::invalid_argument If the text is not well-formed UTF-8, the message naming the byte offset of the first
 *         ill-formed sequence.
 */
[[nodiscard]] std::vector<Class_t> classes_of(const Alphabet& alphabet, std::string_view text);

} // namespace matisse::pretok

#endif // MATISSE_LIBS_PRETOK_INCLUDE_MATISSE_PRETOK_CLASSES_HPP
