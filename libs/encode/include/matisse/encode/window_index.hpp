#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_WINDOW_INDEX_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_WINDOW_INDEX_HPP

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "matisse/pretok/classes.hpp"
#include "matisse/validate/report.hpp"

/**
 * @file
 * @brief The certified windows of one window table by their class strings, Window_index, and the first anchor of a
 *        stretch at or after a byte, first_anchor().
 *
 * An anchor is a position t + o of a stretch's class string for an occurrence at t of a window the table certifies at
 * origin o, and the cut there is a pre-token boundary of every text the stretch stands in, whatever comes before it or
 * after it. The index classes a stretch in blocks of block_bytes from the byte asked for, so that a walk ending at an
 * anchor near it reads the text near it and no more. Window_index is the encoder's own and is not part of matisse's
 * stable surface. The header is installed only because every header of the library lives under include, as munch keeps
 * none under src, and it may change in minor releases.
 */
namespace matisse::encode
{
/**
 * @brief The certified windows of a window table by their class strings, and where they place their anchors in a text.
 *
 * The class table is handed in with each question, as munch's Window_planner takes its token set, so an encoder's index
 * and its streams class by the one table the encoder holds.
 */
class Window_index
{
public:
    /**
     * @brief Indexes the windows of a table by their class strings.
     * @param table The window table.
     */
    explicit Window_index(const validate::Window_table& table);

    /**
     * @brief Returns the widest window the table decided, which the spacing of an encoder's cuts is counted in.
     * @return The width, in class symbols.
     */
    [[nodiscard]] std::size_t width() const noexcept;

    /**
     * @brief Returns the least anchor of a stretch at or after a byte: the byte offset of a code point t + o for an
     *        occurrence in the stretch at t of a window certified at origin o, the stretch's start excluded.
     *
     * The occurrence lies in the stretch, which is read as one text: no window is read across its ends. The first block
     * that holds an anchor holds the least one, since an anchor is where the piece holding its window's last symbol
     * begins in every segmentation, so a window that ends later, one across the block's edge among them, places no
     * anchor before it.
     * @param class_table The class table of the alphabet of the pattern the table was decided of.
     * @param stretch The stretch, well-formed UTF-8.
     * @param from The byte, at the start of a code point of the stretch or at its end.
     * @return The anchor, std::nullopt when the stretch has none at or after the byte.
     * @throws std::invalid_argument If the part of the stretch the walk classes is not well-formed UTF-8, as
     *         classes_of() refuses it.
     */
    [[nodiscard]] std::optional<std::size_t> first_anchor(
            const pretok::Class_table& class_table, std::string_view stretch, std::size_t from) const;

private:
    /**
     * @brief The bytes classed at a time, at least: the walk classes a block, and the next one only when the block
     *        holds no anchor.
     *
     * Anchors stand a few bytes apart on text, so the first block nearly always holds one: eight chunks of the corpora
     * of docs/design.md take 0.05 to 0.07 ms to plan with blocks of 256 bytes, and 0.6 to 0.9 ms with blocks of 4,096.
     */
    static constexpr std::size_t block_bytes{256};

    /**
     * @brief Returns the least anchor of one block of a stretch at or after a byte, the windows read within the block.
     * @param class_table The class table of the alphabet of the pattern the table was decided of.
     * @param stretch The stretch, well-formed UTF-8.
     * @param begin The block's first byte, at the start of a code point.
     * @param end The byte after the block, at the start of a code point or at the stretch's end.
     * @param floor The least byte an anchor may stand at, after the stretch's start.
     * @return The anchor, an offset into the stretch, std::nullopt when the block has none at or after the floor.
     * @throws std::invalid_argument If the block is not well-formed UTF-8, as classes_of() refuses it.
     */
    [[nodiscard]] std::optional<std::size_t> block_anchor(
            const pretok::Class_table& class_table, std::string_view stretch, std::size_t begin, std::size_t end,
            std::size_t floor) const;

    /**
     * @brief The widest window decided.
     */
    std::size_t width_;

    /**
     * @brief The widths of the certified windows, ascending, each once.
     */
    std::vector<std::size_t> widths_;

    /**
     * @brief The origin of every certified window, by its class string.
     */
    std::map<std::string, std::size_t, std::less<>> origins_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_WINDOW_INDEX_HPP
