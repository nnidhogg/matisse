#ifndef MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_SUPPLY_HPP
#define MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_SUPPLY_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "matisse/validate/report.hpp"

/**
 * @file
 * @brief The certified-anchor supply on an input, Supply, measured by supply() from what one inventory of certified
 *        windows finds there, Anchors, and the gaps between those, Gaps; and the supply of both inventories rendered as
 *        JSON, supply_json(), and as the report's own section, supply_section().
 *
 * Anchors and Gaps are munch-audit's, duplicated rather than linked, and Supply adds two figures beside them: the share
 * of the serial pre-token boundaries that are anchors, and the longest piece, which is the floor no cut rule can go
 * below.
 */
namespace matisse::validate
{
/**
 * @brief The gaps between consecutive anchors of one inventory on an input, in bytes: the distance from each anchor to
 *        the next, so the run before the first anchor and the run after the last are in none of them.
 *
 * A percentile is the order statistic at floor(q n) of the n gaps ascending, capped at the last, munch-audit's
 * convention, so the median of two gaps is the larger.
 */
struct Gaps
{
    /**
     * @brief The median gap.
     */
    std::size_t median{};

    /**
     * @brief The ninetieth percentile.
     */
    std::size_t ninetieth{};

    /**
     * @brief The ninety-ninth percentile.
     */
    std::size_t ninety_ninth{};

    /**
     * @brief The longest gap.
     */
    std::size_t longest{};
};

/**
 * @brief What one inventory of certified windows supplies on an input: the anchors it finds there and the gaps between
 *        them.
 */
struct Anchors
{
    /**
     * @brief How many positions of the input are anchors: the distinct interior positions an occurrence of a certified
     *        window places a cut at, its position plus the window's origin, the input's two ends left out since neither
     *        is a cut.
     */
    std::size_t count{};

    /**
     * @brief 1024 times the count over the input's length in bytes; zero on an empty input.
     */
    double per_kibibyte{};

    /**
     * @brief The gaps between consecutive anchors, absent when fewer than two anchors leave none.
     */
    std::optional<Gaps> gaps{};
};

/**
 * @brief The certified-anchor supply of one inventory on one input: how often a certified anchor occurs, how long the
 *        stretches between anchors are, and how close they come to the pre-token boundaries.
 *
 * Every anchor is a decision of the report's applied to the input's classes, each an occurrence of a window the
 * report's table certifies, and every position is counted once, no sampling. Every anchor is a pre-token boundary of
 * the serial segmentation, by the transfer of a cut from class strings to text, so the share is at most one.
 */
struct Supply
{
    /**
     * @brief The input's length in bytes.
     */
    std::size_t bytes{};

    /**
     * @brief The anchors, at byte offsets of the input, and the gaps between them.
     */
    Anchors anchors{};

    /**
     * @brief The anchor count over the count of the serial segmentation's interior pre-token boundaries, zero when the
     *        input has none.
     */
    double boundary_share{};

    /**
     * @brief The longest piece of the serial segmentation in bytes, zero on an empty input.
     */
    std::size_t longest_piece{};
};

/**
 * @brief Measures the certified-anchor supply of one inventory of a report on an input.
 *
 * The input is classed under the pattern's alphabet and every certified window of the inventory's table is looked for
 * at every code point, its origin placing an anchor; the anchors are counted at the byte offsets of their code points,
 * and the input is segmented by the pattern's Segmenter for its pre-token boundaries and its longest piece.
 * @param report The report validate() decided of the pattern, which it holds.
 * @param inventory The inventory whose windows are measured.
 * @param input The input, well-formed UTF-8.
 * @return The supply.
 * @throws std::invalid_argument If the input is not well-formed UTF-8, as classes_of() refuses it.
 */
[[nodiscard]] Supply supply(const Report& report, Inventory inventory, std::string_view input);

/**
 * @brief Renders the supply of both inventories on one input as the JSON object of its account: the input's name and
 *        length, the longest piece, and one object per inventory holding the anchor count, the anchors per kibibyte,
 *        the share of the pre-token boundaries and the four gap figures, each null when the anchors leave no gap.
 * @param factorization The factorization inventory's supply.
 * @param policy The policy inventory's supply, measured on the same input.
 * @param path The input's name, as given.
 * @return The JSON text, one line.
 */
[[nodiscard]] std::string supply_json(const Supply& factorization, const Supply& policy, std::string_view path);

/**
 * @brief Renders the supply of both inventories on one input as the report's own section, a heading naming the input, a
 *        row for the longest piece, and one row per inventory.
 * @param factorization The factorization inventory's supply.
 * @param policy The policy inventory's supply, measured on the same input.
 * @param path The input's name, as given.
 * @return The section, a blank line before its heading and a newline after every row.
 */
[[nodiscard]] std::string supply_section(const Supply& factorization, const Supply& policy, std::string_view path);

} // namespace matisse::validate

#endif // MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_SUPPLY_HPP
