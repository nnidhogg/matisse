#ifndef MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP
#define MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <munch/verifier/verifier_decisions.hpp>

#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The validator's report, Report, decided by validate(): which class windows of a definition's pattern certify a
 *        cut, at which origin and with which right context; and the report rendered as text, render(), and as JSON,
 *        json(), with text as a JSON string, json_string(), and the ending a noun takes for a count, plural().
 *
 * validate() decides every class window of width two to a width, five unless told, under both certificates, the policy
 * certificate of munch::verifier::miscovering() on the split verifier, decided for every window in one walk, and the
 * factorization certificate by is_split_window() on the factorization lexer; each window table carries the right
 * context its inventory needs, decided by munch::verifier::right_context() on the split verifier, and the report
 * carries boundary_gap() of the split verifier. A window is a string of class symbols and its origin counts code
 * points, as the pattern's alphabet classes a text. render() and json() follow munch-audit's report renderings. Two
 * units implement it, as munch-audit's report is implemented: report.cpp decides and report_render.cpp writes what was
 * decided. supply.hpp measures a report's anchors on an input.
 */
namespace matisse::validate
{
/**
 * @brief Which certificate a window table holds, and so which cuts an encoder takes.
 */
enum class Inventory : std::uint8_t
{
    /**
     * @brief The factorization certificate: the cut holds for every complete factorization into the gate-erased
     *        languages of the alternatives, whatever the engine does inside its matches.
     */
    factorization,

    /**
     * @brief The policy certificate: the cut holds for the pattern's own segmentation, decided on its verifier, which
     *        rests on the one-step relation and lands on more boundaries.
     */
    policy
};

/**
 * @brief The widest window validate() decides unless told otherwise, five class symbols.
 *
 * At width five GPT-2's policy anchors are complete, every cut any wider window certifies being one of them, as the
 * report on pre-tokenizer certification proves, and the walk over the windows decides width five in seconds.
 */
inline constexpr std::size_t default_width{5};

/**
 * @brief A certified window: its class symbols and the origin, the symbol of the window at which the piece holding an
 *        occurrence's last symbol begins.
 */
struct Certified_window
{
    /**
     * @brief The window's class symbols.
     */
    std::string window{};

    /**
     * @brief The certified origin, a symbol index of the window.
     */
    std::size_t origin{};
};

/**
 * @brief The windows one certificate certifies up to a width, and the right context their cuts need.
 */
struct Window_table
{
    /**
     * @brief The certificate the windows are certified under.
     */
    Inventory inventory{};

    /**
     * @brief The widest window decided.
     */
    std::size_t width{};

    /**
     * @brief The certified windows of width two to the width, widths ascending, each width's windows in the order the
     *        class symbols give.
     */
    std::vector<Certified_window> windows{};

    /**
     * @brief The right context of the inventory, the largest of its windows': one more than the largest that fails at
     *        some cut, zero when none does.
     */
    std::size_t right_context{};

    /**
     * @brief A truncation at every right context that fails at some cut of the inventory, ascending by context, the
     *        shortest one at its context and the first in the table's order among the shortest.
     */
    std::vector<munch::verifier::Truncation> failing{};
};

/**
 * @brief What validate() decides of a pattern: its window table under each certificate and the boundary gap of its
 *        split verifier.
 */
struct Report
{
    /**
     * @brief The windows the factorization certificate certifies.
     */
    Window_table factorization{};

    /**
     * @brief The windows the policy certificate certifies.
     */
    Window_table policy{};

    /**
     * @brief The supremum of the distance between consecutive boundaries of the split verifier, or the lasso that
     *        makes it unbounded.
     */
    munch::verifier::Boundary_gap gap{};
};

/**
 * @brief Decides every class window of a pattern of width two to a width under both certificates, with the right
 *        context of each inventory and the boundary gap of the split verifier.
 *
 * The factorization table holds the windows is_split_window() certifies on the factorization lexer, at the origin it
 * returns, and the policy table the windows munch::verifier::miscovering() certifies on the split verifier, decided in
 * one walk over the windows: each window carries its cloud, the states the split verifier can be in after an occurrence
 * of it, each with where the piece holding the occurrence's last symbol begins and whether a piece ends after it,
 * extended by one symbol from its prefix's, and the window is certified at an origin when every entry places the piece
 * there. Every window of a total pattern occurs, so none is certified vacuously. An inventory's right context is the
 * largest of its windows' by munch::verifier::right_context() on the split verifier, asked of the windows that hold no
 * shorter certified window whose origin lands on the same symbol, since every cut of one that does is a cut of the
 * shorter one at the same place, with its failing contexts and truncations. GPT-2's and cl100k's patterns need one
 * under the factorization certificate and two under the policy certificate.
 * @param pattern The pattern.
 * @param width The widest window decided, default_width unless told.
 * @return The report.
 * @throws std::invalid_argument If the pattern is not total, as split_verifier() refuses it, the right context of a
 *         certified window is unbounded, or a factorization window is not certified by the split verifier.
 */
[[nodiscard]] Report validate(const pretok::Pattern& pattern, std::size_t width = default_width);

/**
 * @brief Renders a report as text, the verdict first, then each inventory's windows and right context, then the
 *        boundary gap.
 *
 * A window and a truncation's text are class strings, shown quoted with the common controls escaped; the windows are
 * counted by width and the first six shown as examples, printable ones first, and each right context that fails is
 * shown by its shortest truncation, the text with its cut and the boundaries the split verifier accepts it with.
 * @param report The report.
 * @return The text, a newline after every row.
 */
[[nodiscard]] std::string render(const Report& report);

/**
 * @brief Renders a report as JSON, one object with a member per figure.
 *
 * Class strings are JSON strings holding each byte as the code point of its value, so a reader recovers them exactly; a
 * marked string, a truncation's segmentation or a part of the gap's lasso, is an object of its text and the offsets its
 * boundaries stand at; the boundary gap is a number, or the string "unbounded" with its lasso beside it.
 * @param report The report.
 * @return The JSON text, no trailing newline.
 */
[[nodiscard]] std::string json(const Report& report);

/**
 * @brief Returns text as a JSON string, the quote, the backslash and the controls escaped; a byte beyond ASCII stands
 *        as it is where it is part of a well-formed UTF-8 sequence, the text being UTF-8, and is escaped as the code
 *        point of its value where it is not, so that the document is JSON whatever bytes a path holds.
 * @param text The text.
 * @return The JSON text, quotes included.
 */
[[nodiscard]] std::string json_string(std::string_view text);

/**
 * @brief Returns the ending a noun takes for a count.
 * @param count The count.
 * @return Nothing for one, `s` for any other.
 */
[[nodiscard]] constexpr std::string_view plural(const std::size_t count) noexcept
{
    return count == 1 ? "" : "s";
}

} // namespace matisse::validate

#endif // MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP
