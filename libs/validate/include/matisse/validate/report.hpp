#ifndef MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP
#define MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <munch/verifier/verifier_decisions.hpp>

#include "matisse/pretok/pattern.hpp"

/**
 * @file
 * @brief The validator, validate(), supply(), validate_merges(), render() and json(): which class windows of a
 *        definition's pattern certify a cut, at which origin and with which right context, how many anchors they
 *        supply on an input, and which candidate windows of a merge table without a pre-tokenizer certify.
 *
 * validate() decides every class window of width two to four under both certificates, the policy certificate by
 * munch::verifier::miscovering() on the split verifier and the factorization certificate by is_split_window() on the
 * factorization lexer; each window table carries the right context its inventory needs, decided by
 * munch::verifier::right_context() on the split verifier, and the report carries boundary_gap() of the split verifier.
 * A window is a string of class symbols and its origin counts code points, as the pattern's alphabet classes a text.
 * validate_merges() builds munch::verifier::token_automaton() once and decides each candidate window by miscovering()
 * on it, the lazy certification. supply() measures one inventory's anchors on an input as munch-audit's supply does,
 * with the share of the serial pre-token boundaries that are anchors and the longest piece beside them, and render()
 * and json() follow munch-audit's report renderings. Of the functions, only validate() is declared yet; docs/design.md
 * sketches the others.
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
 * @brief The widest window validate() decides unless told otherwise, four class symbols.
 */
inline constexpr std::size_t default_width{4};

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
 * returns, and the policy table the windows munch::verifier::miscovering() certifies on the split verifier. A window
 * certified at an origin is certified at the origin plus one with a symbol prepended, every occurrence of the longer
 * window holding one of the shorter, so a policy window whose last symbols are certified takes its origin from them;
 * every other one is asked of miscovering() at its last symbol, and a refutation leaves at most one origin to ask,
 * where the piece holding the refuted occurrence's last symbol begins, none when that piece begins before the
 * occurrence. Every window of a total pattern occurs, so none is certified vacuously. An inventory's right context is
 * the largest of its windows' by munch::verifier::right_context() on the split verifier, asked of the windows that hold
 * no shorter certified window whose origin lands on the same symbol, since every cut of one that does is a cut of the
 * shorter one at the same place, with its failing contexts and truncations. GPT-2's and cl100k's patterns need one
 * under the factorization certificate and two under the policy certificate.
 * @param pattern The pattern, total.
 * @param width The widest window decided, default_width unless told.
 * @return The report.
 * @throws std::invalid_argument If the right context of a certified window is unbounded, or a factorization window is
 *         not certified by the split verifier.
 * @throws munch::pattern::Syntax_error If an alternative's language is not in munch's pattern syntax.
 */
[[nodiscard]] Report validate(const pretok::Pattern& pattern, std::size_t width = default_width);

} // namespace matisse::validate

#endif // MATISSE_LIBS_VALIDATE_INCLUDE_MATISSE_VALIDATE_REPORT_HPP
