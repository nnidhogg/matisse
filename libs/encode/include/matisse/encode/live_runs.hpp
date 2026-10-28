#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_LIVE_RUNS_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_LIVE_RUNS_HPP

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "matisse/encode/verifier_table.hpp"
#include "matisse/pretok/classes.hpp"

/**
 * @file
 * @brief The live runs of a pattern's split verifier M over the code points of a stretch read so far, Live_runs: every
 *        marking of them that the segmentation of some continuation restricts to, and the boundaries every one of them
 *        keeps.
 *
 * A run is a path of M from the start over a marking of the class string read, and since every state of M is reachable
 * and reaches acceptance, the runs are exactly the restrictions of the segmentations of every continuation; two runs
 * end in two states, so there are at most as many as M has states. Their markings are kept as a tree, one layer per
 * code point read since the last stable one, each node a state with the mark of the code point that led to it, the
 * number of its live children and its parent in the layer before, with the byte its code point ends at and a link to
 * the last node before it on its run after which a piece ends, so that a run's piece ends are read back from the last
 * one without visiting the code points between them; a node whose children all die dies, and the layer before loses it.
 * The layers and nodes are held in two vectors, dropped from the front and erased in bulk, since a deque that is pushed
 * at the back and popped at the front allocates a block and frees one every few code points. The front layer is stable
 * when only one node of it is live, every run passing through it, and the last code point read is stable when every run
 * says the same of a piece ending after it: a run marked there says it ends, one unmarked in a state that does not
 * accept says it does not, one unmarked in an accepting state says both unless the state is a dead end, where it ends.
 * That is the stable marking, the longest prefix of markings every continuation's segmentation shares, as the design of
 * the streaming encoder proves. Live_runs is the streaming encoder's own and is not part of matisse's stable surface.
 * The header is installed only because every header of the library lives under include, as munch keeps none under src,
 * and it may change in minor releases.
 */
namespace matisse::encode
{
/**
 * @brief The live runs of a split verifier over the code points of a stretch, as a tree of their markings since the
 *        last stable code point.
 */
class Live_runs
{
public:
    /**
     * @brief The marking a live run gives the code points read since the last stable one.
     */
    struct Marking
    {
        /**
         * @brief The byte after each code point the run marks, where a piece ends, ascending: the last one at or before
         *        the byte the markings were asked from, where one stands since the last stable code point, and every
         *        one after it.
         */
        std::vector<std::size_t> ends{};

        /**
         * @brief Whether the run stands in a dead end, so that its last piece ends with the stretch.
         */
        bool dead_end{};
    };

    /**
     * @brief Starts the one run of the empty stretch, at M's start.
     * @param table M's table, which must outlive the runs.
     */
    explicit Live_runs(const Verifier_table& table);

    /**
     * @brief Reads a code point: every live run steps on its class with a piece ending after it and without, the steps
     *        M has kept, and a run that has neither dies.
     * @param symbol_class The class of the code point.
     * @param end The byte after the code point, counted from the stretch's start.
     */
    void step(pretok::Class_t symbol_class, std::size_t end);

    /**
     * @brief Drops the stable code points from the front and returns the end of the first piece that one of them ends,
     *        the end of the last one read when every run says a piece ends there.
     *
     * Each piece end is returned once, so that a caller closes the pieces by calling it until it returns none.
     * @return The byte after the piece, std::nullopt when no stable code point ends one.
     */
    [[nodiscard]] std::optional<std::size_t> next_closed();

    /**
     * @brief Returns the marking of every live run since the last stable code point, from its last piece end at or
     *        before a byte.
     *
     * A run's piece ends are read back from its last one through the link each node keeps to the last marked node
     * before it, so the work is the runs times their piece ends after the byte, however many code points the runs have
     * read since the last stable one.
     * @param from The byte, counted from the stretch's start.
     * @return The markings, one per run.
     */
    [[nodiscard]] std::vector<Marking> markings(std::size_t from) const;

    /**
     * @brief Returns the piece ends of the run that accepts, the stretch's segmentation as a complete input since the
     *        last stable code point, its last piece ending with the stretch.
     * @return The byte after each piece the marking ends, ascending.
     */
    [[nodiscard]] std::vector<std::size_t> accepted() const;

    /**
     * @brief Starts again, as for the empty stretch.
     */
    void reset();

private:
    /**
     * @brief The parent of a node of the front layer: the stable state every run passes through.
     */
    static constexpr std::size_t root{std::numeric_limits<std::size_t>::max()};

    /**
     * @brief One step of a run: the state it reached and how.
     *
     * The two flags stand beside the state, which packs a node into 40 bytes; in 56, a stream in pushes of 4 KiB runs
     * 13 per cent slower on cl100k.
     */
    struct Node
    {
        /**
         * @brief The state the step reached.
         */
        Verifier_table::State_t state{};

        /**
         * @brief Whether a piece ends after the code point the step read.
         */
        bool boundary_after{};

        /**
         * @brief Whether a run passes through the node.
         */
        bool live{};

        /**
         * @brief The node of the layer before, by its number among every node made since the reset, root in the front
         *        layer.
         */
        std::size_t parent{};

        /**
         * @brief The number of live nodes of the next layer whose parent this is.
         */
        std::size_t children{};

        /**
         * @brief The last node before this one on its run after whose code point a piece ends, by its number among
         *        every node made since the reset, root when none does since the run left the stable state; a node of a
         *        layer dropped since stands for none.
         */
        std::size_t marked_before{};

        /**
         * @brief The byte after the code point the step read, counted from the stretch's start.
         */
        std::size_t end{};
    };

    /**
     * @brief The nodes of one code point read: where they stand, how many are live, and where the code point ends.
     */
    struct Layer
    {
        /**
         * @brief The number of the layer's first node among every node made since the reset.
         */
        std::size_t first{};

        /**
         * @brief The number of nodes in the layer.
         */
        std::size_t count{};

        /**
         * @brief The number of them that are live.
         */
        std::size_t live{};

        /**
         * @brief The byte after the code point, counted from the stretch's start.
         */
        std::size_t end{};

        /**
         * @brief Whether next_closed() has returned the end already, every run saying a piece ends there.
         */
        bool closed{};
    };

    /**
     * @brief Returns a node by its number among every node made since the reset.
     * @param number The number, of a node not yet dropped.
     * @return The node.
     */
    [[nodiscard]] Node& node(std::size_t number);

    /**
     * @brief Returns a node by its number among every node made since the reset.
     * @param number The number, of a node not yet dropped.
     * @return The node.
     */
    [[nodiscard]] const Node& node(std::size_t number) const;

    /**
     * @brief Kills a node with no live child, and each parent that is left with none.
     * @param number The node's number.
     * @param layer The index of its layer in the layers kept.
     */
    void kill(std::size_t number, std::size_t layer);

    /**
     * @brief Drops the front layer, its one live node becoming the stable state, and erases the dropped layers and
     *        their nodes once they are half of those held, so that the layers and nodes held stay within twice those
     *        kept and each is moved a constant number of times on average.
     */
    void drop_front();

    /**
     * @brief Returns the marking of the run through a node of the last layer from its last piece end at or before a
     *        byte, read back through the marked nodes alone.
     * @param number The node's number, a node of the last layer.
     * @param from The byte, counted from the stretch's start.
     * @return The marking.
     */
    [[nodiscard]] Marking marking_of(std::size_t number, std::size_t from) const;

    /**
     * @brief M's table.
     */
    const Verifier_table& table_;

    /**
     * @brief The nodes made since the reset from the first one not yet erased, front to back, those before the front
     *        layer's dropped.
     */
    std::vector<Node> nodes_;

    /**
     * @brief The layers from the first one not yet erased, one per code point, those before the front one dropped.
     */
    std::vector<Layer> layers_;

    /**
     * @brief The index of the front layer, the first kept, among the layers.
     */
    std::size_t front_;

    /**
     * @brief The number of nodes erased from the front of the nodes, so that a node's index is its number less this.
     */
    std::size_t erased_;

    /**
     * @brief The stable state, where every run stands at the last stable code point.
     */
    Verifier_table::State_t stable_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_LIVE_RUNS_HPP
