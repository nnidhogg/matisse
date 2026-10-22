#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_TOKEN_TRIE_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_TOKEN_TRIE_HPP

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "matisse/load/definition.hpp"

/**
 * @file
 * @brief The tokens that begin a text, Token_trie: a trie over the spellings of a set of tokens, held in a double
 *        array, that gives the longest of them beginning a text and, for each, the longest of them that is a proper
 *        prefix of it.
 *
 * Every node of the trie is a slot, the root slot zero, and the child of a node by a byte is the slot at the node's
 * base plus the byte, which holds the node as its parent; a slot whose parent is another node, or no node, is no child
 * of it. A node's base is the least at which every child's slot is free when the nodes are placed in breadth-first
 * order, so a walk from the root reads one slot a byte, and the slots run 256 past the greatest base so that no child's
 * slot falls outside them. The tokens that are proper prefixes of a token, longest first, are the token's shorter
 * token, its shorter token's, and so on, each read from a table by id.
 */
namespace matisse::encode
{
/**
 * @brief The tokens that begin a text, longest first, over a trie of their spellings in a double array.
 */
class Token_trie
{
public:
    /**
     * @brief A token beginning a text: its id and its length.
     */
    struct Match
    {
        /**
         * @brief The token's id.
         */
        load::Token_id_t id{};

        /**
         * @brief The token's length in bytes, zero when no token begins the text.
         */
        std::uint32_t length{};
    };

    /**
     * @brief Places the spellings of a set of tokens in the trie.
     * @param spellings The bytes of every ordinary token, by id, each shorter than 2^32 bytes.
     * @param ids The ids of the tokens the trie holds, each spelled once among them and none empty.
     */
    Token_trie(std::span<const std::string> spellings, std::span<const load::Token_id_t> ids);

    /**
     * @brief Returns the longest token the trie holds that begins a text.
     * @param text The text.
     * @return The token, of length zero when none begins the text.
     */
    [[nodiscard]] Match longest(std::string_view text) const noexcept;

    /**
     * @brief Returns the longest token the trie holds that is a proper prefix of a token it holds.
     * @param id The token's id.
     * @return The shorter token, std::nullopt when none is a prefix of it.
     */
    [[nodiscard]] std::optional<Match> shorter(load::Token_id_t id) const noexcept;

    /**
     * @brief Returns the length of a token the trie holds.
     * @param id The token's id.
     * @return Its length in bytes.
     */
    [[nodiscard]] std::uint32_t length(load::Token_id_t id) const noexcept;

private:
    /**
     * @brief One slot of the double array: a node's base, its parent and the token it spells, or a free slot.
     */
    struct Slot
    {
        /**
         * @brief The slot of the node's child by byte zero; the child by a byte is that slot plus the byte.
         */
        std::uint32_t base{};

        /**
         * @brief The slot of the node's parent, no_slot for the root and for a free slot.
         */
        std::uint32_t parent{};

        /**
         * @brief The id of the token the node spells, no_token when it spells none.
         */
        load::Token_id_t id{};
    };

    /**
     * @brief What the trie keeps of a token by its id: its length and its shorter token.
     */
    struct Entry
    {
        /**
         * @brief The token's length in bytes.
         */
        std::uint32_t length{};

        /**
         * @brief The id of the longest token the trie holds that is a proper prefix of it, no_token when none is.
         */
        load::Token_id_t shorter{};
    };

    /**
     * @brief The parent of the root and of a free slot, which no node's slot is.
     */
    static constexpr std::uint32_t no_slot{std::numeric_limits<std::uint32_t>::max()};

    /**
     * @brief The id of no token, which no token's id is, a definition holding fewer than 2^32 tokens.
     */
    static constexpr load::Token_id_t no_token{std::numeric_limits<load::Token_id_t>::max()};

    /**
     * @brief Returns the slots of the double array of a set of tokens' spellings.
     * @param spellings The bytes of every ordinary token, by id.
     * @param ids The ids of the tokens the trie holds.
     * @return The slots, the root first.
     */
    [[nodiscard]] static std::vector<Slot> slots_of(
            std::span<const std::string> spellings, std::span<const load::Token_id_t> ids);

    /**
     * @brief Returns the length and the shorter token of every token the trie holds, by id.
     * @param spellings The bytes of every ordinary token, by id.
     * @param ids The ids of the tokens the trie holds.
     * @return The entries, one per spelling, those of the tokens the trie does not hold unread.
     */
    [[nodiscard]] std::vector<Entry> entries_of(
            std::span<const std::string> spellings, std::span<const load::Token_id_t> ids) const;

    /**
     * @brief The slots of the double array, the root first.
     */
    std::vector<Slot> slots_;

    /**
     * @brief The length and the shorter token of every token the trie holds, by id.
     */
    std::vector<Entry> entries_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_TOKEN_TRIE_HPP
