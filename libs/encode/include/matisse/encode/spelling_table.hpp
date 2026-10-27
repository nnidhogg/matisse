#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_SPELLING_TABLE_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_SPELLING_TABLE_HPP

#include <cstddef>
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
 * @brief The id of every spelling of a definition, Spelling_table: a flat table with open addressing that the
 *        whole-piece lookup and the construction of the merge's tables look spellings up in.
 *
 * The table holds the bytes of every spelling once, one after another in a string, and a slot per spelling of its hash
 * by std::hash<std::string_view>, where its bytes stand, their length and its id. Every spelling is placed at the first
 * free slot of its walk from its home, the hash modulo the number of slots, the next slot after each; the slots that
 * placement passed over were full then and no slot is ever emptied, so a lookup meets every spelling the table holds
 * before a free slot and stops at a free slot for any other, and it compares bytes only in a slot of the same hash and
 * length. A spelling met twice keeps its first id. The slots are the least power of two at least twice the spellings,
 * 262,144 on cl100k's file. Spelling_table is the merge's own and is not part of matisse's stable surface. The header
 * is installed only because every header of the library lives under include, as munch keeps none under src, and it may
 * change in minor releases.
 */
namespace matisse::encode
{
/**
 * @brief The id of every spelling of a definition, in a flat table with open addressing.
 */
class Spelling_table
{
public:
    /**
     * @brief Places every spelling in the table with its id, its index.
     * @param spellings The bytes of every ordinary token, by id, fewer than 2^32 of them.
     */
    explicit Spelling_table(std::span<const std::string> spellings);

    /**
     * @brief Returns the id of a spelling.
     * @param spelling The bytes, compared whole.
     * @return The id of the first token that spells them, std::nullopt when none does.
     */
    [[nodiscard]] std::optional<load::Token_id_t> find(std::string_view spelling) const noexcept;

private:
    /**
     * @brief One slot of the table: where a spelling's bytes stand and its id, or the free id.
     */
    struct Slot
    {
        /**
         * @brief The hash of the spelling's bytes.
         */
        std::size_t hash{};

        /**
         * @brief The offset of the spelling's bytes in the stored bytes.
         */
        std::size_t bytes_at{};

        /**
         * @brief The spelling's length in bytes.
         */
        std::uint32_t length{};

        /**
         * @brief The spelling's id, free_id when the slot holds no spelling.
         */
        load::Token_id_t id{};
    };

    /**
     * @brief The id of a free slot, all ones, which no spelling's id is.
     */
    static constexpr load::Token_id_t free_id{std::numeric_limits<load::Token_id_t>::max()};

    /**
     * @brief Returns the number of slots a table of spellings takes: the least power of two at least twice their
     *        number.
     * @param spellings The number of spellings.
     * @return The number of slots, at least one, and more than the spellings when there are any.
     */
    [[nodiscard]] static std::size_t slot_count_of(std::size_t spellings) noexcept;

    /**
     * @brief Returns the slot a spelling's walk ends at: the slot holding it, or the first free slot from its home.
     * @param spelling The bytes.
     * @param hash The hash of the bytes.
     * @return The slot.
     */
    [[nodiscard]] std::size_t slot_of(std::string_view spelling, std::size_t hash) const noexcept;

    /**
     * @brief The number of slots less one, which a hash or a slot is masked by to wrap.
     */
    std::size_t mask_;

    /**
     * @brief The slots.
     */
    std::vector<Slot> slots_;

    /**
     * @brief The bytes of every spelling, one after another.
     */
    std::string bytes_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_SPELLING_TABLE_HPP
