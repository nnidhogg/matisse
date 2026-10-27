#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_PIECE_CACHE_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_PIECE_CACHE_HPP

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
 * @brief The bounded cache of merged pieces, Piece_cache, with its bound, piece_cache_capacity, and the longest piece
 *        it holds, longest_cached_piece.
 *
 * The ids of a piece are a function of the piece's bytes and the definition alone, the merge building its parts afresh
 * for every piece, so a cache keyed by the whole of a piece's bytes and filled under one definition returns the ids the
 * merge would, and never changes an id. The cache fills until it holds its bound and then stops, as Hugging Face's BPE
 * model's cache does, with no eviction; one cache serves one caller, a call of the encoder or a thread's calls, which
 * is where repetition lives. On twitter.json and munch's Python and C++ sources at 15b2863c, GPT-2's pattern makes
 * 111,510 to 1,118,131 pieces of which 3,684 to 14,032 are distinct, and cl100k's leaves 1,527 to 9,218 distinct pieces
 * that are not looked up whole, so 16,384 entries hold every piece the merge makes on them. The cache is a flat table
 * with open addressing, linear probing from a piece's hash modulo a power of two of slots, each slot holding where a
 * piece's bytes and ids stand in a string and a vector of their own; no slot is ever emptied and growth places every
 * piece again, so a lookup meets the piece before a free slot and stops at a free slot for any other piece; the slots
 * start at 64 on the first store and double whenever the pieces fill half of them, so a cache that stores nothing takes
 * no memory. A cache is made for the encoder it serves, which it borrows, and the encoder refuses a cache made for
 * another, whose ids would be another definition's.
 */
namespace matisse::encode
{
class Encoder;

/**
 * @brief The number of pieces an encoder's cache holds before it stops filling.
 */
inline constexpr std::size_t piece_cache_capacity{16384};

/**
 * @brief The longest piece a cache holds, in bytes, as Hugging Face's BPE model leaves a piece of 256 or more out of
 *        its cache.
 */
inline constexpr std::size_t longest_cached_piece{255};

/**
 * @brief A bounded map from the bytes of a merged piece to its ids, filled until it holds its bound.
 *
 * A cache has no lock and is used by one thread at a time, each thread of a pool holding its own, and any number of
 * caches may serve one encoder at once.
 */
class Piece_cache
{
public:
    /**
     * @brief Makes an empty cache for an encoder, which takes no memory until it stores a piece; the cache borrows the
     *        encoder, which must outlive it and must not be moved while the cache is in use.
     * @param encoder The encoder the cache serves.
     * @param capacity The number of pieces it holds before it stops filling.
     */
    Piece_cache(const Encoder& encoder, std::size_t capacity);

    /**
     * @brief A cache is not made for a temporary encoder, which would not outlive it.
     */
    Piece_cache(const Encoder&& encoder, std::size_t capacity) = delete;

    /**
     * @brief Returns whether the cache was made for an encoder.
     *
     * It serves the encoder and is not part of matisse's stable surface.
     * @param encoder The encoder.
     * @return True when it is the encoder the cache serves.
     */
    [[nodiscard]] bool serves(const Encoder& encoder) const noexcept;

    /**
     * @brief Returns the ids stored for a piece.
     *
     * It serves the encoder and the streaming encoder and is not part of matisse's stable surface.
     * @param piece The piece's bytes, compared whole.
     * @return The ids, valid until the next call of store(); std::nullopt when the piece is not stored.
     */
    [[nodiscard]] std::optional<std::span<const load::Token_id_t>> find(std::string_view piece) const;

    /**
     * @brief Stores the ids of a piece not stored yet, unless the cache holds its bound or the piece is longer than
     *        longest_cached_piece.
     *
     * It serves the encoder and the streaming encoder and is not part of matisse's stable surface.
     * @param piece The piece's bytes.
     * @param ids The piece's ids, as the merge gives them.
     */
    void store(std::string_view piece, std::vector<load::Token_id_t> ids);

private:
    /**
     * @brief One slot of the table: where a stored piece's bytes and ids stand, or nothing.
     */
    struct Slot
    {
        /**
         * @brief The hash of the piece's bytes.
         */
        std::size_t hash{};

        /**
         * @brief The offset of the piece's bytes in the stored bytes, free_slot when the slot holds no piece.
         */
        std::size_t bytes_at{};

        /**
         * @brief The offset of the piece's ids in the stored ids.
         */
        std::size_t ids_at{};

        /**
         * @brief The piece's length in bytes.
         */
        std::uint16_t length{};

        /**
         * @brief The number of its ids.
         */
        std::uint16_t count{};
    };

    /**
     * @brief The bytes_at of a free slot.
     */
    static constexpr std::size_t free_slot{std::numeric_limits<std::size_t>::max()};

    /**
     * @brief The number of slots of the table the first stored piece makes, doubled whenever the pieces fill half of
     *        them.
     */
    static constexpr std::size_t first_slot_count{64};

    /**
     * @brief Returns the slot a piece's walk ends at: the slot holding the piece, or the first free slot from its home.
     * @param piece The piece's bytes.
     * @param hash The hash of its bytes.
     * @return The slot, of a table with at least one free slot.
     */
    [[nodiscard]] std::size_t slot_of(std::string_view piece, std::size_t hash) const noexcept;

    /**
     * @brief Doubles the slots, placing every stored piece again at the first free slot from its home.
     */
    void grow();

    /**
     * @brief The encoder the cache serves.
     */
    const Encoder& encoder_;

    /**
     * @brief The number of pieces held before the cache stops filling.
     */
    std::size_t capacity_;

    /**
     * @brief The number of pieces stored.
     */
    std::size_t size_{0};

    /**
     * @brief The slots, a power of two in number and at least twice the pieces stored, none before the first piece is
     *        stored.
     */
    std::vector<Slot> slots_;

    /**
     * @brief The bytes of every stored piece, one after another.
     */
    std::string bytes_;

    /**
     * @brief The ids of every stored piece, one after another.
     */
    std::vector<load::Token_id_t> ids_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_PIECE_CACHE_HPP
