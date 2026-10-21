#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_PIECE_CACHE_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_PIECE_CACHE_HPP

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
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
 * that are not looked up whole, so 16,384 entries hold every piece the merge makes on them.
 */
namespace matisse::encode
{
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
 */
class Piece_cache
{
public:
    /**
     * @brief Makes an empty cache.
     * @param capacity The number of pieces it holds before it stops filling.
     */
    explicit Piece_cache(std::size_t capacity);

    /**
     * @brief Returns the ids stored for a piece.
     * @param piece The piece's bytes, compared whole.
     * @return The ids, valid until the cache is destroyed; std::nullopt when the piece is not stored.
     */
    [[nodiscard]] std::optional<std::span<const load::Token_id_t>> find(std::string_view piece) const;

    /**
     * @brief Stores the ids of a piece not stored yet, unless the cache holds its bound or the piece is longer than
     *        longest_cached_piece.
     * @param piece The piece's bytes.
     * @param ids The piece's ids, as the merge gives them.
     */
    void store(std::string_view piece, std::vector<load::Token_id_t> ids);

private:
    /**
     * @brief A hash of pieces that a string_view looks up without a copy.
     */
    struct Piece_hash
    {
        /**
         * @brief Marks the hash as taking every type that converts to a string_view.
         */
        using is_transparent = void;

        /**
         * @brief Returns the hash of a piece.
         * @param piece The piece's bytes.
         * @return The standard hash of its bytes.
         */
        [[nodiscard]] std::size_t operator()(std::string_view piece) const noexcept;
    };

    /**
     * @brief The ids of every stored piece, by its bytes.
     */
    using Entries_t = std::unordered_map<std::string, std::vector<load::Token_id_t>, Piece_hash, std::equal_to<>>;

    /**
     * @brief The number of pieces held before the cache stops filling.
     */
    std::size_t capacity_;

    /**
     * @brief The stored pieces.
     */
    Entries_t entries_;
};

} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_PIECE_CACHE_HPP
