#include "matisse/encode/piece_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace matisse::encode
{
Piece_cache::Piece_cache(const Encoder& encoder, const std::size_t capacity) : encoder_{encoder}, capacity_{capacity}
{}

bool Piece_cache::serves(const Encoder& encoder) const noexcept
{
    return &encoder_ == &encoder;
}

std::optional<std::span<const load::Token_id_t>> Piece_cache::find(const std::string_view piece) const
{
    if (slots_.empty())
    {
        return std::nullopt;
    }

    const auto hash{std::hash<std::string_view>{}(piece)};

    const auto& [slot_hash, bytes_at, ids_at, length, count]{slots_[slot_of(piece, hash)]};

    if (bytes_at == free_slot)
    {
        return std::nullopt;
    }

    return std::span{ids_}.subspan(ids_at, count);
}

std::size_t Piece_cache::slot_of(const std::string_view piece, const std::size_t hash) const noexcept
{
    const auto mask{slots_.size() - 1};

    for (auto slot{hash & mask};; slot = (slot + 1) & mask)
    {
        const auto& [slot_hash, bytes_at, ids_at, length, count]{slots_[slot]};

        if (bytes_at == free_slot)
        {
            return slot;
        }

        if (slot_hash != hash || length != piece.size())
        {
            continue;
        }

        const auto stored{std::string_view{bytes_}.substr(bytes_at, length)};

        if (stored == piece)
        {
            return slot;
        }
    }
}

void Piece_cache::store(const std::string_view piece, std::vector<load::Token_id_t> ids)
{
    if (size_ == capacity_ || piece.size() > longest_cached_piece)
    {
        return;
    }

    if (2 * (size_ + 1) > slots_.size())
    {
        grow();
    }

    const auto hash{std::hash<std::string_view>{}(piece)};

    auto& slot{slots_[slot_of(piece, hash)]};

    if (slot.bytes_at != free_slot)
    {
        return;
    }

    slot = {.hash = hash,
            .bytes_at = bytes_.size(),
            .ids_at = ids_.size(),
            .length = static_cast<std::uint16_t>(piece.size()),
            .count = static_cast<std::uint16_t>(ids.size())};

    bytes_ += piece;

    ids_.insert(ids_.end(), ids.begin(), ids.end());

    ++size_;
}

void Piece_cache::grow()
{
    const auto slot_count{slots_.empty() ? first_slot_count : 2 * slots_.size()};

    const auto free_one{Slot{.hash = {}, .bytes_at = free_slot, .ids_at = {}, .length = {}, .count = {}}};

    const auto old{std::exchange(slots_, std::vector(slot_count, free_one))};

    const auto mask{slot_count - 1};

    for (const auto& stored : old)
    {
        if (stored.bytes_at == free_slot)
        {
            continue;
        }

        auto slot{stored.hash & mask};

        while (slots_[slot].bytes_at != free_slot)
        {
            slot = (slot + 1) & mask;
        }

        slots_[slot] = stored;
    }
}

} // namespace matisse::encode
