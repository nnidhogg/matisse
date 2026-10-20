#include "matisse/encode/piece_cache.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace matisse::encode
{
Piece_cache::Piece_cache(const std::size_t capacity) : capacity_{capacity}
{}

std::optional<std::span<const load::Token_id_t>> Piece_cache::find(const std::string_view piece) const
{
    const auto found{entries_.find(piece)};

    if (found == entries_.end())
    {
        return std::nullopt;
    }

    return found->second;
}

std::size_t Piece_cache::Piece_hash::operator()(const std::string_view piece) const noexcept
{
    return std::hash<std::string_view>{}(piece);
}

void Piece_cache::store(const std::string_view piece, std::vector<load::Token_id_t> ids)
{
    if (entries_.size() == capacity_ || piece.size() > longest_cached_piece)
    {
        return;
    }

    entries_.try_emplace(std::string{piece}, std::move(ids));
}

} // namespace matisse::encode
