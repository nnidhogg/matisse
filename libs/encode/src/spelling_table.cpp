#include "matisse/encode/spelling_table.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace matisse::encode
{
Spelling_table::Spelling_table(const std::span<const std::string> spellings)
    : mask_{slot_count_of(spellings.size()) - 1}
    , slots_(mask_ + 1, Slot{.hash = {}, .bytes_at = {}, .length = {}, .id = free_id})
{
    for (std::size_t id{0}; id < spellings.size(); ++id)
    {
        const std::string_view spelling{spellings[id]};

        const auto hash{std::hash<std::string_view>{}(spelling)};

        auto& slot{slots_[slot_of(spelling, hash)]};

        if (slot.id != free_id)
        {
            continue;
        }

        slot = {.hash = hash,
                .bytes_at = bytes_.size(),
                .length = static_cast<std::uint32_t>(spelling.size()),
                .id = static_cast<load::Token_id_t>(id)};

        bytes_ += spelling;
    }
}

std::size_t Spelling_table::slot_count_of(const std::size_t spellings) noexcept
{
    const auto least{std::max(2 * spellings, 1UZ)};

    return std::bit_ceil(least);
}

std::size_t Spelling_table::slot_of(const std::string_view spelling, const std::size_t hash) const noexcept
{
    for (auto slot{hash & mask_};; slot = (slot + 1) & mask_)
    {
        const auto& [slot_hash, bytes_at, length, id]{slots_[slot]};

        if (id == free_id)
        {
            return slot;
        }

        if (slot_hash != hash || length != spelling.size())
        {
            continue;
        }

        const auto stored{std::string_view{bytes_}.substr(bytes_at, length)};

        if (stored == spelling)
        {
            return slot;
        }
    }
}

std::optional<load::Token_id_t> Spelling_table::find(const std::string_view spelling) const noexcept
{
    const auto hash{std::hash<std::string_view>{}(spelling)};

    const auto& [slot_hash, bytes_at, length, id]{slots_[slot_of(spelling, hash)]};

    if (id == free_id)
    {
        return std::nullopt;
    }

    return id;
}

} // namespace matisse::encode
