#include "matisse/encode/join_table.hpp"

#include <algorithm>
#include <bit>
#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace matisse::encode
{
Join_table::Join_table(const Joins_t& joins)
    : mask_{slot_count_of(joins.size()) - 1}, slots_(mask_ + 1, Slot{.key = free_key, .join = {}})
{
    for (const auto& [key, join] : joins)
    {
        auto slot{home(key)};

        while (slots_[slot].key != free_key)
        {
            slot = (slot + 1) & mask_;
        }

        slots_[slot] = {.key = key, .join = join};
    }
}

std::size_t Join_table::slot_count_of(const std::size_t joins) noexcept
{
    const auto least{std::max(2 * joins, 1UZ)};

    return std::bit_ceil(least);
}

std::size_t Join_table::home(const std::uint64_t key) const noexcept
{
    std::size_t seed{};

    boost::hash_combine(seed, key);

    return seed & mask_;
}

std::optional<Join_table::Join> Join_table::find(
        const load::Token_id_t left, const load::Token_id_t right) const noexcept
{
    const auto key{key_of(left, right)};

    for (auto slot{home(key)};; slot = (slot + 1) & mask_)
    {
        const auto& [slot_key, join]{slots_[slot]};

        if (slot_key == free_key)
        {
            return std::nullopt;
        }

        if (slot_key == key)
        {
            return join;
        }
    }
}

} // namespace matisse::encode
