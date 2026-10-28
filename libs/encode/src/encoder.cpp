#include "matisse/encode/encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <format>
#include <functional>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "matisse/pretok/verifier.hpp"

namespace matisse::encode
{
namespace
{
/**
 * @brief Returns the window table of an inventory in a report.
 * @param report The report.
 * @param inventory The inventory.
 * @return The table.
 */
[[nodiscard]] const validate::Window_table& table_of(
        const validate::Report& report, const validate::Inventory inventory) noexcept
{
    return inventory == validate::Inventory::factorization ? report.factorization : report.policy;
}

/**
 * @brief Returns whether a byte continues a UTF-8 sequence rather than beginning one.
 * @param byte The byte.
 * @return True for a continuation byte, `10xxxxxx`.
 */
[[nodiscard]] constexpr bool is_continuation(const unsigned char byte) noexcept
{
    return (byte & 0xC0U) == 0x80U;
}

/**
 * @brief Returns the byte offset a number of code points after an offset of a UTF-8 text, or a limit when fewer stand
 *        before it.
 * @param text The text.
 * @param from The offset, at the start of a code point.
 * @param points The number of code points.
 * @param limit The offset not passed, at the start of a code point or at the text's end, at or after the offset.
 * @return The offset, at the start of a code point.
 */
[[nodiscard]] std::size_t byte_after(
        const std::string_view text, const std::size_t from, const std::size_t points, const std::size_t limit) noexcept
{
    auto byte{from};

    for (std::size_t point{0}; point < points && byte < limit; ++point)
    {
        ++byte;

        while (byte < limit && is_continuation(static_cast<unsigned char>(text[byte])))
        {
            ++byte;
        }
    }

    return byte;
}

/**
 * @brief The equal-division targets of a text, the offsets size * index / parts for index 1, 2 and so on, in turn.
 *
 * The product overflows for a large text divided very finely, and so does any form that multiplies the remainder by the
 * index. Adding the quotient each step and carrying the remainder when it fills a whole divisor yields exactly the same
 * offsets and multiplies nothing, the target never exceeding the size and the carry never reaching the divisor; this is
 * munch's Lexer::Division_targets, duplicated rather than shared.
 */
class Division_targets
{
public:
    /**
     * @brief Constructs the walk over a text divided into parts, positioned before the first target.
     * @param size The text's size.
     * @param parts The number of parts; zero leaves every step at zero.
     */
    constexpr Division_targets(const std::size_t size, const std::size_t parts) noexcept
        : step_{parts == 0 ? 0UZ : size / parts}, remainder_{parts == 0 ? 0UZ : size % parts}, parts_{parts}
    {}

    /**
     * @brief Advances to the next target.
     * @return The next equal-division offset.
     */
    constexpr std::size_t next() noexcept
    {
        target_ += step_;

        carry_ += remainder_;

        if (carry_ >= parts_)
        {
            ++target_;

            carry_ -= parts_;
        }

        return target_;
    }

private:
    /**
     * @brief The quotient of the size by the parts, added at every step.
     */
    std::size_t step_;

    /**
     * @brief The remainder of the size by the parts, accumulated into the carry at every step.
     */
    std::size_t remainder_;

    /**
     * @brief The number of parts, the divisor the carry is reduced by.
     */
    std::size_t parts_;

    /**
     * @brief The latest target, zero before the first step.
     */
    std::size_t target_{0};

    /**
     * @brief The accumulated remainders not yet carried into the target, always below the divisor.
     */
    std::size_t carry_{0};
};

/**
 * @brief Returns the first offset of a UTF-8 text at or after an offset at which a code point begins, or the text's
 *        end.
 * @param text The text.
 * @param at The offset, at most the text's size.
 * @return The offset.
 */
[[nodiscard]] std::size_t point_start(const std::string_view text, const std::size_t at) noexcept
{
    auto byte{at};

    while (byte < text.size() && is_continuation(static_cast<unsigned char>(text[byte])))
    {
        ++byte;
    }

    return byte;
}

} // namespace

Encoder::Encoder(
        load::Definition definition, const validate::Report& report, const validate::Inventory inventory,
        const std::span<const std::string> allowed)
    : definition_{std::move(definition)}
    , merger_{definition_}
    , segmenter_{report.pattern}
    , allowed_{allowed_of(definition_.specials, allowed)}
    , windows_{table_of(report, inventory)}
    , right_context_{table_of(report, inventory).right_context}
    , class_table_{report.pattern.alphabet}
    , verifier_{pretok::split_verifier(report.pattern), report.pattern.alphabet}
    , spellings_sorted_{sorted_of(definition_.spellings)}
    , unformed_sorted_{unformed_of(definition_, merger_, spellings_sorted_)}
{}

std::vector<load::Special_token> Encoder::allowed_of(
        const std::span<const load::Special_token> specials, const std::span<const std::string> allowed)
{
    std::vector<load::Special_token> chosen{};

    for (const auto& spelling : allowed)
    {
        const auto found{std::ranges::find(specials, spelling, &load::Special_token::spelling)};

        if (found == specials.end())
        {
            throw std::invalid_argument{std::format(
                    R"(Encoder: the allowed spelling "{}" is not one of the definition's special tokens)", spelling)};
        }

        chosen.push_back(*found);
    }

    return chosen;
}

std::vector<load::Token_id_t> Encoder::sorted_of(const std::span<const std::string> spellings)
{
    std::vector<load::Token_id_t> sorted(spellings.size());

    std::iota(sorted.begin(), sorted.end(), load::Token_id_t{0});

    const auto spelling_of{[spellings](const load::Token_id_t id) { return std::string_view{spellings[id]}; }};

    std::ranges::stable_sort(sorted, std::less{}, spelling_of);

    return sorted;
}

std::vector<load::Token_id_t> Encoder::unformed_of(
        const load::Definition& definition, const Merger& merger, const std::span<const load::Token_id_t> sorted)
{
    std::vector<load::Token_id_t> unformed{};

    if (!definition.whole_piece_lookup)
    {
        return unformed;
    }

    for (const auto id : sorted)
    {
        const auto& spelling{definition.spellings[id]};

        if (spelling.size() < 2)
        {
            continue;
        }

        const auto merged{merger.merged(spelling)};

        const auto forms_itself{merged.size() == 1 && merged.front() == id};

        if (!forms_itself)
        {
            unformed.push_back(id);
        }
    }

    return unformed;
}

std::vector<load::Token_id_t> Encoder::encode(const std::string_view text) const
{
    Piece_cache cache{*this, piece_cache_capacity};

    return encode(text, cache);
}

std::vector<load::Token_id_t> Encoder::encode(const std::string_view text, Piece_cache& cache) const
{
    if (!cache.serves(*this))
    {
        throw std::invalid_argument{"Encoder::encode: the piece cache was made for another encoder"};
    }

    const auto found{occurrences(text)};

    return encode_chunk(text, found, 0, text.size(), cache);
}

std::vector<Encoder::Occurrence> Encoder::occurrences(const std::string_view text) const
{
    // The byte of each token's next occurrence at or after the search, std::string_view::npos when it occurs no more.
    std::vector<std::size_t> next(allowed_.size());

    for (const auto [special, token] : std::views::enumerate(allowed_))
    {
        next[static_cast<std::size_t>(special)] = text.find(token.spelling);
    }

    std::vector<Occurrence> found{};

    std::size_t from{0};

    while (true)
    {
        std::optional<Occurrence> first{};

        for (std::size_t special{0}; special < allowed_.size(); ++special)
        {
            const auto& spelling{allowed_[special].spelling};

            if (next[special] < from)
            {
                next[special] = text.find(spelling, from);
            }

            const auto at{next[special]};

            if (at == std::string_view::npos)
            {
                continue;
            }

            const auto earlier{!first || at < first->at};

            const auto longer{first && at == first->at && spelling.size() > allowed_[first->special].spelling.size()};

            if (earlier || longer)
            {
                first = Occurrence{.at = at, .special = special};
            }
        }

        if (!first)
        {
            return found;
        }

        const auto [at, special]{*first};

        found.push_back(*first);

        from = at + allowed_[special].spelling.size();
    }
}

std::vector<load::Token_id_t> Encoder::encode_chunk(
        const std::string_view text, const std::span<const Occurrence> found, const std::size_t begin,
        const std::size_t end, Piece_cache& cache) const
{
    std::vector<load::Token_id_t> ids{};

    const auto append{
            [&ids](const std::span<const load::Token_id_t> more) { ids.insert(ids.end(), more.begin(), more.end()); }};

    const auto encode_piece{[&](const std::string_view piece) {
        if (const auto whole{merger_.whole_piece(piece)})
        {
            ids.push_back(*whole);

            return;
        }

        if (const auto cached{cache.find(piece)})
        {
            append(*cached);

            return;
        }

        auto merged{merger_.merged(piece)};

        append(merged);

        cache.store(piece, std::move(merged));
    }};

    // Encodes the pieces of a stretch, segmented as a complete input, that end at or before a byte of it.
    const auto encode_stretch{[&](const std::string_view stretch, const std::size_t kept) {
        std::size_t start{0};

        for (const auto piece_end : segmenter_.boundaries(stretch))
        {
            if (piece_end > kept)
            {
                return;
            }

            encode_piece(stretch.substr(start, piece_end - start));

            start = piece_end;
        }
    }};

    const auto before_chunk{[begin](const Occurrence& occurrence) { return occurrence.at < begin; }};

    auto next{std::ranges::partition_point(found, before_chunk)};

    std::size_t start{begin};

    for (; next != found.end() && next->at < end; ++next)
    {
        const auto [at, special]{*next};

        const auto& [spelling, id]{allowed_[special]};

        encode_stretch(text.substr(start, at - start), at - start);

        ids.push_back(id);

        start = at + spelling.size();
    }

    // The last stretch reads the right context past the chunk's end, unless a special token or the text's end comes
    // first.
    const auto stretch_end{next == found.end() ? text.size() : next->at};

    const auto context_end{byte_after(text, end, right_context_, stretch_end)};

    encode_stretch(text.substr(start, context_end - start), end - start);

    return ids;
}

std::vector<std::size_t> Encoder::chunk_boundaries(const std::string_view text, const std::size_t chunks) const
{
    const auto found{occurrences(text)};

    return boundaries_of(text, found, chunks);
}

std::vector<std::size_t> Encoder::boundaries_of(
        const std::string_view text, const std::span<const Occurrence> found, const std::size_t chunks) const
{
    std::vector<std::size_t> boundaries{0};

    // A chunk needs at least one byte, so asking for more chunks than bytes only adds targets that can find nothing.
    const auto usable{std::min(chunks, text.size())};

    Division_targets targets{text.size(), usable};

    for (std::size_t index{1}; index < usable; ++index)
    {
        const auto target{targets.next()};

        // Start at the target or one past the previous boundary, whichever is later, so that adjacent cuts each open a
        // chunk.
        const auto floor{std::max(target, boundaries.back() + 1)};

        const auto cut{next_cut(text, found, boundaries.back(), floor)};

        // A later target starts no earlier, so it finds no cut either.
        if (!cut)
        {
            break;
        }

        boundaries.push_back(*cut);
    }

    boundaries.push_back(text.size());

    return boundaries;
}

std::optional<std::size_t> Encoder::next_cut(
        const std::string_view text, const std::span<const Occurrence> found, const std::size_t previous,
        const std::size_t floor) const
{
    const auto end_of{[this](const Occurrence& occurrence) {
        return occurrence.at + allowed_[occurrence.special].spelling.size();
    }};

    const auto interior{[&text](const std::size_t cut) -> std::optional<std::size_t> {
        if (cut < text.size())
        {
            return cut;
        }

        return std::nullopt;
    }};

    // The first special token that ends after the floor: the floor is inside it, at its start, or before it.
    const auto ends_by_floor{[&](const Occurrence& occurrence) { return end_of(occurrence) <= floor; }};

    const auto next{std::ranges::partition_point(found, ends_by_floor)};

    if (next != found.end() && next->at == floor)
    {
        return floor;
    }

    if (next != found.end() && next->at < floor)
    {
        return interior(end_of(*next));
    }

    const auto stretch_begin{next == found.begin() ? 0UZ : end_of(*std::ranges::prev(next))};

    const auto stretch_end{next == found.end() ? text.size() : next->at};

    // A special token ends at the floor.
    if (floor == stretch_begin)
    {
        return interior(floor);
    }

    const auto spacing_from{std::max(previous, stretch_begin)};

    // The occurrence of the window that certifies an anchor begins at most the width less one code points before it.
    const auto spaced{byte_after(text, spacing_from, windows_.width() - 1, stretch_end)};

    const auto at_code_point{point_start(text, floor)};

    const auto from{std::max(at_code_point, spaced)};

    const auto stretch{text.substr(stretch_begin, stretch_end - stretch_begin)};

    if (const auto anchor{windows_.first_anchor(class_table_, stretch, from - stretch_begin)})
    {
        return stretch_begin + *anchor;
    }

    return interior(stretch_end);
}

std::vector<load::Token_id_t> Encoder::encode_parallel(const std::string_view text, const std::size_t chunks) const
{
    const auto found{occurrences(text)};

    const auto boundaries{boundaries_of(text, found, chunks)};

    const auto count{boundaries.size() - 1};

    std::vector<std::vector<load::Token_id_t>> encoded(count);

    std::mutex failure_mutex{};

    std::exception_ptr failure{};

    const auto encode_one{[&](const std::size_t chunk) {
        try
        {
            Piece_cache cache{*this, piece_cache_capacity};

            encoded[chunk] = encode_chunk(text, found, boundaries[chunk], boundaries[chunk + 1], cache);
        }
        catch (...)
        {
            const std::scoped_lock lock{failure_mutex};

            if (!failure)
            {
                failure = std::current_exception();
            }
        }
    }};

    {
        std::vector<std::jthread> workers{};

        workers.reserve(count - 1);

        for (std::size_t chunk{0}; chunk + 1 < count; ++chunk)
        {
            workers.emplace_back(encode_one, chunk);
        }

        encode_one(count - 1);
    }

    if (failure)
    {
        std::rethrow_exception(failure);
    }

    const auto sizes{encoded | std::views::transform(&std::vector<load::Token_id_t>::size)};

    const auto total{std::ranges::fold_left(sizes, 0UZ, std::plus{})};

    std::vector<load::Token_id_t> ids{};

    // The chunks' ids are copied once, into a vector that holds them all.
    ids.reserve(total);

    for (const auto& chunk_ids : encoded)
    {
        ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
    }

    return ids;
}

std::string Encoder::decode(const std::span<const load::Token_id_t> ids) const
{
    const auto& [merge_rule, whole_piece_lookup, spellings, merges, specials]{definition_};

    std::string text{};

    for (const auto [index, id] : std::views::enumerate(ids))
    {
        if (id < spellings.size())
        {
            text += spellings[id];

            continue;
        }

        const auto special{std::ranges::find(specials, id, &load::Special_token::id)};

        if (special == specials.end())
        {
            throw std::out_of_range{std::format(
                    "Encoder::decode: the id {} at index {} is neither an ordinary nor a special token of the "
                    "definition",
                    id, index)};
        }

        text += special->spelling;
    }

    return text;
}

const load::Definition& Encoder::definition() const noexcept
{
    return definition_;
}

const Merger& Encoder::merger() const noexcept
{
    return merger_;
}

std::span<const load::Special_token> Encoder::allowed() const noexcept
{
    return allowed_;
}

const pretok::Class_table& Encoder::class_table() const noexcept
{
    return class_table_;
}

const Verifier_table& Encoder::verifier() const noexcept
{
    return verifier_;
}

std::span<const load::Token_id_t> Encoder::spellings_sorted() const noexcept
{
    return spellings_sorted_;
}

std::span<const load::Token_id_t> Encoder::unformed_sorted() const noexcept
{
    return unformed_sorted_;
}

} // namespace matisse::encode
