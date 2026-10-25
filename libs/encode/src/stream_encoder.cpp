#include "matisse/encode/stream_encoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace matisse::encode
{
namespace
{
/**
 * @brief Returns the length of the UTF-8 sequence a lead byte begins, read from its high bits alone.
 * @param lead The lead byte of a well-formed sequence.
 * @return One to four.
 */
[[nodiscard]] constexpr std::size_t sequence_length(const unsigned char lead) noexcept
{
    if (lead < 0x80U)
    {
        return 1;
    }

    if (lead < 0xE0U)
    {
        return 2;
    }

    if (lead < 0xF0U)
    {
        return 3;
    }

    return 4;
}

/**
 * @brief Returns the bytes of a text before a UTF-8 sequence its last bytes begin and do not complete, a lead byte of a
 *        well-formed sequence followed by fewer continuation bytes than it calls for.
 * @param bytes The text.
 * @return The length of the text less that sequence, the whole text when it ends at a code point.
 */
[[nodiscard]] std::size_t complete_length(const std::string_view bytes) noexcept
{
    const auto from{bytes.size() < 4 ? 0UZ : bytes.size() - 3};

    for (auto at{bytes.size()}; at > from; --at)
    {
        const auto byte{static_cast<unsigned char>(bytes[at - 1])};

        const auto continues{(byte & 0xC0U) == 0x80U};

        if (continues)
        {
            continue;
        }

        const auto lead_at{at - 1};

        const auto leads{byte >= 0xC2U && byte <= 0xF4U};

        const auto incomplete{leads && lead_at + sequence_length(byte) > bytes.size()};

        return incomplete ? lead_at : bytes.size();
    }

    return bytes.size();
}

/**
 * @brief Returns whether a run of ids has a boundary at a byte: whether the bytes of some prefix of it end there.
 * @param spellings The bytes of every ordinary token, by id.
 * @param ids The ids.
 * @param at The byte.
 * @return True when a prefix of the ids spells exactly the bytes before it.
 */
[[nodiscard]] bool has_boundary(
        const std::span<const std::string> spellings, const std::span<const load::Token_id_t> ids,
        const std::size_t at) noexcept
{
    std::size_t end{0};

    for (const auto id : ids)
    {
        end += spellings[id].size();

        if (end >= at)
        {
            return end == at;
        }
    }

    return at == 0;
}

/**
 * @brief Returns the first of a run of an encoder's ordinary tokens, in the order of their spellings, whose spelling is
 *        not before a string.
 * @param encoder The encoder.
 * @param sorted The ids searched, in the order of their spellings.
 * @param key The string.
 * @return Where the token stands among the sorted ids, their end when every spelling is before the string.
 */
[[nodiscard]] std::span<const load::Token_id_t>::iterator first_not_before(
        const Encoder& encoder, const std::span<const load::Token_id_t> sorted, const std::string_view key)
{
    const auto& spellings{encoder.definition().spellings};

    const auto spelling_of{[&spellings](const load::Token_id_t id) { return std::string_view{spellings[id]}; }};

    return std::ranges::lower_bound(sorted, key, std::less{}, spelling_of);
}

/**
 * @brief Returns whether a string is an ordinary token of an encoder's definition.
 * @param encoder The encoder.
 * @param text The string.
 * @return True when some token spells it.
 */
[[nodiscard]] bool is_token(const Encoder& encoder, const std::string_view text)
{
    const auto sorted{encoder.spellings_sorted()};

    const auto found{first_not_before(encoder, sorted, text)};

    return found != sorted.end() && encoder.definition().spellings[*found] == text;
}

/**
 * @brief Returns whether some ordinary token the merge does not form begins with a string.
 * @param encoder The encoder.
 * @param text The string.
 * @return True when the spelling of such a token has the string as a prefix, itself included.
 */
[[nodiscard]] bool begins_unformed(const Encoder& encoder, const std::string_view text)
{
    const auto unformed{encoder.unformed_sorted()};

    const auto found{first_not_before(encoder, unformed, text)};

    return found != unformed.end() && encoder.definition().spellings[*found].starts_with(text);
}

/**
 * @brief Returns whether no token can straddle a boundary of a piece's merge: no token is a suffix of the token before
 *        the boundary that is itself a token, followed by a nonempty prefix of the bytes after it or by all of them and
 *        more.
 *
 * The first join of any merge of the piece and a continuation that crosses the boundary joins a part ending there, a
 * token that is a suffix of the one before the boundary, to a part beginning there, so with no such token no join
 * crosses it, whatever follows the piece. Every suffix that is a token is tried, which holds every part of the right
 * spine of the token before the boundary.
 * @param encoder The encoder.
 * @param piece The piece's bytes.
 * @param at The boundary, the byte after the token before it.
 * @param before The token before the boundary.
 * @return True when no token straddles the boundary.
 */
[[nodiscard]] bool no_token_straddles(
        const Encoder& encoder, const std::string_view piece, const std::size_t at, const load::Token_id_t before)
{
    const auto& spellings{encoder.definition().spellings};

    const auto sorted{encoder.spellings_sorted()};

    const std::string_view left{spellings[before]};

    const auto after{piece.substr(at)};

    for (std::size_t length{1}; length <= left.size(); ++length)
    {
        const auto suffix{left.substr(left.size() - length)};

        if (!is_token(encoder, suffix))
        {
            continue;
        }

        std::string reach{suffix};

        reach += after;

        // The tokens that begin with the suffix and then the bytes after the boundary, one more byte each time.
        for (auto taken{after.empty() ? 0UZ : 1UZ}; taken <= after.size(); ++taken)
        {
            const auto candidate{std::string_view{reach}.substr(0, length + taken)};

            auto found{first_not_before(encoder, sorted, candidate)};

            if (taken == 0 && found != sorted.end() && spellings[*found] == candidate)
            {
                ++found;
            }

            if (found == sorted.end() || !spellings[*found].starts_with(candidate))
            {
                break;
            }

            const auto spells_it{spellings[*found] == candidate};

            if (spells_it || taken == after.size())
            {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Returns whether a boundary of a piece's merge is kept by the merge of the piece followed by every continuation
 *        a token can straddle the piece's end with: the empty one, and the rest of every token that begins with a
 *        nonempty suffix of the piece.
 *
 * By the projection of the merge every continuation's merge begins with the merge of the piece followed by one of these
 * up to its first boundary at or after the piece's end, so the boundary holds for every continuation exactly when it
 * holds for them. The test stops at the first that crosses it.
 * @param encoder The encoder.
 * @param piece The piece's bytes.
 * @param at The boundary.
 * @param whole_piece Whether a piece that is a token is looked up whole.
 * @return True when every one of them keeps the boundary.
 */
[[nodiscard]] bool straddlers_keep(
        const Encoder& encoder, const std::string_view piece, const std::size_t at, const bool whole_piece)
{
    const auto& spellings{encoder.definition().spellings};

    const auto sorted{encoder.spellings_sorted()};

    const auto keeps{[&](const std::string_view text) {
        if (whole_piece)
        {
            if (const auto whole{encoder.merger().whole_piece(text)})
            {
                const std::array single{*whole};

                return has_boundary(spellings, single, at);
            }
        }

        const auto merged{encoder.merger().merged(text)};

        return has_boundary(spellings, merged, at);
    }};

    if (!keeps(piece))
    {
        return false;
    }

    std::string continued{};

    for (std::size_t start{0}; start < piece.size(); ++start)
    {
        const auto suffix{piece.substr(start)};

        for (auto found{first_not_before(encoder, sorted, suffix)}; found != sorted.end(); ++found)
        {
            const std::string_view token{spellings[*found]};

            if (!token.starts_with(suffix))
            {
                break;
            }

            if (token.size() == suffix.size())
            {
                continue;
            }

            continued.assign(piece);

            continued += token.substr(suffix.size());

            if (!keeps(continued))
            {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Returns the ids of a piece that every byte continuation of it keeps: the ids of its merge up to the last
 *        boundary that no continuation crosses.
 *
 * The boundaries every continuation keeps are closed downward, so the first that holds from the right is the last one.
 * Where pieces are looked up whole, no id is stable while a token the merge does not form begins with the piece, since
 * a continuation can then make the whole piece that one token.
 * @param encoder The encoder.
 * @param piece The piece's bytes.
 * @param whole_piece Whether a piece that is a token is looked up whole.
 * @return The ids, a prefix of the piece's merge.
 */
[[nodiscard]] std::vector<load::Token_id_t> stable_ids(
        const Encoder& encoder, const std::string_view piece, const bool whole_piece)
{
    if (piece.empty() || (whole_piece && begins_unformed(encoder, piece)))
    {
        return {};
    }

    const auto& spellings{encoder.definition().spellings};

    const auto merged{encoder.merger().merged(piece)};

    std::vector<std::size_t> ends{};

    std::size_t end{0};

    for (const auto id : merged)
    {
        end += spellings[id].size();

        ends.push_back(end);
    }

    for (auto kept{merged.size()}; kept > 0; --kept)
    {
        const auto at{ends[kept - 1]};

        const auto stable{
                no_token_straddles(encoder, piece, at, merged[kept - 1]) ||
                straddlers_keep(encoder, piece, at, whole_piece)};

        if (stable)
        {
            return {merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(kept)};
        }
    }

    return {};
}

} // namespace

Stream_encoder::Stream_encoder(const Encoder& encoder)
    : encoder_{encoder}, runs_{encoder.verifier()}, cache_{piece_cache_capacity}, closed_end_{0}
{}

std::vector<load::Token_id_t> Stream_encoder::push(const std::string_view bytes)
{
    std::vector<load::Token_id_t> ids{};

    if (held_.empty())
    {
        const auto taken{read(bytes, false, ids)};

        held_.assign(bytes.substr(taken));
    }
    else
    {
        held_ += bytes;

        const auto taken{read(held_, false, ids)};

        held_.erase(0, taken);
    }

    emit(ids);

    return ids;
}

std::size_t Stream_encoder::read(const std::string_view text, const bool ended, std::vector<load::Token_id_t>& ids)
{
    const auto allowed{encoder_.allowed()};

    const auto spelling_sizes{
            allowed | std::views::transform([](const load::Special_token& token) { return token.spelling.size(); })};

    const auto longest{std::ranges::fold_left(
            spelling_sizes, 0UZ, [](const std::size_t a, const std::size_t b) { return std::max(a, b); })};

    // The earliest byte from which the rest of the text is a nonempty proper prefix of an allowed spelling.
    const auto held_from{[&](const std::size_t from) -> std::optional<std::size_t> {
        const auto back_from{text.size() < longest ? 0UZ : text.size() - longest + 1};

        const auto earliest{std::max(from, back_from)};

        for (auto at{earliest}; at < text.size(); ++at)
        {
            const auto rest{text.substr(at)};

            const auto begins{[rest](const load::Special_token& token) {
                return rest.size() < token.spelling.size() && token.spelling.starts_with(rest);
            }};

            if (std::ranges::any_of(allowed, begins))
            {
                return at;
            }
        }

        return std::nullopt;
    }};

    // The byte of each token's next occurrence at or after the search, std::string_view::npos when it occurs no more.
    std::vector<std::size_t> next(allowed.size());

    for (const auto [special, token] : std::views::enumerate(allowed))
    {
        next[static_cast<std::size_t>(special)] = text.find(token.spelling);
    }

    std::size_t from{0};

    while (true)
    {
        std::optional<std::size_t> first{};

        for (std::size_t special{0}; special < allowed.size(); ++special)
        {
            const auto& spelling{allowed[special].spelling};

            if (next[special] < from)
            {
                next[special] = text.find(spelling, from);
            }

            const auto at{next[special]};

            if (at == std::string_view::npos)
            {
                continue;
            }

            const auto earlier{!first || at < next[*first]};

            const auto longer{first && at == next[*first] && spelling.size() > allowed[*first].spelling.size()};

            if (earlier || longer)
            {
                first = special;
            }
        }

        const auto held{ended ? std::nullopt : held_from(from)};

        const auto stretch_end{first ? next[*first] : text.size()};

        const auto limit{held ? std::min(*held, stretch_end) : stretch_end};

        const auto completes{ended || limit < text.size()};

        const auto stretch{text.substr(from, limit - from)};

        from += feed(stretch, completes, ids);

        const auto commits{first && !(held && *held <= next[*first])};

        if (!commits)
        {
            return from;
        }

        const auto& [spelling, id]{allowed[*first]};

        flush(ids);

        ids.push_back(id);

        from = next[*first] + spelling.size();
    }
}

std::size_t Stream_encoder::feed(const std::string_view bytes, const bool ended, std::vector<load::Token_id_t>& ids)
{
    const auto length{ended ? bytes.size() : complete_length(bytes)};

    const auto points{bytes.substr(0, length)};

    const auto classes{encoder_.class_table().classes(points)};

    std::size_t at{0};

    for (const auto symbol_class : classes)
    {
        const auto size{sequence_length(static_cast<unsigned char>(points[at]))};

        stretch_ += points.substr(at, size);

        at += size;

        runs_.step(symbol_class, closed_end_ + stretch_.size());

        while (const auto end{runs_.next_closed()})
        {
            close_piece(*end, ids);
        }
    }

    return length;
}

void Stream_encoder::close_piece(const std::size_t end, std::vector<load::Token_id_t>& ids)
{
    if (end <= closed_end_)
    {
        return;
    }

    const auto length{end - closed_end_};

    const auto piece{std::string_view{stretch_}.substr(0, length)};

    if (returned_.empty())
    {
        append_ids(piece, ids);

        stretch_.erase(0, length);

        closed_end_ = end;

        return;
    }

    std::vector<load::Token_id_t> closed{};

    append_ids(piece, closed);

    if (returned_.size() < closed.size())
    {
        const auto unreturned{std::span{closed}.subspan(returned_.size())};

        ids.insert(ids.end(), unreturned.begin(), unreturned.end());

        returned_.clear();
    }
    else
    {
        const auto returned_of_piece{static_cast<std::ptrdiff_t>(closed.size())};

        returned_.erase(returned_.begin(), returned_.begin() + returned_of_piece);
    }

    stretch_.erase(0, length);

    closed_end_ = end;
}

void Stream_encoder::append_ids(const std::string_view piece, std::vector<load::Token_id_t>& ids)
{
    if (const auto whole{encoder_.merger().whole_piece(piece)})
    {
        ids.push_back(*whole);

        return;
    }

    if (const auto cached{cache_.find(piece)})
    {
        ids.insert(ids.end(), cached->begin(), cached->end());

        return;
    }

    auto merged{encoder_.merger().merged(piece)};

    ids.insert(ids.end(), merged.begin(), merged.end());

    cache_.store(piece, std::move(merged));
}

void Stream_encoder::flush(std::vector<load::Token_id_t>& ids)
{
    for (const auto end : runs_.accepted())
    {
        close_piece(end, ids);
    }

    close_piece(closed_end_ + stretch_.size(), ids);

    runs_.reset();

    stretch_.clear();

    closed_end_ = 0;

    returned_.clear();
}

void Stream_encoder::emit(std::vector<load::Token_id_t>& ids)
{
    const auto& spellings{encoder_.definition().spellings};

    const auto sizes{returned_ | std::views::transform([&spellings](const load::Token_id_t id) {
                         return spellings[id].size();
                     })};

    const auto returned_bytes{std::ranges::fold_left(sizes, 0UZ, std::plus{})};

    std::optional<std::vector<load::Token_id_t>> common{};

    for (const auto& marking : runs_.markings())
    {
        const auto listed{run_ids(marking, returned_bytes)};

        if (!common)
        {
            common = listed;

            continue;
        }

        const auto [ours, theirs]{std::ranges::mismatch(*common, listed)};

        common->erase(ours, common->end());
    }

    if (!common || common->size() <= returned_.size())
    {
        return;
    }

    const auto unreturned{std::span{*common}.subspan(returned_.size())};

    ids.insert(ids.end(), unreturned.begin(), unreturned.end());

    returned_ = std::move(*common);
}

std::vector<load::Token_id_t> Stream_encoder::run_ids(
        const Live_runs::Marking& marking, const std::size_t returned_bytes)
{
    const auto& [ends, dead_end]{marking};

    const std::string_view stretch{stretch_};

    std::vector<load::Token_id_t> listed{};

    const auto append{[&listed](const std::span<const load::Token_id_t> more) {
        listed.insert(listed.end(), more.begin(), more.end());
    }};

    auto start{closed_end_};

    for (const auto end : ends)
    {
        if (end <= start)
        {
            continue;
        }

        const auto piece{stretch.substr(start - closed_end_, end - start)};

        append_ids(piece, listed);

        start = end;
    }

    const auto open{stretch.substr(start - closed_end_)};

    if (open.empty())
    {
        return listed;
    }

    if (dead_end)
    {
        append_ids(open, listed);

        return listed;
    }

    const auto covered{closed_end_ + returned_bytes};

    if (covered <= start)
    {
        const auto whole_piece{encoder_.definition().whole_piece_lookup};

        const auto stable{stable_ids(encoder_, open, whole_piece)};

        append(stable);

        return listed;
    }

    // The ids returned reach into the open piece, and every continuation keeps the boundary they end at, so the merge
    // of the rest after it is the rest of every continuation's merge, with no whole-piece lookup.
    const auto returned_of_open{std::span{returned_}.subspan(listed.size())};

    append(returned_of_open);

    const auto tail{stretch.substr(covered - closed_end_)};

    const auto stable{stable_ids(encoder_, tail, false)};

    append(stable);

    return listed;
}

std::vector<load::Token_id_t> Stream_encoder::finish()
{
    std::vector<load::Token_id_t> ids{};

    const auto taken{read(held_, true, ids)};

    held_.erase(0, taken);

    flush(ids);

    return ids;
}

} // namespace matisse::encode
