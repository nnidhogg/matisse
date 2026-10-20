#include "matisse/encode/encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "matisse/encode/piece_cache.hpp"

namespace matisse::encode
{
namespace
{
/**
 * @brief An occurrence of an allowed special token in a text: where it begins and which token it is.
 */
struct Occurrence
{
    /**
     * @brief The byte at which the occurrence begins.
     */
    std::size_t at{};

    /**
     * @brief The token's index among the allowed special tokens.
     */
    std::size_t special{};
};

/**
 * @brief The search of a text for its allowed special tokens, front to back.
 *
 * Each token's next occurrence is kept until the search passes it, so that the text is read once for every token
 * however many of its occurrences the search finds.
 */
class Special_search
{
public:
    /**
     * @brief Finds each allowed special token's first occurrence in a text.
     * @param text The text, which outlives the search.
     * @param allowed The allowed special tokens, which outlive the search.
     */
    Special_search(std::string_view text, std::span<const load::Special_token> allowed);

    /**
     * @brief Returns the first occurrence of an allowed special token at or after a byte, the longest token among those
     *        that begin at one byte.
     * @param from The byte, at or after the byte of the previous call.
     * @return The occurrence, std::nullopt when no allowed special token occurs at or after the byte.
     */
    [[nodiscard]] std::optional<Occurrence> next(std::size_t from);

private:
    /**
     * @brief The text.
     */
    std::string_view text_;

    /**
     * @brief The allowed special tokens.
     */
    std::span<const load::Special_token> allowed_;

    /**
     * @brief The byte of each token's next occurrence, std::string_view::npos when it occurs no more.
     */
    std::vector<std::size_t> next_;
};

Special_search::Special_search(const std::string_view text, const std::span<const load::Special_token> allowed)
    : text_{text}, allowed_{allowed}, next_(allowed.size())
{
    for (std::size_t special{0}; special < allowed.size(); ++special)
    {
        next_[special] = text.find(allowed[special].spelling);
    }
}

std::optional<Occurrence> Special_search::next(const std::size_t from)
{
    std::optional<Occurrence> first{};

    for (std::size_t special{0}; special < allowed_.size(); ++special)
    {
        const auto& spelling{allowed_[special].spelling};

        if (next_[special] < from)
        {
            next_[special] = text_.find(spelling, from);
        }

        const auto at{next_[special]};

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

    return first;
}

} // namespace

Encoder::Encoder(
        load::Definition definition, const validate::Report& report, const validate::Inventory inventory,
        const pretok::Pattern& pattern, const std::span<const std::string> allowed)
    : definition_{std::move(definition)}
    , merger_{definition_}
    , segmenter_{pattern}
    , allowed_{allowed_of(definition_.specials, allowed)}
    , table_{inventory == validate::Inventory::factorization ? report.factorization : report.policy}
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

std::vector<load::Token_id_t> Encoder::encode(const std::string_view text) const
{
    std::vector<load::Token_id_t> ids{};

    Piece_cache cache{piece_cache_capacity};

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

    const auto encode_stretch{[&](const std::string_view stretch) {
        std::size_t start{0};

        for (const auto end : segmenter_.boundaries(stretch))
        {
            encode_piece(stretch.substr(start, end - start));

            start = end;
        }
    }};

    Special_search search{text, allowed_};

    std::size_t start{0};

    for (auto found{search.next(start)}; found; found = search.next(start))
    {
        const auto [at, special]{*found};

        const auto& [spelling, id]{allowed_[special]};

        encode_stretch(text.substr(start, at - start));

        ids.push_back(id);

        start = at + spelling.size();
    }

    encode_stretch(text.substr(start));

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
            throw std::invalid_argument{std::format(
                    "Encoder: the id {} at index {} is neither an ordinary nor a special token of the definition", id,
                    index)};
        }

        text += special->spelling;
    }

    return text;
}

} // namespace matisse::encode
