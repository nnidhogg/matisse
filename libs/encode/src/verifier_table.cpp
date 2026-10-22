#include "matisse/encode/verifier_table.hpp"

#include <cstddef>
#include <ranges>
#include <vector>

namespace matisse::encode
{
Verifier_table::Verifier_table(const munch::verifier::Verifier& verifier, const pretok::Alphabet& alphabet)
    : class_count_{alphabet.symbols.size()}
    , next_(verifier.state_count() * class_count_ * 2, no_state)
    , accepting_(verifier.state_count())
    , dead_ends_(verifier.state_count())
{
    for (munch::verifier::Verifier::State_t state{0}; state < verifier.state_count(); ++state)
    {
        auto moves{false};

        for (const auto [symbol_class, symbol] : std::views::enumerate(alphabet.symbols))
        {
            for (const auto boundary_after : {false, true})
            {
                const munch::verifier::Marked marked{
                        .byte = static_cast<unsigned char>(symbol),
                        .boundary_after = boundary_after};

                const auto next{verifier.step(state, marked)};

                if (!next)
                {
                    continue;
                }

                const auto at{(state * class_count_ + static_cast<std::size_t>(symbol_class)) * 2 + boundary_after};

                next_[at] = static_cast<State_t>(*next);

                moves = true;
            }
        }

        accepting_[state] = verifier.accepts(state);

        dead_ends_[state] = accepting_[state] && !moves;
    }
}

Verifier_table::State_t Verifier_table::start() const noexcept
{
    return 0;
}

Verifier_table::State_t Verifier_table::step(
        const State_t state, const pretok::Class_t symbol_class, const bool boundary_after) const noexcept
{
    const auto at{(state * class_count_ + symbol_class) * 2 + boundary_after};

    return next_[at];
}

bool Verifier_table::accepts(const State_t state) const noexcept
{
    return accepting_[state];
}

bool Verifier_table::is_dead_end(const State_t state) const noexcept
{
    return dead_ends_[state];
}

} // namespace matisse::encode
