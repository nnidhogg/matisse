// The extension module of matisse-tokenizer, matisse_tokenizer._core: matisse's loader, validator and encoder as Python
// sees them. Python's matisse_tokenizer package builds the tiktoken interface on top; this module only converts and
// calls, and every decision is matisse's. A text is read from a Python string as its UTF-8, which CPython keeps with
// the string, and every encoding runs with the interpreter lock released, so that threads encode at once.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "matisse/encode/encoder.hpp"
#include "matisse/encode/piece_cache.hpp"
#include "matisse/load/definition.hpp"
#include "matisse/lookup/tokenizers.hpp"
#include "matisse/validate/report.hpp"

namespace nb = nanobind;

namespace
{
/**
 * @brief The ids of a text.
 */
using Ids_t = std::vector<matisse::load::Token_id_t>;

/**
 * @brief The ids of a text as a NumPy array that owns them.
 */
using Id_array_t = nb::ndarray<nb::numpy, const matisse::load::Token_id_t, nb::ndim<1>>;

/**
 * @brief Returns the tokenizer matisse describes under a name.
 * @param name The name, gpt2, cl100k, o200k or llama3.
 * @return The tokenizer.
 * @throws std::invalid_argument If no tokenizer has the name.
 */
[[nodiscard]] matisse::lookup::Tokenizer tokenizer_of(const std::string_view name)
{
    auto tokenizer{matisse::lookup::tokenizer(name)};

    if (!tokenizer)
    {
        throw std::invalid_argument{std::format("matisse describes no tokenizer named '{}'", name)};
    }

    return std::move(*tokenizer);
}

/**
 * @brief Returns the UTF-8 of a Python string, which CPython keeps with the string for as long as it lives.
 * @param text The string.
 * @return The bytes, valid while the string lives.
 * @throws nb::python_error The UnicodeEncodeError CPython raises for a string holding a lone surrogate, which no UTF-8
 *         spells.
 */
[[nodiscard]] std::string_view utf8_of(const nb::str& text)
{
    Py_ssize_t size{};

    const auto* const bytes{PyUnicode_AsUTF8AndSize(text.ptr(), &size)};

    if (bytes == nullptr)
    {
        throw nb::python_error{};
    }

    return {bytes, static_cast<std::size_t>(size)};
}

/**
 * @brief Runs work with the interpreter lock released, taken again before it returns or throws.
 * @tparam Work The type of the work.
 * @param work The work, which touches no Python object.
 * @return What the work returns.
 */
template <typename Work>
[[nodiscard]] auto released(const Work& work)
{
    const nb::gil_scoped_release release{};

    return work();
}

/**
 * @brief Returns the ids of an iterable of Python integers, each read through its `__index__`, as tiktoken reads them.
 * @param tokens The iterable.
 * @return The ids.
 * @throws std::overflow_error If an integer is negative or does not fit 32 bits, as tiktoken's conversion refuses it.
 * @throws nb::python_error If the object is not iterable or an item is not an integer.
 */
[[nodiscard]] Ids_t ids_of(const nb::handle tokens)
{
    Ids_t ids{};

    for (const auto item : tokens)
    {
        const auto index{nb::steal(PyNumber_Index(item.ptr()))};

        if (!index.is_valid())
        {
            throw nb::python_error{};
        }

        const auto value{PyLong_AsLongLong(index.ptr())};

        if (value == -1 && PyErr_Occurred() != nullptr)
        {
            PyErr_Clear();

            throw std::overflow_error{"a token id is not an unsigned 32-bit integer"};
        }

        if (value < 0 || value > std::numeric_limits<matisse::load::Token_id_t>::max())
        {
            throw std::overflow_error{std::format("the token id {} is not an unsigned 32-bit integer", value)};
        }

        ids.push_back(static_cast<matisse::load::Token_id_t>(value));
    }

    return ids;
}

/**
 * @brief Returns ids as a NumPy array that owns them, with no copy.
 * @param ids The ids.
 * @return The array.
 */
[[nodiscard]] Id_array_t array_of(Ids_t ids)
{
    auto owned{std::make_unique<Ids_t>(std::move(ids))};

    const auto data{owned->data()};

    const auto size{owned->size()};

    const nb::capsule owner{owned.release(), [](void* held) noexcept { delete static_cast<Ids_t*>(held); }};

    return Id_array_t{data, {size}, owner};
}

/**
 * @brief Returns the vendor's special tokens of a rank file read under a tokenizer's pattern, as its library supplies
 *        them.
 * @param tokenizer The tokenizer's name.
 * @return The special tokens, each a spelling and an id, by id ascending.
 * @throws std::invalid_argument If no tokenizer has the name.
 */
[[nodiscard]] std::vector<std::pair<std::string, matisse::load::Token_id_t>> vendor_specials(
        const std::string_view tokenizer)
{
    const auto described{tokenizer_of(tokenizer)};

    std::vector<std::pair<std::string, matisse::load::Token_id_t>> specials{};

    for (const auto& [spelling, id] : described.specials)
    {
        specials.emplace_back(spelling, id);
    }

    return specials;
}

/**
 * @brief Reads a tiktoken rank file with special tokens given as spellings and ids.
 * @param path The rank file's path.
 * @param specials The special tokens, each a spelling and an id, by id ascending.
 * @return The definition.
 * @throws std::invalid_argument If the file is not a rank file read_tiktoken() takes.
 * @throws std::runtime_error If the file cannot be read.
 */
[[nodiscard]] matisse::load::Definition read_tiktoken(
        const std::string& path, const std::vector<std::pair<std::string, matisse::load::Token_id_t>>& specials)
{
    std::vector<matisse::load::Special_token> tokens{};

    for (const auto& [spelling, id] : specials)
    {
        tokens.push_back({.spelling = spelling, .id = id});
    }

    return released([&] { return matisse::load::read_tiktoken(path, tokens); });
}

/**
 * @brief Decides the report of a tokenizer's pattern.
 * @param tokenizer The tokenizer's name.
 * @param width The widest window decided.
 * @return The report.
 * @throws std::invalid_argument If no tokenizer has the name, or validate() refuses the width.
 */
[[nodiscard]] matisse::validate::Report validate(const std::string_view tokenizer, const std::size_t width)
{
    const auto described{tokenizer_of(tokenizer)};

    return released([&] { return matisse::validate::validate(described.pattern, width); });
}

/**
 * @brief Reads back the report of a tokenizer's pattern from the text serialize() wrote of it.
 * @param tokenizer The tokenizer's name.
 * @param text The text.
 * @return The report.
 * @throws std::invalid_argument If no tokenizer has the name, or the text is not serialize()'s of a report of a pattern
 *         with the same class symbols.
 */
[[nodiscard]] matisse::validate::Report deserialize(const std::string_view tokenizer, const nb::bytes& text)
{
    const auto described{tokenizer_of(tokenizer)};

    const std::string_view view{text.c_str(), text.size()};

    return released([&] { return matisse::validate::deserialize(view, described.pattern); });
}

/**
 * @brief Returns a definition's special tokens, each its spelling and its id.
 * @param definition The definition.
 * @return The special tokens, in the definition's order.
 */
[[nodiscard]] std::vector<std::pair<std::string, matisse::load::Token_id_t>> specials_of(
        const matisse::load::Definition& definition)
{
    std::vector<std::pair<std::string, matisse::load::Token_id_t>> specials{};

    for (const auto& [spelling, id] : definition.specials)
    {
        specials.emplace_back(spelling, id);
    }

    return specials;
}

/**
 * @brief Returns the bytes of every ordinary token of a definition, by id.
 * @param definition The definition.
 * @return The bytes, a list of bytes objects.
 */
[[nodiscard]] nb::list spellings_of(const matisse::load::Definition& definition)
{
    nb::list spellings{};

    for (const auto& spelling : definition.spellings)
    {
        spellings.append(nb::bytes{spelling.data(), spelling.size()});
    }

    return spellings;
}

/**
 * @brief Constructs an encoder in place, with the interpreter lock released while it builds its tables.
 * @param encoder Where the encoder is constructed.
 * @param definition The definition, copied.
 * @param report The report of the definition's pattern.
 * @param allowed The spellings of the special tokens cut out of a text.
 * @throws std::invalid_argument If the encoder refuses the definition or an allowed spelling.
 */
void construct_encoder(
        matisse::encode::Encoder* encoder, const matisse::load::Definition& definition,
        const matisse::validate::Report& report, const std::vector<std::string>& allowed)
{
    const nb::gil_scoped_release release{};

    new (encoder) matisse::encode::Encoder{definition, report, matisse::validate::Inventory::factorization, allowed};
}

/**
 * @brief Constructs a piece cache in place for an encoder, holding as many pieces as the encoder's own caches.
 * @param cache Where the cache is constructed.
 * @param encoder The encoder, which the Python object keeps alive for as long as the cache.
 */
void construct_cache(matisse::encode::Piece_cache* cache, const matisse::encode::Encoder& encoder)
{
    new (cache) matisse::encode::Piece_cache{encoder, matisse::encode::piece_cache_capacity};
}

/**
 * @brief Returns the ids of a text, with a piece cache the caller keeps when one is given.
 * @param encoder The encoder.
 * @param text The text.
 * @param cache The cache, made for this encoder, or none.
 * @return The ids.
 * @throws std::invalid_argument If the cache was made for another encoder.
 * @throws nb::python_error If the text holds a lone surrogate.
 */
[[nodiscard]] Ids_t encode_ids(
        const matisse::encode::Encoder& encoder, const nb::str& text, matisse::encode::Piece_cache* cache)
{
    const auto view{utf8_of(text)};

    if (cache == nullptr)
    {
        return released([&] { return encoder.encode(view); });
    }

    return released([&] { return encoder.encode(view, *cache); });
}

/**
 * @brief Returns the ids of a text as a NumPy array, with a piece cache the caller keeps when one is given.
 * @param encoder The encoder.
 * @param text The text.
 * @param cache The cache, made for this encoder, or none.
 * @return The ids.
 * @throws std::invalid_argument If the cache was made for another encoder.
 * @throws nb::python_error If the text holds a lone surrogate.
 */
[[nodiscard]] Id_array_t encode_array(
        const matisse::encode::Encoder& encoder, const nb::str& text, matisse::encode::Piece_cache* cache)
{
    return array_of(encode_ids(encoder, text, cache));
}

/**
 * @brief Returns the ids of a text encoded in concurrent chunks cut at certified cuts, which are encode()'s.
 * @param encoder The encoder.
 * @param text The text.
 * @param chunks The number of chunks aimed for, one thread each.
 * @return The ids.
 * @throws nb::python_error If the text holds a lone surrogate.
 */
[[nodiscard]] Ids_t encode_parallel_ids(
        const matisse::encode::Encoder& encoder, const nb::str& text, const std::size_t chunks)
{
    const auto view{utf8_of(text)};

    return released([&] { return encoder.encode_parallel(view, chunks); });
}

/**
 * @brief Returns the ids of a text encoded in concurrent chunks as a NumPy array.
 * @param encoder The encoder.
 * @param text The text.
 * @param chunks The number of chunks aimed for, one thread each.
 * @return The ids.
 * @throws nb::python_error If the text holds a lone surrogate.
 */
[[nodiscard]] Id_array_t encode_parallel_array(
        const matisse::encode::Encoder& encoder, const nb::str& text, const std::size_t chunks)
{
    return array_of(encode_parallel_ids(encoder, text, chunks));
}

/**
 * @brief Returns the ids of every text of a batch, encode() called on each by a pool of threads that take the next text
 *        from a shared counter, the calling thread one of them, each thread keeping one piece cache across the texts it
 *        takes, as matisse's bench encodes a batch.
 * @param encoder The encoder.
 * @param texts The texts.
 * @param threads The threads of the pool, at least one.
 * @return The ids of each text, in the texts' order.
 * @throws nb::python_error If an item is not a string or holds a lone surrogate.
 * @throws std::invalid_argument If a text is refused, the first refusal rethrown once every thread has joined.
 */
[[nodiscard]] std::vector<Ids_t> encode_batch(
        const matisse::encode::Encoder& encoder, const nb::list& texts, const std::size_t threads)
{
    // The strings are held here, so that the bytes read survive a change to the list while the lock is released.
    std::vector<nb::str> held{};

    std::vector<std::string_view> views{};

    for (const auto item : texts)
    {
        const auto& text{held.emplace_back(nb::cast<nb::str>(item))};

        views.push_back(utf8_of(text));
    }

    return released([&] {
        std::vector<Ids_t> encoded(views.size());

        std::atomic<std::size_t> next{0};

        std::mutex failure_mutex{};

        std::exception_ptr failure{};

        const auto work{[&] {
            try
            {
                matisse::encode::Piece_cache cache{encoder, matisse::encode::piece_cache_capacity};

                for (auto index{next.fetch_add(1)}; index < views.size(); index = next.fetch_add(1))
                {
                    encoded[index] = encoder.encode(views[index], cache);
                }
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

            for (std::size_t worker{1}; worker < threads && worker < views.size(); ++worker)
            {
                workers.emplace_back(work);
            }

            work();
        }

        if (failure)
        {
            std::rethrow_exception(failure);
        }

        return encoded;
    });
}

/**
 * @brief Returns the bytes a run of ids spells, raw.
 * @param encoder The encoder.
 * @param tokens The ids, an iterable of integers.
 * @return The bytes.
 * @throws nb::key_error If an id is neither an ordinary nor a special token, as tiktoken raises KeyError.
 * @throws std::overflow_error If an id is negative or does not fit 32 bits.
 */
[[nodiscard]] nb::bytes decode_bytes(const matisse::encode::Encoder& encoder, const nb::handle tokens)
{
    const auto ids{ids_of(tokens)};

    try
    {
        const auto bytes{released([&] { return encoder.decode(ids); })};

        return nb::bytes{bytes.data(), bytes.size()};
    }
    catch (const std::out_of_range& error)
    {
        throw nb::key_error(error.what());
    }
}

} // namespace

NB_MODULE(_core, module)
{
    module.doc() = "matisse's loader, validator and encoder, as matisse_tokenizer calls them.";

    nb::class_<matisse::load::Definition>(module, "Definition", "A tokenizer's definition, as a loader read it.")
            .def_prop_ro(
                    "ordinary_count",
                    [](const matisse::load::Definition& definition) { return definition.spellings.size(); },
                    "The number of ordinary tokens, whose ids are dense from zero.")
            .def_prop_ro("specials", &specials_of, "The special tokens, each a spelling and an id.")
            .def("spellings", &spellings_of, "The bytes of every ordinary token, by id.");

    nb::class_<matisse::validate::Report>(module, "Report", "The windows of a pattern that certify a cut.")
            .def_prop_ro(
                    "width", [](const matisse::validate::Report& report) { return report.factorization.width; },
                    "The widest window decided.")
            .def("serialize", &matisse::validate::serialize, "The report as the text deserialize() reads back.");

    nb::class_<matisse::encode::Encoder>(module, "Encoder", "matisse's encoder under the factorization inventory.")
            .def("__init__", &construct_encoder, nb::arg("definition"), nb::arg("report"), nb::arg("allowed"))
            .def("encode", &encode_ids, nb::arg("text"), nb::arg("cache").none(), "The ids of a text.")
            .def("encode_to_numpy", &encode_array, nb::arg("text"), nb::arg("cache").none(), "The ids as an array.")
            .def("encode_parallel", &encode_parallel_ids, nb::arg("text"), nb::arg("chunks"),
                 "The ids of a text encoded in chunks at certified cuts.")
            .def("encode_parallel_to_numpy", &encode_parallel_array, nb::arg("text"), nb::arg("chunks"),
                 "The ids of a text encoded in chunks at certified cuts, as an array.")
            .def("encode_batch", &encode_batch, nb::arg("texts"), nb::arg("threads"),
                 "The ids of every text, encoded by a pool of threads.")
            .def("decode_bytes", &decode_bytes, nb::arg("tokens"), "The bytes a run of ids spells.");

    nb::class_<matisse::encode::Piece_cache>(module, "Piece_cache", "A piece cache one thread keeps across its calls.")
            .def("__init__", &construct_cache, nb::arg("encoder"), nb::keep_alive<1, 2>());

    module.def(
            "vendor_specials", &vendor_specials, nb::arg("tokenizer"),
            "The vendor's special tokens of a rank file read under a tokenizer's pattern.");
    module.def(
            "read_tiktoken", &read_tiktoken, nb::arg("path"), nb::arg("specials"),
            "Reads a rank file with the special tokens given.");
    module.def(
            "validate", &validate, nb::arg("tokenizer"), nb::arg("width"),
            "Decides the report of a tokenizer's pattern.");
    module.def(
            "deserialize", &deserialize, nb::arg("tokenizer"), nb::arg("text"),
            "Reads back the report of a tokenizer's pattern from its serialized text.");
}
