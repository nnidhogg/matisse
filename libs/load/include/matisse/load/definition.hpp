#ifndef MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP
#define MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <munch/verifier/merge_tower.hpp>

/**
 * @file
 * @brief The loaders, read_gpt2() and read_tiktoken(), each reading a real tokenizer's files into one Definition: its
 *        merge rule, the bytes of every ordinary token by id, the merge table in rank order and the special tokens.
 *
 * A definition is held in raw bytes throughout, with GPT-2's byte-to-character map undone by the loader. Content a
 * loader cannot take exactly is refused with std::invalid_argument naming the item and where it is, ids that are not
 * dense from zero and an empty token among them, and a file that cannot be read throws std::runtime_error. A tiktoken
 * rank file names neither its pattern nor its special tokens, so the caller gives the vendor's special tokens, which
 * the tokenizer library of the pattern supplies; the pattern itself is the tokenizer library's description, which the
 * stages take beside the definition. read_tokenizer_json(), the Hugging Face reader, is not declared yet;
 * docs/design.md sketches it.
 */
namespace matisse::load
{
/**
 * @brief A token id, what a model reads; 32 bits, which every vocabulary in scope fits.
 */
using Token_id_t = std::uint32_t;

/**
 * @brief How the bytes of a piece are merged into tokens.
 */
enum class Merge_rule : std::uint8_t
{
    /**
     * @brief The merge table's pairs in rank order, encoder.py's loop and Hugging Face's BPE model.
     */
    pairs,

    /**
     * @brief tiktoken's rank merge, which joins the adjacent pair whose concatenation has the least rank first and so
     *        needs the ranks only.
     */
    ranks
};

/**
 * @brief A special token: the text that stands for it and its id.
 */
struct Special_token
{
    /**
     * @brief The text the token stands for.
     */
    std::string spelling{};

    /**
     * @brief The token's id.
     */
    Token_id_t id{};
};

/**
 * @brief A real tokenizer's definition, held in raw bytes.
 */
struct Definition
{
    /**
     * @brief How the bytes of a piece are merged.
     */
    Merge_rule merge_rule{};

    /**
     * @brief The bytes of every ordinary token, by id, dense from zero.
     */
    std::vector<std::string> spellings{};

    /**
     * @brief The merge table in rank order under Merge_rule::pairs, empty under Merge_rule::ranks.
     */
    std::vector<munch::verifier::Merge> merges{};

    /**
     * @brief The special tokens, each with its id.
     */
    std::vector<Special_token> specials{};
};

/**
 * @brief Reads GPT-2's encoder.json and vocab.bpe as openai/gpt-2's src/encoder.py reads them.
 *
 * encoder.json is read as the one flat object it is, each token's spelling under GPT-2's byte-to-character map and its
 * id, with JSON's escapes; vocab.bpe as a version line and then one merge per line, its two parts separated by white
 * space, the text after the last newline dropped as encoder.py drops it. The byte map is undone, so "Ġthe" is read as "
 * the". The definition has the pair rule and no special tokens, since encoder.py reads `<|endoftext|>` as text.
 * @param encoder The path of encoder.json.
 * @param vocab The path of vocab.bpe.
 * @return The definition.
 * @throws std::runtime_error If a file cannot be read.
 * @throws std::invalid_argument If encoder.json is not a flat object of spellings and ids, a spelling is empty or holds
 *         a character the byte map does not carry, the ids are not dense from zero, or a merge line is not two parts of
 *         such characters; the message names the file and the byte offset, the token or the line.
 */
[[nodiscard]] Definition read_gpt2(const std::filesystem::path& encoder, const std::filesystem::path& vocab);

/**
 * @brief Reads a tiktoken rank file, one base64 token and its rank per line, as tiktoken's load_tiktoken_bpe() reads
 *        it, with the special tokens the caller gives.
 *
 * The rank of a token is its id, and empty lines are skipped, as tiktoken skips them. The file names neither its
 * pattern nor its special tokens, so the special tokens are the vendor's for the encoding, which the tokenizer library
 * supplies, quoted from tiktoken's openai_public.py. The definition has the rank rule and no merge table.
 * @param path The path of the rank file.
 * @param specials The vendor's special tokens, by id ascending.
 * @return The definition.
 * @throws std::runtime_error If the file cannot be read.
 * @throws std::invalid_argument If a line is not a base64 token and a decimal rank, or the ranks are not dense from
 *         zero; the message names the file and the line.
 */
[[nodiscard]] Definition read_tiktoken(const std::filesystem::path& path, std::span<const Special_token> specials);

} // namespace matisse::load

#endif // MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP
