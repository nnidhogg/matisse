#ifndef MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP
#define MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <munch/verifier/merge_tower.hpp>

/**
 * @file
 * @brief The loaders, read_gpt2(), read_tiktoken() and read_tokenizer_json(), each reading a real tokenizer's files
 *        into one Definition: its merge rule and whether it looks a whole piece up, the bytes of every ordinary token
 *        by id, the merge table in rank order and the special tokens; and tokenizer_json_spelling(), which names the
 *        pattern a tokenizer.json is read under.
 *
 * A definition is held in raw bytes throughout, with GPT-2's byte-to-character map undone by the loader. Content a
 * loader cannot take exactly is refused with std::invalid_argument naming the item and where it is, ids that are not
 * dense from zero, an empty token and a spelling a file lists twice among them, and a file that cannot be read throws
 * std::runtime_error. A tiktoken rank file names neither its pattern nor its special tokens, so the caller gives the
 * vendor's special tokens, which the tokenizer library of the pattern supplies; the pattern itself is the tokenizer
 * library's description, which the stages take beside the definition. A tokenizer.json names both, and its reader takes
 * the published spelling of the pattern, which the tokenizer library also gives, and recognizes the file's
 * pre-tokenizer by it.
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
     * @brief The merge table's pairs, one join at a time: the adjacent pair of least rank, at its leftmost occurrence,
     *        as Hugging Face's BPE model merges.
     *
     * encoder.py's loop joins every occurrence of the least pair in one pass instead. The two give the same tokens on
     * every piece of a table in which no part of a pair is formed by a pair of a later rank, each pass then being a run
     * of single joins, and can differ on a table that is not: under (ab, a) then (a, b) the pass gives ab ab on abab
     * and the single joins aba b. read_gpt2() refuses such a table, so on every file it reads the rule is encoder.py's.
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
     * @brief The text the token stands for, UTF-8 as every vendor's special token is, which the Encoder requires.
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
     * @brief Whether a piece that is itself an ordinary token is that one token, looked up before any merge.
     *
     * tiktoken looks every piece up and Hugging Face's BPE model does when its `ignore_merges` is true; encoder.py
     * never does. A piece spelling a token that the merge rule would not form from the piece's bytes alone is one id
     * with the lookup and several without it.
     */
    bool whole_piece_lookup{};

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
 * encoder.json is read as the one flat object it is, each token's spelling under GPT-2's byte-to-character map, with
 * JSON's escapes, and its id a JSON number that Python's json reads as a non-negative integer, `-?(0|[1-9][0-9]*)` with
 * no fraction and no exponent, so that `-0` is zero as Python reads it; vocab.bpe, UTF-8 throughout as encoder.py's
 * text-mode read requires, as a version line and then one merge per line, its two parts separated by the white space
 * Python's str.split() splits at, the ASCII white space, U+001C to U+001F and the rest of Unicode's White_Space, the
 * text after the last newline dropped as encoder.py drops it. The byte map is undone, so "Ġthe" is read as " the". The
 * definition has the pair rule, no whole-piece lookup, and no special tokens, since encoder.py reads `<|endoftext|>` as
 * text. The merge table is checked once: no pair is listed twice, and no part of a merge is formed by a merge of a
 * later rank, so that the pair rule is encoder.py's loop on it.
 * @param encoder The path of encoder.json.
 * @param vocab The path of vocab.bpe.
 * @return The definition.
 * @throws std::runtime_error If a file cannot be read.
 * @throws std::invalid_argument If encoder.json is not a flat object of spellings and ids, a spelling is empty, is not
 *         UTF-8 or holds a character the byte map does not carry, the ids are not dense from zero, two tokens have one
 *         spelling, vocab.bpe is not UTF-8 throughout, a merge line is not two parts of such characters, a pair is
 *         listed twice, or a part of a merge is formed by a merge of a later rank; the message names the file and the
 *         byte offset, the token, both ids or the line, or the merge, its rank and its line.
 */
[[nodiscard]] Definition read_gpt2(const std::filesystem::path& encoder, const std::filesystem::path& vocab);

/**
 * @brief Reads a tiktoken rank file, one base64 token and its rank per line, as tiktoken's load_tiktoken_bpe() reads
 *        it, with the special tokens the caller gives.
 *
 * The rank of a token is its id, and empty lines are skipped, as tiktoken skips them, where a line of white space alone
 * is refused, as tiktoken refuses it. The file names neither its pattern nor its special tokens, so the special tokens
 * are the vendor's for the encoding, which the tokenizer library supplies, quoted from tiktoken's openai_public.py. The
 * definition has the rank rule with the whole-piece lookup, which tiktoken makes of every piece before it merges, and
 * no merge table.
 * @param path The path of the rank file.
 * @param specials The vendor's special tokens, by id ascending.
 * @return The definition.
 * @throws std::runtime_error If the file cannot be read.
 * @throws std::invalid_argument If a line is not a base64 token and a decimal rank written as digits alone, the ranks
 *         are not dense from zero, or two tokens have one spelling; the message names the file and the line or the
 *         ranks.
 */
[[nodiscard]] Definition read_tiktoken(const std::filesystem::path& path, std::span<const Special_token> specials);

/**
 * @brief Reads a Hugging Face tokenizer.json with a BPE model, as Hugging Face's tokenizers encodes with it, its
 *        pre-tokenizer recognized only by the published spelling of the pattern the file is read under.
 *
 * Three pre-tokenizers are recognized, each with `add_prefix_space` false: a ByteLevel with its regex, which is GPT-2's
 * pattern as encoder.py spells it, when that is the published spelling; a Sequence of a Split on the published spelling
 * followed by a ByteLevel without its regex, the Split isolated and not inverted, or removed and inverted, which keeps
 * the same pieces of a total pattern, as every pattern matisse describes is; and for an empty spelling a ByteLevel
 * without its regex alone, which splits nothing, for a merge table read without a pattern. A null pre-tokenizer is none
 * of them, since Hugging Face's BPE then merges the text's characters and not its bytes. A Split on a spelling that
 * holds an interval followed by a plus is refused, since Oniguruma, Hugging Face's engine, reads `{1,3}+` as the
 * interval repeated and not as possessive, so cl100k's published pattern is read otherwise than it is published; the
 * cl100k conversions on the Hugging Face Hub, Xenova/gpt-4 among them, carry its spelling from before the possessive
 * rewrite, which is Llama 3's pattern.
 *
 * The vocabulary's spellings and the merges' parts are read with GPT-2's byte map undone, as the ByteLevel
 * pre-tokenizer maps the bytes. The `added_tokens` marked special are the special tokens, by id ascending, and the
 * ordinary tokens are the vocabulary's other entries. Hugging Face reads no added token's own id but gives it one: its
 * id in the vocabulary when its spelling is there, and otherwise the vocabulary's size, or one past the greatest id
 * given before it in the file's order when that is no less, so a token whose id is another is refused. The definition
 * has the pair rule, which is Hugging Face's own, with the merges in the file's order, each a string of two parts
 * separated by a space or an array of two strings, and not every part of a merge need be formed before it, as
 * read_gpt2() requires of its files; the whole-piece lookup is the model's `ignore_merges`, false when it is absent. Its ids are
 * those Hugging Face's encode() gives with `add_special_tokens` false: the tokens a post-processor adds under
 * `add_special_tokens` are the caller's to add. The model's `unk_token` is not read, since the encoder refuses a
 * definition in which a byte is no token, and under one in which every byte is a token no text holds an unknown
 * character.
 * @param path The path of tokenizer.json.
 * @param published The published spelling of the pattern the file is read under, as the pattern's tokenizer library
 *        gives it, or empty for a pre-tokenizer that splits nothing.
 * @return The definition.
 * @throws std::runtime_error If the file cannot be read.
 * @throws std::invalid_argument If the file is not JSON, its model is not BPE or has dropout, a continuing-subword
 *         prefix, an end-of-word suffix or byte fallback, its normalizer, truncation or padding is not null, its
 *         pre-tokenizer is not one of the three under the published spelling or its pattern holds an interval followed
 *         by a plus or is empty, its post-processor is not null, ByteLevel, TemplateProcessing or a Sequence of them,
 *         an added token is not special, strips white space, matches single words or has another id than the one
 *         Hugging Face gives it, the ordinary ids are not dense from zero, a token is empty or holds a character the
 *         byte map does not carry, two tokens have one spelling, a merge is not two parts, or a pair is listed twice;
 *         the message names the member of the file and what it holds, or the ids or the tokens.
 */
[[nodiscard]] Definition read_tokenizer_json(const std::filesystem::path& path, std::string_view published);

/**
 * @brief Returns the published spelling of the pattern a tokenizer.json's pre-tokenizer runs, as read_tokenizer_json()
 *        recognizes it, for a caller that finds the tokenizer library of a file by the file itself: GPT-2's for a
 *        ByteLevel with its regex, the Split's for a Split followed by a ByteLevel without its regex, and empty for a
 *        ByteLevel without its regex alone.
 * @param path The path of tokenizer.json.
 * @return The spelling.
 * @throws std::runtime_error If the file cannot be read.
 * @throws std::invalid_argument If the file is not JSON, or its pre-tokenizer is none of the three or its pattern holds
 *         an interval followed by a plus; the message names the member of the file and what it holds.
 */
[[nodiscard]] std::string tokenizer_json_spelling(const std::filesystem::path& path);

} // namespace matisse::load

#endif // MATISSE_LIBS_LOAD_INCLUDE_MATISSE_LOAD_DEFINITION_HPP
