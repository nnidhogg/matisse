#ifndef MATISSE_TOOLS_FUZZ_INCLUDE_MATISSE_TOOLS_FUZZ_HARNESS_HPP
#define MATISSE_TOOLS_FUZZ_INCLUDE_MATISSE_TOOLS_FUZZ_HARNESS_HPP

#include <filesystem>
#include <span>
#include <string>
#include <string_view>

/**
 * @file
 * @brief What the fuzz harnesses share: the invariant check, require(), the fuzz input written to the file a loader
 *        reads, written(), and the check every definition a loader returns passes, require_spelled_apart().
 */
namespace matisse::tools::fuzz
{
/**
 * @brief Aborts on a violated invariant, which the fuzzer reports as a crash on this input.
 * @param condition The invariant.
 */
void require(bool condition);

/**
 * @brief Writes bytes to a file of the process's own directory under the system's temporary directory, since the
 *        loaders read paths; the directory is made on the first call and removed at exit.
 * @param name The file's name, the same file overwritten by every call that names it.
 * @param bytes The file's bytes.
 * @return The file's path.
 */
[[nodiscard]] std::filesystem::path written(std::string_view name, std::string_view bytes);

/**
 * @brief Requires the spellings of a definition a loader returned to be what both loaders promise of the tokens they
 *        read: none empty and no two alike, since each loader refuses an empty token and a spelling listed twice.
 * @param spellings The bytes of every ordinary token, by id.
 */
void require_spelled_apart(std::span<const std::string> spellings);

} // namespace matisse::tools::fuzz

#endif // MATISSE_TOOLS_FUZZ_INCLUDE_MATISSE_TOOLS_FUZZ_HARNESS_HPP
