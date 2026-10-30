#include "matisse/tools/fuzz/harness.hpp"

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>

namespace matisse::tools::fuzz
{
namespace
{
/**
 * @brief The process's own directory under the system's temporary directory, made when it is built and removed with
 *        everything in it when it is destroyed, at exit.
 */
class Scratch
{
public:
    /**
     * @brief Makes the directory, named by the process's id so that the jobs of one fuzzing run never share it.
     */
    Scratch() : directory_{std::filesystem::temp_directory_path() / std::format("matisse-fuzz-{}", ::getpid())}
    {
        std::filesystem::create_directories(directory_);
    }

    /**
     * @brief A scratch is not copied: the directory is removed once, by the one scratch that made it.
     */
    Scratch(const Scratch& other) = delete;

    /**
     * @brief A scratch is not moved, as for the copy.
     */
    Scratch(Scratch&& other) = delete;

    /**
     * @brief A scratch is not assigned to, as for the copy.
     */
    Scratch& operator=(const Scratch& other) = delete;

    /**
     * @brief A scratch is not assigned to, as for the copy.
     */
    Scratch& operator=(Scratch&& other) = delete;

    /**
     * @brief Removes the directory and its files; a failure to remove them leaves them behind and is not reported.
     */
    ~Scratch()
    {
        std::error_code ignored{};

        std::filesystem::remove_all(directory_, ignored);
    }

    /**
     * @brief Returns the directory.
     * @return Its path.
     */
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }

private:
    /**
     * @brief The directory's path.
     */
    std::filesystem::path directory_;
};

} // namespace

void require(const bool condition)
{
    if (!condition)
    {
        std::abort();
    }
}

std::filesystem::path written(const std::string_view name, const std::string_view bytes)
{
    static const Scratch scratch{};

    const auto path{scratch.directory() / name};

    std::ofstream stream{path, std::ios::binary | std::ios::trunc};

    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    stream.close();

    // A file the harness cannot write is the harness's own failure, never the loader's answer.
    require(!stream.fail());

    return path;
}

void require_spelled_apart(const std::span<const std::string> spellings)
{
    std::set<std::string_view> seen{};

    for (const auto& spelling : spellings)
    {
        require(!spelling.empty());

        require(seen.insert(spelling).second);
    }
}

} // namespace matisse::tools::fuzz
