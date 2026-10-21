# The installed matisse package depends on munch, whose verifiers and merge tables every library reads through; munch is
# found as its own installed package. Boost's container_hash, which the encoder hashes with, is a build-time dependency
# that never appears in a public header, and everything else matisse uses is the standard library.
include(CMakeFindDependencyMacro)
find_dependency(munch)

include("${CMAKE_CURRENT_LIST_DIR}/matisse-targets.cmake")
