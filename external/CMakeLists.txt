if (MATISSE_BUILD_TESTS AND NOT USE_SYSTEM_GTEST)
    add_subdirectory(googletest)
endif ()

# munch is the submodule by default, pinned to the release matisse builds against; a build that will be installed
# takes an installed munch instead, found at the top level so its imported targets are visible to every library.
if (NOT MATISSE_SYSTEM_MUNCH)
    add_subdirectory(munch)
endif ()
