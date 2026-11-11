if (MATISSE_BUILD_TESTS AND NOT USE_SYSTEM_GTEST)
    add_subdirectory(googletest)
endif ()

# munch is the submodule by default, pinned to the release matisse builds against; a build that will be installed
# takes an installed munch instead, found at the top level so its imported targets are visible to every library.
if (NOT MATISSE_SYSTEM_MUNCH)
    add_subdirectory(munch)
else ()
    # The encoder's join table hashes with Boost's container_hash, the header-only module munch fetches for its own
    # libraries as munch_boost; an installed munch does not carry it, so the submodule's fetch provides it here.
    add_subdirectory(munch/external/boost)
endif ()

# hopper's JSON parser reads tokenizer.json. hopper 1.0.0's top level adds its own munch submodule unless hopper is
# built against an installed munch, which would define munch's targets twice beside the submodule above, so its two
# libraries, parse and json, are added directly; both link munch::munch, the submodule's or the installed one.
add_subdirectory(hopper/libs/parse)
add_subdirectory(hopper/libs/json)
