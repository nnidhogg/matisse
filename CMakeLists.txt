cmake_minimum_required(VERSION 3.20.6)
project(matisse VERSION 0.1.0 LANGUAGES CXX)


if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MATISSE_IS_TOP_LEVEL ON)
else ()
    set(MATISSE_IS_TOP_LEVEL OFF)
endif ()

# The default build type is matisse's own choice only when it is the top-level project; a parent keeps its own.
if (MATISSE_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# matisse's own root for test data paths; CMAKE_SOURCE_DIR names the embedding project's root under add_subdirectory().
set(MATISSE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# GCC-style flags; threading is carried by munch's targets, not by a global flag.
add_compile_options(-Wall -Wunused -Wextra)

# Warnings are errors for matisse's own development; a consumer's newer compiler must not break their build over a
# warning matisse has not seen yet.
option(MATISSE_WERROR "Treat warnings as errors" ${MATISSE_IS_TOP_LEVEL})
if (MATISSE_WERROR)
    add_compile_options(-Werror)
endif ()

add_compile_options(
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-O2>
)

# Developer-only targets stay out of consumer builds.
option(MATISSE_BUILD_TESTS "Build tests for matisse" ${MATISSE_IS_TOP_LEVEL})
option(USE_SYSTEM_GTEST "Use an installed googletest found with find_package instead of fetching it" OFF)
if (MATISSE_BUILD_TESTS)
    enable_testing()
endif ()

# The pinned vocabulary files, GPT-2's and cl100k_base's, are not in the tree; the loaders' tests read them from this
# directory, each held to its digest, and skip when it is not given.
set(MATISSE_VOCAB_DIR "" CACHE PATH "A directory holding the pinned vocabulary files the tests read")

# The three corpora of the test strategy are not in the tree either: a directory holding twitter.json,
# munch-python@15b2863c.txt and munch-cpp@15b2863c.txt, built as docs/benchmarks.md says, which the corpus probe holds
# to the fixtures' digests when it is given beside the vocabulary directory.
set(MATISSE_CORPUS_DIR "" CACHE PATH "A directory holding the three corpora of the test strategy")

include(GNUInstallDirs)

# Installing matisse means installing targets that link munch, and an exported target can only refer to munch as an
# installed package, so the install rules require the system munch; the default build uses the submodule and does not
# install.
option(MATISSE_SYSTEM_MUNCH "Use an installed munch found with find_package instead of the submodule" OFF)
option(MATISSE_INSTALL "Generate install rules for matisse (requires MATISSE_SYSTEM_MUNCH)" OFF)

if (MATISSE_INSTALL AND NOT MATISSE_SYSTEM_MUNCH)
    message(FATAL_ERROR
            "MATISSE_INSTALL needs MATISSE_SYSTEM_MUNCH=ON: the exported targets refer to munch's installed package")
endif ()

# Found here rather than under external/, because imported targets are visible only from the directory that finds
# them downward: every library under libs/ links munch, and every test beside them googletest.
if (MATISSE_SYSTEM_MUNCH)
    find_package(munch REQUIRED)
endif ()

if (MATISSE_BUILD_TESTS AND USE_SYSTEM_GTEST)
    find_package(GTest 1.12 REQUIRED CONFIG)
endif ()

add_subdirectory(external)
add_subdirectory(libs)
add_subdirectory(tools)

# The umbrella carries matisse's own libraries, the stages, every tokenizer and the lookup over them; munch arrives
# transitively through matisse_load.
add_library(${PROJECT_NAME}
        INTERFACE
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
add_library(${PROJECT_NAME}::load ALIAS matisse_load)
add_library(${PROJECT_NAME}::pretok ALIAS matisse_pretok)
add_library(${PROJECT_NAME}::validate ALIAS matisse_validate)
add_library(${PROJECT_NAME}::encode ALIAS matisse_encode)
add_library(${PROJECT_NAME}::gpt2 ALIAS matisse_gpt2)
add_library(${PROJECT_NAME}::cl100k ALIAS matisse_cl100k)
add_library(${PROJECT_NAME}::lookup ALIAS matisse_lookup)

target_link_libraries(${PROJECT_NAME}
        INTERFACE
        matisse_load
        matisse_pretok
        matisse_validate
        matisse_encode
        matisse_gpt2
        matisse_cl100k
        matisse_lookup
)

if (MATISSE_INSTALL)
    # The installed names are the aliases the build offers: matisse::load, matisse::pretok, matisse::validate,
    # matisse::encode, matisse::gpt2, matisse::cl100k, matisse::lookup, matisse::matisse.
    set_target_properties(matisse_load PROPERTIES EXPORT_NAME load)
    set_target_properties(matisse_pretok PROPERTIES EXPORT_NAME pretok)
    set_target_properties(matisse_validate PROPERTIES EXPORT_NAME validate)
    set_target_properties(matisse_encode PROPERTIES EXPORT_NAME encode)
    set_target_properties(matisse_gpt2 PROPERTIES EXPORT_NAME gpt2)
    set_target_properties(matisse_cl100k PROPERTIES EXPORT_NAME cl100k)
    set_target_properties(matisse_lookup PROPERTIES EXPORT_NAME lookup)

    install(TARGETS ${PROJECT_NAME} matisse_load matisse_pretok matisse_validate matisse_encode matisse_gpt2
            matisse_cl100k matisse_lookup matisse_validate_cli
            EXPORT matisse_targets)

    install(DIRECTORY libs/load/include/ libs/pretok/include/ libs/validate/include/ libs/encode/include/
            libs/gpt2/include/ libs/cl100k/include/ libs/lookup/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # The Unicode License v3 requires its complete notice to accompany copies of the generated class tables.
    install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/matisse)

    install(EXPORT matisse_targets
            FILE matisse-targets.cmake
            NAMESPACE matisse::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matisse
    )

    include(CMakePackageConfigHelpers)

    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/matisse-config-version.cmake
            COMPATIBILITY SameMajorVersion
    )

    install(FILES
            cmake/matisse-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/matisse-config-version.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matisse
    )
endif ()
