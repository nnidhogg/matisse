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
