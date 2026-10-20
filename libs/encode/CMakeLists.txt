cmake_minimum_required(VERSION 3.20.6)
cmake_policy(SET CMP0097 NEW)
project(matisse_encode)


add_library(${PROJECT_NAME}
        src/encoder.cpp
        src/merger.cpp
        src/piece_cache.cpp
        src/window_index.cpp
)

# The exported target carries the standard it needs.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

# The build reads the headers from the tree; an installed consumer reads them from the prefix's include directory.
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# The encoder is built from a definition, a report and the pattern it segments by, and munch arrives through them.
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        matisse_load
        matisse_pretok
        matisse_validate
)

if (MATISSE_BUILD_TESTS)
    add_executable(${PROJECT_NAME}_tests
            tests/encoder_test.cpp
            tests/merger_test.cpp
            tests/piece_cache_test.cpp
            tests/window_index_test.cpp
    )

    target_link_libraries(${PROJECT_NAME}_tests
            PRIVATE
            ${PROJECT_NAME}
            matisse_common
            GTest::gtest_main
    )

    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)

    # The merge is held to the plain references on every vocabulary the loaders read, from outside the tree; the tests
    # that need them skip when none is named.
    target_compile_definitions(${PROJECT_NAME}_tests
            PRIVATE
            VOCAB_DIR="${MATISSE_VOCAB_DIR}"
    )
endif ()
