cmake_minimum_required(VERSION 3.20.6)
cmake_policy(SET CMP0097 NEW)
project(matisse_gpt2)


add_library(${PROJECT_NAME}
        src/tokenizer.cpp
)

# The exported target carries the standard it needs.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

# The build reads the headers from the tree; an installed consumer reads them from the prefix's include directory.
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# The description is a pattern for pretok and special tokens for load, so consumers link both.
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        matisse_load
        matisse_pretok
)

if (MATISSE_BUILD_TESTS)
    add_executable(${PROJECT_NAME}_tests
            tests/classes_test.cpp
            tests/definition_test.cpp
            tests/report_test.cpp
            tests/segmenter_test.cpp
            tests/verifier_test.cpp
    )

    target_link_libraries(${PROJECT_NAME}_tests
            PRIVATE
            ${PROJECT_NAME}
            matisse_common
            matisse_validate
            GTest::gtest_main
    )

    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)

    # The fixtures are shared, every tokenizer's universe in one file per subject, and the pinned vocabularies are read
    # from outside the tree; the tests that need them skip when none is named.
    target_compile_definitions(${PROJECT_NAME}_tests
            PRIVATE
            SOURCE_DIR="${MATISSE_SOURCE_DIR}"
            VOCAB_DIR="${MATISSE_VOCAB_DIR}"
    )
endif ()
