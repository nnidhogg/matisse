cmake_minimum_required(VERSION 3.20.6)
cmake_policy(SET CMP0097 NEW)
project(matisse_lookup)


add_library(${PROJECT_NAME}
        src/tokenizers.cpp
)

# The exported target carries the standard it needs.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

# The build reads the headers from the tree; an installed consumer reads them from the prefix's include directory.
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# A tokenizer is a pattern for pretok and special tokens for load, so consumers link both; the lookup lists every
# tokenizer library, which only its source names.
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        matisse_load
        matisse_pretok
        PRIVATE
        matisse_gpt2
        matisse_cl100k
)

if (MATISSE_BUILD_TESTS)
    add_executable(${PROJECT_NAME}_tests
            tests/tokenizers_test.cpp
    )

    target_link_libraries(${PROJECT_NAME}_tests
            PRIVATE
            ${PROJECT_NAME}
            GTest::gtest_main
    )

    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)
endif ()
