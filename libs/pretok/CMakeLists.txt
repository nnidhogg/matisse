cmake_minimum_required(VERSION 3.20.6)
cmake_policy(SET CMP0097 NEW)
project(matisse_pretok)


add_library(${PROJECT_NAME}
        src/classes.cpp
        src/segmenter.cpp
)

# The exported target carries the standard it needs.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

# The build reads the headers from the tree; an installed consumer reads them from the prefix's include directory.
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# munch is header-visible: the segmenter runs munch's DFAs and a pattern's verifier is a munch::verifier::Verifier, so
# consumers link it too. The umbrella target carries the pattern parser, which reads the alternatives, and the core,
# which compiles them.
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        matisse_load
        munch::munch
)

if (MATISSE_BUILD_TESTS)
    add_executable(${PROJECT_NAME}_tests
            tests/classes_test.cpp
            tests/segmenter_test.cpp
    )

    target_link_libraries(${PROJECT_NAME}_tests
            PRIVATE
            ${PROJECT_NAME}
            GTest::gtest_main
    )

    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)

    target_compile_definitions(${PROJECT_NAME}_tests PRIVATE SOURCE_DIR="${MATISSE_SOURCE_DIR}")
endif ()
