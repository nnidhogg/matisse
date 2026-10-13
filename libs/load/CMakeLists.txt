cmake_minimum_required(VERSION 3.20.6)
cmake_policy(SET CMP0097 NEW)
project(matisse_load)


add_library(${PROJECT_NAME}
        src/definition.cpp
)

# The exported target carries the standard it needs.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

# The build reads the headers from the tree; an installed consumer reads them from the prefix's include directory.
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# munch is header-visible: a definition holds its merge table as munch::verifier::Merge, so consumers link it too.
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        munch::munch_verifier
)

if (MATISSE_BUILD_TESTS)
    add_executable(${PROJECT_NAME}_tests
            tests/definition_test.cpp
    )

    target_link_libraries(${PROJECT_NAME}_tests
            PRIVATE
            ${PROJECT_NAME}
            GTest::gtest_main
    )

    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)
endif ()
